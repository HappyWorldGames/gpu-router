package main

import (
	"bytes"
	"errors"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeBackend поднимает тестовый сервер и возвращает host, port и список заголовков Authorization,
// которые он получил.
type seenAuth struct {
	mu   sync.Mutex
	auth []string
	xkey []string
	path []string
}

func (s *seenAuth) add(r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.auth = append(s.auth, r.Header.Get("Authorization"))
	s.xkey = append(s.xkey, r.Header.Get("X-Api-Key"))
	s.path = append(s.path, r.URL.Path)
}

func fakeBackend(t *testing.T, status int, body string) (host, port string, seen *seenAuth) {
	t.Helper()
	seen = &seenAuth{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen.add(r)
		w.WriteHeader(status)
		w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	h, p, _ := net.SplitHostPort(strings.TrimPrefix(srv.URL, "http://"))
	return h, p, seen
}

func authServer(t *testing.T, mutate func(*Config)) *Server {
	t.Helper()
	cfg := &Config{}
	if mutate != nil {
		mutate(cfg)
	}
	applyDefaults(cfg)
	s := NewServer(cfg)
	if err := s.loadAuth(); err != nil {
		t.Fatal(err)
	}
	return s
}

func TestResolveSecret(t *testing.T) {
	dir := t.TempDir()
	kf := filepath.Join(dir, "key")
	os.WriteFile(kf, []byte("  from-file-key \nsecond line ignored\n"), 0o600)
	empty := filepath.Join(dir, "empty")
	os.WriteFile(empty, []byte("\n"), 0o600)
	t.Setenv("GPU_TEST_KEY", " from-env ")
	t.Setenv("GPU_TEST_EMPTY", "")

	ok := map[string]string{
		"":                 "",
		"plain-key":        "plain-key",
		"file:" + kf:       "from-file-key",
		"env:GPU_TEST_KEY": "from-env",
	}
	for in, want := range ok {
		got, _, err := resolveSecret(in)
		if err != nil || got != want {
			t.Errorf("%q -> %q (%v), want %q", in, got, err, want)
		}
	}
	for _, bad := range []string{"file:" + filepath.Join(dir, "nope"), "file:" + empty, "file:", "env:", "env:GPU_TEST_EMPTY", "env:GPU_TEST_UNSET"} {
		if _, _, err := resolveSecret(bad); err == nil {
			t.Errorf("%q: ожидалась ошибка", bad)
		}
	}
	// описание источника не содержит сам ключ
	_, src, _ := resolveSecret("file:" + kf)
	if strings.Contains(src, "from-file-key") {
		t.Fatal("источник не должен содержать ключ")
	}
}

func TestLoadAuthValidation(t *testing.T) {
	// проверка у клиентов без единого ключа — ошибка настройки
	s := NewServer(func() *Config {
		c := &Config{}
		c.Proxy.LlamaRequireKey = true
		applyDefaults(c)
		return c
	}())
	if err := s.loadAuth(); err == nil {
		t.Fatal("llama_require_key без ключей должен давать ошибку")
	}
	// недоступный файл — ошибка, а не молчаливая работа без ключа
	s = NewServer(func() *Config {
		c := &Config{}
		c.LLAMA.APIKey = "file:/nonexistent/llama.key"
		applyDefaults(c)
		return c
	}())
	if err := s.loadAuth(); err == nil {
		t.Fatal("несуществующий файл с ключом должен давать ошибку")
	}
}

func TestKeySentOnlyToLlama(t *testing.T) {
	lh, lp, lseen := fakeBackend(t, 200, `{"data":[]}`)
	ch, cp, cseen := fakeBackend(t, 200, `{}`)
	s := authServer(t, func(c *Config) {
		c.LLAMA.Host, c.LLAMA.Port, c.LLAMA.APIKey = lh, lp, "REAL-LLAMA-KEY"
		c.ComfyUI.Host, c.ComfyUI.Port = ch, cp
	})
	if _, err := s.llamaLoaded(); err != nil {
		t.Fatal(err)
	}
	s.call("POST", s.Config.llamaURL("/models/unload"), map[string]string{"model": "m"}, time.Second)
	s.call("POST", s.Config.comfyURL("/free"), map[string]interface{}{}, time.Second)
	s.call("GET", s.Config.comfyURL("/queue"), nil, time.Second)

	for _, a := range lseen.auth {
		if a != "Bearer REAL-LLAMA-KEY" {
			t.Fatalf("llama получила %q", a)
		}
	}
	if len(lseen.auth) != 2 {
		t.Fatalf("запросов к llama: %d", len(lseen.auth))
	}
	for _, a := range cseen.auth {
		if a != "" {
			t.Fatalf("ключ llama ушёл в ComfyUI: %q", a)
		}
	}
	if len(cseen.auth) != 2 {
		t.Fatalf("запросов к ComfyUI: %d", len(cseen.auth))
	}
}

func TestExecBackendSendsKey(t *testing.T) {
	lh, lp, seen := fakeBackend(t, 200, `{"ok":1}`)
	s := authServer(t, func(c *Config) { c.LLAMA.Host, c.LLAMA.Port, c.LLAMA.APIKey = lh, lp, "REAL-LLAMA-KEY" })
	task := newTask("llama", "m")
	task.Path, task.Body = "/v1/chat/completions", []byte(`{}`)
	task.W = httptest.NewRecorder()
	s.execBackend(task)
	if len(seen.auth) != 1 || seen.auth[0] != "Bearer REAL-LLAMA-KEY" {
		t.Fatalf("execBackend отправил %v", seen.auth)
	}

	// без ключа в конфиге заголовок не добавляется
	lh2, lp2, seen2 := fakeBackend(t, 200, `{}`)
	s2 := authServer(t, func(c *Config) { c.LLAMA.Host, c.LLAMA.Port = lh2, lp2 })
	task2 := newTask("llama", "m")
	task2.Path, task2.Body, task2.W = "/v1/chat/completions", []byte(`{}`), httptest.NewRecorder()
	s2.execBackend(task2)
	if len(seen2.auth) != 1 || seen2.auth[0] != "" {
		t.Fatalf("без ключа не должно быть Authorization: %v", seen2.auth)
	}
}

func TestReverseProxyAuth(t *testing.T) {
	h, p, seen := fakeBackend(t, 200, "ok")
	s := authServer(t, nil)
	base := "http://" + h + ":" + p + "/"

	// llama: ключ прокси заменяет заголовок клиента, X-Api-Key убирается
	front := httptest.NewServer(s.reverseProxy(base, "Bearer REAL"))
	defer front.Close()
	req, _ := http.NewRequest("GET", front.URL+"/v1/models", nil)
	req.Header.Set("Authorization", "Bearer client-key")
	req.Header.Set("X-Api-Key", "other-key")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if seen.auth[0] != "Bearer REAL" || seen.xkey[0] != "" {
		t.Fatalf("бэкенд получил Authorization=%q X-Api-Key=%q", seen.auth[0], seen.xkey[0])
	}

	// ComfyUI (auth пустой): заголовки клиента не трогаем
	front2 := httptest.NewServer(s.reverseProxy(base, ""))
	defer front2.Close()
	req2, _ := http.NewRequest("GET", front2.URL+"/queue", nil)
	req2.Header.Set("Authorization", "Bearer comfy-user")
	resp2, _ := http.DefaultClient.Do(req2)
	resp2.Body.Close()
	if seen.auth[1] != "Bearer comfy-user" {
		t.Fatalf("без ключа прокси не должен менять заголовок: %q", seen.auth[1])
	}
}

func TestAuthMiddleware(t *testing.T) {
	mk := func(require bool) http.Handler {
		s := authServer(t, func(c *Config) {
			c.LLAMA.APIKey = "REAL-LLAMA-KEY"
			c.Proxy.LlamaRequireKey = require
			c.Proxy.LlamaAPIKeys = []string{"client-two"}
		})
		return s.llamaAuthMW(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("passed")) }))
	}
	do := func(h http.Handler, method, path, auth string) (int, string) {
		req := httptest.NewRequest(method, path, nil)
		if auth != "" {
			req.Header.Set("Authorization", auth)
		}
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		return rec.Code, rec.Body.String()
	}

	on := mk(true)
	type c struct {
		method, path, auth string
		want               int
	}
	for _, tc := range []c{
		{"POST", "/v1/chat/completions", "", 401},
		{"POST", "/v1/chat/completions", "Bearer wrong", 401},
		{"POST", "/v1/chat/completions", "Basic REAL-LLAMA-KEY", 401},
		{"POST", "/v1/chat/completions", "Bearer REAL-LLAMA-KEY", 200}, // ключ самого llama
		{"POST", "/v1/chat/completions", "bearer client-two", 200},     // отдельный ключ клиента, схема без учёта регистра
		{"POST", "/completion", "", 401},
		{"GET", "/v1/models", "", 401},
		{"GET", "/props", "", 401},
		{"GET", "/slots", "Bearer client-two", 200},
		{"GET", "/status", "", 401},
		{"GET", "/health", "", 200}, // публичные, как у самого llama-server
		{"GET", "/v1/health", "", 200},
		{"GET", "/", "", 200}, // статика веб-интерфейса
		{"GET", "/assets/index.js", "", 200},
		{"OPTIONS", "/v1/chat/completions", "", 200}, // CORS preflight
		{"POST", "/", "", 401},                       // POST никогда не публичный
	} {
		if code, _ := do(on, tc.method, tc.path, tc.auth); code != tc.want {
			t.Errorf("require=on %s %s auth=%q -> %d, want %d", tc.method, tc.path, tc.auth, code, tc.want)
		}
	}
	// ответ 401 в формате OpenAI и с WWW-Authenticate
	rec := httptest.NewRecorder()
	on.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", nil))
	if rec.Header().Get("WWW-Authenticate") != "Bearer" || !strings.Contains(rec.Body.String(), "Invalid API Key") {
		t.Fatalf("формат 401: %v %s", rec.Header(), rec.Body.String())
	}

	// проверка клиентов выключена: пускаем всех (прокси сам подставит ключ)
	off := mk(false)
	if code, _ := do(off, "POST", "/v1/chat/completions", ""); code != 200 {
		t.Fatalf("при выключенной проверке: %d", code)
	}
}

func TestUnauthenticatedRequestNeverQueues(t *testing.T) {
	s := authServer(t, func(c *Config) {
		c.LLAMA.APIKey = "REAL-LLAMA-KEY"
		c.Proxy.LlamaRequireKey = true
	})
	h := loggingMW("LLAMA", s.llamaAuthMW(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.gpuTask("llama")(w, r)
	})))

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"m"}`)))
	if rec.Code != 401 {
		t.Fatalf("без ключа: %d", rec.Code)
	}
	if len(s.listTasks()) != 0 || len(s.taskQueue) != 0 {
		t.Fatal("запрос без ключа не должен попадать в очередь (иначе он мог бы выгрузить ComfyUI)")
	}

	// с ключом задача регистрируется (воркера нет: она лежит в очереди)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Authorization", "Bearer REAL-LLAMA-KEY")
	done := make(chan struct{})
	go func() { h.ServeHTTP(httptest.NewRecorder(), req); close(done) }()
	waitFor(t, "task registered", func() bool { return len(s.listTasks()) == 1 })
	ts := s.listTasks()
	s.cancelTask(ts[0].ID, "test")
	<-done
}

func TestRejectLogHasNoKeyAndIsRateLimited(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(os.Stderr)
	s := authServer(t, func(c *Config) {
		c.LLAMA.APIKey = "REAL-LLAMA-KEY"
		c.Proxy.LlamaRequireKey = true
	})
	h := s.llamaAuthMW(http.NewServeMux())
	buf.Reset()
	for i := 0; i < 5; i++ {
		req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
		req.Header.Set("Authorization", "Bearer guess-super-secret-"+string(rune('a'+i)))
		req.RemoteAddr = "10.1.2.3:555"
		h.ServeHTTP(httptest.NewRecorder(), req)
	}
	out := buf.String()
	if strings.Contains(out, "guess-super-secret") || strings.Contains(out, "REAL-LLAMA-KEY") {
		t.Fatalf("в логе есть ключ:\n%s", out)
	}
	if n := strings.Count(out, "[Auth] отказ"); n != 1 {
		t.Fatalf("отказы должны логироваться не чаще раза в 10 с, записей: %d\n%s", n, out)
	}
	if !strings.Contains(out, "ключ не подошёл") || !strings.Contains(out, "10.1.2.3") {
		t.Fatalf("в логе должна быть причина и адрес:\n%s", out)
	}
}

func TestBackendRejectedIsReported(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(os.Stderr)
	lh, lp, _ := fakeBackend(t, 401, `{"error":"bad key"}`)
	s := authServer(t, func(c *Config) { c.LLAMA.Host, c.LLAMA.Port, c.LLAMA.APIKey = lh, lp, "WRONG-KEY" })
	buf.Reset()
	if _, err := s.llamaLoaded(); !errors.Is(err, errAuthRejected) {
		t.Fatalf("ожидалась errAuthRejected, получили %v", err)
	}
	s.llamaLoaded() // второй раз — не спамим
	if n := strings.Count(buf.String(), "[Auth] llama-server ответил 401"); n != 1 || strings.Contains(buf.String(), "WRONG-KEY") {
		t.Fatalf("подсказка про ключ: %d раз\n%s", n, buf.String())
	}
	st := s.buildDashStatus()
	if !st.LLAMA.AuthErr || st.LLAMA.Auth != "proxy" {
		t.Fatalf("на странице: auth_error=%v auth=%q", st.LLAMA.AuthErr, st.LLAMA.Auth)
	}
}
