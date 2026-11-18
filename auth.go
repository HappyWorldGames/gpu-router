package main

// Bearer-авторизация для llama.cpp.
//
// Две независимые вещи:
//
//  1. Ключ прокси для llama-server (llama.api_key). С ним прокси ходит в llama-server, если
//     тот запущен с --api-key: в служебных вызовах (список моделей, выгрузка), при передаче
//     запросов клиентов и в обратном прокси. В ComfyUI этот ключ никогда не отправляется.
//
//  2. Проверка ключа у клиентов прокси (proxy.llama_require_key). Пока она выключена, прокси
//     пускает любого, кто дошёл до его порта, и сам подставляет свой ключ: защита llama-server
//     тогда держится только на файрволе. Когда включена, клиент должен прислать допустимый ключ
//     (llama.api_key или любой из proxy.llama_api_keys). Проверка идёт ДО очереди, поэтому
//     запрос без ключа не может ни занять очередь, ни заставить прокси выгрузить ComfyUI.
//
// Ключи можно задать значением, как env:ИМЯ (переменная окружения) или file:/путь
// (первая строка файла); последние два варианта не оставляют ключ в config.yaml.

import (
	"crypto/sha256"
	"crypto/subtle"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// errAuthRejected — llama-server ответил 401/403: ключ прокси не подошёл (или не задан).
var errAuthRejected = errors.New("llama-server не принял ключ прокси (HTTP 401/403)")

type authState struct {
	backendHdr string     // "Bearer <ключ>" для llama-server или ""
	clients    [][32]byte // sha256 допустимых ключей клиентов (llama.api_key + proxy.llama_api_keys)
	require    bool

	mu             sync.Mutex
	lastReject     time.Time
	suppressed     int
	lastBackend401 time.Time
}

// resolveSecret разбирает значение ключа: литерал, env:ИМЯ или file:/путь.
// Возвращает ключ и короткое описание источника (без самого ключа).
func resolveSecret(v string) (secret, source string, err error) {
	v = strings.TrimSpace(v)
	switch {
	case v == "":
		return "", "", nil
	case strings.HasPrefix(v, "env:"):
		name := strings.TrimSpace(strings.TrimPrefix(v, "env:"))
		if name == "" {
			return "", "", errors.New("после env: не указано имя переменной")
		}
		s := strings.TrimSpace(os.Getenv(name))
		if s == "" {
			return "", "", fmt.Errorf("переменная окружения %s пуста или не задана", name)
		}
		return s, "переменная " + name, nil
	case strings.HasPrefix(v, "file:"):
		path := strings.TrimSpace(strings.TrimPrefix(v, "file:"))
		if path == "" {
			return "", "", errors.New("после file: не указан путь")
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return "", "", fmt.Errorf("не удалось прочитать файл с ключом %s: %w", path, err)
		}
		s := strings.TrimSpace(strings.SplitN(string(b), "\n", 2)[0])
		if s == "" {
			return "", "", fmt.Errorf("файл с ключом %s пуст", path)
		}
		if st, err := os.Stat(path); err == nil && st.Mode().Perm()&0o077 != 0 {
			log.Printf("[Auth] ВНИМАНИЕ: файл с ключом %s доступен другим пользователям (права %v); лучше chmod 600", path, st.Mode().Perm())
		}
		return s, "файл " + path, nil
	}
	return v, "config.yaml (открытым текстом)", nil
}

func hashKey(k string) [32]byte { return sha256.Sum256([]byte(k)) }

// loadAuth читает ключи из конфига. Ошибка = неверная настройка: на старте лучше упасть
// с понятным сообщением, чем молча работать без ключа и ловить 401.
func (s *Server) loadAuth() error {
	a := &authState{require: s.Config.Proxy.LlamaRequireKey}

	key, src, err := resolveSecret(s.Config.LLAMA.APIKey)
	if err != nil {
		return fmt.Errorf("llama.api_key: %w", err)
	}
	if key != "" {
		a.backendHdr = "Bearer " + key
		a.clients = append(a.clients, hashKey(key))
		log.Printf("[Auth] ключ для llama-server загружен (источник: %s, %d симв.)", src, len(key))
	}
	for i, raw := range s.Config.Proxy.LlamaAPIKeys {
		k, _, err := resolveSecret(raw)
		if err != nil {
			return fmt.Errorf("proxy.llama_api_keys[%d]: %w", i, err)
		}
		if k == "" {
			continue
		}
		a.clients = append(a.clients, hashKey(k))
	}
	if a.require && len(a.clients) == 0 {
		return errors.New("proxy.llama_require_key включён, но нет ни одного ключа: задайте llama.api_key и/или proxy.llama_api_keys")
	}
	switch {
	case a.require:
		log.Printf("[Auth] проверка ключа у клиентов llama-прокси: ВКЛЮЧЕНА (допустимых ключей: %d)", len(a.clients))
	case a.backendHdr != "":
		log.Printf("[Auth] ВНИМАНИЕ: проверка ключа у клиентов выключена (proxy.llama_require_key: false): любой, кто достучится " +
			"до порта llama-прокси, получит доступ к llama без ключа. Закройте порт файрволом или включите проверку.")
	}
	s.auth = a
	return nil
}

// llamaAuthHeader — значение заголовка Authorization для llama-server ("" = ключ не нужен).
func (s *Server) llamaAuthHeader() string {
	if s.auth == nil {
		return ""
	}
	return s.auth.backendHdr
}

// isLlamaTarget: адрес принадлежит llama-серверу (а не ComfyUI).
func (s *Server) isLlamaTarget(target string) bool {
	return strings.HasPrefix(target, s.Config.llamaURL("/"))
}

// authFor — заголовки, которые нужно добавить к служебному запросу на этот адрес.
func (s *Server) authFor(target string) http.Header {
	if h := s.llamaAuthHeader(); h != "" && s.isLlamaTarget(target) {
		return http.Header{"Authorization": {h}}
	}
	return nil
}

// call — служебный запрос к бэкенду (ComfyUI или llama); к llama добавляет ключ.
func (s *Server) call(method, target string, payload interface{}, timeout time.Duration) (int, []byte, error) {
	return httpDoH(method, target, payload, timeout, s.authFor(target))
}

// noteBackendRejected: llama-server ответил 401/403 — пишем подсказку (не чаще раза в минуту).
func (s *Server) noteBackendRejected(where string) {
	if s.auth == nil {
		return
	}
	a := s.auth
	a.mu.Lock()
	defer a.mu.Unlock()
	if time.Since(a.lastBackend401) < time.Minute {
		return
	}
	a.lastBackend401 = time.Now()
	if a.backendHdr == "" {
		log.Printf("[Auth] llama-server ответил 401/403 (%s), а llama.api_key не задан: запущен ли он с --api-key? задайте ключ в config.yaml", where)
	} else {
		log.Printf("[Auth] llama-server ответил 401/403 (%s): ключ llama.api_key не совпадает с --api-key сервера", where)
	}
}

// ---- проверка ключа у клиентов ----

// apiPrefixes — пути API llama-server: к ним ключ нужен всегда. Остальное при GET (статика
// веб-интерфейса) остаётся публичным, как и у самого llama-server, иначе страница не откроется.
var apiPrefixes = []string{
	"/v1", "/api/", "/props", "/slots", "/models", "/tokenize", "/detokenize", "/embedding",
	"/completion", "/chat", "/infill", "/rerank", "/lora", "/apply-template", "/metrics",
	"/cors-proxy", "/mcp", "/status",
}

func isPublicLlamaRequest(r *http.Request) bool {
	if r.Method == http.MethodOptions { // CORS preflight браузера
		return true
	}
	p := r.URL.Path
	if p == "/health" || p == "/v1/health" {
		return true
	}
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		return false
	}
	for _, pre := range apiPrefixes {
		if strings.HasPrefix(p, pre) {
			return false
		}
	}
	return true // статика веб-интерфейса
}

func bearerToken(r *http.Request) string {
	h := strings.TrimSpace(r.Header.Get("Authorization"))
	if len(h) > 7 && strings.EqualFold(h[:7], "bearer ") {
		return strings.TrimSpace(h[7:])
	}
	return ""
}

// valid сравнивает ключ со всеми допустимыми за постоянное время (без раннего выхода).
func (a *authState) valid(key string) bool {
	if key == "" {
		return false
	}
	got := hashKey(key)
	ok := 0
	for _, want := range a.clients {
		ok |= subtle.ConstantTimeCompare(got[:], want[:])
	}
	return ok == 1
}

// llamaAuthMW проверяет ключ клиента до очереди. Без включённой проверки ничего не делает.
func (s *Server) llamaAuthMW(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		a := s.auth
		if a == nil || !a.require || isPublicLlamaRequest(r) {
			next.ServeHTTP(w, r)
			return
		}
		if a.valid(bearerToken(r)) {
			next.ServeHTTP(w, r)
			return
		}
		a.logReject(r)
		w.Header().Set("WWW-Authenticate", "Bearer")
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error":{"code":401,"message":"Invalid API Key","type":"authentication_error"}}`))
	})
}

// logReject пишет отказ, не чаще раза в 10 секунд (иначе перебор ключей забьёт лог).
// Сам присланный ключ в лог не попадает никогда.
func (a *authState) logReject(r *http.Request) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if time.Since(a.lastReject) < 10*time.Second {
		a.suppressed++
		return
	}
	why := "ключ не передан"
	if bearerToken(r) != "" {
		why = "ключ не подошёл"
	}
	extra := ""
	if a.suppressed > 0 {
		extra = fmt.Sprintf(" (ещё %d отказов за это время)", a.suppressed)
	}
	a.lastReject, a.suppressed = time.Now(), 0
	log.Printf("[Auth] отказ: %s %s от %s — %s%s", r.Method, r.URL.Path, clientAddr(r), why, extra)
}
