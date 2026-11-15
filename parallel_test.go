package main

import (
	"bytes"
	"context"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"
)

func TestRequestModel(t *testing.T) {
	big := `{"messages":[{"role":"user","content":"` + strings.Repeat("x", 100000) + `"}],"stream":true,"model":"abc"}`
	cases := map[string]string{
		`{"model":"m1","a":1}`:    "m1",
		big:                       "abc",
		`{"a":{"model":"inner"}}`: "",
		`not json`:                "",
		`{"model":5}`:             "",
		`[{"model":"x"}]`:         "",
	}
	for in, want := range cases {
		if got := requestModel([]byte(in)); got != want {
			t.Errorf("%.40q -> %q, want %q", in, got, want)
		}
	}
}

func TestParseINI(t *testing.T) {
	def, m := parseParallelINI("testdata/models.ini")
	if def != 1 {
		t.Fatalf("defaults=%d", def)
	}
	if m["big-model"] != 2 {
		t.Fatalf("big: %v", m)
	}
	if m["small-model"] != 1 || m["embed-model"] != 1 {
		t.Fatalf("inherit: %v", m)
	}
	if stripINIComment("2 # много") != "2" || stripINIComment("a#b") != "a#b" {
		t.Fatal("comment")
	}
	ini := newModelsINI("testdata/models.ini")
	if ini.parallelFor("BIG-model") != 2 {
		t.Fatal("case-insensitive")
	}
	if ini.parallelFor("unknown") != 1 || ini.parallelFor("") != 1 {
		t.Fatal("unknown")
	}
	var nilIni *modelsINI
	if nilIni.parallelFor("x") != 1 {
		t.Fatal("nil")
	}
}

func newTask(service, model string) *Task {
	ctx, cancel := context.WithCancel(context.Background())
	return &Task{Service: service, Model: model, Ctx: ctx, Cancel: cancel,
		Turn: make(chan struct{}), Done: make(chan struct{})}
}

// regTask — задача, зарегистрированная в реестре (как в настоящем обработчике).
func regTask(s *Server, service, model, id string, state int32) *Task {
	t := newTask(service, model)
	t.ID, t.Client = id, "10.0.0.9"
	t.EnqueuedAt = time.Now()
	t.state.Store(state)
	s.register(t)
	return t
}

func isClosed(c chan struct{}) bool {
	select {
	case <-c:
		return true
	default:
		return false
	}
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for i := 0; i < 200; i++ {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("не дождались: %s", what)
}

func TestAdmit(t *testing.T) {
	cfg := &Config{}
	cfg.LLAMA.ModelsINI = "testdata/models.ini"
	applyDefaults(cfg)
	s := NewServer(cfg)
	const big = "big-model"

	// Идёт один запрос к большой модели (parallel=2).
	s.activeService, s.activeReady, s.llamaInflight, s.llamaModel = "llama", true, 1, big

	// 1) второй запрос к той же модели присоединяется сразу
	b := newTask("llama", big)
	if !s.admit(b) || !isClosed(b.Turn) || s.llamaInflight != 2 {
		t.Fatalf("join failed: inflight=%d", s.llamaInflight)
	}

	// 2) третий — слотов нет, ждёт
	c := newTask("llama", big)
	res := make(chan bool, 1)
	go func() { res <- s.admit(c) }()
	time.Sleep(100 * time.Millisecond)
	if isClosed(c.Turn) {
		t.Fatal("третий не должен был стартовать")
	}
	// b закончился -> слот освободился -> c стартует
	close(b.Done)
	waitFor(t, "c joined", func() bool { return isClosed(c.Turn) })
	if !<-res {
		t.Fatal("admit(c) должен вернуть true")
	}

	// 3) запрос к другой модели и ComfyUI ждут полного освобождения
	d := newTask("llama", "small-model")
	resD := make(chan bool, 1)
	go func() { resD <- s.admit(d) }()
	time.Sleep(100 * time.Millisecond)
	if isClosed(d.Turn) {
		t.Fatal("другая модель не должна присоединяться")
	}
	// все идущие заканчиваются -> admit вернул false (можно готовить GPU обычным путём)
	close(c.Done)
	s.mu.Lock()
	// a (первый, "запущенный вручную") тоже завершается
	s.llamaInflight--
	s.cond.Broadcast()
	s.mu.Unlock()
	waitFor(t, "d released", func() bool {
		select {
		case v := <-resD:
			return !v
		default:
			return false
		}
	})
	if isClosed(d.Turn) {
		t.Fatal("для обычного пути Turn закрывает воркер, а не admit")
	}

	// 4) клиент ушёл во время ожидания
	s.mu.Lock()
	s.llamaInflight, s.llamaModel = 1, big
	s.mu.Unlock()
	e := newTask("comfyui", "")
	resE := make(chan bool, 1)
	go func() { resE <- s.admit(e) }()
	time.Sleep(50 * time.Millisecond)
	e.Cancel()
	select {
	case v := <-resE:
		if !v {
			t.Fatal("ушедший клиент: admit должен вернуть true (пропустить задачу)")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("ожидание не прервалось после ухода клиента")
	}
}

func TestMemIdleProtect(t *testing.T) {
	// crit=512, порог простоя по умолчанию 256
	if !memIdleProtect(400, 512, 0) {
		t.Fatal("400 МБ при простое: модель щадим")
	}
	if memIdleProtect(200, 512, 0) {
		t.Fatal("200 МБ при простое: уже последний рубеж, выгружаем")
	}
	if !memIdleProtect(50, 512, -1) {
		t.Fatal("-1: при простое не трогаем никогда")
	}
	if memIdleProtect(900, 512, 1000) {
		t.Fatal("свой порог 1000: 900 МБ — выгружаем")
	}
}

func TestWaitDoneOrLimit(t *testing.T) {
	cfg := &Config{}
	applyDefaults(cfg)
	s := NewServer(cfg)

	// 1) предел истёк, но очередь пуста: задачу не трогаем
	a := newTask("llama", "m")
	got := make(chan bool, 1)
	go func() { got <- s.waitDoneOrLimit(a, 50*time.Millisecond) }()
	time.Sleep(400 * time.Millisecond)
	if a.Ctx.Err() != nil {
		t.Fatal("задачу прервали, хотя за ней никто не ждёт")
	}
	// 2) пришёл второй в очередь: теперь прерываем
	q1 := regTask(s, "llama", "m", "q1", stQueued)
	s.taskQueue <- q1
	waitFor(t, "cancel after contention", func() bool { return a.Ctx.Err() != nil })
	close(a.Done)
	if !<-got {
		t.Fatal("должно вернуться true (прервали принудительно)")
	}

	// 3) задача сама закончилась — без вмешательства
	b := newTask("llama", "m")
	go func() { time.Sleep(20 * time.Millisecond); close(b.Done) }()
	if s.waitDoneOrLimit(b, time.Hour) {
		t.Fatal("false ожидался")
	}

	// 4) воркер держит ждущего в admit() — это тоже «кто-то ждёт»
	for len(s.taskQueue) > 0 {
		<-s.taskQueue
	}
	s.unregister(q1)
	if s.someoneWaits() {
		t.Fatal("очередь пуста")
	}
	held := regTask(s, "llama", "m", "h1", stWaiting)
	if !s.someoneWaits() {
		t.Fatal("ждущий в admit должен считаться")
	}
	s.unregister(held)
}

func TestBackendErrorLogged(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(os.Stderr)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		w.Write([]byte(`{"error":{"code":500,"message":"Compute error.","type":"server_error"}}`))
	}))
	defer srv.Close()
	u, _ := url.Parse(srv.URL)
	host, port, _ := net.SplitHostPort(u.Host)

	cfg := &Config{}
	cfg.LLAMA.Host, cfg.LLAMA.Port = host, port
	applyDefaults(cfg)
	s := NewServer(cfg)
	task := newTask("llama", "big-model")
	task.Path, task.Body = "/v1/chat/completions", []byte(`{}`)
	rec := httptest.NewRecorder()
	task.W = rec
	if code := s.execBackend(task); code != 500 || rec.Code != 500 {
		t.Fatalf("status passthrough: %d / %d", code, rec.Code)
	}
	out := buf.String()
	if !strings.Contains(out, "HTTP 500") || !strings.Contains(out, "Compute error") {
		t.Fatalf("в логе нет причины ошибки:\n%s", out)
	}
}

func TestNoJoinWhenRAMLow(t *testing.T) {
	// контроль: с выключенной проверкой ОЗУ (-1) тот же запрос присоединяется сразу
	ctl := &Config{}
	ctl.LLAMA.ModelsINI = "testdata/models.ini"
	ctl.GPU.MinAvailableMB = -1
	applyDefaults(ctl)
	sc := NewServer(ctl)
	sc.activeService, sc.activeReady, sc.llamaInflight, sc.llamaModel = "llama", true, 1, "big-model"
	if !sc.admit(newTask("llama", "big-model")) {
		t.Fatal("контроль: без проверки ОЗУ запрос должен присоединиться")
	}

	cfg := &Config{}
	cfg.LLAMA.ModelsINI = "testdata/models.ini"
	cfg.GPU.MinAvailableMB = 1 << 30 // заведомо больше любой свободной ОЗУ
	applyDefaults(cfg)
	s := NewServer(cfg)
	const big = "big-model"
	s.activeService, s.activeReady, s.llamaInflight, s.llamaModel = "llama", true, 1, big

	b := newTask("llama", big)
	res := make(chan bool, 1)
	go func() { res <- s.admit(b) }()
	time.Sleep(300 * time.Millisecond)
	if isClosed(b.Turn) {
		t.Fatal("при нехватке ОЗУ запрос не должен был присоединиться к идущим")
	}
	// идущий запрос закончился -> ждущий получает обычный путь (false), без зависания
	s.mu.Lock()
	s.llamaInflight = 0
	s.cond.Broadcast()
	s.mu.Unlock()
	select {
	case v := <-res:
		if v {
			t.Fatal("ожидался false (обычная подготовка GPU)")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("ожидание не завершилось")
	}
}

func TestLoggingMWDiagnostics(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(os.Stderr)
	old := slowBypassThreshold
	slowBypassThreshold = 30 * time.Millisecond
	defer func() { slowBypassThreshold = old }()
	logRequests.Store(true)

	slow := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(80 * time.Millisecond)
		w.Write([]byte("ok"))
	})
	h := loggingMW("LLAMA", slow)

	// GET мимо очереди, долгий -> попадает в лог
	req := httptest.NewRequest("GET", "/v1/models", nil)
	req.RemoteAddr = "172.17.0.3:5555"
	h.ServeHTTP(httptest.NewRecorder(), req)
	if !strings.Contains(buf.String(), "МИМО очереди") || !strings.Contains(buf.String(), "172.17.0.3") || !strings.Contains(buf.String(), "HTTP 200") {
		t.Fatalf("нет записи о медленном запросе мимо очереди:\n%s", buf.String())
	}
	buf.Reset()

	// долгоживущий поток событий — не считается зависшим
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/models/sse", nil))
	if strings.Contains(buf.String(), "МИМО") {
		t.Fatalf("SSE не должен попадать в лог зависших:\n%s", buf.String())
	}

	// POST в очередь: строка REQ содержит адрес клиента
	req = httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req.RemoteAddr = "127.0.0.1:4444"
	h.ServeHTTP(httptest.NewRecorder(), req)
	if !strings.Contains(buf.String(), "[LLAMA REQ] POST /v1/chat/completions от 127.0.0.1:4444") {
		t.Fatalf("в REQ нет адреса клиента:\n%s", buf.String())
	}
}

func TestStreamResumeLogged(t *testing.T) {
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(os.Stderr)
	logRequests.Store(true)
	h := loggingMW("LLAMA", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	for _, p := range []string{"/v1/stream/abc", "/v1/streams/lookup"} {
		req := httptest.NewRequest("GET", p, nil)
		req.RemoteAddr = "10.0.0.5:1"
		h.ServeHTTP(httptest.NewRecorder(), req)
		if !strings.Contains(buf.String(), "[LLAMA REQ] GET "+p+" от 10.0.0.5:1") {
			t.Fatalf("запрос %s не залогирован:\n%s", p, buf.String())
		}
	}
	buf.Reset()
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest("GET", "/props", nil))
	if buf.Len() != 0 {
		t.Fatalf("/props не должен логироваться:\n%s", buf.String())
	}
}

func TestQueueLenCountsWaiterHeldByWorker(t *testing.T) {
	cfg := &Config{}
	applyDefaults(cfg)
	s := NewServer(cfg)
	if s.queueLen() != 0 {
		t.Fatal("пусто")
	}
	regTask(s, "llama", "m", "a", stQueued)
	regTask(s, "llama", "m", "b", stWaiting) // воркер держит ещё одну в admit()
	regTask(s, "llama", "m", "c", stRunning) // идущая в очередь не считается
	if s.queueLen() != 2 {
		t.Fatalf("queueLen=%d, ждали 2", s.queueLen())
	}
	st := s.buildDashStatus()
	if st.QueueSize != 2 || len(st.Tasks) != 3 {
		t.Fatalf("на странице queue_size=%d, задач %d (ждали 2 и 3)", st.QueueSize, len(st.Tasks))
	}
	if st.Tasks[0].ID != "c" || st.Tasks[0].State != "running" {
		t.Fatalf("первой должна идти выполняющаяся: %+v", st.Tasks)
	}
}

func TestCancelTask(t *testing.T) {
	cfg := &Config{}
	applyDefaults(cfg)
	s := NewServer(cfg)

	w := regTask(s, "llama", "m", "w1", stWaiting)
	if code, _ := s.cancelTask("w1", "test"); code != 200 || w.Ctx.Err() == nil || !w.kicked.Load() {
		t.Fatalf("ждущую задачу llama надо снять: code=%d", code)
	}
	r := regTask(s, "llama", "m", "r1", stRunning)
	if code, _ := s.cancelTask("r1", "test"); code != 200 || r.Ctx.Err() == nil {
		t.Fatalf("идущий запрос llama можно остановить: code=%d", code)
	}
	cq := regTask(s, "comfyui", "", "c1", stQueued)
	if code, _ := s.cancelTask("c1", "test"); code != 200 || cq.Ctx.Err() == nil {
		t.Fatalf("ComfyUI из очереди снимается: code=%d", code)
	}
	cr := regTask(s, "comfyui", "", "c2", stRunning)
	if code, _ := s.cancelTask("c2", "test"); code != 409 || cr.Ctx.Err() != nil {
		t.Fatalf("идущую генерацию ComfyUI отсюда не трогаем: code=%d", code)
	}
	if code, _ := s.cancelTask("nope", "test"); code != 404 {
		t.Fatalf("неизвестная задача: %d", code)
	}
	for _, d := range s.listTasks() {
		if d.ID == "c2" && d.CanCancel {
			t.Fatal("для идущей ComfyUI-задачи кнопки быть не должно")
		}
	}
}

func TestCancelEndpointGuards(t *testing.T) {
	cfg := &Config{}
	applyDefaults(cfg)
	s := NewServer(cfg)
	regTask(s, "llama", "m", "x1", stQueued)

	do := func(method, ctype, hdr, body string) int {
		req := httptest.NewRequest(method, "/api/queue/cancel", strings.NewReader(body))
		if ctype != "" {
			req.Header.Set("Content-Type", ctype)
		}
		if hdr != "" {
			req.Header.Set("X-GPU-Router", hdr)
		}
		rec := httptest.NewRecorder()
		s.handleCancel(rec, req)
		return rec.Code
	}
	if c := do("GET", "", "", ""); c != 405 {
		t.Fatalf("GET: %d", c)
	}
	// «простой» кросс-сайтовый POST (форма): нет ни JSON, ни своего заголовка
	if c := do("POST", "text/plain", "", `{"id":"x1"}`); c != 403 {
		t.Fatalf("без заголовков: %d", c)
	}
	if c := do("POST", "application/json", "", `{"id":"x1"}`); c != 403 {
		t.Fatalf("без X-GPU-Router: %d", c)
	}
	if c := do("POST", "application/json", "1", `{}`); c != 400 {
		t.Fatalf("без id: %d", c)
	}
	if c := do("POST", "application/json", "1", `{"id":"x1"}`); c != 200 {
		t.Fatalf("штатный вызов: %d", c)
	}
	no := false
	s.Config.Dashboard.AllowCancel = &no
	regTask(s, "llama", "m", "x2", stQueued)
	if c := do("POST", "application/json", "1", `{"id":"x2"}`); c != 403 {
		t.Fatalf("при allow_cancel=false: %d", c)
	}
}

func TestKickedClientGetsError(t *testing.T) {
	cfg := &Config{}
	applyDefaults(cfg)
	s := NewServer(cfg) // воркера нет: задача так и лежит в очереди
	h := loggingMW("LLAMA", s.gpuTask("llama"))

	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"m"}`))
	req.RemoteAddr = "192.168.1.5:1234"
	rec := httptest.NewRecorder()
	finished := make(chan struct{})
	go func() { h.ServeHTTP(rec, req); close(finished) }()

	waitFor(t, "task registered", func() bool { return len(s.listTasks()) == 1 })
	ts := s.listTasks()
	if ts[0].State != "queued" || ts[0].Client != "192.168.1.5" || ts[0].Model != "m" {
		t.Fatalf("карточка задачи: %+v", ts[0])
	}
	if code, _ := s.cancelTask(ts[0].ID, "test"); code != 200 {
		t.Fatalf("cancel: %d", code)
	}
	select {
	case <-finished:
	case <-time.After(2 * time.Second):
		t.Fatal("обработчик не завершился после снятия")
	}
	if rec.Code != 503 || !strings.Contains(rec.Body.String(), "снят") {
		t.Fatalf("клиент должен получить понятную ошибку: %d %s", rec.Code, rec.Body.String())
	}
	if len(s.listTasks()) != 0 || s.queueLen() != 0 {
		t.Fatal("после снятия задача должна пропасть из реестра")
	}
}
