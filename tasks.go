package main

// Реестр задач для страницы состояния: что стоит в очереди, что ждёт слот, что выполняется,
// и возможность снять задачу с очереди (или остановить идущий запрос llama) с этой страницы.
//
// Канал очереди перечислить нельзя, поэтому каждая задача регистрируется здесь при создании
// и удаляется, когда её обработчик завершился.

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// Состояния задачи (Task.state).
const (
	stQueued    int32 = iota // лежит в очереди
	stWaiting                // воркер взял её и держит: ждёт слот или конца идущих запросов
	stPreparing              // идёт передача GPU (выгрузка прежнего сервиса)
	stRunning                // выполняется
)

func stateName(v int32) string {
	switch v {
	case stWaiting:
		return "waiting"
	case stPreparing:
		return "preparing"
	case stRunning:
		return "running"
	}
	return "queued"
}

func (s *Server) nextTaskID() string {
	return fmt.Sprintf("t%d", s.taskSeq.Add(1))
}

// clientAddr — кто прислал запрос (для показа на странице). За Caddy берём X-Forwarded-For.
func clientAddr(r *http.Request) string {
	if xf := strings.TrimSpace(strings.Split(r.Header.Get("X-Forwarded-For"), ",")[0]); xf != "" {
		return xf
	}
	if h, _, err := net.SplitHostPort(r.RemoteAddr); err == nil {
		return h
	}
	return r.RemoteAddr
}

func (s *Server) register(t *Task) {
	s.tasksMu.Lock()
	if s.tasks == nil {
		s.tasks = map[string]*Task{}
	}
	s.tasks[t.ID] = t
	s.tasksMu.Unlock()
}

func (s *Server) unregister(t *Task) {
	s.tasksMu.Lock()
	delete(s.tasks, t.ID)
	s.tasksMu.Unlock()
	s.dash.invalidate()
}

func (s *Server) getTask(id string) *Task {
	s.tasksMu.Lock()
	defer s.tasksMu.Unlock()
	return s.tasks[id]
}

// queueLen — сколько живых задач ждёт своей очереди (в очереди + удерживаемые воркером).
// Снятые и завершённые из реестра уже удалены, поэтому мёртвые задачи не считаются.
func (s *Server) queueLen() int {
	s.tasksMu.Lock()
	defer s.tasksMu.Unlock()
	n := 0
	for _, t := range s.tasks {
		if st := t.state.Load(); st == stQueued || st == stWaiting {
			n++
		}
	}
	return n
}

// ---- откуда пришёл запрос: название, приложение ----

type clientRule struct {
	net  *net.IPNet
	name string
}

func (s *Server) initClients() {
	s.clientExact = map[string]string{}
	s.clientNets = nil
	for k, name := range s.Config.Dashboard.Clients {
		k = strings.TrimSpace(k)
		if strings.Contains(k, "/") {
			if _, n, err := net.ParseCIDR(k); err == nil {
				s.clientNets = append(s.clientNets, clientRule{n, name})
				continue
			}
			log.Printf("[Dashboard] dashboard.clients: не понимаю подсеть %q — пропускаю", k)
			continue
		}
		if ip := net.ParseIP(k); ip != nil {
			s.clientExact[ip.String()] = name
		} else {
			log.Printf("[Dashboard] dashboard.clients: не понимаю адрес %q — пропускаю", k)
		}
	}
}

// clientName: своё название адреса (точный адрес), затем имя контейнера Docker, затем
// самая узкая подсеть из конфига, затем имя из DNS, иначе пусто.
func (s *Server) clientName(ip string) string {
	p := net.ParseIP(ip)
	if p == nil {
		return ""
	}
	if n, ok := s.clientExact[p.String()]; ok {
		return n
	}
	// Имя контейнера от Docker точнее общего названия подсети.
	if n := s.dockerName(p.String()); n != "" {
		return n
	}
	best, name := -1, ""
	for _, r := range s.clientNets {
		if r.net.Contains(p) {
			if ones, _ := r.net.Mask.Size(); ones > best {
				best, name = ones, r.name
			}
		}
	}
	if name != "" {
		return name
	}
	if h := s.Config.Dashboard.ResolveHostnames; h == nil || *h {
		return s.rdns.lookup(p)
	}
	return ""
}

// rdnsCache — имена хостов из DNS: запрос идёт в фоне, пока его нет — пусто, результат
// держим 10 минут. Страницу и очередь это не задерживает.
type rdnsCache struct {
	mu sync.Mutex
	m  map[string]rdnsEntry
}

type rdnsEntry struct {
	name string
	at   time.Time
}

func (c *rdnsCache) lookup(ip net.IP) string {
	if ip.IsLoopback() || ip.IsUnspecified() {
		return ""
	}
	key := ip.String()
	c.mu.Lock()
	if c.m == nil {
		c.m = map[string]rdnsEntry{}
	}
	e, ok := c.m[key]
	if ok && time.Since(e.at) < 10*time.Minute {
		c.mu.Unlock()
		return e.name
	}
	c.m[key] = rdnsEntry{name: e.name, at: time.Now()} // чтобы не запускать повторно, пока идёт запрос
	c.mu.Unlock()
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 700*time.Millisecond)
		defer cancel()
		name := ""
		if names, err := net.DefaultResolver.LookupAddr(ctx, key); err == nil && len(names) > 0 {
			name = strings.TrimSuffix(names[0], ".")
		}
		c.mu.Lock()
		c.m[key] = rdnsEntry{name: name, at: time.Now()}
		c.mu.Unlock()
	}()
	return e.name
}

// clipUA ограничивает длину User-Agent и убирает управляющие символы.
func clipUA(ua string) string {
	ua = strings.Map(func(r rune) rune {
		if r < 32 || r == 127 {
			return -1
		}
		return r
	}, ua)
	if len(ua) > 160 {
		ua = ua[:160]
	}
	return ua
}

// appFromUA — короткое название приложения по User-Agent: "browser" для браузеров,
// иначе имя известной библиотеки или первого продукта в строке.
func appFromUA(ua string) string {
	l := strings.ToLower(ua)
	if strings.TrimSpace(l) == "" {
		return ""
	}
	for _, k := range []string{"aiohttp", "python-httpx", "httpx", "python-requests", "curl", "okhttp", "axios", "openai", "litellm", "langchain", "node-fetch", "undici", "go-http-client", "postman"} {
		if strings.Contains(l, k) {
			return k
		}
	}
	if strings.Contains(l, "mozilla") || strings.Contains(l, "applewebkit") {
		return "browser"
	}
	first := strings.Fields(ua)[0]
	if i := strings.IndexByte(first, '/'); i > 0 {
		first = first[:i]
	}
	if len(first) > 24 {
		first = first[:24]
	}
	return first
}

type dashTask struct {
	ID        string `json:"id"`
	Service   string `json:"service"`
	Model     string `json:"model,omitempty"`
	State     string `json:"state"` // queued | waiting | preparing | running
	Client    string `json:"client,omitempty"`
	Name      string `json:"client_name,omitempty"` // своё название или имя из DNS
	App       string `json:"app,omitempty"`         // "browser" или имя библиотеки/приложения
	UA        string `json:"ua,omitempty"`          // полный User-Agent (для подсказки)
	Path      string `json:"path,omitempty"`
	Text      string `json:"text,omitempty"`       // хвост генерируемого текста (если включён show_text)
	Thinking  bool   `json:"thinking,omitempty"`   // этот хвост — «рассуждение» модели
	Reason    string `json:"reason,omitempty"`     // почему ждёт: hold | slot | drain | mem
	ReasonSec int    `json:"reason_sec,omitempty"` // hold: сколько ещё секунд до жёсткого предела ожидания
	AgeSec    int    `json:"age_sec"`
	SizeKB    int    `json:"size_kb"`
	CanCancel bool   `json:"can_cancel"`
}

const maxDashTasks = 50

func (s *Server) listTasks() []dashTask {
	s.tasksMu.Lock()
	ts := make([]*Task, 0, len(s.tasks))
	for _, t := range s.tasks {
		ts = append(ts, t)
	}
	s.tasksMu.Unlock()

	// Сначала выполняющиеся, потом по времени постановки.
	sort.Slice(ts, func(i, j int) bool {
		si, sj := ts[i].state.Load(), ts[j].state.Load()
		if (si >= stPreparing) != (sj >= stPreparing) {
			return si >= stPreparing
		}
		return ts[i].EnqueuedAt.Before(ts[j].EnqueuedAt)
	})
	if len(ts) > maxDashTasks {
		ts = ts[:maxDashTasks]
	}
	allowed := s.cancelAllowed()
	out := make([]dashTask, 0, len(ts))
	for _, t := range ts {
		st := t.state.Load()
		var reason string
		var reasonSec int
		if st == stQueued || st == stWaiting {
			reason = reasonName(t.reason.Load())
			if reason == "hold" {
				if left, capped := s.capLeft(t); capped {
					reasonSec = int(left.Seconds())
				}
			}
		}
		var text string
		var thinking bool
		if t.Live != nil && st == stRunning {
			text, thinking = t.Live.snapshot()
		}
		out = append(out, dashTask{
			ID: t.ID, Service: t.Service, Model: t.Model, State: stateName(st), Client: t.Client,
			Name: s.clientName(t.Client), App: appFromUA(t.UA), UA: t.UA, Path: t.Path,
			Text: text, Thinking: thinking, Reason: reason, ReasonSec: reasonSec,
			AgeSec: int(time.Since(t.EnqueuedAt).Seconds()), SizeKB: len(t.Body) / 1024,
			// Идущую задачу ComfyUI отсюда не прерываем (у неё свой механизм остановки в самом ComfyUI).
			CanCancel: allowed && !(t.Service == "comfyui" && st == stRunning),
		})
	}
	return out
}

func (s *Server) cancelAllowed() bool {
	a := s.Config.Dashboard.AllowCancel
	return a == nil || *a
}

// cancelTask снимает задачу с очереди (или останавливает идущий запрос llama).
// Возвращает HTTP-код и сообщение.
func (s *Server) cancelTask(id, who string) (int, string) {
	t := s.getTask(id)
	if t == nil {
		return http.StatusNotFound, "задача не найдена (возможно, уже завершилась)"
	}
	st := t.state.Load()
	if t.Service == "comfyui" && st == stRunning {
		return http.StatusConflict, "идущую генерацию ComfyUI останавливайте в самом ComfyUI"
	}
	t.kicked.Store(true)
	t.Cancel()
	log.Printf("[Admin] %s снят(а) с очереди со страницы состояния: задача %s, модель %q, клиент %s, состояние %s, шла %ds (запрос от %s)",
		s.name(t.Service), t.ID, t.Model, t.Client, stateName(st), int(time.Since(t.EnqueuedAt).Seconds()), who)
	s.dash.invalidate()
	return http.StatusOK, "ok"
}

// kickedReply: если задачу сняли вручную и клиенту ещё ничего не отправлено, отвечаем ему
// понятной ошибкой, а не пустым «200 OK».
func (s *Server) kickedReply(w http.ResponseWriter, t *Task) {
	if !t.kicked.Load() {
		return
	}
	if sw, ok := w.(*statusWriter); ok && sw.status != 0 {
		return // ответ уже начат (идёт поток) — оборвать его всё, что можно
	}
	s.jsonError(w, http.StatusServiceUnavailable, "запрос снят с очереди администратором")
}

// handleCancel — POST /api/queue/cancel {"id": "t12"} на сервере страницы состояния.
// Защита от чужих сайтов: нужен собственный заголовок и JSON — из чужой вкладки браузер
// без CORS-разрешения такой запрос не отправит.
func (s *Server) handleCancel(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	reply := func(code int, msg string) {
		w.WriteHeader(code)
		json.NewEncoder(w).Encode(map[string]interface{}{"ok": code == http.StatusOK, "message": msg})
	}
	if r.Method != http.MethodPost {
		reply(http.StatusMethodNotAllowed, "нужен POST")
		return
	}
	if !s.cancelAllowed() {
		reply(http.StatusForbidden, "снятие задач отключено (dashboard.allow_cancel: false)")
		return
	}
	if r.Header.Get("X-GPU-Router") != "1" || !strings.HasPrefix(r.Header.Get("Content-Type"), "application/json") {
		reply(http.StatusForbidden, "нужны заголовки Content-Type: application/json и X-GPU-Router: 1")
		return
	}
	var req struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096)).Decode(&req); err != nil || req.ID == "" {
		reply(http.StatusBadRequest, "ожидался JSON вида {\"id\":\"t12\"}")
		return
	}
	code, msg := s.cancelTask(req.ID, clientAddr(r))
	reply(code, msg)
}
