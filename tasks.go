package main

// Реестр задач для страницы состояния: что стоит в очереди, что ждёт слот, что выполняется,
// и возможность снять задачу с очереди (или остановить идущий запрос llama) с этой страницы.
//
// Канал очереди перечислить нельзя, поэтому каждая задача регистрируется здесь при создании
// и удаляется, когда её обработчик завершился.

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"sort"
	"strings"
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

type dashTask struct {
	ID        string `json:"id"`
	Service   string `json:"service"`
	Model     string `json:"model,omitempty"`
	State     string `json:"state"` // queued | waiting | preparing | running
	Client    string `json:"client,omitempty"`
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
		out = append(out, dashTask{
			ID: t.ID, Service: t.Service, Model: t.Model, State: stateName(st), Client: t.Client,
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
