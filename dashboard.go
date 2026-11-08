package main

// Панель состояния для home.home: отдельный маленький HTTP-сервер с одним
// JSON-эндпоинтом /api/status. Слушает отдельный порт (по умолчанию
// 127.0.0.1:8190), не проходит через loggingMW и очередь GPU, поэтому
// опрос страницы не засоряет лог и не мешает задачам.

import (
	"context"
	"encoding/json"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// DashService — сервис, чью доступность показываем на странице.
type DashService struct {
	Name string `yaml:"name" json:"name"`
	Addr string `yaml:"addr" json:"-"`    // host:port для TCP-проверки
	URL  string `yaml:"url" json:"url"`   // ссылка для кнопки на странице
	Desc string `yaml:"desc" json:"desc"` // короткое описание
}

type dashSvcStatus struct {
	DashService
	Up        bool `json:"up"`
	LatencyMS int  `json:"latency_ms"`
}

type dashGPUMem struct {
	VRAMUsedMB  int `json:"vram_used_mb"`
	VRAMTotalMB int `json:"vram_total_mb"`
	GTTUsedMB   int `json:"gtt_used_mb"`
	GTTTotalMB  int `json:"gtt_total_mb"`
}

type dashBackend struct {
	Up       bool       `json:"up"`
	State    string     `json:"state"` // busy | loaded | idle | down
	VRAMMB   *int       `json:"vram_mb"`
	Dirty    bool       `json:"dirty"`
	Running  int        `json:"running,omitempty"`  // ComfyUI: задач в работе
	Pending  int        `json:"pending,omitempty"`  // ComfyUI: задач в очереди самого ComfyUI
	Models   []string   `json:"models,omitempty"`   // llama: загруженные модели
	Speed    *tokenSnap `json:"speed,omitempty"`    // llama: токенов в секунду
	Inflight int        `json:"inflight,omitempty"` // llama: запросов идёт сейчас
	Slots    int        `json:"slots,omitempty"`    // llama: слотов у текущей модели (parallel)
	Model    string     `json:"model,omitempty"`    // llama: модель, к которой идут запросы
}

type dashStatus struct {
	Time   int64 `json:"time"`
	Active struct {
		Service  string `json:"service"` // "", "comfyui", "llama"
		Phase    string `json:"phase"`   // "", preparing, running
		SinceSec int    `json:"since_sec"`
	} `json:"active"`
	QueueSize  int             `json:"queue_size"`
	QueueMax   int             `json:"queue_max"`
	GPUMode    string          `json:"gpu_mode"`
	MemAvailMB *int            `json:"mem_available_mb"`
	GPUMem     *dashGPUMem     `json:"gpu_mem"`
	ComfyUI    dashBackend     `json:"comfyui"`
	LLAMA      dashBackend     `json:"llama"`
	Services   []dashSvcStatus `json:"services"`
}

// defaultDashServices — набор по умолчанию, если в config.yaml нет dashboard.services.
func defaultDashServices(cfg *Config) []DashService {
	return []DashService{
		{Name: "Open WebUI", Addr: "127.0.0.1:3000", URL: "https://openwebui.home", Desc: "Чат с моделями"},
		{Name: "ComfyUI", Addr: net.JoinHostPort(cfg.ComfyUI.Host, cfg.ComfyUI.Port), URL: "https://comfyui.home", Desc: "Картинки и видео"},
		{Name: "llama.cpp", Addr: net.JoinHostPort(cfg.LLAMA.Host, cfg.LLAMA.Port), URL: "", Desc: "Сервер языковых моделей"},
		{Name: "n8n", Addr: "127.0.0.1:5678", URL: "https://n8n.home", Desc: "Автоматизации"},
		{Name: "SillyTavern", Addr: "127.0.0.1:8300", URL: "https://sillytavern.home", Desc: "Ролевые чаты"},
	}
}

// ---- кэш снимка (страниц может быть несколько, замер делаем не чаще раза в 2 с) ----

type dashCache struct {
	mu   sync.Mutex
	at   time.Time
	data []byte
}

const dashTTL = 2 * time.Second

func (s *Server) dashSnapshot() []byte {
	c := &s.dash
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.data != nil && time.Since(c.at) < dashTTL {
		return c.data
	}
	b, _ := json.Marshal(s.buildDashStatus())
	c.data, c.at = b, time.Now()
	return b
}

func (s *Server) buildDashStatus() *dashStatus {
	st := &dashStatus{Time: time.Now().Unix(), QueueMax: cap(s.taskQueue), QueueSize: len(s.taskQueue)}

	s.mu.Lock()
	active := s.activeService
	ready := s.activeReady
	since := s.activeSince
	inflight, lmodel, llimit := s.llamaInflight, s.llamaModel, s.llamaLimit
	dirty := map[string]bool{"comfyui": s.dirty["comfyui"], "llama": s.dirty["llama"]}
	s.mu.Unlock()

	if active != "" {
		st.Active.Service = active
		st.Active.Phase = "preparing"
		if ready {
			st.Active.Phase = "running"
		}
		if !since.IsZero() {
			st.Active.SinceSec = int(time.Since(since).Seconds())
		}
	}
	st.GPUMode = s.gpuMode()

	var wg sync.WaitGroup

	// ComfyUI
	wg.Add(1)
	go func() {
		defer wg.Done()
		b := &st.ComfyUI
		b.Dirty = dirty["comfyui"]
		r, p, err := s.comfyBusy()
		b.Up = err == nil
		b.Running, b.Pending = r, p
	}()

	// llama
	wg.Add(1)
	go func() {
		defer wg.Done()
		b := &st.LLAMA
		b.Dirty = dirty["llama"]
		models, err := s.llamaLoaded()
		b.Up = err == nil
		b.Models = models
	}()

	// VRAM по сервисам
	var usage map[string]*vramUsage
	wg.Add(1)
	go func() {
		defer wg.Done()
		usage = s.gpuUsage()
	}()

	// Остальные сервисы — параллельно, TCP-проверка
	svcs := s.Config.Dashboard.Services
	st.Services = make([]dashSvcStatus, len(svcs))
	for i, sv := range svcs {
		st.Services[i].DashService = sv
		wg.Add(1)
		go func(i int, sv DashService) {
			defer wg.Done()
			t0 := time.Now()
			ctx, cancel := context.WithTimeout(context.Background(), 1500*time.Millisecond)
			defer cancel()
			var d net.Dialer
			conn, err := d.DialContext(ctx, "tcp", sv.Addr)
			if err == nil {
				conn.Close()
				st.Services[i].Up = true
				st.Services[i].LatencyMS = int(time.Since(t0).Milliseconds())
			}
		}(i, sv)
	}

	if avail, err := memAvailableMB(); err == nil {
		st.MemAvailMB = &avail
	}
	st.GPUMem = readGPUMem()

	wg.Wait()

	sp := s.tokens.snap()
	st.LLAMA.Speed = &sp
	st.LLAMA.Inflight, st.LLAMA.Model, st.LLAMA.Slots = inflight, lmodel, llimit

	for svc, b := range map[string]*dashBackend{"comfyui": &st.ComfyUI, "llama": &st.LLAMA} {
		if u := usage[svc]; u != nil && u.OK {
			mb := u.MB
			b.VRAMMB = &mb
		}
		limit := s.Config.backend(svc).ResidualMB
		switch {
		case !b.Up:
			b.State = "down"
		case active == svc && ready:
			b.State = "busy"
		case svc == "comfyui" && b.Running+b.Pending > 0:
			b.State = "busy"
		case len(b.Models) > 0 || (b.VRAMMB != nil && *b.VRAMMB > limit):
			b.State = "loaded" // модель в памяти, но прямо сейчас не работает
		default:
			b.State = "idle"
		}
	}
	return st
}

// readGPUMem — занятость VRAM и GTT из sysfs amdgpu (nil, если недоступно).
func readGPUMem() *dashGPUMem {
	cards, _ := filepath.Glob("/sys/class/drm/card*/device/mem_info_gtt_used")
	for _, f := range cards {
		dir := filepath.Dir(f)
		rd := func(name string) int {
			b, err := os.ReadFile(filepath.Join(dir, name))
			if err != nil {
				return -1
			}
			n, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
			if err != nil {
				return -1
			}
			return int(n >> 20)
		}
		m := &dashGPUMem{
			VRAMUsedMB: rd("mem_info_vram_used"), VRAMTotalMB: rd("mem_info_vram_total"),
			GTTUsedMB: rd("mem_info_gtt_used"), GTTTotalMB: rd("mem_info_gtt_total"),
		}
		if m.VRAMUsedMB < 0 || m.GTTUsedMB < 0 {
			continue
		}
		return m
	}
	return nil
}

// startDashboard поднимает сервер панели. Не блокирует. port <= 0 — панель выключена.
func (s *Server) startDashboard() *http.Server {
	d := &s.Config.Dashboard
	if d.Port <= 0 {
		return nil
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/status", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		w.Write(s.dashSnapshot())
	})
	srv := &http.Server{
		Addr:              net.JoinHostPort(d.ListenHost, strconv.Itoa(d.Port)),
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		log.Printf("[Dashboard] http://%s/api/status", srv.Addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("[Dashboard] Сервер: %v", err)
		}
	}()
	return srv
}
