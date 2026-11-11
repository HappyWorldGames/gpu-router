package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/google/uuid"
	"gopkg.in/yaml.v3"
)

// ==================== КОНФИГ ====================

// BackendCfg — настройки одного бэкенда (ComfyUI или llama.cpp).
type BackendCfg struct {
	Host string `yaml:"host"`
	Port string `yaml:"port"`
	// Unit — имя systemd-юнита. По нему из /proc/<pid>/cgroup определяется,
	// какие процессы на GPU принадлежат сервису (учёт VRAM и «выбивание»).
	Unit string `yaml:"unit"`
	// ResidualMB — сколько VRAM сервис вправе удерживать после выгрузки
	// (CUDA-контекст и прочее). Больше — считается утечкой.
	ResidualMB int `yaml:"residual_mb"`
	// IdleRelease — через сколько секунд простоя выгружать сервис заранее.
	// 0 = выгружать только когда GPU понадобился другому сервису.
	IdleRelease int `yaml:"idle_release"`
	// ModelsINI — путь к models.ini llama.cpp (только для секции llama). Из него берётся
	// parallel каждой модели: столько запросов к ней прокси пускает одновременно.
	// Пусто = параллельные запросы выключены (строго по одному, как раньше).
	ModelsINI string `yaml:"models_ini"`
}

type Config struct {
	ComfyUI BackendCfg `yaml:"comfyui"`
	LLAMA   BackendCfg `yaml:"llama"`
	Proxy   struct {
		ComfyUIPort int `yaml:"comfyui_port"`
		LLAMAPort   int `yaml:"llama_port"`
		// ListenHost — на каком адресе слушать. Пусто = все интерфейсы.
		// Если все клиенты (Open WebUI, n8n) на этой же машине — ставьте 127.0.0.1.
		ListenHost string `yaml:"listen_host"`
	} `yaml:"proxy"`
	Queue struct {
		MaxSize          int `yaml:"max_size"`
		WaitTimeout      int `yaml:"wait_timeout"`
		PollInterval     int `yaml:"poll_interval"`
		ComfyIdleTimeout int `yaml:"comfy_idle_timeout"`
		ExecIdleTimeout  int `yaml:"exec_idle_timeout"` // нет данных от бэкенда дольше — обрыв
		TaskTimeout      int `yaml:"task_timeout"`      // жёсткий потолок на одну задачу
	} `yaml:"queue"`
	// Dashboard — страница состояния для home.home (см. dashboard.go).
	Dashboard struct {
		Port       int           `yaml:"port"`        // 0 = 8190, -1 = выключить
		ListenHost string        `yaml:"listen_host"` // по умолчанию 127.0.0.1 (к нему ходит Caddy)
		Services   []DashService `yaml:"services"`    // пусто = набор по умолчанию
	} `yaml:"dashboard"`
	Logging struct {
		Verbose  *bool `yaml:"verbose"`   // false = только важное; по умолчанию true
		Requests *bool `yaml:"requests"`  // логировать входящие запросы (если verbose)
		MemTrace *int  `yaml:"mem_trace"` // период срезов памяти во время задачи, сек; 0 = выкл (по умолчанию 30)
	} `yaml:"logging"`
	GPU struct {
		ReleaseTimeout   int    `yaml:"release_timeout"`   // сколько ждать мягкой выгрузки, потом SIGKILL
		WatchdogInterval int    `yaml:"watchdog_interval"` // период проверки конфликтов VRAM
		RestartWait      int    `yaml:"restart_wait"`      // сколько ждать подъёма сервиса после SIGKILL
		KillOnLeak       *bool  `yaml:"kill_on_leak"`      // по умолчанию true
		NvidiaSMI        string `yaml:"nvidia_smi"`
		Backend          string `yaml:"backend"`               // auto | fdinfo (AMD/Intel) | nvidia | none
		MinAvailableMB   int    `yaml:"min_available_mb"`      // порог MemAvailable (ОЗУ); -1 = не проверять
		MemWait          int    `yaml:"mem_wait"`              // сколько ждать восстановления ОЗУ перед задачей, сек
		CriticalAvailMB  int    `yaml:"critical_available_mb"` // ниже — аварийная остановка работающего GPU-сервиса; -1 = выкл
		// CriticalIdleMB — тот же порог, но когда GPU никому не нужен (нет идущей задачи и очередь пуста).
		// Тогда модель держит память впустую, а виновата обычно не она (например, сборка у агента),
		// поэтому выбиваем её, только если ОЗУ упала совсем низко. 0 = половина critical_available_mb,
		// -1 = при простое никогда не выбивать (риск: OOM-killer ядра выберет жертву сам).
		CriticalIdleMB int `yaml:"critical_idle_available_mb"`
	} `yaml:"gpu"`
}

func (c *Config) comfyURL(p string) string {
	return "http://" + c.ComfyUI.Host + ":" + c.ComfyUI.Port + p
}
func (c *Config) llamaURL(p string) string { return "http://" + c.LLAMA.Host + ":" + c.LLAMA.Port + p }

func (c *Config) backend(service string) *BackendCfg {
	if service == "llama" {
		return &c.LLAMA
	}
	return &c.ComfyUI
}

func (c *Config) verbose() bool { return c.Logging.Verbose == nil || *c.Logging.Verbose }

func (c *Config) logRequests() bool {
	return c.verbose() && (c.Logging.Requests == nil || *c.Logging.Requests)
}

// memTraceEvery — период периодических срезов памяти во время задачи (0 = выключено).
func (c *Config) memTraceEvery() time.Duration {
	if !c.verbose() {
		return 0
	}
	if c.Logging.MemTrace == nil {
		return 30 * time.Second
	}
	if *c.Logging.MemTrace <= 0 {
		return 0
	}
	return time.Duration(*c.Logging.MemTrace) * time.Second
}

func (c *Config) killOnLeak() bool { return c.GPU.KillOnLeak == nil || *c.GPU.KillOnLeak }

func loadConfig(path string) (*Config, error) {
	cfg := &Config{}
	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, fmt.Errorf("cannot read config %s: %w", path, err)
	}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return cfg, fmt.Errorf("cannot parse config %s: %w", path, err)
	}
	return cfg, nil
}

func applyDefaults(cfg *Config) {
	if cfg.ComfyUI.Host == "" {
		cfg.ComfyUI.Host = "127.0.0.1"
	}
	if cfg.ComfyUI.Port == "" {
		cfg.ComfyUI.Port = "8288"
	}
	if cfg.LLAMA.Host == "" {
		cfg.LLAMA.Host = "127.0.0.1"
	}
	if cfg.LLAMA.Port == "" {
		cfg.LLAMA.Port = "8180"
	}
	if cfg.ComfyUI.ResidualMB <= 0 {
		cfg.ComfyUI.ResidualMB = 2048
	}
	if cfg.LLAMA.ResidualMB <= 0 {
		cfg.LLAMA.ResidualMB = 512
	}
	if cfg.Proxy.ComfyUIPort <= 0 {
		cfg.Proxy.ComfyUIPort = 8188
	}
	if cfg.Proxy.LLAMAPort <= 0 {
		cfg.Proxy.LLAMAPort = 8080
	}
	if cfg.Queue.MaxSize <= 0 {
		cfg.Queue.MaxSize = 100
	}
	if cfg.Queue.WaitTimeout <= 0 {
		cfg.Queue.WaitTimeout = 30
	}
	if cfg.Queue.PollInterval <= 0 {
		cfg.Queue.PollInterval = 2
	}
	// comfy_idle_timeout = 0 → безлимит, оставляем как есть
	if cfg.Queue.ExecIdleTimeout <= 0 {
		cfg.Queue.ExecIdleTimeout = 300
	}
	if cfg.Queue.TaskTimeout <= 0 {
		cfg.Queue.TaskTimeout = 14400
	}
	if cfg.Queue.ComfyIdleTimeout > 0 && cfg.Queue.TaskTimeout < cfg.Queue.ComfyIdleTimeout+600 {
		cfg.Queue.TaskTimeout = cfg.Queue.ComfyIdleTimeout + 600
	}
	if cfg.GPU.ReleaseTimeout <= 0 {
		cfg.GPU.ReleaseTimeout = 30
	}
	if cfg.GPU.WatchdogInterval <= 0 {
		cfg.GPU.WatchdogInterval = 2
	}
	if cfg.GPU.RestartWait <= 0 {
		cfg.GPU.RestartWait = 180
	}
	if cfg.GPU.MinAvailableMB == 0 {
		cfg.GPU.MinAvailableMB = 2048
	}
	if cfg.GPU.CriticalAvailMB == 0 {
		cfg.GPU.CriticalAvailMB = 1024
	}
	if cfg.GPU.MemWait <= 0 {
		cfg.GPU.MemWait = 20
	}
	if cfg.GPU.NvidiaSMI == "" {
		cfg.GPU.NvidiaSMI = "nvidia-smi"
	}
	if cfg.Dashboard.Port == 0 {
		cfg.Dashboard.Port = 8190
	}
	if cfg.Dashboard.ListenHost == "" {
		cfg.Dashboard.ListenHost = "127.0.0.1"
	}
	if len(cfg.Dashboard.Services) == 0 {
		cfg.Dashboard.Services = defaultDashServices(cfg)
	}
}

// ==================== ОБЩИЕ HTTP-КЛИЕНТЫ ====================
//
// backendClient — для проксируемых запросов (стримы, без глобального Timeout).
// controlClient — для служебных вызовов (queue/free/unload...), таймаут задаётся
// через context в httpDo.

var backendClient = &http.Client{
	Transport: &http.Transport{
		MaxIdleConns:        32,
		MaxIdleConnsPerHost: 8,
		IdleConnTimeout:     90 * time.Second,
	},
}

var controlClient = &http.Client{
	Transport: &http.Transport{
		MaxIdleConns:        8,
		MaxIdleConnsPerHost: 4,
		IdleConnTimeout:     30 * time.Second,
	},
}

// httpDo — короткий служебный запрос с JSON-телом (payload может быть nil).
func httpDo(method, target string, payload interface{}, timeout time.Duration) (int, []byte, error) {
	var rd io.Reader
	if payload != nil {
		b, err := json.Marshal(payload)
		if err != nil {
			return 0, nil, err
		}
		rd = bytes.NewReader(b)
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, method, target, rd)
	if err != nil {
		return 0, nil, err
	}
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := controlClient.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	return resp.StatusCode, data, nil
}

// isDown — бэкенд не запущен / упал (в отличие от «не успел ответить»).
func isDown(err error) bool {
	return errors.Is(err, syscall.ECONNREFUSED) ||
		errors.Is(err, syscall.ECONNRESET) ||
		errors.Is(err, io.EOF)
}

// ==================== СЕРВЕР И ОЧЕРЕДЬ ====================

type Task struct {
	Service    string
	Path       string
	Body       []byte
	W          http.ResponseWriter
	Ctx        context.Context
	Cancel     context.CancelFunc
	EnqueuedAt time.Time
	Turn       chan struct{}
	Done       chan struct{}
	Err        error
	PromptID   string
	Model      string // llama: значение поля "model" из тела запроса
}

// abandoned — задача уже не нужна (клиент ушёл / обработчик завершился).
func (t *Task) abandoned() bool {
	select {
	case <-t.Done:
		return true
	case <-t.Ctx.Done():
		return true
	default:
		return false
	}
}

// discardResponseWriter — заглушка: ответ уже отправлен клиенту,
// execBackend для ComfyUI-задачи пишет в пустоту.
type discardResponseWriter struct{}

func (discardResponseWriter) Header() http.Header         { return http.Header{} }
func (discardResponseWriter) Write(p []byte) (int, error) { return len(p), nil }
func (discardResponseWriter) WriteHeader(int)             {}

// logRequests включает построчный лог входящих запросов (читает loggingMW).
var logRequests atomic.Bool

type Server struct {
	Config *Config

	mu            sync.Mutex // защищает поля ниже
	activeService string
	activeReady   bool                 // GPU передан сервису, задача реально выполняется
	activeSince   time.Time            // когда воркер взял задачу
	cond          *sync.Cond           // будит воркер, когда освободился слот llama (на mu)
	llamaInflight int                  // сколько запросов llama идёт прямо сейчас
	llamaModel    string               // модель, к которой они обращаются
	llamaLimit    int                  // сколько слотов у этой модели (parallel)
	ini           *modelsINI           // models.ini; nil = параллели нет
	lastUsed      map[string]time.Time // когда сервис последний раз закончил задачу
	lastKill      map[string]time.Time // когда сервис последний раз «выбивали»
	dirty         map[string]bool      // сервис работал с GPU и ещё не выгружался

	taskQueue chan *Task

	// opMu сериализует операции передачи GPU (worker) и починки (watchdog):
	// две выгрузки/захвата одновременно — это и есть гонка, из-за которой падает OOM.
	opMu sync.Mutex

	smiWarned      atomic.Bool
	fdDeniedWarned atomic.Bool
	lastMemWarn    atomic.Int64 // unix-nano последнего предупреждения о нехватке ОЗУ
	lastMemTrace   atomic.Int64 // unix-nano последней трассировки памяти во время задачи

	dash   dashCache  // кэш снимка для страницы состояния
	tokens tokenMeter // скорость генерации llama (токенов/с), см. tokens.go

	admitWaiting atomic.Int32 // сколько задач воркер держит в admit() в ожидании (см. parallel.go)

	modeOnce sync.Once
	mode     string // выбранный источник данных о VRAM
}

func NewServer(cfg *Config) *Server {
	now := time.Now()
	logRequests.Store(cfg.logRequests())
	s := &Server{
		Config:    cfg,
		lastUsed:  map[string]time.Time{"comfyui": now, "llama": now},
		lastKill:  map[string]time.Time{},
		dirty:     map[string]bool{},
		taskQueue: make(chan *Task, cfg.Queue.MaxSize),
		ini:       newModelsINI(cfg.LLAMA.ModelsINI),
	}
	s.cond = sync.NewCond(&s.mu)
	return s
}

// vlogf — подробный лог; в режиме logging.verbose=false не печатается.
func (s *Server) vlogf(format string, args ...interface{}) {
	if s.Config.verbose() {
		log.Printf(format, args...)
	}
}

// memSuffix — сводка памяти в конце строк [Start]/[Done] (только в подробном режиме).
func (s *Server) memSuffix() string {
	if !s.Config.verbose() {
		return ""
	}
	return ". " + memStatus()
}

func otherService(service string) string {
	if service == "llama" {
		return "comfyui"
	}
	return "llama"
}

// ==================== ЛОГИРОВАНИЕ ====================

// ComfyUI: пути, которые фронт опрашивает десятками в секунду — не логируем.
//
//	var comfyNoisyPrefixes = []string{
//		"/assets/",
//		"/extensions/",
//		"/scripts/",
//		"/fonts/",
//		"/history/",                 // /history/<uuid> — опрос результата генерации
//		"/api/jobs",
//		"/api/view",
//		"/api/object_info",
//		"/api/global_subgraphs/",
//		"/api/manager/",
//		"/api/experiments/",
//		"/api/node_replacements",
//		"/api/system_stats",
//		"/api/features",
//		"/api/users",
//		"/api/settings",
//		"/api/i18n",
//		"/api/extensions",
//		"/api/customnode/",
//		"/api/queue",                // очередь ComfyUI (не путать с нашей)
//		"/manager/",
//		"/user.css",
//		"/materialdesignicons.min.css",
//	}
//
// ComfyUI: пути, которые реально влияют на GPU или важны для отладки.
// Всё остальное — UI-шум, не логируем.
var comfyWorkPaths = map[string]bool{
	// Очередь GPU
	"/prompt":     true, // POST — постановка задачи
	"/api/prompt": true,

	// Управление текущей генерацией
	"/interrupt":     true, // POST — отмена активной генерации
	"/api/interrupt": true,
	"/queue":         true, // POST — очистка очереди ComfyUI
	"/api/queue":     true,
}

// LLAMA: пути, которые вообще НЕ логируем (служебные).
// Всё, чего нет в этом списке, логируется — только реальная работа.
var llamaSilentExact = map[string]bool{
	"/v1/models":         true,
	"/models":            true,
	"/models/sse":        true,
	"/props":             true,
	"/health":            true,
	"/metrics":           true,
	"/slots":             true,
	"/tokenize":          true,
	"/detokenize":        true,
	"/tools":             true,
	"/build.json":        true,
	"/_app/version.json": true,
}

// Пути llama.cpp, которые НЕ грузят модель в VRAM и не должны
// попадать в FIFO-очередь. Проксируются напрямую в llama-server.
//
// Источник: tools/server/server.cpp (control-plane маршруты).
var llamaServicePaths = map[string]bool{
	// Server metadata
	"/props":             true,
	"/v1/streams/lookup": true,
	"/v1/streams":        true, // на случай GET-листинга в старых сборках
	"/v1/stream/":        true, // префикс для GET/DELETE /v1/stream/<conv_id>
	"/health":            true,
	"/v1/health":         true,
	"/metrics":           true,
	"/build.json":        true,
	"/_app/version.json": true,

	// Model catalog (router-mode)
	"/models":     true,
	"/v1/models":  true,
	"/models/sse": true,

	// Выгрузка модели освобождает VRAM — очередь ей не нужна
	"/models/unload": true,

	// Slots (native)
	"/slots": true,

	// Tokenization — использует токенизатор, но не GPU
	"/tokenize":       true,
	"/detokenize":     true,
	"/apply-template": true,

	// LoRA metadata
	"/lora-adapters": true,

	// MCP tools (WebUI --webui-mcp-proxy)
	"/tools":    true,
	"/v1/tools": true, // на случай OpenAI-совместимого алиаса

	// WebUI assets
	"/manifest.webmanifest": true,
	"/favicon.ico":          true,
}

func isLlamaServicePath(p string) bool {
	if llamaServicePaths[p] {
		return true
	}
	if strings.HasPrefix(p, "/_app/") {
		return true
	}
	// /v1/stream/<conv_id> — GET (replay) или DELETE (cancel)
	if strings.HasPrefix(p, "/v1/stream/") {
		return true
	}
	return false
}

// loggingMW логирует только значимые запросы. Спам от поллинга ComfyUI
// и служебных эндпоинтов llama (модели, props, tokenize, WebUI) — гасится.
// slowBypassThreshold: запрос llama МИМО очереди (служебный путь или GET), который
// идёт дольше этого времени, считается подозрительным и попадает в лог.
var slowBypassThreshold = 5 * time.Second

// statusWriter запоминает код ответа, не ломая стриминг (Flush) и ResponseController.
type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	if w.status == 0 {
		w.status = code
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Write(b []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return w.ResponseWriter.Write(b)
}

func (w *statusWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (w *statusWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

// longLived: пути, которые по природе висят долго (потоки событий) — не считаем «зависшими».
func longLived(path string) bool {
	return path == "/models/sse" || strings.HasPrefix(path, "/v1/stream")
}

func loggingMW(name string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Path
		var show bool
		start := time.Now()
		// Запросы к llama мимо очереди: если такой запрос застрянет (например, router занят
		// загрузкой модели), без этой записи в логе не было бы ничего.
		bypass := name == "LLAMA" && !longLived(path) && (isLlamaServicePath(path) || r.Method != http.MethodPost)
		sw := &statusWriter{ResponseWriter: w}
		w = sw
		if bypass {
			defer func() {
				if d := time.Since(start); d > slowBypassThreshold {
					log.Printf("[LLAMA] запрос МИМО очереди шёл %v: %s %s → HTTP %d (клиент %s)",
						d.Round(time.Millisecond), r.Method, path, sw.status, r.RemoteAddr)
				}
			}()
		}

		switch name {
		case "LLAMA":
			// Возобновление/просмотр потоков встроенного WebUI (/v1/stream*): это не работа с GPU,
			// но именно такие запросы могут «висеть», если WebUI пытается продолжить оборванный
			// ответ, поэтому их видно в логе.
			if strings.HasPrefix(path, "/v1/stream") {
				show = true
				break
			}
			// Остальные служебные пути (мимо очереди, GPU не трогают) — не логируем.
			if isLlamaServicePath(path) {
				show = false
				break
			}
			// Любой оставшийся POST — потенциальная работа, всегда логируем.
			if r.Method == http.MethodPost {
				show = true
				break
			}
			// GET-спам от WebUI (/props, /models, /_app/*) — гасим.
			show = !llamaSilentExact[path] &&
				!strings.HasPrefix(path, "/_app/") &&
				path != "/"
		// case "ComfyUI":
		// 	// Исключение: сохранение workflow — полезно видеть
		// 	if path == "/api/userdata/workflows" || strings.HasPrefix(path, "/api/userdata/workflows/") {
		// 		show = true
		// 	} else {
		// 		show = true
		// 		for _, p := range comfyNoisyPrefixes {
		// 			if strings.HasPrefix(path, p) {
		// 				show = false
		// 				break
		// 			}
		// 		}
		// 	}
		case "ComfyUI":
			// Логируем только то, что реально работает с GPU-очередью.
			show = comfyWorkPaths[path]

			// Исключение: сохранение/загрузка workflow — полезно видеть
			// (юзер работает с воркфлоу, понимаем что происходит).
			if strings.HasPrefix(path, "/api/userdata/workflows/") {
				show = true
			}
		default:
			show = true
		}

		if show && logRequests.Load() {
			log.Printf("[%s REQ] %s %s от %s", name, r.Method, path, r.RemoteAddr)
		}
		next.ServeHTTP(w, r)
	})
}

// ==================== СТАРТ / ОСТАНОВКА ====================

func (s *Server) Start() {
	log.Println("GPU Resource Proxy — Go версия")
	log.Printf("ComfyUI backend: %s:%s", s.Config.ComfyUI.Host, s.Config.ComfyUI.Port)
	log.Printf("LLAMA   backend: %s:%s", s.Config.LLAMA.Host, s.Config.LLAMA.Port)
	s.gpuSelfCheck()

	// --- ComfyUI-прокси ---
	muxComfy := http.NewServeMux()
	// Путь на бэкенд = r.URL.Path, поэтому /api/prompt → /api/prompt, /prompt → /prompt.
	// Это устраняет класс ошибок "вход один, на бэкенд отдаём другой".
	muxComfy.HandleFunc("/prompt", s.gpuTask("comfyui"))
	muxComfy.HandleFunc("/api/prompt", s.gpuTask("comfyui"))
	muxComfy.HandleFunc("/status", s.status)
	muxComfy.Handle("/", s.reverseProxy(s.Config.comfyURL("/")))

	// --- LLAMA-прокси ---
	muxLlama := http.NewServeMux()
	muxLlama.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// Служебные пути llama.cpp — напрямую, минуя очередь.
		// Они не грузят модель в VRAM: /props, /tokenize, /models, /slots и т.д.
		if isLlamaServicePath(r.URL.Path) {
			s.reverseProxy(s.Config.llamaURL("/")).ServeHTTP(w, r)
			return
		}
		// Любой другой POST — потенциальная генерация (autoload модели
		// у router-сборки), поэтому через очередь.
		if r.Method == http.MethodPost {
			s.gpuTask("llama")(w, r)
			return
		}
		// GET/HEAD/OPTIONS — читаем напрямую: health/models/props/WebUI-статика
		// модель не загружают.
		s.reverseProxy(s.Config.llamaURL("/")).ServeHTTP(w, r)
	})
	// /status оставляем своим, он отладочный
	muxLlama.HandleFunc("/status", s.status)

	serverLlama := &http.Server{
		Addr:    s.listenAddr(s.Config.Proxy.LLAMAPort),
		Handler: loggingMW("LLAMA", muxLlama),
	}
	serverComfy := &http.Server{
		Addr:    s.listenAddr(s.Config.Proxy.ComfyUIPort),
		Handler: loggingMW("ComfyUI", muxComfy),
	}

	go func() {
		if err := serverComfy.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("[ComfyUI] Сервер: %v", err)
		}
	}()
	go func() {
		if err := serverLlama.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("[LLAMA] Сервер: %v", err)
		}
	}()

	serverDash := s.startDashboard()

	go s.worker()
	go s.watchdog()
	go s.memGuard()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan
	log.Println("Завершение работы...")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	serverComfy.Shutdown(ctx)
	serverLlama.Shutdown(ctx)
	if serverDash != nil {
		serverDash.Shutdown(ctx)
	}
}

func (s *Server) listenAddr(port int) string {
	return net.JoinHostPort(s.Config.Proxy.ListenHost, strconv.Itoa(port))
}

func (s *Server) reverseProxy(backend string) http.Handler {
	target, _ := url.Parse(backend)
	p := httputil.NewSingleHostReverseProxy(target)
	if s.Config.verbose() {
		p.ErrorLog = log.New(os.Stderr, "[Proxy] ", 0)
	} else {
		p.ErrorLog = log.New(io.Discard, "", 0) // «connection refused» при каждом перезапуске бэкенда
	}
	return p
}

// ==================== ОБРАБОТЧИКИ ВХОДА ====================

// gpuTask ставит задачу в очередь и ждёт своей очереди, стримя ответ бэкенда клиенту.
// Путь на бэкенд = r.URL.Path — не переписываем.
func (s *Server) gpuTask(service string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		readStart := time.Now()
		body, err := io.ReadAll(io.LimitReader(r.Body, 64<<20))
		if d := time.Since(readStart); d > 3*time.Second || err != nil {
			log.Printf("[%s] тело запроса от %s читалось %v (%d КБ, ошибка: %v) — клиент медленно шлёт или оборвал соединение",
				s.name(service), r.RemoteAddr, d.Round(time.Millisecond), len(body)/1024, err)
		}
		if err != nil {
			s.jsonError(w, http.StatusBadRequest, "не удалось прочитать тело")
			return
		}

		// ComfyUI /prompt — async submit: отдаём клиенту свой prompt_id
		// сразу, а реальную работу кладём в очередь с detached context.
		if service == "comfyui" {
			s.comfyPromptAsync(w, r, body)
			return
		}

		// LLAMA и всё остальное — держим коннект, пока задача не выполнится.
		ctx, cancel := context.WithCancel(r.Context())
		t := &Task{
			Service:    service,
			Path:       r.URL.Path,
			Body:       body,
			W:          w,
			Ctx:        ctx,
			Cancel:     cancel,
			EnqueuedAt: time.Now(),
			Turn:       make(chan struct{}),
			Done:       make(chan struct{}),
		}
		if service == "llama" {
			t.Model = requestModel(body)
		}
		// Done закрываем только после cancel: воркер не должен принять
		// уже умершую задачу за живую.
		defer func() {
			cancel()
			close(t.Done)
		}()

		select {
		case s.taskQueue <- t:
			s.vlogf("[Queue] %s в очереди (ждут: %d)", s.name(service), s.queueLen())
		default:
			s.jsonError(w, http.StatusServiceUnavailable, "очередь GPU переполнена")
			return
		}

		timer := time.NewTimer(time.Duration(s.Config.Queue.WaitTimeout) * time.Second)
		defer timer.Stop()
		select {
		case <-t.Turn:
			if t.Err != nil {
				s.jsonError(w, http.StatusBadGateway, "не удалось подготовить GPU: "+t.Err.Error())
			} else if ctx.Err() == nil {
				s.execBackend(t)
			} else {
				log.Printf("[Queue] %s: клиент ушёл в момент очереди", s.name(service))
			}
		case <-ctx.Done():
			log.Printf("[Queue] %s: клиент отключился, пропускаем", s.name(service))
		case <-timer.C:
			s.jsonError(w, http.StatusGatewayTimeout, "таймаут ожидания в очереди")
		}
	}
}

// comfyPromptAsync — обрабатывает POST /prompt и /api/prompt:
//  1. генерирует свой UUID и ставит задачу в очередь (переполнена → честный 503)
//  2. сразу отвечает клиенту {"prompt_id": <uuid>}
//  3. задача живёт в отдельной горутине с контекстом, не привязанным к HTTP-соединению
//  4. когда очередь дойдёт — внедряет prompt_id в тело и шлёт в ComfyUI,
//     после чего держит GPU-слот до реального завершения генерации
//
// Клиент (ComfyUI UI) опрашивает /history/<uuid>, который через reverse proxy
// уходит в ComfyUI — и находит там результат.
func (s *Server) comfyPromptAsync(w http.ResponseWriter, r *http.Request, body []byte) {
	promptID := uuid.NewString()

	ctx, cancel := context.WithCancel(context.Background())
	t := &Task{
		Service:    "comfyui",
		Path:       r.URL.Path,
		Body:       body,
		W:          discardResponseWriter{},
		Ctx:        ctx,
		Cancel:     cancel,
		EnqueuedAt: time.Now(),
		Turn:       make(chan struct{}),
		Done:       make(chan struct{}),
		PromptID:   promptID,
	}

	select {
	case s.taskQueue <- t:
		s.vlogf("[Queue] ComfyUI в очереди (размер: %d, prompt_id: %s)", len(s.taskQueue), promptID)
	default:
		cancel()
		log.Printf("[Queue] ComfyUI: очередь переполнена, prompt_id: %s", promptID)
		// Формат ошибки ComfyUI — фронт покажет сообщение, а не «успех» с пустой историей.
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error": map[string]interface{}{
				"type":       "gpu_queue_full",
				"message":    "очередь GPU переполнена",
				"details":    "",
				"extra_info": map[string]interface{}{},
			},
			"node_errors": map[string]interface{}{},
		})
		return
	}

	// Мгновенный ответ с явным Content-Length (иначе Go перейдёт на chunked).
	payload, _ := json.Marshal(map[string]interface{}{
		"prompt_id":   promptID,
		"number":      0,
		"node_errors": map[string]interface{}{},
	})
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
	w.WriteHeader(http.StatusOK)
	w.Write(payload)

	// Дальше задача живёт сама; HTTP-обработчик возвращается сразу.
	go s.runComfyTask(t)
}

// runComfyTask ждёт очереди, отправляет граф в ComfyUI и держит GPU-слот,
// пока ComfyUI реально не закончит. Если генерация не заканчивается
// (comfy_idle_timeout / task_timeout) — ComfyUI принудительно прерывается,
// а не «продолжаем как будто всё хорошо» (это и приводило к двойной загрузке VRAM).
func (s *Server) runComfyTask(t *Task) {
	defer func() {
		t.Cancel()
		close(t.Done)
	}()

	timer := time.NewTimer(time.Duration(s.Config.Queue.WaitTimeout) * time.Second)
	defer timer.Stop()

	select {
	case <-t.Turn:
		if t.Err != nil {
			log.Printf("[ComfyUI] %s: подготовка GPU упала: %v", t.PromptID, t.Err)
			return
		}
		if t.Ctx.Err() != nil {
			return
		}
		status := s.execBackend(t)
		if status < 200 || status >= 300 {
			log.Printf("[ComfyUI] %s: ComfyUI не принял задачу (статус %d) — слот свободен", t.PromptID, status)
			return
		}
		// Держим GPU-слот до реального завершения генерации.
		idle, wentDown := s.waitForComfyIdle(t.Ctx)
		if wentDown {
			log.Printf("[ComfyUI] %s: ComfyUI пропал во время задачи (упал/перезапущен — вероятно OOM), "+
				"результата не будет. %s. Больше всего ОЗУ держат: %s",
				t.PromptID, memStatus(), topMemProcs(5))
		}
		if !idle {
			log.Printf("[ComfyUI] %s: генерация не завершилась вовремя — прерываю ComfyUI", t.PromptID)
			s.opMu.Lock()
			s.release("comfyui", "генерация не завершилась вовремя")
			s.opMu.Unlock()
		}
	case <-t.Ctx.Done():
		log.Printf("[Queue] ComfyUI: context отменён, prompt_id: %s", t.PromptID)
	case <-timer.C:
		log.Printf("[Queue] ComfyUI: таймаут очереди, prompt_id: %s", t.PromptID)
	}
}

func (s *Server) status(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	active := s.activeService
	dirty := map[string]bool{"comfyui": s.dirty["comfyui"], "llama": s.dirty["llama"]}
	s.mu.Unlock()

	vram := map[string]interface{}{}
	for svc, u := range s.gpuUsage() {
		if u.OK {
			vram[svc] = u.MB
		} else {
			vram[svc] = nil
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"active_service": active,
		"queue_size":     s.queueLen(),
		"vram_mb":        vram,
		"dirty":          dirty,
	})
}

func (s *Server) name(service string) string {
	if service == "llama" {
		return "LLAMA"
	}
	return "ComfyUI"
}

// ==================== ВОРКЕР ====================

func (s *Server) worker() {
	taskLimit := time.Duration(s.Config.Queue.TaskTimeout) * time.Second

	for t := range s.taskQueue {
		if t.abandoned() {
			log.Printf("[Queue] %s: клиент ушёл, пропускаем", s.name(t.Service))
			continue
		}

		// Если llama уже работает: присоединяемся к идущим (свободные слоты модели)
		// либо ждём, пока они закончат. true = задача уже обработана.
		if s.admit(t) {
			continue
		}

		// Передача GPU идёт под opMu — watchdog в это время ничего не трогает.
		s.opMu.Lock()
		if t.abandoned() {
			s.opMu.Unlock()
			log.Printf("[Queue] %s: клиент ушёл, пропускаем", s.name(t.Service))
			continue
		}
		s.mu.Lock()
		s.activeService = t.Service
		s.activeReady = false
		s.activeSince = time.Now()
		s.mu.Unlock()

		taskStart := time.Now()
		prepErr := s.prepareGPU(t.Service)
		prepDone := time.Now()
		s.opMu.Unlock()

		t.Err = prepErr
		if prepErr != nil {
			log.Printf("[Switch] ошибка подготовки GPU: %v", prepErr)
		} else {
			log.Printf("[Start] %s — ваш ход (ждал %v, подготовка %v)%s",
				s.name(t.Service), time.Since(t.EnqueuedAt).Round(time.Millisecond),
				prepDone.Sub(taskStart).Round(time.Millisecond), s.memSuffix())
		}
		if prepErr == nil {
			s.mu.Lock()
			s.activeReady = true
			s.activeSince = time.Now() // отсчёт «работает уже N» — с момента выдачи GPU
			s.mu.Unlock()
		}
		close(t.Turn)

		// llama не ждём: следующие запросы к этой модели могут идти рядом (см. parallel.go).
		if t.Service == "llama" && prepErr == nil {
			s.startLlamaGroup(t, taskStart, prepDone)
			continue
		}

		// Ждём конец задачи, но не вечно: зависший обработчик не должен вешать всю очередь.
		s.waitDoneOrLimit(t, taskLimit)

		s.mu.Lock()
		s.activeService = ""
		s.activeReady = false
		s.lastUsed[t.Service] = time.Now()
		s.dirty[t.Service] = true
		s.mu.Unlock()
		log.Printf("[Done] %s завершил (всего %v, из них подготовка %v)%s",
			s.name(t.Service),
			time.Since(taskStart).Round(time.Millisecond),
			prepDone.Sub(taskStart).Round(time.Millisecond), s.memSuffix())
	}
}

// prepareGPU вызывается воркером под opMu. После возврата другой сервис
// гарантированно не держит VRAM (проверено по API и по реальному VRAM),
// а если не отпускал — выбит.
func (s *Server) prepareGPU(service string) error {
	if service == "comfyui" {
		s.releaseIfNeeded("llama", "очередь ComfyUI")
	} else {
		// Свои comfy-задачи слот отдают уже после idle; тут ждём только чужую
		// (внеочередную) генерацию, а по таймауту release её прервёт.
		_, _ = s.waitForComfyIdle(context.Background())
		s.releaseIfNeeded("comfyui", "очередь LLAMA")
	}
	// Сервис мог упасть (OOM) и перезапускаться systemd'ом — ждём подъёма.
	if err := s.waitBackend(service); err != nil {
		return err
	}
	s.waitForMemory(service)
	return nil
}

// releaseIfNeeded выгружает сервис, только если он работал после прошлой выгрузки
// или реально держит VRAM. Подряд идущие задачи одного сервиса не дёргают выгрузку.
func (s *Server) releaseIfNeeded(service, reason string) {
	s.mu.Lock()
	dirty := s.dirty[service]
	s.mu.Unlock()
	if !dirty && s.serviceState(service).OK {
		return
	}
	s.release(service, reason)
}

// ==================== УЧЁТ VRAM ====================

type gpuProc struct {
	PID int
	MB  int
}

type vramUsage struct {
	MB   int
	PIDs []int
	OK   bool // false — измерить нельзя (нет источника данных или не задан unit)
}

// gpuMode выбирает источник данных о VRAM один раз за запуск:
//
//	nvidia — nvidia-smi (видеокарты NVIDIA);
//	fdinfo — DRM fdinfo ядра + KFD sysfs (AMD, в т.ч. iGPU вроде 780M; Intel);
//	none   — не измерять, работать только по API.
//
// auto: nvidia-smi, если он есть и видит карту, иначе fdinfo.
func (s *Server) gpuMode() string {
	s.modeOnce.Do(func() {
		m := strings.ToLower(strings.TrimSpace(s.Config.GPU.Backend))
		switch m {
		case "nvidia", "fdinfo", "none":
		default:
			m = "fdinfo"
			if _, err := exec.LookPath(s.Config.GPU.NvidiaSMI); err == nil {
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				if exec.CommandContext(ctx, s.Config.GPU.NvidiaSMI, "-L").Run() == nil {
					m = "nvidia"
				}
				cancel()
			}
		}
		s.mode = m
	})
	return s.mode
}

func (s *Server) gpuProcs() ([]gpuProc, error) {
	switch s.gpuMode() {
	case "nvidia":
		return s.nvidiaProcs()
	case "fdinfo":
		return s.fdinfoProcs()
	}
	return nil, fmt.Errorf("измерение VRAM отключено (gpu.backend: none)")
}

func (s *Server) nvidiaProcs() ([]gpuProc, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, s.Config.GPU.NvidiaSMI,
		"--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil, err
	}
	sum := map[int]int{}
	for _, line := range strings.Split(string(out), "\n") {
		parts := strings.Split(strings.TrimSpace(line), ",")
		if len(parts) < 2 {
			continue
		}
		pid, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			continue
		}
		mb, err := strconv.Atoi(strings.TrimSpace(parts[1]))
		if err != nil {
			mb = 0 // "[N/A]"
		}
		sum[pid] += mb
	}
	procs := make([]gpuProc, 0, len(sum))
	for pid, mb := range sum {
		procs = append(procs, gpuProc{PID: pid, MB: mb})
	}
	return procs, nil
}

// fdinfoProcs — память GPU по процессам из ядра, без внешних утилит (так же работает nvtop).
// Просматриваются только процессы из cgroup'ов наших юнитов — это дёшево.
// Нужен root (или тот же пользователь, что у сервисов), иначе /proc/<pid>/fdinfo недоступен.
func (s *Server) fdinfoProcs() ([]gpuProc, error) {
	units := []string{normUnit(s.Config.ComfyUI.Unit), normUnit(s.Config.LLAMA.Unit)}
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return nil, err
	}
	var out []gpuProc
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		cg, err := os.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid))
		if err != nil {
			continue
		}
		match := false
		for _, u := range units {
			if u != "" && cgroupHasUnit(string(cg), u) {
				match = true
				break
			}
		}
		if !match {
			continue
		}
		if _, err := os.ReadDir(fmt.Sprintf("/proc/%d/fdinfo", pid)); err != nil && os.IsPermission(err) {
			if s.fdDeniedWarned.CompareAndSwap(false, true) {
				log.Printf("[GPU] ВНИМАНИЕ: нет прав читать /proc/%d/fdinfo — процесс другого пользователя. "+
					"Запустите gpu-proxy от того же пользователя, что и comfyui/llama (root не нужен)", pid)
			}
			continue
		}
		if mb, ok := procGPUMemoryMB(pid); ok {
			out = append(out, gpuProc{PID: pid, MB: mb})
		}
	}
	return out, nil
}

// parseDRMSize разбирает значение вида "123456 KiB" / "12 MiB" / "4096" (байты) в КиБ.
func parseDRMSize(v string) int64 {
	f := strings.Fields(v)
	if len(f) == 0 {
		return 0
	}
	n, err := strconv.ParseInt(f[0], 10, 64)
	if err != nil {
		return 0
	}
	if len(f) > 1 {
		switch f[1] {
		case "MiB":
			return n * 1024
		case "GiB":
			return n * 1024 * 1024
		case "B":
			return n / 1024
		}
		return n // KiB
	}
	return n / 1024
}

// procGPUMemoryMB — сколько памяти GPU (VRAM + GTT) держит процесс.
// Источники: drm-memory-vram/gtt из fdinfo (по уникальным drm-client-id, чтобы не
// считать дубликаты fd дважды) и KFD sysfs (ROCm/HIP-аллокации). Берём большее.
func procGPUMemoryMB(pid int) (int, bool) {
	found := false
	var fdKiB int64

	dir := fmt.Sprintf("/proc/%d/fdinfo", pid)
	if ents, err := os.ReadDir(dir); err == nil {
		seen := map[string]bool{}
		for _, e := range ents {
			b, err := os.ReadFile(dir + "/" + e.Name())
			if err != nil || !bytes.Contains(b, []byte("drm-driver")) {
				continue
			}
			var client string
			var vram, gtt int64
			isDRM := false
			for _, line := range strings.Split(string(b), "\n") {
				k, v, ok := strings.Cut(line, ":")
				if !ok {
					continue
				}
				v = strings.TrimSpace(v)
				switch strings.TrimSpace(k) {
				case "drm-driver":
					isDRM = v == "amdgpu" || v == "i915" || v == "xe"
				case "drm-client-id":
					client = v
				case "drm-memory-vram", "drm-resident-vram0":
					vram = parseDRMSize(v)
				case "drm-memory-gtt":
					gtt = parseDRMSize(v)
				}
			}
			if !isDRM {
				continue
			}
			found = true
			if client == "" {
				client = e.Name()
			}
			if seen[client] {
				continue
			}
			seen[client] = true
			fdKiB += vram + gtt
		}
	}

	var kfdBytes int64
	if files, err := os.ReadDir(fmt.Sprintf("/sys/class/kfd/kfd/proc/%d", pid)); err == nil {
		for _, f := range files {
			if !strings.HasPrefix(f.Name(), "vram_") {
				continue
			}
			if b, err := os.ReadFile(fmt.Sprintf("/sys/class/kfd/kfd/proc/%d/%s", pid, f.Name())); err == nil {
				if n, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64); err == nil {
					kfdBytes += n
					found = true
				}
			}
		}
	}

	mb := int(fdKiB / 1024)
	if k := int(kfdBytes >> 20); k > mb {
		mb = k
	}
	return mb, found
}

// normUnit: имя юнита без суффикса .service ("comfyui.service" и "comfyui" равнозначны).
func normUnit(u string) string {
	return strings.TrimSuffix(strings.TrimSpace(u), ".service")
}

// cgroupHasUnit — в содержимом /proc/<pid>/cgroup есть сегмент пути, равный юниту
// (cgroup v1 и v2, системные и пользовательские юниты).
func cgroupHasUnit(cgroup, unit string) bool {
	for _, line := range strings.Split(cgroup, "\n") {
		parts := strings.SplitN(line, ":", 3)
		if len(parts) < 3 {
			continue
		}
		for _, seg := range strings.Split(parts[2], "/") {
			if seg != "" && strings.TrimSuffix(seg, ".service") == unit {
				return true
			}
		}
	}
	return false
}

// pidInUnit — процесс лежит в cgroup указанного systemd-юнита
// (дочерние процессы, например модели llama-router, наследуют cgroup).
func pidInUnit(pid int, unit string) bool {
	b, err := os.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	if err != nil {
		return false
	}
	return cgroupHasUnit(string(b), unit)
}

// gpuUsage — один замер VRAM, разложенный по сервисам.
func (s *Server) gpuUsage() map[string]*vramUsage {
	res := map[string]*vramUsage{"comfyui": {}, "llama": {}}
	procs, err := s.gpuProcs()
	if err != nil {
		if s.smiWarned.CompareAndSwap(false, true) {
			log.Printf("[GPU] ВНИМАНИЕ: VRAM не измеряется (%v) — "+
				"работаю только по API (проверка и SIGKILL по VRAM отключены)", err)
		}
		return res
	}
	for svc, u := range res {
		unit := normUnit(s.Config.backend(svc).Unit)
		if unit == "" {
			continue
		}
		u.OK = true
		for _, p := range procs {
			if pidInUnit(p.PID, unit) {
				u.MB += p.MB
				if p.MB > 0 {
					u.PIDs = append(u.PIDs, p.PID)
				}
			}
		}
	}
	return res
}

func (s *Server) serviceVRAM(service string) vramUsage { return *s.gpuUsage()[service] }

func (s *Server) gpuSelfCheck() {
	for _, svc := range []string{"comfyui", "llama"} {
		if normUnit(s.Config.backend(svc).Unit) == "" {
			log.Printf("[GPU] ВНИМАНИЕ: %s.unit не задан — VRAM сервиса %s не отслеживается и выбить его нельзя",
				strings.ToLower(s.name(svc)), s.name(svc))
		}
	}
	mode := s.gpuMode()
	if _, err := s.gpuProcs(); err != nil {
		log.Printf("[GPU] ВНИМАНИЕ: учёт VRAM (%s) недоступен: %v — работаю только по API", mode, err)
		s.smiWarned.Store(true)
		return
	}
	u := s.gpuUsage()
	log.Printf("[GPU] учёт VRAM: %s. ComfyUI %d МБ (процессов с GPU: %d, лимит остатка %d), LLAMA %d МБ (процессов: %d, лимит %d), kill_on_leak=%v",
		mode, u["comfyui"].MB, len(u["comfyui"].PIDs), s.Config.ComfyUI.ResidualMB,
		u["llama"].MB, len(u["llama"].PIDs), s.Config.LLAMA.ResidualMB, s.Config.killOnLeak())
	if avail, err := memAvailableMB(); err == nil {
		log.Printf("[Mem] MemAvailable %d МБ (порог предупреждения %d МБ, ожидание до %d с)",
			avail, s.Config.GPU.MinAvailableMB, s.Config.GPU.MemWait)
	}
}

// ==================== ОПЕРАТИВНАЯ ПАМЯТЬ ====================
//
// У APU (например, Radeon 780M) «видеопамять» — это обычная ОЗУ, поэтому нехватка
// ОЗУ убивает и ComfyUI, и llama, даже когда VRAM между ними поделён правильно.

// memAvailableMB — MemAvailable из /proc/meminfo, МБ.
func memAvailableMB() (int, error) {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, err
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "MemAvailable:") {
			f := strings.Fields(line)
			if len(f) >= 2 {
				kb, err := strconv.ParseInt(f[1], 10, 64)
				if err != nil {
					return 0, err
				}
				return int(kb / 1024), nil
			}
		}
	}
	return 0, fmt.Errorf("MemAvailable не найден")
}

// gpuMemSummary — занятость VRAM и GTT из sysfs amdgpu (у APU GTT — это системная ОЗУ).
func gpuMemSummary() string {
	cards, _ := filepath.Glob("/sys/class/drm/card*/device/mem_info_gtt_used")
	for _, f := range cards {
		dir := filepath.Dir(f)
		rd := func(name string) float64 {
			b, err := os.ReadFile(filepath.Join(dir, name))
			if err != nil {
				return -1
			}
			n, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
			if err != nil {
				return -1
			}
			return float64(n) / (1 << 30)
		}
		vu, vt, gu, gt := rd("mem_info_vram_used"), rd("mem_info_vram_total"), rd("mem_info_gtt_used"), rd("mem_info_gtt_total")
		if vu < 0 || gu < 0 {
			continue
		}
		return fmt.Sprintf("GPU: VRAM %.1f/%.1f ГиБ, GTT %.1f/%.1f ГиБ", vu, vt, gu, gt)
	}
	return ""
}

// memStatus — короткая сводка для логов: свободная ОЗУ и занятость GPU-памяти.
func memStatus() string {
	out := "ОЗУ свободно ?"
	if avail, err := memAvailableMB(); err == nil {
		out = fmt.Sprintf("ОЗУ свободно %d МБ", avail)
	}
	if g := gpuMemSummary(); g != "" {
		out += "; " + g
	}
	return out
}

// topMemProcs — самые «тяжёлые» по RSS процессы вместе с systemd-юнитом.
// Именно это показывает, кто на самом деле съел ОЗУ.
func topMemProcs(n int) string {
	type pm struct {
		pid  int
		name string
		kb   int64
		unit string
	}
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return ""
	}
	var all []pm
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		b, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid))
		if err != nil {
			continue
		}
		p := pm{pid: pid}
		for _, line := range strings.Split(string(b), "\n") {
			if strings.HasPrefix(line, "Name:") {
				p.name = strings.TrimSpace(strings.TrimPrefix(line, "Name:"))
			} else if strings.HasPrefix(line, "VmRSS:") {
				if f := strings.Fields(line); len(f) >= 2 {
					p.kb, _ = strconv.ParseInt(f[1], 10, 64)
				}
			}
		}
		if p.kb == 0 {
			continue
		}
		if cg, err := os.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid)); err == nil {
			for _, line := range strings.Split(string(cg), "\n") {
				parts := strings.SplitN(line, ":", 3)
				if len(parts) < 3 {
					continue
				}
				for _, seg := range strings.Split(parts[2], "/") {
					if strings.HasSuffix(seg, ".service") {
						p.unit = seg
					}
				}
				if p.unit != "" {
					break
				}
			}
		}
		all = append(all, p)
	}
	for i := 0; i < len(all); i++ { // простая сортировка по убыванию RSS
		for j := i + 1; j < len(all); j++ {
			if all[j].kb > all[i].kb {
				all[i], all[j] = all[j], all[i]
			}
		}
	}
	if len(all) > n {
		all = all[:n]
	}
	parts := make([]string, 0, len(all))
	for _, p := range all {
		u := p.unit
		if u == "" {
			u = "-"
		}
		parts = append(parts, fmt.Sprintf("%s(%d) %.1f ГБ [%s]", p.name, p.pid, float64(p.kb)/1048576, u))
	}
	return strings.Join(parts, "; ")
}

// warnLowMemory пишет предупреждение, если ОЗУ мало (не чаще раза в минуту).
// Возвращает true, если ОЗУ ниже порога.
func (s *Server) warnLowMemory(where string) bool {
	min := s.Config.GPU.MinAvailableMB
	if min <= 0 {
		return false
	}
	avail, err := memAvailableMB()
	if err != nil || avail >= min {
		return false
	}
	now := time.Now().UnixNano()
	if last := s.lastMemWarn.Load(); now-last > int64(time.Minute) && s.lastMemWarn.CompareAndSwap(last, now) {
		log.Printf("[Mem] ВНИМАНИЕ (%s): свободно ОЗУ %d МБ < порога %d МБ — риск OOM-killer (%s). Больше всего держат: %s",
			where, avail, min, gpuMemSummary(), topMemProcs(5))
	}
	return true
}

// waitForMemory перед выдачей GPU ждёт, пока ОЗУ восстановится (mem_wait сек).
// Если не восстановилась — задача всё равно идёт, но предупреждение уже в логе.
func (s *Server) waitForMemory(service string) {
	if !s.warnLowMemory("перед задачей " + s.name(service)) {
		return
	}
	deadline := time.Now().Add(time.Duration(s.Config.GPU.MemWait) * time.Second)
	for time.Now().Before(deadline) {
		time.Sleep(2 * time.Second)
		if avail, err := memAvailableMB(); err == nil && avail >= s.Config.GPU.MinAvailableMB {
			log.Printf("[Mem] ОЗУ восстановилась (%d МБ) — запускаю %s", avail, s.name(service))
			return
		}
	}
	avail, _ := memAvailableMB()
	log.Printf("[Mem] ОЗУ не восстановилась за %d с (%d МБ) — запускаю %s на свой риск",
		s.Config.GPU.MemWait, avail, s.name(service))
}

// ==================== ВЫГРУЗКА / ВЫБИВАНИЕ ====================

type relState struct {
	OK       bool // сервис не держит GPU
	Measured bool // VRAM удалось измерить
	MB       int
	Why      string
}

// serviceState отвечает на вопрос «сервис отпустил GPU?» двумя независимыми способами:
// по API (модели загружены / ComfyUI занят) и по реальному VRAM (fdinfo/nvidia-smi).
func (s *Server) serviceState(service string) relState {
	st := relState{OK: true}

	if service == "llama" {
		if loaded, err := s.llamaLoaded(); err == nil && len(loaded) > 0 {
			st.OK = false
			st.Why = fmt.Sprintf("загружены модели %v", loaded)
		}
	} else if r, p, err := s.comfyBusy(); err == nil && r+p > 0 {
		st.OK = false
		st.Why = fmt.Sprintf("ComfyUI занят (running %d, pending %d)", r, p)
	}

	if u := s.serviceVRAM(service); u.OK {
		st.Measured = true
		st.MB = u.MB
		if limit := s.Config.backend(service).ResidualMB; u.MB > limit {
			st.OK = false
			if st.Why == "" {
				st.Why = fmt.Sprintf("держит %d МБ VRAM (лимит %d)", u.MB, limit)
			}
		}
	}
	return st
}

// release освобождает GPU от сервиса: мягко (unload / free) → ожидание с проверкой
// реального VRAM → если не отпустил за release_timeout, evict (SIGKILL).
// Вызывать под opMu.
func (s *Server) release(service, reason string) {
	name := s.name(service)
	start := time.Now()
	timeout := time.Duration(s.Config.GPU.ReleaseTimeout) * time.Second
	deadline := start.Add(timeout)

	acted := s.gracefulRelease(service)
	lastNudge := time.Now()
	var st relState

	for first := true; ; first = false {
		if !first {
			time.Sleep(700 * time.Millisecond)
		}
		st = s.serviceState(service)
		if st.OK {
			if acted && !st.Measured {
				time.Sleep(2 * time.Second) // VRAM не измерить — даём драйверу время
			}
			if acted {
				log.Printf("[Release] %s: GPU свободен за %v (%s; остаток %d МБ)",
					name, time.Since(start).Round(100*time.Millisecond), reason, st.MB)
			}
			s.markClean(service)
			return
		}
		if time.Now().After(deadline) {
			break
		}
		if time.Since(lastNudge) >= 5*time.Second {
			if s.gracefulRelease(service) {
				acted = true
			}
			lastNudge = time.Now()
		}
	}

	log.Printf("[Release] %s НЕ отпустил GPU за %v (%s): %s", name, timeout, reason, st.Why)
	s.evict(service, st.Why)
}

func (s *Server) markClean(service string) {
	s.mu.Lock()
	s.dirty[service] = false
	s.mu.Unlock()
}

// gracefulRelease просит сервис отпустить VRAM его собственными средствами.
// Идемпотентно. Возвращает true, если что-то было отправлено.
func (s *Server) gracefulRelease(service string) bool {
	if service == "llama" {
		loaded, err := s.llamaLoaded()
		if err != nil || len(loaded) == 0 {
			return false // llama недоступна или модели уже выгружены
		}
		log.Printf("[Unload] Выгружаю модели LLAMA (%d): %v", len(loaded), loaded)
		for _, id := range loaded {
			code, _, err := httpDo("POST", s.Config.llamaURL("/models/unload"),
				map[string]string{"model": id}, 30*time.Second)
			if err != nil {
				log.Printf("[Unload] %s: %v", id, err)
			} else if code >= 300 {
				log.Printf("[Unload] %s: HTTP %d", id, code)
			}
		}
		return true
	}

	// ComfyUI
	r, p, err := s.comfyBusy()
	if err != nil {
		return false // не отвечает/не запущен — выгружать нечего
	}
	if r+p > 0 {
		log.Printf("[Release] ComfyUI занят (running %d, pending %d) — прерываю и чищу очередь", r, p)
		httpDo("POST", s.Config.comfyURL("/queue"), map[string]interface{}{"clear": true}, 5*time.Second)
		httpDo("POST", s.Config.comfyURL("/interrupt"), map[string]interface{}{}, 5*time.Second)
		for i := 0; i < 20; i++ {
			time.Sleep(500 * time.Millisecond)
			if r, p, err := s.comfyBusy(); err != nil || r+p == 0 {
				break
			}
		}
	}
	log.Printf("[Release] ComfyUI: /free (unload_models + free_memory)")
	code, _, err := httpDo("POST", s.Config.comfyURL("/free"),
		map[string]interface{}{"unload_models": true, "free_memory": true}, 10*time.Second)
	if err != nil {
		log.Printf("[Release] ComfyUI /free: %v", err)
	} else if code >= 300 {
		log.Printf("[Release] ComfyUI /free: HTTP %d", code)
	}
	return true
}

// evict — крайняя мера: SIGKILL процессам сервиса, которые держат VRAM.
// systemd поднимет сервис заново (Restart=...), а llama-router перезапустит модель по требованию.
func (s *Server) evict(service, why string) {
	name := s.name(service)
	if !s.Config.killOnLeak() {
		log.Printf("[Evict] %s: kill_on_leak=false — не выбиваю (%s)", name, why)
		return
	}
	u := s.serviceVRAM(service)
	if !u.OK {
		log.Printf("[Evict] %s: не могу найти процессы (задайте unit и проверьте gpu.backend / права root) — выбить нечем", name)
		return
	}
	if len(u.PIDs) == 0 {
		log.Printf("[Evict] %s: процессов на GPU нет — выбивать некого", name)
		return
	}

	log.Printf("[Evict] %s держит %d МБ — SIGKILL pid %v", name, u.MB, u.PIDs)
	for _, pid := range u.PIDs {
		if err := syscall.Kill(pid, syscall.SIGKILL); err != nil {
			log.Printf("[Evict] kill %d: %v (нужны права root или тот же пользователь)", pid, err)
		}
	}
	s.mu.Lock()
	s.lastKill[service] = time.Now()
	s.dirty[service] = false
	s.mu.Unlock()

	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		time.Sleep(time.Second)
		if st := s.serviceState(service); st.OK {
			log.Printf("[Evict] %s: VRAM освобождён (остаток %d МБ)", name, st.MB)
			return
		}
	}
	log.Printf("[Evict] %s: после SIGKILL VRAM всё ещё занят — проверьте вручную", name)
}

// waitBackend: если сервис недоступен (упал по OOM, перезапускается systemd'ом,
// выбит нами) — ждём его подъёма до restart_wait. Иначе задача получила бы
// «бэкенд недоступен» вместо обычной паузы.
func (s *Server) waitBackend(service string) error {
	probe := s.Config.comfyURL("/queue")
	if service == "llama" {
		probe = s.Config.llamaURL("/health")
	}
	maxWait := time.Duration(s.Config.GPU.RestartWait) * time.Second
	deadline := time.Now().Add(maxWait)
	start := time.Now()
	logged := false
	for {
		if _, _, err := httpDo("GET", probe, nil, 3*time.Second); err == nil {
			if logged {
				log.Printf("[Wait] %s снова доступен (ждали %v)", s.name(service), time.Since(start).Round(time.Second))
			}
			return nil
		}
		if !logged {
			log.Printf("[Wait] %s недоступен (перезапускается после падения?) — жду до %v", s.name(service), maxWait)
			logged = true
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("%s не поднялся за %v", s.name(service), maxWait)
		}
		time.Sleep(2 * time.Second)
	}
}

// ==================== АВАРИЙНАЯ ЗАЩИТА ОЗУ ====================

// pidsInUnit — все процессы юнита (по cgroup), независимо от того, держат ли они GPU.
func pidsInUnit(unit string) []int {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return nil
	}
	var pids []int
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		if pidInUnit(pid, unit) {
			pids = append(pids, pid)
		}
	}
	return pids
}

// killUnit — SIGKILL всем процессам сервиса. systemd поднимет его заново.
func (s *Server) killUnit(service string) {
	unit := normUnit(s.Config.backend(service).Unit)
	if unit == "" {
		log.Printf("[MemGuard] %s: unit не задан — убить нечем", s.name(service))
		return
	}
	pids := pidsInUnit(unit)
	if len(pids) == 0 {
		log.Printf("[MemGuard] %s: процессов юнита %s не найдено", s.name(service), unit)
		return
	}
	log.Printf("[MemGuard] SIGKILL %s (юнит %s, pid %v)", s.name(service), unit, pids)
	for _, pid := range pids {
		if err := syscall.Kill(pid, syscall.SIGKILL); err != nil {
			log.Printf("[MemGuard] kill %d: %v", pid, err)
		}
	}
	s.mu.Lock()
	s.lastKill[service] = time.Now()
	s.dirty[service] = false
	s.mu.Unlock()
}

// memGuard — защита от «мёртвой» нехватки ОЗУ. Когда MemAvailable падает ниже
// critical_available_mb, система начинает swap-шторм и зависает минутами, прежде чем
// сработает OOM-killer (у ядра при этом жертвами становятся ещё и невиновные сервисы
// с OOMScoreAdjust=1000). Прокси реагирует раньше и точечно: останавливает того, кто
// сейчас держит GPU. Работает независимо от очереди и opMu.
func (s *Server) memGuard() {
	crit := s.Config.GPU.CriticalAvailMB
	if crit <= 0 {
		return
	}
	low := 0
	var cooldown time.Time
	for range time.Tick(time.Second) {
		if time.Now().Before(cooldown) {
			continue
		}
		avail, err := memAvailableMB()
		if err != nil || avail >= crit {
			low = 0
			continue
		}
		low++
		if low < 2 { // два замера подряд — отсечь случайный провал
			continue
		}
		low = 0
		s.memEmergency(avail)
		cooldown = time.Now().Add(20 * time.Second)
	}
}

// memIdleProtect решает, щадить ли модель при нехватке ОЗУ, пока GPU простаивает:
// true — не трогать. Трогаем только когда свободной памяти осталось меньше порога простоя
// (по умолчанию половина критического) — это последний рубеж перед OOM-killer'ом ядра.
func memIdleProtect(avail, crit, idleFloor int) bool {
	if idleFloor < 0 {
		return true
	}
	if idleFloor == 0 {
		idleFloor = crit / 2
	}
	return avail >= idleFloor
}

func (s *Server) memEmergency(avail int) {
	crit := s.Config.GPU.CriticalAvailMB

	s.mu.Lock()
	victim := s.activeService
	s.mu.Unlock()
	idle := victim == "" && len(s.taskQueue) == 0
	if victim == "" { // очередь пуста — берём того, кто реально держит GPU-память
		usage := s.gpuUsage()
		best := 0
		for _, svc := range []string{"comfyui", "llama"} {
			if u := usage[svc]; u.OK && u.MB > s.Config.backend(svc).ResidualMB && u.MB > best {
				best, victim = u.MB, svc
			}
		}
	}

	log.Printf("[MemGuard] КРИТИЧНО: свободно ОЗУ %d МБ < %d МБ (%s). Больше всего держат: %s",
		avail, crit, gpuMemSummary(), topMemProcs(5))
	if victim == "" {
		log.Printf("[MemGuard] GPU-сервисы не при чём — память держит другой процесс (см. список выше)")
		return
	}
	if idle && memIdleProtect(avail, crit, s.Config.GPU.CriticalIdleMB) {
		log.Printf("[MemGuard] GPU сейчас никому не нужен (нет задачи, очередь пуста) — %s не выгружаю, ОЗУ съедает другой процесс (см. список выше)",
			s.name(victim))
		return
	}
	if !s.Config.killOnLeak() {
		log.Printf("[MemGuard] kill_on_leak=false — %s не трогаю", s.name(victim))
		return
	}

	log.Printf("[MemGuard] спасаю систему: останавливаю %s", s.name(victim))
	// 1) мягко
	if victim == "comfyui" {
		httpDo("POST", s.Config.comfyURL("/queue"), map[string]interface{}{"clear": true}, 2*time.Second)
		httpDo("POST", s.Config.comfyURL("/interrupt"), map[string]interface{}{}, 2*time.Second)
		httpDo("POST", s.Config.comfyURL("/free"),
			map[string]interface{}{"unload_models": true, "free_memory": true}, 2*time.Second)
	} else if loaded, err := s.llamaLoaded(); err == nil {
		for _, id := range loaded {
			httpDo("POST", s.Config.llamaURL("/models/unload"), map[string]string{"model": id}, 5*time.Second)
		}
	}
	for i := 0; i < 8; i++ {
		time.Sleep(500 * time.Millisecond)
		if a, err := memAvailableMB(); err == nil && a >= crit*3/2 {
			log.Printf("[MemGuard] ОЗУ восстановилась (%d МБ) без SIGKILL", a)
			return
		}
	}
	// 2) жёстко
	s.killUnit(victim)
}

// ==================== WATCHDOG ====================

// watchdog раз в watchdog_interval проверяет инвариант «GPU держит один сервис»
// и чинит нарушения независимо от очереди: ловит всё, что попало на GPU мимо неё
// (autoload модели от служебных запросов, прямой запуск в ComfyUI, перезапуск прокси
// посреди работы). Заодно выгружает простаивающие сервисы заранее (idle_release).
func (s *Server) watchdog() {
	ticker := time.NewTicker(time.Duration(s.Config.GPU.WatchdogInterval) * time.Second)
	defer ticker.Stop()

	suspect := "" // жертва VRAM-конфликта, замеченного в прошлом тике (антидребезг)
	for {
		if s.opMu.TryLock() {
			suspect = s.watchdogTick(suspect)
			s.opMu.Unlock()
		} else {
			suspect = "" // идёт передача GPU — не мешаем
		}
		<-ticker.C
	}
}

func (s *Server) watchdogTick(suspect string) string {
	s.warnLowMemory("watchdog")

	s.mu.Lock()
	active := s.activeService
	s.mu.Unlock()

	// Пока идёт задача — раз в 30 с (раз в 5 с, если ОЗУ на исходе) пишем срез памяти:
	// по нему видно, растёт ли GTT/ОЗУ во время генерации и что было перед OOM.
	if active != "" {
		interval := s.Config.memTraceEvery() // 0 = выключено
		if avail, err := memAvailableMB(); err == nil && s.Config.GPU.MinAvailableMB > 0 &&
			avail < 2*s.Config.GPU.MinAvailableMB {
			interval = 5 * time.Second
		}
		now := time.Now().UnixNano()
		if last := s.lastMemTrace.Load(); interval > 0 && now-last > int64(interval) && s.lastMemTrace.CompareAndSwap(last, now) {
			log.Printf("[Mem] идёт %s: %s", s.name(active), memStatus())
		}
	}

	// 1) Конфликт по API: чужак работает в момент, когда GPU принадлежит другому.
	switch active {
	case "comfyui":
		if loaded, err := s.llamaLoaded(); err == nil && len(loaded) > 0 {
			log.Printf("[Watchdog] сейчас ход ComfyUI, но в LLAMA загружено %v — выгружаю", loaded)
			s.release("llama", "llama загрузилась мимо очереди")
			return ""
		}
	case "llama":
		if r, p, err := s.comfyBusy(); err == nil && r+p > 0 {
			log.Printf("[Watchdog] сейчас ход LLAMA, но ComfyUI занят (running %d, pending %d) — прерываю", r, p)
			s.release("comfyui", "ComfyUI работает мимо очереди")
			return ""
		}
	}

	// 2) Конфликт по реальному VRAM: оба держат больше допустимого остатка.
	usage := s.gpuUsage()
	c, l := usage["comfyui"], usage["llama"]
	if c.OK && l.OK && c.MB > s.Config.ComfyUI.ResidualMB && l.MB > s.Config.LLAMA.ResidualMB {
		var victim string
		switch active {
		case "comfyui":
			victim = "llama"
		case "llama":
			victim = "comfyui"
		default:
			s.mu.Lock()
			lc, ll := s.lastUsed["comfyui"], s.lastUsed["llama"]
			s.mu.Unlock()
			if lc.Before(ll) {
				victim = "comfyui"
			} else {
				victim = "llama"
			}
		}
		if victim != suspect {
			return victim // подтвердим в следующем тике
		}
		log.Printf("[Watchdog] VRAM занят обоими (ComfyUI %d МБ, LLAMA %d МБ) — выгружаю %s",
			c.MB, l.MB, s.name(victim))
		s.release(victim, "конфликт VRAM")
		return ""
	}

	// 3) Простой: выгрузить заранее, не дожидаясь, пока GPU понадобится другому.
	if active == "" && len(s.taskQueue) == 0 {
		for _, svc := range []string{"comfyui", "llama"} {
			idle := s.Config.backend(svc).IdleRelease
			if idle <= 0 {
				continue
			}
			s.mu.Lock()
			since := time.Since(s.lastUsed[svc])
			dirty := s.dirty[svc]
			s.mu.Unlock()
			if since < time.Duration(idle)*time.Second {
				continue
			}
			u := usage[svc]
			if dirty || (u.OK && u.MB > s.Config.backend(svc).ResidualMB) {
				s.release(svc, fmt.Sprintf("простой %v", since.Round(time.Second)))
			}
		}
	}
	return ""
}

// ==================== LLAMA: СОСТОЯНИЕ ====================

// llamaLoaded возвращает ID моделей, реально загруженных в VRAM.
// В router-сборке llama.cpp у каждой модели есть status.value = "unloaded"
// | "loading" | "loaded". /models — это каталог, он никогда не пустеет.
func (s *Server) llamaLoaded() ([]string, error) {
	code, data, err := httpDo("GET", s.Config.llamaURL("/models"), nil, 5*time.Second)
	if err != nil {
		return nil, err
	}
	if code != http.StatusOK {
		return nil, fmt.Errorf("/models: HTTP %d", code)
	}
	var doc struct {
		Data []struct {
			ID     string `json:"id"`
			Status struct {
				Value string `json:"value"`
			} `json:"status"`
		} `json:"data"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("разбор /models: %w", err)
	}
	out := make([]string, 0, len(doc.Data))
	for _, m := range doc.Data {
		switch m.Status.Value {
		case "", "unloaded":
			// пусто/не загружено — пропускаем
		default:
			out = append(out, m.ID)
		}
	}
	return out, nil
}

// ==================== COMFYUI: СОСТОЯНИЕ ====================

func (s *Server) comfyBusy() (running, pending int, err error) {
	code, data, err := httpDo("GET", s.Config.comfyURL("/queue"), nil, 5*time.Second)
	if err != nil {
		return 0, 0, err
	}
	if code != http.StatusOK {
		return 0, 0, fmt.Errorf("/queue: HTTP %d", code)
	}
	var q struct {
		Running []json.RawMessage `json:"queue_running"`
		Pending []json.RawMessage `json:"queue_pending"`
	}
	if err := json.Unmarshal(data, &q); err != nil {
		return 0, 0, err
	}
	return len(q.Running), len(q.Pending), nil
}

// waitForComfyIdle ждёт, пока ComfyUI освободит GPU (running+pending == 0).
// idle=true — свободен (или не запущен вовсе); wentDown=true — ComfyUI оказался
// недоступен (упал/перезапускается), т.е. идущая задача, скорее всего, потеряна;
// idle=false — не дождались: comfy_idle_timeout, отмена ctx или ComfyUI завис и не отвечает.
// Вызывающий в этом случае обязан прервать ComfyUI (release), а не идти дальше.
// comfy_idle_timeout = 0 → безлимит.
func (s *Server) waitForComfyIdle(ctx context.Context) (idle bool, wentDown bool) {
	interval := time.Duration(s.Config.Queue.PollInterval) * time.Second

	var deadline time.Time
	timeout := time.Duration(s.Config.Queue.ComfyIdleTimeout) * time.Second
	if timeout > 0 {
		deadline = time.Now().Add(timeout)
	}

	waitingLogged := false
	unknown := 0 // подряд идущие «не ответил» (не путать с «не запущен»)

	for {
		r, p, err := s.comfyBusy()
		switch {
		case err == nil:
			unknown = 0
			if r+p == 0 {
				if waitingLogged {
					s.vlogf("[Wait] ComfyUI завершил")
				}
				return true, false
			}
			if !waitingLogged {
				s.vlogf("[Wait] ComfyUI активен (running: %d, pending: %d), ждём завершения", r, p)
				waitingLogged = true
			}
		case isDown(err):
			if waitingLogged {
				log.Printf("[Wait] ComfyUI недоступен (%v) — GPU он не держит", err)
			}
			return true, true
		default:
			// таймаут и т.п.: ComfyUI может быть жив и занят — считать его свободным нельзя
			unknown++
			if unknown >= 6 {
				log.Printf("[Wait] ComfyUI не отвечает уже %d опросов подряд (%v) — считаю зависшим", unknown, err)
				return false, false
			}
		}

		if ctx.Err() != nil {
			return false, false
		}
		if !deadline.IsZero() && time.Now().After(deadline) {
			log.Printf("[Wait] Предельное ожидание ComfyUI (%v) истекло", timeout)
			return false, false
		}
		select {
		case <-ctx.Done():
			return false, false
		case <-time.After(interval):
		}
	}
}

// ==================== ИСПОЛНЕНИЕ ЗАДАЧИ ====================

// execBackend отправляет задачу бэкенду и стримит ответ клиенту.
// Возвращает HTTP-статус бэкенда или 0, если запрос не удался.
func (s *Server) execBackend(t *Task) int {
	target := s.Config.llamaURL(t.Path)
	if t.Service == "comfyui" {
		target = s.Config.comfyURL(t.Path)
		t.Body = comfyPrepareBody(t.Body, t.PromptID)
	}

	ctx, cancel := context.WithCancel(t.Ctx)
	defer cancel()

	var resp *http.Response
	for attempt := 0; ; attempt++ {
		req, err := http.NewRequestWithContext(ctx, "POST", target, bytes.NewReader(t.Body))
		if err != nil {
			s.jsonError(t.W, http.StatusBadGateway, fmt.Sprintf("ошибка запроса к бэкенду: %v", err))
			return 0
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err = backendClient.Do(req)
		if err == nil {
			break
		}
		if t.Ctx.Err() != nil {
			log.Printf("[Exec] %s: клиент отключился во время запроса", s.name(t.Service))
			return 0
		}
		// Бэкенд не принял соединение (упал/перезапускается): клиенту ничего ещё не
		// отправлено, поэтому безопасно дождаться подъёма и повторить один раз.
		if attempt == 0 && isDown(err) {
			log.Printf("[%s] бэкенд недоступен (%v) — жду перезапуска и повторю запрос", s.name(t.Service), err)
			if werr := s.waitBackend(t.Service); werr == nil {
				continue
			}
		}
		s.jsonError(t.W, http.StatusBadGateway, "бэкенд недоступен")
		log.Printf("[%s] Ошибка: %v", s.name(t.Service), err)
		return 0
	}
	defer resp.Body.Close()

	for k, vs := range resp.Header {
		for _, v := range vs {
			t.W.Header().Add(k, v)
		}
	}
	t.W.WriteHeader(resp.StatusCode)
	if resp.StatusCode >= 400 {
		// Прокси ошибку не придумывает, а передаёт клиенту как есть, поэтому в логе
		// нужно видеть, кто и что ответил (текст ошибки пишется ниже из тела).
		log.Printf("[%s] бэкенд ответил HTTP %d на %s (модель %q, идёт запросов: %d)",
			s.name(t.Service), resp.StatusCode, t.Path, t.Model, s.inflightNow())
	}

	idleLimit := time.Duration(s.Config.Queue.ExecIdleTimeout) * time.Second
	var lastRead atomic.Int64
	lastRead.Store(time.Now().UnixNano())
	stopWatch := make(chan struct{})
	defer close(stopWatch)
	go func() {
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-stopWatch:
				return
			case <-ticker.C:
				// Тишина обрывает запрос, только если за ним кто-то ждёт: пока очередь пуста,
				// долгая обработка промпта или «раздумья» модели никому не мешают.
				if t.Ctx.Err() == nil && time.Since(time.Unix(0, lastRead.Load())) > idleLimit && s.someoneWaits() {
					log.Printf("[%s] нет данных от бэкенда %v, а за ним ждут другие — обрываю (подозрение на зависание)",
						s.name(t.Service), idleLimit)
					cancel()
				}
			}
		}
	}()

	flusher, _ := t.W.(http.Flusher)
	buf := make([]byte, 32*1024)
	var meter *tokenFeed
	if t.Service == "llama" {
		meter = s.tokens.newFeed()
		defer meter.end()
	}
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			lastRead.Store(time.Now().UnixNano())
			if meter != nil {
				meter.feed(buf[:n])
			}
			t.W.Write(buf[:n])
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			if err != io.EOF && t.Ctx.Err() == nil {
				log.Printf("[%s] поток оборван бэкендом: %v (вероятно упал/OOM). %s. Больше всего ОЗУ держат: %s",
					s.name(t.Service), err, memStatus(), topMemProcs(5))
			}
			break
		}
	}

	if t.Ctx.Err() != nil {
		log.Printf("[Exec] %s: стрим завершился — клиент отключился", s.name(t.Service))
	}
	return resp.StatusCode
}

// / comfyPrepareBody готовит тело для ComfyUI:
//   - если тело — голый граф (нет ключа "prompt"), оборачивает в {"prompt": ...}
//   - если задан promptID и клиент не передал свой — внедряет "prompt_id"
//
// Так UI получает от прокси ранний prompt_id, а ComfyUI сохраняет результат
// в /history под тем же ID.
func comfyPrepareBody(body []byte, promptID string) []byte {
	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return body // не JSON — пропускаем как есть
	}

	// Голый граф?
	if _, hasPrompt := parsed["prompt"]; !hasPrompt {
		wrapped := map[string]interface{}{"prompt": parsed}
		// Пересохраняем остальные ключи, если они там были (client_id, extra_data и т.п.).
		// На практике голый граф — это только граф, без обёрток.
		parsed = wrapped
	}

	if promptID != "" {
		if _, exists := parsed["prompt_id"]; !exists {
			parsed["prompt_id"] = promptID
		}
	}

	out, err := json.Marshal(parsed)
	if err != nil {
		return body
	}
	return out
}

func (s *Server) jsonError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	fmt.Fprintf(w, `{"error":%q}`, msg)
}

// ==================== MAIN ====================

func main() {
	cfgPath := "config.yaml"
	if len(os.Args) > 1 {
		cfgPath = os.Args[1]
	}
	cfg, err := loadConfig(cfgPath)
	applyDefaults(cfg)
	if err != nil {
		log.Println("WARNING:", err)
	} else if abs, aerr := filepath.Abs(cfgPath); aerr == nil {
		log.Printf("Конфиг: %s", abs)
	}
	srv := NewServer(cfg)
	srv.Start()
}
