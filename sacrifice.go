package main

// «Жертвы» MemGuard: дешёвые процессы, которые прокси останавливает первыми, когда ОЗУ
// на исходе, чтобы не трогать llama вместе с её кэшем контекста (его пересчёт стоит минут).
//
// Список задаёт gpu.sacrifice (по умолчанию пустой): имена процессов, совпадение по началу
// имени без учёта регистра ("Godot" подойдёт для Godot_v4.3-stab). Из кандидатов исключаются
// сам прокси, процесс №1, процессы в контейнерах и процессы юнитов самих сервисов.
// За раз гасится один, самый тяжёлый по RSS: сначала SIGTERM, через 2 с SIGKILL.

import (
	"log"
	"os"
	"sort"
	"strings"
	"syscall"
	"time"
)

const maxSacrificesPerMinute = 5

// sacrificeCandidates — процессы, подходящие под список, от самого тяжёлого.
func (s *Server) sacrificeCandidates(procs []memProc) []memProc {
	names := s.Config.GPU.Sacrifice
	if len(names) == 0 {
		return nil
	}
	want := make([]string, 0, len(names))
	for _, n := range names {
		if n = strings.ToLower(strings.TrimSpace(n)); n != "" {
			want = append(want, n)
		}
	}
	protectedUnits := map[string]bool{
		normUnit(s.Config.ComfyUI.Unit): true,
		normUnit(s.Config.LLAMA.Unit):   true,
	}
	if cg, err := os.ReadFile("/proc/self/cgroup"); err == nil {
		if u, _ := parseCgroup(string(cg)); u != "" {
			protectedUnits[normUnit(u)] = true // сам прокси
		}
	}
	var out []memProc
	for _, p := range procs {
		if p.PID <= 1 || p.PID == os.Getpid() || p.Container || (p.Unit != "" && protectedUnits[normUnit(p.Unit)]) {
			continue
		}
		low := strings.ToLower(p.Name)
		for _, w := range want {
			if strings.HasPrefix(low, w) {
				out = append(out, p)
				break
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].kb > out[j].kb })
	return out
}

// sacrifice гасит самый тяжёлый процесс из списка. true — кого-то остановили (тогда llama
// на этом шаге не трогаем, а на следующем замере смотрим, хватило ли).
func (s *Server) sacrifice(avail int) bool {
	if len(s.Config.GPU.Sacrifice) == 0 || !s.Config.killOnLeak() {
		return false
	}
	now := time.Now()
	recent := s.sacrificed[:0]
	for _, at := range s.sacrificed {
		if now.Sub(at) < time.Minute {
			recent = append(recent, at)
		}
	}
	s.sacrificed = recent
	if len(s.sacrificed) >= maxSacrificesPerMinute {
		log.Printf("[MemGuard] уже остановлено %d процессов из gpu.sacrifice за минуту — больше не трогаю, перехожу к llama", len(s.sacrificed))
		return false
	}
	cands := s.sacrificeCandidates(readProcs())
	if len(cands) == 0 {
		return false
	}
	v := cands[0]
	log.Printf("[MemGuard] свободно ОЗУ %d МБ: останавливаю %s (pid %d, %d МБ) из gpu.sacrifice, чтобы не трогать кэш llama",
		avail, v.Name, v.PID, v.MB)
	s.sacrificed = append(s.sacrificed, now)
	_ = syscall.Kill(v.PID, syscall.SIGTERM)
	for i := 0; i < 8; i++ {
		time.Sleep(250 * time.Millisecond)
		if syscall.Kill(v.PID, 0) == syscall.ESRCH {
			return true
		}
	}
	log.Printf("[MemGuard] %s (pid %d) не завершился за 2 с — SIGKILL", v.Name, v.PID)
	_ = syscall.Kill(v.PID, syscall.SIGKILL)
	time.Sleep(500 * time.Millisecond)
	return true
}
