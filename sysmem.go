package main

// Память для страницы состояния: кто сколько держит в ОЗУ, сколько у юнитов сервисов,
// и короткая история свободной ОЗУ. Раз в 5 секунд один проход по /proc (миллисекунды);
// страница читает готовый снимок и ничего не считает сама.

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type memProc struct {
	PID       int
	Name      string
	MB        int
	Unit      string // systemd-юнит процесса или ""
	Container bool   // процесс внутри контейнера (docker/podman/containerd)
	kb        int64
}

// readProcs — все процессы с ненулевым RSS: имя, RSS, юнит, признак контейнера.
func readProcs() []memProc {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return nil
	}
	var all []memProc
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		b, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid))
		if err != nil {
			continue
		}
		p := memProc{PID: pid}
		for _, line := range strings.Split(string(b), "\n") {
			if strings.HasPrefix(line, "Name:") {
				p.Name = strings.TrimSpace(strings.TrimPrefix(line, "Name:"))
			} else if strings.HasPrefix(line, "VmRSS:") {
				if f := strings.Fields(line); len(f) >= 2 {
					p.kb, _ = strconv.ParseInt(f[1], 10, 64)
				}
			}
		}
		if p.kb == 0 {
			continue
		}
		p.MB = int(p.kb / 1024)
		if cg, err := os.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid)); err == nil {
			p.Unit, p.Container = parseCgroup(string(cg))
		}
		all = append(all, p)
	}
	return all
}

// parseCgroup достаёт из содержимого /proc/<pid>/cgroup имя systemd-юнита и признак контейнера.
func parseCgroup(cg string) (unit string, container bool) {
	for _, line := range strings.Split(cg, "\n") {
		parts := strings.SplitN(line, ":", 3)
		if len(parts) < 3 {
			continue
		}
		for _, seg := range strings.Split(parts[2], "/") {
			switch {
			case strings.HasSuffix(seg, ".service") && unit == "":
				unit = seg
			case strings.HasPrefix(seg, "docker-"), seg == "docker", strings.HasPrefix(seg, "libpod-"),
				strings.Contains(seg, "containerd"), strings.HasPrefix(seg, "crio-"):
				container = true
			}
		}
	}
	return unit, container
}

// heavyProcs — n самых тяжёлых по RSS.
func heavyProcs(all []memProc, n int) []memProc {
	c := append([]memProc(nil), all...)
	sort.Slice(c, func(i, j int) bool { return c[i].kb > c[j].kb })
	if len(c) > n {
		c = c[:n]
	}
	return c
}

// topMemProcs — строка «самые тяжёлые процессы» для записей в лог.
func topMemProcs(n int) string {
	top := heavyProcs(readProcs(), n)
	parts := make([]string, 0, len(top))
	for _, p := range top {
		u := p.Unit
		if u == "" {
			u = "-"
		}
		parts = append(parts, fmt.Sprintf("%s(%d) %.1f ГБ [%s]", p.Name, p.PID, float64(p.kb)/1048576, u))
	}
	return strings.Join(parts, "; ")
}

// unitRSSMB — сумма RSS процессов юнита.
func unitRSSMB(all []memProc, unit string) int {
	unit = normUnit(unit)
	if unit == "" {
		return 0
	}
	sum := 0
	for _, p := range all {
		if p.Unit != "" && normUnit(p.Unit) == unit {
			sum += p.MB
		}
	}
	return sum
}

func memTotalMB() int {
	b, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "MemTotal:") {
			if f := strings.Fields(line); len(f) >= 2 {
				if kb, err := strconv.Atoi(f[1]); err == nil {
					return kb / 1024
				}
			}
		}
	}
	return 0
}

// ---- снимок для страницы ----

const (
	memSampleEvery = 5 * time.Second
	memHistLen     = 180 // 15 минут
)

type memSnap struct {
	mu       sync.Mutex
	hist     []int // свободная ОЗУ, МБ, от старых к новым
	top      []memProc
	unitMB   map[string]int // service -> RSS юнита
	totalMB  int
	sampleAt time.Time
}

func (s *Server) sampleMem() {
	procs := readProcs()
	avail, err := memAvailableMB()
	m := &s.mem
	m.mu.Lock()
	defer m.mu.Unlock()
	if err == nil {
		m.hist = append(m.hist, avail)
		if len(m.hist) > memHistLen {
			m.hist = m.hist[len(m.hist)-memHistLen:]
		}
	}
	m.top = heavyProcs(procs, 5)
	m.unitMB = map[string]int{
		"comfyui": unitRSSMB(procs, s.Config.ComfyUI.Unit),
		"llama":   unitRSSMB(procs, s.Config.LLAMA.Unit),
	}
	if m.totalMB == 0 {
		m.totalMB = memTotalMB()
	}
	m.sampleAt = time.Now()
}

// memSampler раз в 5 секунд обновляет снимок. Запускается, если включена страница состояния.
func (s *Server) memSampler() {
	s.sampleMem()
	for range time.Tick(memSampleEvery) {
		s.sampleMem()
	}
}

type dashProc struct {
	Name      string `json:"name"`
	PID       int    `json:"pid"`
	MB        int    `json:"mb"`
	Unit      string `json:"unit,omitempty"`
	Container bool   `json:"container,omitempty"`
}

// memData — готовые данные для страницы: тяжёлые процессы, история, RSS юнитов.
func (s *Server) memData() (top []dashProc, hist []int, totalMB int, unitMB map[string]int) {
	m := &s.mem
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, p := range m.top {
		top = append(top, dashProc{p.Name, p.PID, p.MB, p.Unit, p.Container})
	}
	hist = append([]int(nil), m.hist...)
	unitMB = map[string]int{}
	for k, v := range m.unitMB {
		unitMB[k] = v
	}
	return top, hist, m.totalMB, unitMB
}
