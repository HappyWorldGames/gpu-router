package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestParseCgroup(t *testing.T) {
	cases := []struct {
		cg        string
		unit      string
		container bool
	}{
		{"0::/system.slice/llama.service\n", "llama.service", false},
		{"0::/system.slice/docker-abcdef.scope\n", "", true},
		{"0::/kubepods.slice/kubepods-pod1.slice/cri-containerd-abc.scope\n", "", true},
		{"0::/system.slice/containerd.service\n", "containerd.service", false}, // сам демон — не контейнер
		{"0::/user.slice/user-1000.slice/session-3.scope\n", "", false},
		{"", "", false},
	}
	for _, c := range cases {
		u, ctr := parseCgroup(c.cg)
		if u != c.unit || ctr != c.container {
			t.Errorf("%q -> %q,%v want %q,%v", c.cg, u, ctr, c.unit, c.container)
		}
	}
}

func TestReadProcsAndTop(t *testing.T) {
	procs := readProcs()
	if len(procs) == 0 {
		t.Skip("/proc недоступен")
	}
	found := false
	for _, p := range procs {
		if p.PID == os.Getpid() {
			found = true
			if p.MB <= 0 || p.Name == "" {
				t.Fatalf("свой процесс: %+v", p)
			}
		}
	}
	if !found {
		t.Fatal("свой процесс не найден")
	}
	top := heavyProcs(procs, 3)
	if len(top) > 3 || (len(top) == 3 && (top[0].kb < top[1].kb || top[1].kb < top[2].kb)) {
		t.Fatalf("сортировка по RSS: %+v", top)
	}
	if s := topMemProcs(2); s == "" || !strings.Contains(s, "ГБ") {
		t.Fatalf("строка для лога: %q", s)
	}
	if unitRSSMB([]memProc{{Unit: "x.service", MB: 5}, {Unit: "x.service", MB: 7}, {Unit: "y.service", MB: 100}}, "x") != 12 {
		t.Fatal("сумма RSS по юниту")
	}
}

func TestMemSnapshotRing(t *testing.T) {
	cfg := &Config{}
	applyDefaults(cfg)
	s := NewServer(cfg)
	for i := 0; i < memHistLen+20; i++ {
		s.sampleMem()
	}
	top, hist, total, _ := s.memData()
	if len(hist) != memHistLen {
		t.Fatalf("история должна быть ограничена %d, а не %d", memHistLen, len(hist))
	}
	if len(top) == 0 || total <= 0 {
		t.Fatalf("top=%d total=%d", len(top), total)
	}
	hist[0] = -1 // копия: правка снаружи не портит внутреннее состояние
	if _, h2, _, _ := s.memData(); h2[0] == -1 {
		t.Fatal("memData должна отдавать копию")
	}
}

func TestSacrifice(t *testing.T) {
	dir := t.TempDir()
	bin := filepath.Join(dir, "sacrtestproc")
	src, err := os.ReadFile("/bin/sleep")
	if err != nil {
		t.Skip("нет /bin/sleep")
	}
	if err := os.WriteFile(bin, src, 0o755); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(bin, "300")
	if err := cmd.Start(); err != nil {
		t.Skip("не удалось запустить:", err)
	}
	exited := make(chan struct{})
	go func() { cmd.Wait(); close(exited) }()
	defer cmd.Process.Kill()
	time.Sleep(150 * time.Millisecond)

	cfg := &Config{}
	cfg.GPU.Sacrifice = []string{"SACRTEST"} // без учёта регистра, по началу имени
	cfg.ComfyUI.Unit, cfg.LLAMA.Unit = "comfyui.service", "llama.service"
	applyDefaults(cfg)
	s := NewServer(cfg)

	// кандидат найден, а сам тестовый процесс и контейнеры из кандидатов исключены
	procs := readProcs()
	procs = append(procs,
		memProc{PID: os.Getpid(), Name: "sacrtest-self", kb: 1 << 30},
		memProc{PID: 99999, Name: "sacrtest-in-docker", kb: 1 << 30, Container: true},
		memProc{PID: 1, Name: "sacrtest-init", kb: 1 << 30},
		memProc{PID: 99998, Name: "sacrtest-llama", kb: 1 << 30, Unit: "llama.service"},
		memProc{PID: 99997, Name: "sacrtest-comfy", kb: 1 << 30, Unit: "comfyui.service"})
	c := s.sacrificeCandidates(procs)
	if len(c) != 1 || c[0].PID != cmd.Process.Pid {
		t.Fatalf("кандидат должен быть один — наш процесс: %+v", c)
	}

	// выключено kill_on_leak=false — ничего не трогаем
	off := false
	s.Config.GPU.KillOnLeak = &off
	if s.sacrifice(100) {
		t.Fatal("при kill_on_leak=false жертвовать нельзя")
	}
	select {
	case <-exited:
		t.Fatal("процесс убит, хотя kill_on_leak=false")
	default:
	}
	s.Config.GPU.KillOnLeak = nil

	if !s.sacrifice(100) {
		t.Fatal("процесс из списка должен быть остановлен")
	}
	select {
	case <-exited:
	case <-time.After(3 * time.Second):
		t.Fatal("процесс не завершился")
	}
	// пустой список — ничего не делаем
	s2 := NewServer(&Config{})
	if s2.sacrifice(100) {
		t.Fatal("пустой gpu.sacrifice: действий быть не должно")
	}
}
