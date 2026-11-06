package main

import (
	"context"
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
	s.taskQueue <- newTask("llama", "m")
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
	if s.someoneWaits() {
		t.Fatal("очередь пуста")
	}
	s.admitWaiting.Add(1)
	if !s.someoneWaits() {
		t.Fatal("ждущий в admit должен считаться")
	}
	s.admitWaiting.Add(-1)
}
