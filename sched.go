package main

// Планировщик очереди: «удержание» llama перед передачей GPU ComfyUI.
//
// Агент делает паузы между запросами к llama (инструменты, сборки). Если в такую паузу
// в очереди стоит задача ComfyUI, GPU уходит к ней, и llama теряет кэш контекста
// (десятки тысяч токенов пересчитываются минутами). Поэтому пока llama «горячая»
// (идёт запрос или он закончился меньше queue.llama_hold секунд назад), задача ComfyUI
// откладывается, а новые запросы llama могут её обгонять. Жёсткий предел: ComfyUI не ждёт
// из-за удержания дольше queue.comfy_max_wait секунд с момента постановки в очередь
// (по умолчанию 3600). Идущий запрос llama при этом не прерывается: предел ограничивает
// только добровольное ожидание.

import (
	"log"
	"time"
)

// Причины ожидания задачи (Task.reason) — для страницы состояния.
const (
	reasonNone  int32 = iota
	reasonHold        // llama держит GPU ради кэша
	reasonSlot        // ждёт свободный слот модели
	reasonDrain       // ждёт, пока завершатся идущие запросы llama
	reasonMem         // мало свободной ОЗУ
)

func reasonName(v int32) string {
	switch v {
	case reasonHold:
		return "hold"
	case reasonSlot:
		return "slot"
	case reasonDrain:
		return "drain"
	case reasonMem:
		return "mem"
	}
	return ""
}

func (s *Server) llamaHoldDur() time.Duration {
	return time.Duration(s.Config.Queue.LlamaHold) * time.Second
}

// comfyMaxWait — предел удержания; 0 = без предела.
func (s *Server) comfyMaxWait() time.Duration {
	if s.Config.Queue.ComfyMaxWait < 0 {
		return 0
	}
	return time.Duration(s.Config.Queue.ComfyMaxWait) * time.Second
}

// llamaHot: у llama есть кэш, который жалко терять — идёт запрос или он завершился
// недавно. Без llama_hold всегда false.
func (s *Server) llamaHot() bool {
	hold := s.llamaHoldDur()
	if hold <= 0 {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.llamaInflight > 0 {
		return true
	}
	if !s.dirty["llama"] {
		return false
	}
	return time.Since(s.lastUsed["llama"]) < hold
}

// holdLeft — сколько ещё llama продержит GPU, если запросов не будет (0, если не держит).
func (s *Server) holdLeft() time.Duration {
	hold := s.llamaHoldDur()
	if hold <= 0 {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.llamaInflight > 0 || !s.dirty["llama"] {
		return 0
	}
	if left := hold - time.Since(s.lastUsed["llama"]); left > 0 {
		return left
	}
	return 0
}

// capLeft — сколько ещё задача ComfyUI может ждать из-за удержания (до жёсткого предела).
// Второе значение false, если предела нет.
func (s *Server) capLeft(t *Task) (time.Duration, bool) {
	mw := s.comfyMaxWait()
	if mw <= 0 {
		return 0, false
	}
	left := mw - time.Since(t.EnqueuedAt)
	if left < 0 {
		left = 0
	}
	return left, true
}

// holdOver: удерживать задачу ComfyUI больше не нужно — llama остыла или вышел предел.
func (s *Server) holdOver(t *Task) bool {
	if left, capped := s.capLeft(t); capped && left == 0 {
		return true
	}
	return !s.llamaHot()
}

func (s *Server) deferTask(t *Task) {
	first := !t.held.Load()
	t.held.Store(true)
	t.reason.Store(reasonHold)
	t.state.Store(stWaiting)
	s.deferred = append(s.deferred, t)
	if first {
		left, capped := s.capLeft(t)
		lim := "без предела"
		if capped {
			lim = left.Round(time.Second).String()
		}
		log.Printf("[Queue] ComfyUI ждёт: llama держит GPU ради кэша контекста (удержание %v после последнего запроса, предел ожидания %s)",
			s.llamaHoldDur(), lim)
	}
}

func (s *Server) undefer(t *Task, why string) {
	t.held.Store(false)
	t.reason.Store(reasonNone)
	log.Printf("[Queue] ComfyUI: %s — передаю GPU", why)
}

// nextTask отдаёт воркеру следующую задачу. Задачи ComfyUI, которым пока нельзя
// забирать GPU, откладываются, а запросы llama за ними идут вперёд.
// nil — очередь закрыта.
func (s *Server) nextTask() *Task {
	for {
		// 1) Отложенная задача ComfyUI, которой пора?
		for len(s.deferred) > 0 {
			d := s.deferred[0]
			if d.abandoned() {
				s.deferred = s.deferred[1:]
				d.held.Store(false)
				log.Printf("[Queue] ComfyUI: клиент ушёл, пропускаем")
				continue
			}
			if !s.holdOver(d) {
				break
			}
			s.deferred = s.deferred[1:]
			if left, capped := s.capLeft(d); capped && left == 0 {
				s.undefer(d, "предел ожидания исчерпан")
			} else {
				s.undefer(d, "llama простаивает дольше удержания")
			}
			return d
		}

		// 2) Следующая из очереди.
		var t *Task
		if len(s.deferred) == 0 {
			var ok bool
			if t, ok = <-s.taskQueue; !ok {
				return nil
			}
		} else {
			select {
			case x, ok := <-s.taskQueue:
				if !ok {
					return nil
				}
				t = x
			case <-time.After(time.Second): // перепроверить отложенные
				continue
			}
		}
		// Задачи ComfyUI не обгоняют друг друга: пока есть отложенная, остальные встают за ней.
		if t.Service == "comfyui" && !t.abandoned() && (len(s.deferred) > 0 || !s.holdOver(t)) {
			s.deferTask(t)
			continue
		}
		return t
	}
}
