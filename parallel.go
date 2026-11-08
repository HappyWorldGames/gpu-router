package main

// Параллельные запросы к llama.cpp.
//
// Раньше воркер отдавал GPU одной задаче и ждал её конца. Теперь для llama он не ждёт:
// пока у загруженной модели есть свободные слоты (parallel из models.ini), следующие
// запросы к той же модели запускаются рядом. Всё остальное по-прежнему строго по очереди:
//   - запрос к другой модели (router может выгрузить текущую) ждёт, пока все идущие закончат;
//   - ComfyUI ждёт, пока все идущие запросы llama закончат, и только потом получает GPU;
//   - порядок очереди сохраняется: пока первым в очереди стоит «чужой», новые запросы
//     к текущей модели к идущим не присоединяются.

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"time"
)

// requestModel достаёт поле "model" верхнего уровня из JSON-тела, не разбирая остальное
// (в теле могут быть мегабайты base64-картинок). Нет поля / не JSON — пустая строка.
func requestModel(body []byte) string {
	dec := json.NewDecoder(bytes.NewReader(body))
	tok, err := dec.Token()
	if err != nil {
		return ""
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return ""
	}
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return ""
		}
		key, _ := kt.(string)
		if key == "model" {
			var v interface{}
			if dec.Decode(&v) != nil {
				return ""
			}
			name, _ := v.(string)
			return name
		}
		var skip json.RawMessage
		if err := dec.Decode(&skip); err != nil && err != io.EOF {
			return ""
		}
	}
	return ""
}

// someoneWaits — за идущей задачей кто-то ждёт своей очереди: либо лежит в очереди,
// либо воркер уже взял его и держит в admit() (ждёт слот или конец идущих запросов).
func (s *Server) someoneWaits() bool {
	return len(s.taskQueue) > 0 || s.admitWaiting.Load() > 0
}

// waitDoneOrLimit ждёт конец задачи. Потолок task_timeout нужен, чтобы зависшая задача
// не вешала очередь, поэтому срабатывает только когда очередь действительно стоит:
// пока за задачей никто не ждёт, она может идти сколько угодно. Возвращает true, если
// задачу пришлось прервать.
func (s *Server) waitDoneOrLimit(t *Task, limit time.Duration) bool {
	timer := time.NewTimer(limit)
	defer timer.Stop()
	select {
	case <-t.Done:
		return false
	case <-timer.C:
	}
	poll := time.NewTicker(2 * time.Second)
	defer poll.Stop()
	for !s.someoneWaits() {
		select {
		case <-t.Done:
			return false
		case <-poll.C:
		}
	}
	log.Printf("[Worker] %s: задача идёт дольше %v, и за ней уже ждут другие — принудительно завершаю",
		s.name(t.Service), limit)
	t.Cancel()
	grace := time.NewTimer(20 * time.Second)
	defer grace.Stop()
	select {
	case <-t.Done:
	case <-grace.C:
		log.Printf("[Worker] %s: обработчик не завершился — иду дальше", s.name(t.Service))
	}
	return true
}

// admit вызывается воркером под управлением очереди для каждой взятой задачи.
// Возвращает true, если задача уже обработана здесь: запущена параллельно с идущими
// (Turn закрыт) или клиент ушёл, пока она ждала. false — llama свободна, можно
// проходить обычную подготовку GPU.
func (s *Server) admit(t *Task) bool {
	stop := make(chan struct{})
	defer close(stop)
	// Если клиент уйдёт, пока задача ждёт, разбудим ожидание.
	go func() {
		select {
		case <-t.Ctx.Done():
		case <-t.Done:
		case <-stop:
			return
		}
		s.mu.Lock()
		s.cond.Broadcast()
		s.mu.Unlock()
	}()

	s.mu.Lock()
	defer s.mu.Unlock()
	logged := false
	defer func() {
		if logged {
			s.admitWaiting.Add(-1)
		}
	}()
	for {
		if t.abandoned() {
			log.Printf("[Queue] %s: клиент ушёл, пропускаем", s.name(t.Service))
			return true
		}
		if s.llamaInflight == 0 {
			return false
		}
		if t.Service == "llama" && t.Model != "" && t.Model == s.llamaModel {
			limit := s.ini.parallelFor(t.Model)
			s.llamaLimit = limit
			if s.llamaInflight < limit {
				s.llamaInflight++
				n := s.llamaInflight
				start := time.Now()
				close(t.Turn)
				go s.watchLlama(t, start, 0)
				log.Printf("[Start] LLAMA — параллельно с идущими (%d из %d слотов), модель %s%s",
					n, limit, t.Model, s.memSuffix())
				return true
			}
		}
		if !logged {
			logged = true
			s.admitWaiting.Add(1)
			if t.Service == "llama" && t.Model == s.llamaModel && t.Model != "" {
				s.vlogf("[Queue] LLAMA ждёт свободный слот модели %s (занято %d из %d)", t.Model, s.llamaInflight, s.llamaLimit)
			} else {
				s.vlogf("[Queue] %s ждёт, пока завершатся запросы LLAMA (идёт %d, модель %q)", s.name(t.Service), s.llamaInflight, s.llamaModel)
			}
		}
		s.cond.Wait()
	}
}

// startLlamaGroup регистрирует первый запрос группы (GPU уже подготовлен).
func (s *Server) startLlamaGroup(t *Task, taskStart, prepDone time.Time) {
	limit := s.ini.parallelFor(t.Model)
	s.mu.Lock()
	s.llamaInflight = 1
	s.llamaModel = t.Model
	s.llamaLimit = limit
	s.mu.Unlock()
	if limit > 1 {
		log.Printf("[Parallel] модель %s принимает до %d запросов одновременно", t.Model, limit)
	}
	go s.watchLlama(t, taskStart, prepDone.Sub(taskStart))
}

// watchLlama ждёт конец одного запроса llama (с потолком по времени) и освобождает слот.
// Когда слотов занято 0 — GPU считается свободным.
func (s *Server) watchLlama(t *Task, taskStart time.Time, prep time.Duration) {
	s.waitDoneOrLimit(t, time.Duration(s.Config.Queue.TaskTimeout)*time.Second)

	s.mu.Lock()
	s.llamaInflight--
	left := s.llamaInflight
	if left <= 0 {
		s.llamaInflight = 0
		s.llamaModel = ""
		s.activeService = ""
		s.activeReady = false
	}
	s.lastUsed["llama"] = time.Now()
	s.dirty["llama"] = true
	s.cond.Broadcast()
	s.mu.Unlock()

	if prep > 0 {
		log.Printf("[Done] LLAMA завершил (всего %v, из них подготовка %v)%s",
			time.Since(taskStart).Round(time.Millisecond), prep.Round(time.Millisecond), s.memSuffix())
	} else {
		log.Printf("[Done] LLAMA завершил параллельный запрос (%v), осталось %d%s",
			time.Since(taskStart).Round(time.Millisecond), left, s.memSuffix())
	}
}
