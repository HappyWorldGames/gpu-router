package main

// Скорость генерации llama.cpp (токенов/с) для панели.
//
// Прокси и так стримит ответ llama клиенту, поэтому просто подглядываем в
// проходящие байты (ничего не меняя и не задерживая):
//   - «живая» скорость: считаем SSE-чанки с текстом (в llama-server один чанк ≈ один токен)
//     в скользящем окне последних секунд;
//   - итоговая скорость: llama-server в последнем чанке (или в теле не-стримового ответа)
//     отдаёт поле "timings" с точными predicted_per_second / prompt_per_second.
// Если клиент просит сжатый ответ, разобрать поток не получится — тогда цифр не будет,
// но на проксирование это никак не влияет.

import (
	"bytes"
	"encoding/json"
	"log"
	"sync"
	"time"
)

// tokenMeter — общий счётчик по всем запросам llama (их может быть несколько
// одновременно, если у модели parallel > 1). Каждый запрос читает свой поток через tokenFeed.
type tokenMeter struct {
	mu     sync.Mutex
	stamps []time.Time // моменты прихода последних токенов (со всех запросов)

	lastTPS       float64 // итоговая скорость генерации последнего завершённого запроса
	lastPromptTPS float64 // скорость обработки промпта у него же
	lastTokens    int     // сколько токенов он сгенерировал
	lastAt        time.Time
}

const (
	tokWindow   = 5 * time.Second // окно для «живой» скорости
	tokFresh    = 2 * time.Second // если токенов не было дольше — генерация не идёт
	tokMaxStamp = 512
	tokMaxLine  = 1 << 20 // не копим строки длиннее 1 МиБ
)

// tokenFeed — разбор потока одного запроса (свой буфер неполной строки).
type tokenFeed struct {
	m       *tokenMeter
	pending []byte
}

func (m *tokenMeter) newFeed() *tokenFeed { return &tokenFeed{m: m} }

// feed получает очередной кусок ответа бэкенда.
func (f *tokenFeed) feed(p []byte) {
	f.pending = append(f.pending, p...)
	for {
		i := bytes.IndexByte(f.pending, '\n')
		if i < 0 {
			break
		}
		f.m.line(f.pending[:i])
		f.pending = f.pending[i+1:]
	}
	if len(f.pending) > tokMaxLine {
		f.pending = f.pending[:0]
	}
}

// end вызывается в конце ответа: разбирает остаток без перевода строки
// (не-стримовый ответ llama — это один JSON без \n).
func (f *tokenFeed) end() {
	if len(f.pending) > 0 {
		f.m.line(f.pending)
		f.pending = f.pending[:0]
	}
}

func (m *tokenMeter) line(b []byte) {
	b = bytes.TrimSpace(b)
	b = bytes.TrimPrefix(b, []byte("data:"))
	b = bytes.TrimSpace(b)
	if len(b) == 0 || b[0] != '{' {
		return
	}
	// Ошибка от llama: отдельным JSON ({"error":...}) или событием внутри потока.
	// Причина HTTP 500 (нехватка KV-кэша, контекст слота и т.п.) написана именно здесь.
	if bytes.HasPrefix(b, []byte(`{"error"`)) {
		msg := b
		if len(msg) > 600 {
			msg = msg[:600]
		}
		log.Printf("[LLAMA] ошибка от llama-server: %s", msg)
		return
	}
	if bytes.Contains(b, []byte(`"timings"`)) {
		var v struct {
			Timings struct {
				PredictedN  int     `json:"predicted_n"`
				PredictedPS float64 `json:"predicted_per_second"`
				PromptPS    float64 `json:"prompt_per_second"`
			} `json:"timings"`
		}
		if json.Unmarshal(b, &v) == nil && (v.Timings.PredictedPS > 0 || v.Timings.PromptPS > 0) {
			m.mu.Lock()
			m.lastTPS, m.lastPromptTPS = v.Timings.PredictedPS, v.Timings.PromptPS
			m.lastTokens, m.lastAt = v.Timings.PredictedN, time.Now()
			m.mu.Unlock()
		}
		return
	}
	if hasText(b) {
		now := time.Now()
		m.mu.Lock()
		m.stamps = append(m.stamps, now)
		if len(m.stamps) > tokMaxStamp {
			m.stamps = m.stamps[len(m.stamps)-tokMaxStamp:]
		}
		m.mu.Unlock()
	}
}

// hasText: в чанке есть непустой content / reasoning_content.
func hasText(b []byte) bool {
	key := []byte(`content":"`)
	for off := 0; ; {
		i := bytes.Index(b[off:], key)
		if i < 0 {
			return false
		}
		j := off + i + len(key)
		if j < len(b) && b[j] != '"' {
			return true
		}
		off = j
	}
}

// live — скорость прямо сейчас (nil, если токены не идут).
func (m *tokenMeter) live() *float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	n := len(m.stamps)
	if n < 3 || time.Since(m.stamps[n-1]) > tokFresh {
		return nil
	}
	cut := m.stamps[n-1].Add(-tokWindow)
	first := 0
	for first < n-1 && m.stamps[first].Before(cut) {
		first++
	}
	span := m.stamps[n-1].Sub(m.stamps[first]).Seconds()
	if n-first < 3 || span <= 0 {
		return nil
	}
	v := float64(n-first-1) / span
	return &v
}

type tokenSnap struct {
	Live       *float64 `json:"tps_live"`
	Last       *float64 `json:"tps_last"`
	PromptLast *float64 `json:"prompt_tps_last"`
	LastTokens int      `json:"last_tokens,omitempty"`
	LastAgeSec int      `json:"last_age_sec,omitempty"`
}

func (m *tokenMeter) snap() tokenSnap {
	s := tokenSnap{Live: m.live()}
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.lastAt.IsZero() {
		if m.lastTPS > 0 {
			v := m.lastTPS
			s.Last = &v
		}
		if m.lastPromptTPS > 0 {
			v := m.lastPromptTPS
			s.PromptLast = &v
		}
		s.LastTokens = m.lastTokens
		s.LastAgeSec = int(time.Since(m.lastAt).Seconds())
	}
	return s
}
