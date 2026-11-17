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
	"strings"
	"sync"
	"time"
	"unicode/utf8"
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
	sink    *liveText // nil = живой текст выключен: разбирать содержимое чанков не нужно
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
		f.m.line(f.pending[:i], f.sink)
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
		f.m.line(f.pending, f.sink)
		f.pending = f.pending[:0]
	}
}

func (m *tokenMeter) line(b []byte, sink *liveText) {
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
		if sink != nil {
			sink.addChunk(b)
		}
		now := time.Now()
		m.mu.Lock()
		m.stamps = append(m.stamps, now)
		if len(m.stamps) > tokMaxStamp {
			m.stamps = m.stamps[len(m.stamps)-tokMaxStamp:]
		}
		m.mu.Unlock()
	}
}

// liveText — хвост генерируемого текста одного запроса для страницы состояния.
// Включается настройкой dashboard.show_text; добавление — микросекунды на токен.
type liveText struct {
	mu       sync.Mutex
	buf      []byte // последние байты текста (с запасом)
	cut      bool   // раньше был ещё текст, который уже отброшен
	thinking bool   // последний фрагмент был «рассуждением» модели
	max      int    // сколько последних символов показывать
}

func newLiveText(maxRunes int) *liveText {
	if maxRunes < 40 {
		maxRunes = 40
	}
	if maxRunes > 2000 {
		maxRunes = 2000
	}
	return &liveText{max: maxRunes}
}

// addChunk достаёт текст из одного чанка потока llama-server (OpenAI-формат или родной /completion).
func (l *liveText) addChunk(b []byte) {
	var v struct {
		Content string `json:"content"`
		Choices []struct {
			Delta struct {
				Content   string `json:"content"`
				Reasoning string `json:"reasoning_content"`
			} `json:"delta"`
			Text string `json:"text"`
		} `json:"choices"`
	}
	if json.Unmarshal(b, &v) != nil {
		return
	}
	text, thinking := v.Content, false
	if len(v.Choices) > 0 {
		c := v.Choices[0]
		switch {
		case c.Delta.Content != "":
			text = c.Delta.Content
		case c.Delta.Reasoning != "":
			text, thinking = c.Delta.Reasoning, true
		case c.Text != "":
			text = c.Text
		}
	}
	if text == "" {
		return
	}
	l.mu.Lock()
	l.buf = append(l.buf, text...)
	l.thinking = thinking
	// держим не больше 4 байт на символ показа (UTF-8) и режем только по границе символа
	if limit := l.max * 4; len(l.buf) > 2*limit {
		l.cut = true
		keep := l.buf[len(l.buf)-limit:]
		for len(keep) > 0 && !utf8.RuneStart(keep[0]) {
			keep = keep[1:]
		}
		l.buf = append(l.buf[:0], keep...)
	}
	l.mu.Unlock()
}

// snapshot — последние символы текста (с «…» в начале, если это не всё) и признак рассуждения.
func (l *liveText) snapshot() (string, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if len(l.buf) == 0 {
		return "", false
	}
	r := []rune(string(l.buf))
	more := l.cut
	if len(r) > l.max {
		r, more = r[len(r)-l.max:], true
	}
	text := strings.TrimLeft(string(r), "\n ")
	if more {
		text = "…" + text
	}
	return text, l.thinking
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
