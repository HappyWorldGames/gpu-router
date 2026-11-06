package main

import (
	"testing"
	"time"
)

func TestMeter(t *testing.T) {
	var m tokenMeter
	f := m.newFeed()
	chunk := "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n"
	for i := 0; i < 6; i++ {
		f.feed([]byte(chunk[:20])) // чанк, разрезанный посреди строки
		f.feed([]byte(chunk[20:]))
		time.Sleep(50 * time.Millisecond)
	}
	f.feed([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n"))
	if l := m.live(); l == nil || *l < 10 || *l > 40 {
		t.Fatalf("live=%v", l)
	}
	f.feed([]byte("data: {\"choices\":[],\"timings\":{\"predicted_n\":42,\"predicted_per_second\":17.5,\"prompt_per_second\":230.1}}\n"))
	if s := m.snap(); s.Last == nil || *s.Last != 17.5 || s.LastTokens != 42 {
		t.Fatalf("%+v", s)
	}
	var m2 tokenMeter
	f2 := m2.newFeed()
	f2.feed([]byte(`{"content":"x","timings":{"predicted_n":5,"predicted_per_second":9.0}}`))
	f2.end()
	if s2 := m2.snap(); s2.Last == nil || *s2.Last != 9.0 {
		t.Fatalf("%+v", s2)
	}
	if !hasText([]byte(`{"delta":{"reasoning_content":"a"}}`)) || hasText([]byte(`{"delta":{"content":""}}`)) {
		t.Fatal("hasText")
	}
}
