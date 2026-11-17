package main

// Имена контейнеров Docker для страницы очереди.
//
// Адрес контейнера (например, Open WebUI) меняется при каждом пересоздании, поэтому
// привязать к нему название в конфиге нельзя. Если задан dashboard.docker_socket, прокси
// спрашивает у Docker список контейнеров (GET /containers/json, только чтение) и по
// текущему адресу показывает имя сервиса из compose (или имя контейнера).
// Запрос идёт в фоне и кэшируется: очередь и страницу он не задерживает.

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

type dockerNames struct {
	mu     sync.Mutex
	m      map[string]string // адрес контейнера -> имя
	at     time.Time         // когда последний раз пытались обновить
	busy   bool
	warned bool
}

var (
	dockerTTL      = 60 * time.Second // плановое обновление
	dockerMissWait = 5 * time.Second  // неизвестный адрес: перечитать, но не чаще раза в 5 с
)

// dockerName возвращает имя контейнера по адресу или "" (пока неизвестно / Docker не настроен).
func (s *Server) dockerName(ip string) string {
	sock := s.Config.Dashboard.DockerSocket
	if sock == "" {
		return ""
	}
	d := &s.docker
	d.mu.Lock()
	defer d.mu.Unlock()
	name, ok := d.m[ip]
	age := time.Since(d.at)
	if !d.busy && (age > dockerTTL || (!ok && age > dockerMissWait)) {
		d.busy = true
		go s.refreshDocker(sock)
	}
	return name
}

type dockerContainer struct {
	Names           []string          `json:"Names"`
	Labels          map[string]string `json:"Labels"`
	NetworkSettings struct {
		Networks map[string]struct {
			IPAddress string `json:"IPAddress"`
		} `json:"Networks"`
	} `json:"NetworkSettings"`
}

func (s *Server) refreshDocker(sock string) {
	d := &s.docker
	m, err := fetchDockerNames(sock)
	d.mu.Lock()
	defer d.mu.Unlock()
	d.busy = false
	d.at = time.Now()
	if err != nil {
		if !d.warned {
			d.warned = true
			log.Printf("[Dashboard] docker: не удалось прочитать список контейнеров через %s: %v (имена контейнеров не будут показаны; "+
				"нужен доступ к сокету: запуск от root или группа docker)", sock, err)
		}
		return
	}
	d.warned = false
	d.m = m
}

func fetchDockerNames(sock string) (map[string]string, error) {
	tr := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			var dl net.Dialer
			return dl.DialContext(ctx, "unix", sock)
		},
	}
	defer tr.CloseIdleConnections()
	cl := &http.Client{Transport: tr, Timeout: 2 * time.Second}
	resp, err := cl.Get("http://docker/containers/json")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, &httpStatusError{resp.StatusCode}
	}
	var list []dockerContainer
	if err := json.NewDecoder(io.LimitReader(resp.Body, 8<<20)).Decode(&list); err != nil {
		return nil, err
	}
	out := map[string]string{}
	for _, c := range list {
		name := c.Labels["com.docker.compose.service"]
		if name == "" && len(c.Names) > 0 {
			name = strings.TrimPrefix(c.Names[0], "/")
		}
		if name == "" {
			continue
		}
		for _, n := range c.NetworkSettings.Networks {
			if n.IPAddress != "" {
				out[n.IPAddress] = name
			}
		}
	}
	return out, nil
}

type httpStatusError struct{ code int }

func (e *httpStatusError) Error() string { return "HTTP " + http.StatusText(e.code) }
