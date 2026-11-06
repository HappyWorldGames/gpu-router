# gpu-router

**English** | [Русский](README.ru.md)

A queueing proxy for one shared GPU (or an APU with shared memory) that hosts
both **ComfyUI** and **llama.cpp** (`llama-server` in router mode). It hands the
GPU to one service at a time, unloads the models of whichever service gives
way, and shows on a web page who is using the GPU right now.

A single Go binary with only two library dependencies.

> **Note:** the status page is available in English and Russian (detected from
> the browser, switchable in the page footer). Log messages are currently in
> Russian.

## Why

If ComfyUI and llama.cpp keep their models in the same video memory (especially
on an APU such as the Radeon 780M, where "VRAM" is the same physical RAM as
system memory), running both at once ends in OOM, freezes, and processes killed
by the kernel. Clients such as Open WebUI or n8n know nothing about this and
send requests whenever they like.

gpu-router sits between the clients and the services and fixes that:

- requests that need the GPU are lined up in a **FIFO queue** instead of running
  simultaneously;
- before the GPU goes to the other service, the previous one is **unloaded**
  (softly, through its API), and if it does not give the memory back, its
  processes are **killed**;
- it watches **system RAM** and keeps the machine from reaching the kernel OOM
  killer;
- it draws a **status page**: GPU usage, service health, and generation speed in
  tokens per second.

## Features

- **Request queue** for ComfyUI (`/prompt`) and llama.cpp (any POST that can
  start a generation). Service paths (`/props`, `/models`, `/tokenize`, `/slots`
  , and so on) go straight through, bypassing the queue.
- **GPU hand-over:** ComfyUI is freed through `/interrupt` \+ `/free`, llama
  through `/models/unload`. If memory is still not released after `
  release_timeout`, the unit's processes get `SIGKILL`.
- **Per-process VRAM accounting** from the kernel's `fdinfo` (AMD, Intel, no
  external tools) or via `nvidia-smi`. Service processes are found through their
  systemd units.
- **Parallel requests to llama.cpp:** the number of slots (`parallel`) is read
  from your `models.ini`. Details below.
- **Memory protection:** warnings, waiting for RAM to recover before a task, and
  emergency stop when memory is critically low. While the GPU is idle, a model
  is not evicted that easily.
- **Queue-aware timeouts:** a stuck request is aborted only if someone is
  actually waiting behind it.
- **Idle unload** for ComfyUI (configurable).
- **Status page** for a home-server landing page: who holds the GPU, how many
  requests are running, memory, tokens per second, service availability.

## How it works

```
 Open WebUI, n8n, ...                      browser
        │                                     │
        ▼                                     ▼
  :8080  llama proxy                   :8188  ComfyUI proxy
        └──────────── gpu-router ─────────────┘
          queue · VRAM accounting · unloading · watchdog
        │                                     │
        ▼                                     ▼
  :8180  llama-server (router)         :8288  ComfyUI

  :8190  /api/status  → status page (through Caddy or directly)
```
Clients switch to the proxy ports (`8080` and `8188`); the services themselves
move to different ports (`8180` and `8288`).

How a single task is processed:

1.  The task joins the queue (`queue.max_size`; waiting is limited by `
    queue.wait_timeout`).
2.  When its turn comes, the proxy prepares the GPU: it waits for ComfyUI to
    finish its current generation, unloads the competitor's models, checks that
    memory was really released (taking `residual_mb` into account), and waits
    for enough free RAM (`gpu.min_available_mb`).
3.  The request is proxied to the backend. llama's response is streamed
    through; for ComfyUI the proxy waits for the generation to finish.
4.  Every `gpu.watchdog_interval` seconds a watchdog checks whether both
    services hold memory at once (for example, someone talked to a backend
    directly) and evicts the one that should not be there.

## Requirements

- Linux with systemd (service processes are identified by their cgroup).
- Go 1.21+ to build.
- ComfyUI and llama.cpp (`llama-server` in router mode with a `models.ini`),
  running as systemd units.
- Permission to read `/proc/\<pid>/fdinfo` of the services' processes and to
  signal them: run as root, or as the same user as the services.
- VRAM data source: AMD/Intel work through `fdinfo` (main testing: AMD Radeon
  780M APU). There is an `nvidia-smi` mode for NVIDIA, but verify it on your own
  card.

## Quick start

### 1\. Build

```bash
go mod tidy
go build -o gpu-router .
```
### 2\. Move the services to other ports

The proxy takes the familiar ports, so the services must listen elsewhere. For
example, in their units: ComfyUI on `8288`, llama-server on `8180`. Put these
values into `comfyui.port` and `llama.port`.

### 3\. Config

```bash
cp config.example.yaml config.yaml
```
Make sure to check:

- `comfyui.unit` and `llama.unit`: the names of your systemd units (`systemctl
  list-units | grep -i comfy`);
- backend ports and proxy ports;
- `proxy.listen_host`: by default it listens on all interfaces and **the proxy
  has no authentication of its own**, so block ports `8080`/`8188` from the
  outside network with a firewall;
- `comfyui.residual_mb` and `llama.residual_mb`: how much memory a service may
  keep after unloading. Tune them from the `residual N MB` log line.

Every option is described in the comments of `config.example.yaml` (a
Russian-commented copy is in `config.example.ru.yaml`).

### 4\. Run

```bash
./gpu-router config.yaml
```
For permanent use, take the unit from `deploy/gpu-router.service`:

```bash
sudo install -D gpu-router /opt/gpu-router/gpu-router
sudo install -m 600 config.yaml /opt/gpu-router/config.yaml
sudo cp deploy/gpu-router.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-router
```
### 5\. Point the clients at the proxy

Open WebUI, n8n and everything else should use `:8080` (llama) and `:8188`
(ComfyUI), not the services directly. Direct access to the backends bypasses
the queue; the watchdog will notice and resolve the conflict, but it is more
reliable not to rely on that.

## Parallel requests to llama.cpp

By default llama gets one request at a time. If a model in `models.ini` has `
parallel` greater than 1, the proxy lets that many requests run on the already
loaded model at once.

```yaml
llama:
  models_ini: "/path/to/models.ini"
```
Rules:

- A request joins the running ones only if it targets the same model (the `model`
  field in its body) and a slot is free. Otherwise it waits.
- A request for a different model, and any ComfyUI task, waits until all
  running llama requests finish. Queue order is preserved, so ComfyUI does not
  starve.
- `parallel` comes from the model's section, or from `[*]` if absent. The keys `
  parallel`, `np` and `n-parallel` are recognised. Invalid values and `\-1`
  (auto) count as 1.
- The file is read once when the proxy starts. After editing `models.ini`,
  restart llama.cpp and gpu-router together: llama.cpp itself applies `parallel`
  only on (re)start.

Keep in mind that without `kv-unified`, `ctx-size` is split between the slots,
so each request gets a smaller context. Check `total_slots` and the slot context
size in your build's `/props`.

## Memory protection and timeouts


|Option|What it does|
|-|-|
|`gpu.min_available_mb`, `gpu.mem_wait`|Before a task, waits for available RAM to rise above the threshold (up to `mem_wait` seconds), then proceeds at its own risk. The log lists the heaviest processes.|
|`gpu.critical_available_mb`|If available RAM stays below the threshold for two consecutive samples, the proxy stops whoever holds the GPU: softly first, then `SIGKILL` on the unit after ~4 s without improvement. Faster and more targeted than the kernel OOM killer.|
|`gpu.critical_idle_available_mb`|The same, but while there is no task and the queue is empty. The model is not evicted until memory falls below this threshold (default: half of the critical one; `\-1` = never evict while idle).|
|`gpu.kill_on_leak`|`false` = only log, never kill.|
|`queue.task_timeout`|Ceiling for one task. Fires only if someone is waiting behind it.|
|`queue.exec_idle_timeout`|No data from the backend for this many seconds: abort the request. Also only if someone is waiting. For long contexts set it generously: processing a 90k-token prompt can stay silent for minutes.|
|`queue.comfy_idle_timeout`|Maximum ComfyUI generation time.|

## Status page

The proxy starts a separate small server (`127.0.0.1:8190` by default) with a
single endpoint, `/api/status`, that returns JSON: who holds the GPU, queue
length, ComfyUI and llama state, memory, generation speed, service
availability. It bypasses the queue and does not write to the log, so polling
the page never gets in anyone's way.

`web/index.html` can be served by any web server. A Caddy example is in `
deploy/Caddyfile`:

```caddy
home.home {
    handle /api/* {
        reverse_proxy 127.0.0.1:8190
    }
    handle {
        root * /usr/share/caddy   # put web/index.html here
        file_server
    }
}
```
The page polls the status every 2 seconds, loads nothing from external
addresses, follows the light/dark theme, and switches between English and
Russian (picked from the browser language; the choice is remembered). The
service list is set in `dashboard.services` (availability is checked over TCP).

Tokens per second are computed by peeking at llama's response as it passes
through (without modifying it): a live rate over a sliding window, and a final
figure from the `timings` field of the last chunk. If a client asks for a
compressed response, no numbers appear, but proxying is unaffected.

## Endpoints


|Address|Purpose|
|-|-|
|`:8188`, `:8080` `/status`|Debug status of the proxy.|
|`127.0.0.1:8190` `/api/status`|JSON for the status page.|

## Logs

Log lines carry tags: `\[Start]`/`\[Done]` (task started/finished), `\[Queue]`, `
\[Release]`/`\[Unload]` (unloading), `\[Evict]` (forced eviction), `\[MemGuard]`/`
\[Mem]` (memory), `\[Worker]` (timeouts), `\[Parallel]` (slots). If something
unloaded your model, start with:

```bash
journalctl -u gpu-router | grep -E "MemGuard|Evict|Release|Unload|Worker"
```
`logging.verbose`, `logging.requests` and `logging.mem_trace` control the
verbosity.

## Limitations

- No authentication: protect the ports with a firewall.
- Must run on the same machine as the services (it needs `/proc` and systemd).
- Direct access to the backends bypasses the queue.
- Only requests to the same llama model run in parallel.
- Per-process memory as reported by the kernel includes both VRAM and GTT
  (shared memory).
- Log messages are in Russian for now (the status page is bilingual).

## Development

```bash
go test -race ./...
```
The tests cover `models.ini` parsing, the admission logic for parallel requests
(joining, waiting for a slot, another model, ComfyUI, client disconnect),
stream parsing for the token counter, and the queue-dependent timeouts. They
need neither a GPU nor running services.

