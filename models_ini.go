package main

// Чтение количества параллельных слотов (parallel) из models.ini llama.cpp.
//
// Формат — тот же, что у --models-preset: секции [имя-модели], ключи совпадают с
// аргументами командной строки без дефисов; секция [*] задаёт значения по умолчанию
// для всех моделей. Прокси нужен только ключ parallel (он же np / n-parallel).
// Файл читается один раз при запуске прокси: llama.cpp тоже применяет parallel только
// при (пере)запуске, поэтому после правки models.ini перезапускайте их вместе.

import (
	"bufio"
	"log"
	"os"
	"strconv"
	"strings"
)

type modelsINI struct {
	path string

	defaults int            // parallel из [*]; 0 = не задан
	models   map[string]int // имя секции -> parallel
}

func newModelsINI(path string) *modelsINI {
	if strings.TrimSpace(path) == "" {
		return nil
	}
	m := &modelsINI{path: path, models: map[string]int{}}
	if _, err := os.Stat(path); err != nil {
		log.Printf("[Parallel] не могу прочитать %s: %v — параллельные запросы выключены", path, err)
		return m
	}
	m.defaults, m.models = parseParallelINI(path)

	var multi []string
	for name, n := range m.models {
		if n > 1 {
			multi = append(multi, name+"="+strconv.Itoa(n))
		}
	}
	if len(multi) == 0 && m.defaults > 1 {
		multi = append(multi, "[*]="+strconv.Itoa(m.defaults))
	}
	if len(multi) == 0 {
		log.Printf("[Parallel] %s загружен: параллельных моделей нет (у всех parallel=1)", path)
	} else {
		log.Printf("[Parallel] %s загружен, параллельные модели: %s", path, strings.Join(multi, ", "))
	}
	return m
}

// parallelFor — сколько запросов модель принимает одновременно (минимум 1).
func (m *modelsINI) parallelFor(model string) int {
	if m == nil || model == "" {
		return 1
	}
	n, ok := m.models[model]
	if !ok {
		for name, v := range m.models {
			if strings.EqualFold(name, model) {
				n, ok = v, true
				break
			}
		}
	}
	if !ok {
		n = m.defaults
	}
	if n < 1 {
		return 1
	}
	return n
}

// parseParallelINI возвращает parallel из секции [*] и из каждой модели.
// Модель без собственного parallel наследует значение из [*].
func parseParallelINI(path string) (defaults int, models map[string]int) {
	models = map[string]int{}
	f, err := os.Open(path)
	if err != nil {
		return 0, models
	}
	defer f.Close()

	section := ""
	seen := map[string]bool{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || line[0] == '#' || line[0] == ';' {
			continue
		}
		if line[0] == '[' {
			if end := strings.IndexByte(line, ']'); end > 0 {
				section = strings.TrimSpace(line[1:end])
				if section != "*" {
					seen[section] = true
				}
			}
			continue
		}
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		k = strings.ToLower(strings.TrimSpace(k))
		k = strings.ReplaceAll(strings.TrimLeft(k, "-"), "_", "-")
		if k != "parallel" && k != "np" && k != "n-parallel" {
			continue
		}
		v = stripINIComment(v)
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 { // -1 («авто») и мусор считаем за 1: так безопаснее
			n = 1
		}
		switch section {
		case "":
		case "*":
			defaults = n
		default:
			models[section] = n
		}
	}
	// Модели без своего parallel — наследуют [*]; их заводим явно, чтобы было видно в логе.
	for name := range seen {
		if _, ok := models[name]; !ok && defaults > 0 {
			models[name] = defaults
		}
	}
	return defaults, models
}

// stripINIComment убирает хвостовой комментарий: "2 # много" -> "2".
func stripINIComment(v string) string {
	for i := 0; i < len(v); i++ {
		if (v[i] == '#' || v[i] == ';') && (i == 0 || v[i-1] == ' ' || v[i-1] == '\t') {
			v = v[:i]
			break
		}
	}
	return strings.TrimSpace(v)
}
