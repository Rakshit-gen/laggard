package main

import (
	"fmt"
	"os/exec"
	"runtime"
	"slices"
	"strings"
	"time"

	"github.com/Rakshit-gen/laggard/locate"
	"github.com/Rakshit-gen/laggard/sim"
	"github.com/Rakshit-gen/laggard/trace"
)

// bench times Observe, the one call a job pays for every step, on traces
// from three job sizes.
func bench(dir string) {
	var b strings.Builder
	fmt.Fprintf(&b, "## Cost of reading a step\n\nMachine: %s, %s. One goroutine. Each job is warmed for 150 steps, then 200 steps are timed one call at a time.\n\n", cpuName(), runtime.Version())
	b.WriteString("| GPUs | layout (D x S x G) | operations per step | member records per step | Observe p50 | Observe p99 | per record | localizer memory |\n|---:|---|---:|---:|---:|---:|---:|---:|\n")
	for _, t := range []trace.Topology{{D: 4, S: 4, G: 8}, {D: 16, S: 8, G: 8}, {D: 32, S: 16, G: 8}} {
		cfg := sim.Default()
		cfg.Topo = t
		w := sim.NewWorld(cfg, 1, nil)
		runtime.GC()
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		l := locate.New(t, locate.DefaultConfig())
		var lat []time.Duration
		ops, recs := 0, 0
		for i := range 350 {
			st := w.Step()
			start := time.Now()
			l.Observe(st)
			el := time.Since(start)
			if i >= 150 {
				lat = append(lat, el)
			}
			ops = len(st.Ops)
			recs = 0
			for _, op := range st.Ops {
				recs += len(op.Ranks)
			}
		}
		slices.Sort(lat)
		runtime.GC()
		runtime.ReadMemStats(&after)
		runtime.KeepAlive(l)
		runtime.KeepAlive(w)
		p50, p99 := lat[len(lat)/2], lat[len(lat)*99/100]
		fmt.Fprintf(&b, "| %d | %d x %d x %d | %d | %d | %s | %s | %.1f ns | %.2f MB |\n", t.Ranks(), t.D, t.S, t.G, ops, recs, ms3(p50), ms3(p99), float64(p50.Nanoseconds())/float64(recs), float64(after.HeapAlloc-before.HeapAlloc)/1e6)
	}
	b.WriteString("\nMemory is heap growth from creating the localizer and running it, measured after a GC.\n\n")
	emit(dir, "bench.md", b.String())
}

func ms3(d time.Duration) string { return fmt.Sprintf("%.3f ms", float64(d.Nanoseconds())/1e6) }

func cpuName() string {
	if out, err := exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output(); err == nil {
		return strings.TrimSpace(string(out))
	}
	return runtime.GOARCH
}
