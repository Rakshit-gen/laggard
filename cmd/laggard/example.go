package main

import (
	"fmt"
	"math"
	"strings"

	"github.com/Rakshit-gen/laggard/internal/svg"
	"github.com/Rakshit-gen/laggard/sim"
)

// example follows one job with a NIC that starts running at two thirds of
// its speed, and shows when each detector noticed and whom it blamed.
func example(dir string) {
	cfg := sim.Default()
	t := cfg.Topo
	bad := t.Rank(2, 1, 5)
	f := sim.Fault{Kind: sim.SlowNIC, Where: bad, Factor: 1.5, Start: 800}
	cs := sim.Contenders(cfg, 0, 0)
	mks, _ := sim.Calibrated(cfg, cs, sim.Healthy(calibSeed, calibRuns, steps))
	dets := make([]sim.Detector, len(mks))
	for i, mk := range mks {
		dets[i] = mk()
	}
	w := sim.NewWorld(cfg, 7, []sim.Fault{f})
	first := make([]string, len(dets))
	var xs, ys []float64
	prev := 0.0
	for range 1400 {
		st := w.Step()
		now := w.Now()
		xs, ys = append(xs, float64(st.Index)), append(ys, now-prev)
		prev = now
		for i, d := range dets {
			d.Observe(st)
			if a, ok := d.Alarm(); ok && first[i] == "" && st.Index >= f.Start {
				who := "the job, no location"
				if a.Node >= 0 {
					who = fmt.Sprintf("%s %d on node %d", a.Target, a.Where, a.Node)
					if a.Node != f.Node(t) {
						who += " (wrong node)"
					}
				}
				first[i] = fmt.Sprintf("step %d, %d after the fault: %s", st.Index, st.Index-f.Start, who)
			}
		}
	}
	ys[0] = math.NaN()

	var b strings.Builder
	d, s, g := t.Coords(bad)
	fmt.Fprintf(&b, "## One slow NIC\n\nThe NIC of rank %d (replica %d, stage %d, position %d, node %d) runs at two thirds of its speed from step %d. Every DP all-reduce and pipeline transfer through it takes 1.5x as long. First alarm after the fault:\n\n| detector | first alarm |\n|---|---|\n", bad, d, s, g, f.Node(t), f.Start)
	for i, c := range cs {
		a := first[i]
		if a == "" {
			a = "none in 600 steps"
		}
		fmt.Fprintf(&b, "| %s | %s |\n", c.Name, a)
	}
	b.WriteString("\n")
	emit(dir, "example.md", b.String())

	ch := svg.Chart{Title: "Step time, NIC slows to 2/3 speed at step 800", XLabel: "step", YLabel: "seconds per step", XMin: 600, XMax: 1000, YMin: 0.8, YMax: 1.2,
		Series: []svg.Series{{Name: "step time", X: xs, Y: ys}}, Marks: []svg.Mark{{X: 800, Label: "NIC slows"}}}
	writeFile(dir, "example.svg", ch.Write)
}
