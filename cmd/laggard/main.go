// Command laggard runs the experiments behind the README. Every number in
// the README comes from `laggard all -out results`.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/Rakshit-gen/laggard/internal/svg"
	"github.com/Rakshit-gen/laggard/sim"
)

const usage = `usage: laggard <command> [-out dir]
       laggard watch [-restart s] [-steps n] [-patience n] [-thresholds g,n,c] < trace.jsonl
       laggard export [-steps n] [-seed n] [-fault kind:where:factor:start] > trace.jsonl

commands:
  detect    calibrate every detector on healthy jobs, then score it on one-fault jobs
  decide    let each detector drain nodes and compare total job time
  example   one job with a slow NIC, step by step
  bench     what reading one step's trace costs, at three job sizes
  all       everything above, written to -out
  watch     read a trace from stdin and say what's slow and when to drain
  export    write a simulated job's trace
`

// Sizes of the experiments. Steps per job is about 43 minutes of training.
const (
	steps        = 3000
	calibRuns    = 32
	healthyRuns  = 32
	detectReps   = 6
	decideReps   = 3
	calibSeed    = 1_000
	healthySeed  = 2_000
	faultySeed   = 100_000
	decisionSeed = 900_000
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	cmd := os.Args[1]
	switch cmd {
	case "watch":
		watchCmd(os.Args[2:])
		return
	case "export":
		exportCmd(os.Args[2:])
		return
	}
	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	out := fs.String("out", "", "directory for markdown and SVG output (default: print only)")
	fs.Parse(os.Args[2:])
	if *out != "" {
		if err := os.MkdirAll(*out, 0o755); err != nil {
			fatal(err)
		}
	}
	run := map[string]func(string){"detect": detect, "decide": decide, "example": example, "bench": bench}
	names := []string{cmd}
	if cmd == "all" {
		names = []string{"detect", "decide", "example", "bench"}
	}
	for _, name := range names {
		f, ok := run[name]
		if !ok {
			fmt.Fprint(os.Stderr, usage)
			os.Exit(2)
		}
		start := time.Now()
		fmt.Fprintf(os.Stderr, "running %s...\n", name)
		f(*out)
		fmt.Fprintf(os.Stderr, "  %s done in %.1fs\n", name, time.Since(start).Seconds())
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}

func emit(dir, name, body string) {
	fmt.Print(body)
	if dir == "" {
		return
	}
	if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
		fatal(err)
	}
}

func writeFile(dir, name string, write func(io.Writer) error) {
	if dir == "" {
		return
	}
	f, err := os.Create(filepath.Join(dir, name))
	if err != nil {
		fatal(err)
	}
	if err := write(f); err != nil {
		fatal(err)
	}
	if err := f.Close(); err != nil {
		fatal(err)
	}
}

func pct(n, d int) string {
	if d == 0 {
		return "-"
	}
	return fmt.Sprintf("%.0f%%", 100*float64(n)/float64(d))
}

func medianInt(v []int) string {
	if len(v) == 0 {
		return "-"
	}
	slices.Sort(v)
	return fmt.Sprint(v[len(v)/2])
}

// group names the row a fault falls in.
func group(f sim.Fault) string {
	if f.Transient() {
		return "transient GPU"
	}
	return "slow " + f.Kind.String()
}

var groups = []string{"slow GPU", "slow NIC", "slow node", "transient GPU"}

func detect(dir string) {
	cfg := sim.Default()
	faulty := sim.Faulty(cfg, faultySeed, detectReps, steps)
	rep := sim.Detection(cfg, sim.Contenders(cfg, 0, 0), sim.Healthy(calibSeed, calibRuns, steps), sim.Healthy(healthySeed, healthyRuns, steps), faulty)

	var b strings.Builder
	t := cfg.Topo
	fmt.Fprintf(&b, "## Detection\n\nJob: %d GPUs (%d replicas x %d pipeline stages x %d-way tensor parallel), %d steps per run. Thresholds were set on %d healthy runs; false alarms are counted on %d different healthy runs (%d steps) and before the fault in every faulty run. Faulty runs: %d, one fault each, %d per kind and size.\n\n",
		t.Ranks(), t.D, t.S, t.G, steps, calibRuns, healthyRuns, rep.HealthySteps, len(faulty), detectReps)
	b.WriteString("A fault counts as caught when the first alarm after it starts names its node. \"Wrong node\" means the first alarm named a healthy one. Transient faults count if flagged before they end plus 20 steps.\n\n")
	b.WriteString("| detector | caught | wrong node first | no location | missed | median steps to catch | false alarms, healthy runs | false alarms, before the fault |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
	for i, name := range rep.Names {
		vs := rep.Verdicts[i]
		var c, w, u, fa int
		var delays []int
		for _, v := range vs {
			fa += v.FalseBefore
			switch {
			case v.Caught:
				c++
				delays = append(delays, v.Delay)
			case v.WrongFirst:
				w++
			case v.Unlocated:
				u++
			}
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %s | %s | %s | %d | %d |\n", name, pct(c, len(vs)), pct(w, len(vs)), pct(u, len(vs)), pct(len(vs)-c-w-u, len(vs)), medianInt(delays), rep.HealthyAlarms[i], fa)
	}

	b.WriteString("\nCaught, by kind of fault:\n\n| detector |")
	for _, g := range groups {
		fmt.Fprintf(&b, " %s |", g)
	}
	b.WriteString("\n|---|" + strings.Repeat("---:|", len(groups)) + "\n")
	for i, name := range rep.Names {
		fmt.Fprintf(&b, "| %s |", name)
		for _, g := range groups {
			n, c := 0, 0
			for _, v := range rep.Verdicts[i] {
				if group(v.Fault) == g {
					n++
					if v.Caught {
						c++
					}
				}
			}
			fmt.Fprintf(&b, " %s |", pct(c, n))
		}
		b.WriteString("\n")
	}

	b.WriteString("\nlaggard by fault size. The first run of each persistent size has the fault from step 0; \"kind right\" means it also said GPU, NIC or node correctly.\n\n")
	b.WriteString("| fault | slowdown | caught | kind right | median steps to catch | caught when present from step 0 |\n|---|---:|---:|---:|---:|---:|\n")
	for _, c := range sim.Cases {
		var n, caught, kind, n0, c0 int
		var delays []int
		for _, v := range rep.Verdicts[0] {
			f := v.Fault
			if f.Kind != c.Kind || f.Factor != c.Factor || f.Transient() != c.Transient {
				continue
			}
			n++
			if f.Start == 0 && !f.Transient() {
				n0++
				if v.Caught {
					c0++
				}
			}
			if v.Caught {
				caught++
				delays = append(delays, v.Delay)
				if v.KindRight {
					kind++
				}
			}
		}
		name := group(sim.Fault{Kind: c.Kind, End: map[bool]int{true: 1}[c.Transient]})
		start0 := fmt.Sprintf("%d of %d", c0, n0)
		if c.Transient {
			start0 = "-"
		}
		fmt.Fprintf(&b, "| %s | %.2fx | %d of %d | %d of %d | %s | %s |\n", name, c.Factor, caught, n, kind, caught, medianInt(delays), start0)
	}
	b.WriteString("\nThresholds after calibration:\n\n")
	for i, name := range rep.Names {
		if rep.Thresholds[i] == nil {
			fmt.Fprintf(&b, "- %s: package defaults\n", name)
			continue
		}
		fmt.Fprintf(&b, "- %s: %s\n", name, fmtFloats(rep.Thresholds[i]))
	}
	b.WriteString("\n")
	emit(dir, "detection.md", b.String())
	catchChart(dir, rep)
}

func fmtFloats(v []float64) string {
	var s []string
	for _, x := range v {
		s = append(s, fmt.Sprintf("%.3g", x))
	}
	return strings.Join(s, ", ")
}

// catchChart plots the share of slow GPUs caught at each size.
func catchChart(dir string, rep *sim.DetectionReport) {
	var factors []float64
	for _, c := range sim.Cases {
		if c.Kind == sim.SlowGPU && !c.Transient {
			factors = append(factors, c.Factor)
		}
	}
	ch := svg.Chart{Title: "Slow GPUs caught on the right node, by slowdown", XLabel: "slowdown (x)", YLabel: "caught", XMin: 1, XMax: 1.5, YMin: 0, YMax: 1, YPercent: true}
	for i, name := range rep.Names {
		if name == "step-time alert" {
			continue
		}
		var ys []float64
		for _, f := range factors {
			n, c := 0, 0
			for _, v := range rep.Verdicts[i] {
				if v.Fault.Kind == sim.SlowGPU && !v.Fault.Transient() && v.Fault.Factor == f {
					n++
					if v.Caught {
						c++
					}
				}
			}
			ys = append(ys, float64(c)/float64(n))
		}
		ch.Series = append(ch.Series, svg.Series{Name: shortName(name), X: factors, Y: ys, Dash: i > 0})
	}
	writeFile(dir, "gpu-catch.svg", ch.Write)
}

func shortName(n string) string {
	return strings.NewReplacer("NVRx-style, defaults (0.7, 300 s)", "NVRx-style 0.7", "NVRx-style, calibrated, ", "NVRx-style cal. ").Replace(n)
}

func decide(dir string) {
	cfg := sim.Default()
	eps := sim.Faulty(cfg, decisionSeed, decideReps, steps)
	calib := sim.Healthy(calibSeed, calibRuns, steps)
	var b strings.Builder
	fmt.Fprintf(&b, "## Draining decisions\n\nEach policy acts on its own alarms: the node it names is drained and the job restarts from a checkpoint on a spare, which costs the whole job a fixed number of seconds. %d one-fault runs of %d steps, %d per kind and size. Loss is extra job time over the same run with no fault; regret is extra time over the oracle, which knows the fault and picks the cheaper of draining the moment it starts and never draining. laggard is told how many steps the job has left and waits %d steps before draining.\n\n", len(eps), steps, decideReps, sim.Patience)
	for _, r := range []float64{120, 600} {
		rep := sim.Decision(cfg, r, calib, eps)
		fmt.Fprintf(&b, "Restart costs %.0f s (about %.0f steps):\n\n| policy | mean loss | mean regret | drains | wrong node or no fault | waited-out transient drained |\n|---|---:|---:|---:|---:|---:|\n", r, r/0.85)
		for _, row := range rep.Rows {
			fmt.Fprintf(&b, "| %s | %.2f%% | %.2f%% | %d | %d | %d |\n", row.Name, 100*row.Loss, 100*row.Regret, row.Evictions, row.Wrong, row.Transient)
		}
		b.WriteString("\n")
	}
	emit(dir, "decision.md", b.String())
}
