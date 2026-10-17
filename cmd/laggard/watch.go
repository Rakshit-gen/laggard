package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/Rakshit-gen/laggard/locate"
	"github.com/Rakshit-gen/laggard/sim"
	"github.com/Rakshit-gen/laggard/trace"
)

// watchCmd reads a trace from stdin and prints a line whenever the most
// confident alarm changes, and when laggard would drain a node.
func watchCmd(args []string) {
	fs := flag.NewFlagSet("watch", flag.ExitOnError)
	restart := fs.Float64("restart", 120, "seconds one restart costs the job")
	steps := fs.Int("steps", 0, "steps the job has left to run (0: unknown, use ski rental)")
	patience := fs.Int("patience", sim.Patience, "steps an alarm must last before a drain")
	thr := fs.String("thresholds", "", "GPU,node,NIC thresholds (default: the package defaults)")
	fs.Parse(args)
	c := locate.DefaultConfig()
	c.Restart, c.Steps, c.Patience = *restart, *steps, *patience
	if *thr != "" {
		parts := strings.Split(*thr, ",")
		if len(parts) != locate.Families {
			fatal(fmt.Errorf("-thresholds wants %d numbers", locate.Families))
		}
		for i, p := range parts {
			v, err := strconv.ParseFloat(p, 64)
			if err != nil || v <= 0 {
				fatal(fmt.Errorf("-thresholds: bad number %q", p))
			}
			c.Thresholds[i] = v
		}
	}
	if err := watch(os.Stdin, os.Stdout, c); err != nil {
		fatal(err)
	}
}

// watch prints a line whenever the most confident alarm changes, and one
// when a node should be drained. After a drain it reads to the end of the
// input without saying more, since the job would have restarted.
func watch(in io.Reader, w io.Writer, c locate.Config) error {
	rd, err := trace.NewReader(in)
	if err != nil {
		return err
	}
	l := locate.New(rd.Topo, c)
	last := ""
	for {
		st, err := rd.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		l.Observe(st)
		now := "quiet"
		if a, ok := l.Alarm(); ok {
			now = fmt.Sprintf("%s %d on node %d", a.Target, a.Where, a.Node)
			if a.Excess > 0 {
				now += fmt.Sprintf(", costing %.3f s per step", a.Excess)
			}
		}
		if key := strings.SplitN(now, ",", 2)[0]; key != strings.SplitN(last, ",", 2)[0] {
			fmt.Fprintf(w, "step %d: %s\n", st.Index, now)
		}
		last = now
		if n, ok := l.Evict(); ok {
			fmt.Fprintf(w, "step %d: drain node %d\n", st.Index, n)
			_, err := io.Copy(io.Discard, in)
			return err
		}
	}
}

// exportCmd writes a simulated job's trace, for trying watch.
func exportCmd(args []string) {
	fs := flag.NewFlagSet("export", flag.ExitOnError)
	steps := fs.Int("steps", 400, "steps to run")
	seed := fs.Uint64("seed", 1, "random seed")
	fault := fs.String("fault", "", "kind:where:factor:start, kind is gpu, nic or node; e.g. nic:77:1.5:200")
	fs.Parse(args)
	cfg := sim.Default()
	var faults []sim.Fault
	if *fault != "" {
		f, err := parseFault(*fault, cfg.Topo)
		if err != nil {
			fatal(err)
		}
		faults = append(faults, f)
	}
	w := sim.NewWorld(cfg, *seed, faults)
	tw, err := trace.NewWriter(os.Stdout, cfg.Topo)
	if err != nil {
		fatal(err)
	}
	for range *steps {
		if err := tw.Write(w.Step()); err != nil {
			fatal(err)
		}
	}
	if err := tw.Flush(); err != nil {
		fatal(err)
	}
}

func parseFault(s string, t trace.Topology) (sim.Fault, error) {
	p := strings.Split(s, ":")
	if len(p) != 4 {
		return sim.Fault{}, fmt.Errorf("fault %q: want kind:where:factor:start", s)
	}
	kind, ok := map[string]sim.FaultKind{"gpu": sim.SlowGPU, "nic": sim.SlowNIC, "node": sim.SlowNode}[p[0]]
	if !ok {
		return sim.Fault{}, fmt.Errorf("fault kind %q: want gpu, nic or node", p[0])
	}
	where, err1 := strconv.Atoi(p[1])
	factor, err2 := strconv.ParseFloat(p[2], 64)
	start, err3 := strconv.Atoi(p[3])
	if err1 != nil || err2 != nil || err3 != nil || factor < 1 || start < 0 {
		return sim.Fault{}, fmt.Errorf("fault %q: bad number", s)
	}
	limit := t.Ranks()
	if kind == sim.SlowNode {
		limit = t.Nodes()
	}
	if where < 0 || where >= limit {
		return sim.Fault{}, fmt.Errorf("fault %q: %d out of range (0 to %d)", s, where, limit-1)
	}
	return sim.Fault{Kind: kind, Where: where, Factor: factor, Start: start}, nil
}
