package sim

import (
	"math"
	"math/rand/v2"
	"slices"

	"github.com/Rakshit-gen/laggard/locate"
	"github.com/Rakshit-gen/laggard/trace"
)

// Detector is anything that reads traces and may blame a component.
type Detector interface {
	Name() string
	Observe(*trace.Step)
	Alarm() (locate.Alarm, bool)
	// Evict says which node the detector would drain now, if any.
	Evict() (int, bool)
	// Scores is the current alarm statistic per family of evidence, used
	// to set thresholds on healthy runs.
	Scores() []float64
	SetThresholds([]float64)
	Reset()
}

// Laggard wraps a Localizer as a Detector.
type Laggard struct {
	*locate.Localizer
	Label string
}

func (l Laggard) Name() string { return l.Label }

// Episode is one job run.
type Episode struct {
	Seed   uint64
	Steps  int
	Faults []Fault
}

// Onset is an alarm appearing, or moving to a different node.
type Onset struct {
	Step  int
	Alarm locate.Alarm
}

// Outcome is what one detector did over one episode.
type Outcome struct {
	Time      float64 // seconds to finish every step, restarts included
	Onsets    []Onset
	Evictions []Onset
	MaxScores []float64
}

// Watch runs one episode with several detectors reading the same trace.
// Nobody acts on anything, so they can share one world.
func Watch(cfg Config, ep Episode, dets []Detector) []Outcome {
	w := NewWorld(cfg, ep.Seed, ep.Faults)
	out := make([]Outcome, len(dets))
	prev := make([]int, len(dets))
	for i := range prev {
		prev[i] = -2
	}
	for range ep.Steps {
		st := w.Step()
		for i, d := range dets {
			d.Observe(st)
			track(&out[i], &prev[i], st.Index, d)
		}
	}
	for i := range out {
		out[i].Time = w.Now()
	}
	return out
}

// Act runs one episode where the detector's evictions happen: the named
// node is drained and the job restarts, costing restart seconds.
func Act(cfg Config, ep Episode, det Detector, restart float64) Outcome {
	w := NewWorld(cfg, ep.Seed, ep.Faults)
	var out Outcome
	prev := -2
	for range ep.Steps {
		st := w.Step()
		det.Observe(st)
		track(&out, &prev, st.Index, det)
		if n, ok := det.Evict(); ok && n >= 0 {
			out.Evictions = append(out.Evictions, Onset{st.Index, locate.Alarm{Node: n}})
			w.Evict(n, restart)
			det.Reset()
			prev = -2
		}
	}
	out.Time = w.Now()
	return out
}

func track(o *Outcome, prev *int, step int, d Detector) {
	sc := d.Scores()
	if o.MaxScores == nil {
		o.MaxScores = make([]float64, len(sc))
		for i := range o.MaxScores {
			o.MaxScores[i] = math.Inf(-1)
		}
	}
	for i, s := range sc {
		o.MaxScores[i] = max(o.MaxScores[i], s)
	}
	a, ok := d.Alarm()
	node := -2
	if ok {
		node = a.Node
	}
	if ok && node != *prev {
		o.Onsets = append(o.Onsets, Onset{step, a})
	}
	*prev = node
}

// thresholds sets each family's threshold just above the largest score it
// reached on healthy runs, so every detector is held to the same rule: no
// false alarms on these jobs.
func thresholds(maxScores [][]float64) []float64 {
	var h []float64
	for _, m := range maxScores {
		if h == nil {
			h = slices.Repeat([]float64{math.Inf(-1)}, len(m))
		}
		for i, v := range m {
			h[i] = max(h[i], v)
		}
	}
	for i := range h {
		h[i] *= 1.05
	}
	return h
}

// Healthy builds n fault-free episodes.
func Healthy(seed uint64, n, steps int) []Episode {
	eps := make([]Episode, n)
	for i := range eps {
		eps[i] = Episode{Seed: seed + uint64(i), Steps: steps}
	}
	return eps
}

// Case is one kind and size of fault.
type Case struct {
	Kind      FaultKind
	Factor    float64
	Transient bool
}

// Cases is the fault mix every experiment draws from.
var Cases = []Case{
	{SlowGPU, 1.03, false}, {SlowGPU, 1.05, false}, {SlowGPU, 1.1, false}, {SlowGPU, 1.2, false}, {SlowGPU, 1.5, false},
	{SlowNIC, 1.25, false}, {SlowNIC, 1.5, false}, {SlowNIC, 2, false}, {SlowNIC, 4, false},
	{SlowNode, 1.05, false}, {SlowNode, 1.1, false}, {SlowNode, 1.2, false},
	{SlowGPU, 1.5, true}, {SlowGPU, 2, true},
}

// Faulty builds reps episodes per case, each with one fault at a random
// place. For persistent faults the first rep has the fault from step 0,
// the way a bad GPU arrives with a fresh allocation; the rest start
// somewhere in the first half of the run. Transient faults last 5 to 40
// steps.
func Faulty(cfg Config, seed uint64, reps, steps int) []Episode {
	t := cfg.Topo
	var eps []Episode
	for ci, c := range Cases {
		for rep := range reps {
			s := seed + uint64(ci*1000+rep)
			rng := rand.New(rand.NewPCG(s, 0xfa17))
			f := Fault{Kind: c.Kind, Factor: c.Factor}
			if c.Kind == SlowNode {
				f.Where = rng.IntN(t.Nodes())
			} else {
				f.Where = rng.IntN(t.Ranks())
			}
			switch {
			case c.Transient:
				f.Start = 300 + rng.IntN(steps-400)
				f.End = f.Start + 5 + rng.IntN(36)
			case rep > 0:
				f.Start = 300 + rng.IntN(steps/2-300)
			}
			eps = append(eps, Episode{Seed: s, Steps: steps, Faults: []Fault{f}})
		}
	}
	return eps
}
