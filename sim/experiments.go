package sim

import (
	"math"
	"slices"

	"github.com/Rakshit-gen/laggard/baseline"
	"github.com/Rakshit-gen/laggard/locate"
	"github.com/Rakshit-gen/laggard/trace"
)

// Contender is one detector setup in the comparison.
type Contender struct {
	Name      string
	New       func() Detector
	Calibrate bool // set thresholds from healthy runs; otherwise keep defaults
}

// Contenders is the line-up every experiment uses. restart and steps go to
// laggard's drain rule; the baselines drain as soon as they alarm.
func Contenders(cfg Config, restart float64, steps int) []Contender {
	t := cfg.Topo
	return []Contender{
		{"laggard", func() Detector {
			c := locate.DefaultConfig()
			c.Restart, c.Steps, c.Patience = restart, steps, Patience
			return Laggard{locate.New(t, c), "laggard"}
		}, true},
		{"step-time alert", func() Detector { return baseline.NewStepTime() }, true},
		{"last-arriver blame", func() Detector { return baseline.NewBlame(t) }, true},
		{"NVRx-style, defaults (0.7, 300 s)", func() Detector { return baseline.NewNVRx(t, 300, "NVRx-style, defaults (0.7, 300 s)") }, false},
		{"NVRx-style, calibrated, 300 s", func() Detector { return baseline.NewNVRx(t, 300, "NVRx-style, calibrated, 300 s") }, true},
		{"NVRx-style, calibrated, 30 s", func() Detector { return baseline.NewNVRx(t, 30, "NVRx-style, calibrated, 30 s") }, true},
	}
}

// Calibrated returns constructors with thresholds fixed from healthy runs.
// Every contender watches the same healthy jobs at once.
func Calibrated(cfg Config, cs []Contender, healthy []Episode) ([]func() Detector, [][]float64) {
	outs := parallel(len(healthy), func(i int) []Outcome {
		dets := make([]Detector, len(cs))
		for j, c := range cs {
			dets[j] = c.New()
		}
		return Watch(cfg, healthy[i], dets)
	})
	mks := make([]func() Detector, len(cs))
	hs := make([][]float64, len(cs))
	for i, c := range cs {
		mks[i] = c.New
		if !c.Calibrate {
			continue
		}
		var per [][]float64
		for _, os := range outs {
			per = append(per, os[i].MaxScores)
		}
		h := thresholds(per)
		hs[i] = h
		mk := c.New
		mks[i] = func() Detector {
			d := mk()
			d.SetThresholds(h)
			return d
		}
	}
	return mks, hs
}

// Patience is how long laggard waits out an alarm before draining. The
// simulated transient slowdowns last at most 40 steps.
const Patience = 50

// Verdict is how one detector did on one faulty episode.
type Verdict struct {
	Fault       Fault
	FalseBefore int  // alarms before the fault began
	Caught      bool // the first alarm after the fault began named its node
	WrongFirst  bool // the first alarm after the fault began named another node
	Unlocated   bool // the first alarm after the fault began named no node
	Delay       int  // steps from fault start to the first alarm
	KindRight   bool // and it named the right kind of component
}

func targetMatches(t locate.Target, k FaultKind) bool {
	return (t == locate.GPU && k == SlowGPU) || (t == locate.NIC && k == SlowNIC) || (t == locate.Node && k == SlowNode)
}

// Judge scores one detector's onsets against the episode's fault.
func Judge(cfg Config, ep Episode, o Outcome) Verdict {
	f := ep.Faults[0]
	v := Verdict{Fault: f, Delay: -1}
	node := f.Node(cfg.Topo)
	for _, on := range o.Onsets {
		if on.Step < f.Start {
			v.FalseBefore++
			continue
		}
		if f.Transient() && on.Step >= f.End+20 {
			break
		}
		v.Delay = on.Step - f.Start
		switch {
		case on.Alarm.Node == node:
			v.Caught = true
			v.KindRight = targetMatches(on.Alarm.Target, f.Kind)
		case on.Alarm.Node < 0:
			v.Unlocated = true
		default:
			v.WrongFirst = true
		}
		break
	}
	return v
}

// DetectionReport is the outcome of the detection experiment.
type DetectionReport struct {
	Names      []string
	Thresholds [][]float64
	Verdicts   [][]Verdict // [contender][episode]
	// Healthy test runs, separate from the calibration runs.
	HealthyRuns, HealthySteps int
	HealthyAlarms             []int
}

// Detection calibrates every contender on one set of healthy jobs, then
// watches fresh healthy jobs and one-fault jobs.
func Detection(cfg Config, calib, healthy, faulty []Episode) *DetectionReport {
	cs := Contenders(cfg, 0, 0)
	mks, hs := Calibrated(cfg, cs, calib)
	rep := &DetectionReport{Thresholds: hs, Verdicts: make([][]Verdict, len(cs)), HealthyAlarms: make([]int, len(cs))}
	for _, c := range cs {
		rep.Names = append(rep.Names, c.Name)
	}
	all := append(slices.Clone(healthy), faulty...)
	outs := parallel(len(all), func(i int) []Outcome {
		dets := make([]Detector, len(mks))
		for j, mk := range mks {
			dets[j] = mk()
		}
		return Watch(cfg, all[i], dets)
	})
	for i, os := range outs {
		for j, o := range os {
			if i < len(healthy) {
				rep.HealthyAlarms[j] += len(o.Onsets)
				continue
			}
			rep.Verdicts[j] = append(rep.Verdicts[j], Judge(cfg, all[i], o))
		}
	}
	rep.HealthyRuns = len(healthy)
	for _, e := range healthy {
		rep.HealthySteps += e.Steps
	}
	return rep
}

// scripted drains a fixed node at a fixed step. It's how the oracle acts.
type scripted struct {
	step, node, now int
	done            bool
}

func (s *scripted) Name() string                { return "oracle" }
func (s *scripted) Observe(st *trace.Step)      { s.now = st.Index }
func (s *scripted) Alarm() (locate.Alarm, bool) { return locate.Alarm{}, false }
func (s *scripted) Scores() []float64           { return nil }
func (s *scripted) SetThresholds([]float64)     {}
func (s *scripted) Reset()                      {}
func (s *scripted) Evict() (int, bool) {
	if s.done || s.now < s.step {
		return -1, false
	}
	s.done = true
	return s.node, true
}

// jobTime is how long the episode takes with nobody acting.
func jobTime(cfg Config, ep Episode) float64 {
	w := NewWorld(cfg, ep.Seed, ep.Faults)
	for range ep.Steps {
		w.Step()
	}
	return w.Now()
}

// DecisionRow is one policy's cost over the decision experiment.
type DecisionRow struct {
	Name      string
	Loss      float64 // mean extra job time over the healthy run, as a fraction
	Regret    float64 // mean extra job time over the oracle, as a fraction of healthy time
	Evictions int
	Wrong     int // drained a node that had no fault
	Transient int // drained a node whose fault would have ended by itself
}

// DecisionReport compares what each policy's evictions cost.
type DecisionReport struct {
	Restart  float64
	Episodes int
	Rows     []DecisionRow
}

// Decision lets every policy act on the same episodes and compares total
// job time, restarts included. The oracle knows the fault and takes the
// cheaper of draining its node the moment it starts and leaving it alone.
// The step-time alert can't name a node, so it isn't in this one.
func Decision(cfg Config, restart float64, calib, eps []Episode) *DecisionReport {
	var cs []Contender
	for _, c := range Contenders(cfg, restart, eps[0].Steps) {
		if c.Name != "step-time alert" {
			cs = append(cs, c)
		}
	}
	// laggard with other drain rules, to separate what the rule adds from
	// what the detector adds.
	variant := func(name string, set func(*locate.Config)) Contender {
		return Contender{name, func() Detector {
			c := locate.DefaultConfig()
			set(&c)
			return Laggard{locate.New(cfg.Topo, c), name}
		}, true}
	}
	cs = append(cs[:1], append([]Contender{
		variant("laggard, ski rental only", func(c *locate.Config) { c.Restart = restart }),
		variant("laggard, drain on first alarm", func(c *locate.Config) {}),
	}, cs[1:]...)...)
	mks, _ := Calibrated(cfg, cs, calib)

	type res struct {
		healthy, never, oracle float64
		outs                   []Outcome
	}
	rs := parallel(len(eps), func(i int) res {
		ep := eps[i]
		f := ep.Faults[0]
		r := res{healthy: jobTime(cfg, Episode{Seed: ep.Seed, Steps: ep.Steps}), never: jobTime(cfg, ep)}
		r.oracle = r.never
		if !f.Transient() {
			o := Act(cfg, ep, &scripted{step: f.Start, node: f.Node(cfg.Topo)}, restart)
			r.oracle = min(r.oracle, o.Time)
		}
		for _, mk := range mks {
			r.outs = append(r.outs, Act(cfg, ep, mk(), restart))
		}
		return r
	})

	rep := &DecisionReport{Restart: restart, Episodes: len(eps)}
	rows := []DecisionRow{{Name: "never drain"}, {Name: "oracle"}}
	for _, c := range cs {
		rows = append(rows, DecisionRow{Name: c.Name})
	}
	n := float64(len(eps))
	for i, r := range rs {
		f := eps[i].Faults[0]
		node := f.Node(cfg.Topo)
		add := func(row *DecisionRow, t float64) {
			row.Loss += (t/r.healthy - 1) / n
			row.Regret += (t - r.oracle) / r.healthy / n
		}
		add(&rows[0], r.never)
		add(&rows[1], r.oracle)
		if r.oracle < r.never {
			rows[1].Evictions++
		}
		for j, o := range r.outs {
			row := &rows[2+j]
			add(row, o.Time)
			for _, e := range o.Evictions {
				row.Evictions++
				live := e.Step >= f.Start && (!f.Transient() || e.Step < f.End)
				switch {
				case e.Alarm.Node != node || !live:
					row.Wrong++
				case f.Transient():
					row.Transient++
				}
			}
		}
	}
	rep.Rows = rows
	return rep
}

// Mean is a small helper for reports.
func Mean(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s / math.Max(1, float64(len(v)))
}
