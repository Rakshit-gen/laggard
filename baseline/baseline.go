// Package baseline holds the detectors laggard is measured against. Each is
// a fair version of something people run today, given the same trace.
package baseline

import (
	"math"
	"slices"

	"github.com/Rakshit-gen/laggard/locate"
	"github.com/Rakshit-gen/laggard/trace"
)

// computeTracker sums each rank's time between its own operations.
type computeTracker struct {
	lastExit []float64
	seen     []bool
	comp     []float64
}

func newComputeTracker(n int) computeTracker {
	return computeTracker{make([]float64, n), make([]bool, n), make([]float64, n)}
}

func (c *computeTracker) observe(st *trace.Step) bool {
	first := !c.seen[0]
	clear(c.comp)
	for i := range st.Ops {
		op := &st.Ops[i]
		for j, r := range op.Ranks {
			if c.seen[r] {
				c.comp[r] += op.Enter[j] - c.lastExit[r]
			}
			c.lastExit[r], c.seen[r] = op.Exit[j], true
		}
	}
	return !first
}

func (c *computeTracker) reset() { clear(c.seen) }

// StepTime alarms when the job's step time stays above its recent median.
// It's what a dashboard alert on throughput does. It can say the job is
// slow but not why, so it never evicts.
type StepTime struct {
	Window, Run int
	thr         float64
	hist        []float64 // step times, newest last
	last        float64
	seen        bool
	ratios      []float64
	score       float64
}

func NewStepTime() *StepTime { return &StepTime{Window: 200, Run: 5, thr: 0.15} }

func (s *StepTime) Name() string { return "step-time alert" }

func (s *StepTime) Observe(st *trace.Step) {
	// Rank 0's own clock is enough to time the job: every rank finishes
	// together at the step's last collective.
	end := 0.0
	for i := range st.Ops {
		op := &st.Ops[i]
		if op.Ranks[0] == 0 {
			end = op.Exit[0]
		}
	}
	if !s.seen {
		s.last, s.seen = end, true
		return
	}
	dt := end - s.last
	s.last = end
	if len(s.hist) >= s.Window/4 {
		m := median(slices.Clone(s.hist))
		s.ratios = append(s.ratios, dt/m-1)
		if len(s.ratios) > s.Run {
			s.ratios = s.ratios[1:]
		}
	}
	s.hist = append(s.hist, dt)
	if len(s.hist) > s.Window {
		s.hist = s.hist[1:]
	}
	s.score = 0
	if len(s.ratios) == s.Run {
		s.score = slices.Min(s.ratios)
	}
}

func (s *StepTime) Alarm() (locate.Alarm, bool) {
	if s.score < s.thr {
		return locate.Alarm{}, false
	}
	return locate.Alarm{Target: locate.Job, Where: -1, Node: -1, Score: s.score / s.thr}, true
}

func (s *StepTime) Evict() (int, bool)        { return -1, false }
func (s *StepTime) Scores() []float64         { return []float64{s.score} }
func (s *StepTime) SetThresholds(h []float64) { s.thr = h[0] }
func (s *StepTime) Reset()                    { *s = StepTime{Window: s.Window, Run: s.Run, thr: s.thr} }

// Blame charges every collective's waiting to the member who arrived last,
// and alarms on the rank holding the largest share of the blame over a
// sliding window. It's the "who are the others waiting for" question that
// flight-recorder analyses ask. PP transfers are left out, because a
// pipeline's warm-up bubble would put the first stages on top for no fault
// of their own.
type Blame struct {
	Window int
	topo   trace.Topology
	thr    float64
	ring   [][]float64
	sum    []float64
	n      int
	score  float64
	top    int
}

func NewBlame(t trace.Topology) *Blame {
	b := &Blame{Window: 20, topo: t, thr: 0.07}
	b.ring = make([][]float64, b.Window)
	for i := range b.ring {
		b.ring[i] = make([]float64, t.Ranks())
	}
	b.sum = make([]float64, t.Ranks())
	return b
}

func (b *Blame) Name() string { return "last-arriver blame" }

func (b *Blame) Observe(st *trace.Step) {
	cur := b.ring[b.n%b.Window]
	for r, v := range cur {
		b.sum[r] -= v
	}
	clear(cur)
	for i := range st.Ops {
		op := &st.Ops[i]
		if op.Kind == trace.PP {
			continue
		}
		// Durations come from one clock each. The shortest belongs to the
		// member who arrived last; everyone else's extra is waiting.
		last, xfer := 0, math.Inf(1)
		for j := range op.Ranks {
			if d := op.Exit[j] - op.Enter[j]; d < xfer {
				last, xfer = j, d
			}
		}
		w := 0.0
		for j := range op.Ranks {
			w += op.Exit[j] - op.Enter[j] - xfer
		}
		cur[op.Ranks[last]] += w
	}
	total := 0.0
	for r, v := range cur {
		b.sum[r] += v
	}
	for _, v := range b.sum {
		total += v
	}
	b.n++
	b.score, b.top = 0, -1
	if b.n < b.Window || total <= 0 {
		return
	}
	for r, v := range b.sum {
		if v/total > b.score {
			b.score, b.top = v/total, r
		}
	}
}

func (b *Blame) Alarm() (locate.Alarm, bool) {
	if b.top < 0 || b.score < b.thr {
		return locate.Alarm{}, false
	}
	return locate.Alarm{Target: locate.GPU, Where: b.top, Node: b.topo.Node(b.top), Score: b.score / b.thr}, true
}

func (b *Blame) Evict() (int, bool) {
	a, ok := b.Alarm()
	return a.Node, ok
}

func (b *Blame) Scores() []float64         { return []float64{b.score} }
func (b *Blame) SetThresholds(h []float64) { b.thr = h[0] }

func (b *Blame) Reset() {
	for _, r := range b.ring {
		clear(r)
	}
	clear(b.sum)
	b.n, b.score, b.top = 0, 0, -1
}

// NVRx is modeled on the straggler detection in NVIDIA's Resiliency
// Extension. At the end of every report interval each rank gets two scores
// between 0 and 1: the median rank's compute time over its own (relative),
// and its own best interval over its current one (individual). A rank
// scoring below the threshold on either is a straggler; the package default
// is 0.7, with reports every 300 s. This is my reimplementation of that
// idea, not their code, and it times compute only.
type NVRx struct {
	Interval float64 // seconds of job time between reports
	label    string
	topo     trace.Topology
	thr      float64 // alarm when 1 - score reaches this; 0.3 is the 0.7 default
	ct       computeTracker
	samples  [][]float64
	best     []float64
	start    float64
	started  bool
	score    float64
	top      int
	tmp      []float64
}

func NewNVRx(t trace.Topology, interval float64, label string) *NVRx {
	n := t.Ranks()
	x := &NVRx{Interval: interval, label: label, topo: t, thr: 0.3, ct: newComputeTracker(n), best: make([]float64, n), tmp: make([]float64, n), top: -1}
	x.samples = make([][]float64, n)
	return x
}

func (x *NVRx) Name() string { return x.label }

func (x *NVRx) Observe(st *trace.Step) {
	if !x.ct.observe(st) {
		return
	}
	now := x.ct.lastExit[0]
	if !x.started {
		x.start, x.started = now, true
	}
	for r, c := range x.ct.comp {
		x.samples[r] = append(x.samples[r], c)
	}
	if now-x.start < x.Interval {
		return
	}
	x.start = now
	for r := range x.samples {
		x.tmp[r] = median(x.samples[r])
		x.samples[r] = x.samples[r][:0]
	}
	cur := slices.Clone(x.tmp)
	med := median(x.tmp)
	x.score, x.top = 0, -1
	for r, c := range cur {
		if x.best[r] == 0 || c < x.best[r] {
			x.best[r] = c
		}
		s := 1 - min(med/c, x.best[r]/c)
		if s > x.score {
			x.score, x.top = s, r
		}
	}
}

func (x *NVRx) Alarm() (locate.Alarm, bool) {
	if x.top < 0 || x.score < x.thr {
		return locate.Alarm{}, false
	}
	return locate.Alarm{Target: locate.GPU, Where: x.top, Node: x.topo.Node(x.top), Score: x.score / x.thr}, true
}

func (x *NVRx) Evict() (int, bool) {
	a, ok := x.Alarm()
	return a.Node, ok
}

func (x *NVRx) Scores() []float64         { return []float64{x.score} }
func (x *NVRx) SetThresholds(h []float64) { x.thr = h[0] }

func (x *NVRx) Reset() {
	x.ct.reset()
	for r := range x.samples {
		x.samples[r] = x.samples[r][:0]
	}
	clear(x.best)
	x.started, x.score, x.top = false, 0, -1
}

func median(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	slices.Sort(v)
	n := len(v)
	if n%2 == 1 {
		return v[n/2]
	}
	return (v[n/2-1] + v[n/2]) / 2
}
