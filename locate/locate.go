// Package locate finds the component that is slowing a synchronous training
// job down: a GPU, a NIC, or a whole node. It works from per-operation
// timestamps alone (see package trace) and decides when the slowdown has
// cost enough to be worth a restart.
//
// The hard part is that a synchronous job hides its stragglers. Every
// collective ends at the same moment for every member, so every rank's step
// takes the same time. What differs is how each rank spent it. locate splits
// each rank's time into compute (the gaps between its own operations) and
// communication, and splits each operation into waiting for the last member
// to arrive and the transfer itself. A slow GPU shows up as more compute
// than its peers doing the same work. A slow NIC shows up as longer
// transfers in exactly the groups that NIC belongs to. Neither is confused
// by the lateness it causes elsewhere, which is what fools "who do the
// others wait for" heuristics.
package locate

import (
	"math"
	"slices"

	"github.com/Rakshit-gen/laggard/trace"
)

// Target is what an alarm blames.
type Target uint8

const (
	GPU  Target = iota // Where is a rank
	NIC                // Where is the rank the NIC belongs to
	Node               // Where is a node
	Job                // the job is slow, nobody in particular
)

func (t Target) String() string {
	return [...]string{"GPU", "NIC", "node", "job"}[t]
}

// Alarm is a component that is slow right now.
type Alarm struct {
	Target Target
	Where  int
	Node   int     // the node to drain to get rid of it, -1 if unknown
	Score  float64 // how far past its threshold, 1 = just crossed
	Excess float64 // seconds per step it's costing, over the last few steps
	Loss   float64 // seconds it's estimated to have cost since the alarm went up
	Age    int     // steps since the alarm went up
}

// Idea is one part of the method that can be switched off.
type Idea uint8

const (
	// PeerGroups compares a GPU with its TP peers rather than every GPU.
	PeerGroups Idea = 1 << iota
	// Intersect blames a NIC only when all of its groups are slow, rather
	// than whenever its DP group is.
	Intersect
	// Polish removes replica and stage effects before comparing nodes.
	Polish
)

// Families of evidence. Each has its own threshold.
const (
	FamGPU = iota
	FamNode
	FamNIC
	Families
)

// Config tunes a Localizer. Thresholds are in units of accumulated noise
// and are best set by running healthy jobs and taking the largest score
// seen (see sim.Calibrate).
type Config struct {
	// Warmup is how many steps each component is watched before it can
	// raise an alarm. Its normal level relative to peers is learned here.
	Warmup int
	// K is the CUSUM allowance: residuals smaller than K noise units drain
	// the statistic instead of growing it.
	K float64
	// Clip caps any one step's contribution, so one long hiccup (a GC
	// pause, a page fault storm) can't raise an alarm alone.
	Clip float64
	// MaxNative is the most, as a log ratio, a component may be slower than
	// its peers and still have that counted as its normal. It's what lets a
	// GPU that was already bad at job start get caught.
	MaxNative float64
	// Adapt is how quickly the normal level follows slow drift, per step,
	// while a component's statistic is below Quiet.
	Adapt float64
	// Quiet is the statistic below which a component counts as unsuspected.
	// Gating on exactly zero would be wrong: the statistic only touches zero
	// after below-normal steps, so the learned level would drift low.
	Quiet float64
	// Thresholds per family: GPU, node, NIC.
	Thresholds [Families]float64
	// Without switches ideas off, for measuring what each one is worth.
	Without Idea
	// Restart is what one restart costs the job, in seconds of job time.
	// Zero means drain as soon as an alarm goes up.
	Restart float64
	// Steps is how many steps the job has left to run when the Localizer
	// starts, 0 if unknown. With it, an alarm that has lasted Patience
	// steps leads to a drain when its current cost per step, over the
	// steps that remain, adds up to more than a restart. Without it, the
	// drain happens once the alarm has cost a restart's worth, the
	// break-even rule from ski rental, which never pays more than twice
	// what the best choice in hindsight would have.
	Steps int
	// Patience is how many steps an alarm must last before a drain is
	// considered. Set it past how long transient slowdowns usually last
	// on your fleet, so they're waited out.
	Patience int
}

func DefaultConfig() Config {
	return Config{
		Warmup:     100,
		K:          1,
		Clip:       4,
		MaxNative:  math.Log(1.05),
		Adapt:      0.002,
		Quiet:      5,
		Thresholds: [Families]float64{20, 20, 20},
	}
}

// family tracks one kind of entity with a CUSUM each.
type family struct {
	mu, s   []float64
	warm    [][]float64
	sigma   float64
	n       int // steps seen
	resid   []float64
	excess  []float64
	scratch []float64
}

func newFamily(n, warmup int) *family {
	f := &family{mu: make([]float64, n), s: make([]float64, n), resid: make([]float64, n), excess: make([]float64, n), scratch: make([]float64, n)}
	f.warm = make([][]float64, n)
	for i := range f.warm {
		f.warm[i] = make([]float64, 0, warmup)
	}
	return f
}

func (f *family) reset() {
	clear(f.mu)
	clear(f.s)
	for i := range f.warm {
		f.warm[i] = f.warm[i][:0]
	}
	f.sigma, f.n = 0, 0
}

// update feeds this step's residuals (already in f.resid) through each
// entity's CUSUM.
func (f *family) update(c *Config) {
	// Noise level: the median absolute residual across entities, which one
	// faulty entity can't move, smoothed over steps.
	for i, r := range f.resid {
		f.scratch[i] = math.Abs(r - f.mu[i])
	}
	step := 1.4826 * median(f.scratch)
	if f.n == 0 {
		f.sigma = step
	} else {
		f.sigma += 0.05 * (step - f.sigma)
	}
	f.sigma = max(f.sigma, 1e-4)
	f.n++

	if f.n <= c.Warmup {
		for i, r := range f.resid {
			f.warm[i] = append(f.warm[i], r)
		}
		if f.n == c.Warmup {
			for i := range f.mu {
				f.mu[i] = min(median(f.warm[i]), c.MaxNative)
			}
		}
		return
	}
	for i, r := range f.resid {
		z := max(-c.Clip, min(c.Clip, (r-f.mu[i])/f.sigma))
		f.s[i] = max(0, f.s[i]+z-c.K)
		if f.s[i] < c.Quiet {
			f.mu[i] = min(f.mu[i]+c.Adapt*(r-f.mu[i]), c.MaxNative)
		}
	}
}

func (f *family) ready(c *Config) bool { return f.n > c.Warmup }

// excessOf turns a log residual into seconds over normal, given the
// measured seconds.
func (f *family) excessOf(i int, seconds float64) float64 {
	return seconds * (1 - math.Exp(-(f.resid[i] - f.mu[i])))
}

// Localizer watches one job.
type Localizer struct {
	cfg    Config
	topo   trace.Topology
	tp     [][]int32
	gpu    *family // per rank: compute vs the rest of its TP group
	node   *family // per node: compute vs other nodes, with replica and stage effects removed
	dp     *family // per DP group: transfer time vs other DP groups at the same stage
	pp     *family // per PP link: transfer time vs other links
	nicSc  []float64
	loss   [Families][]float64 // accumulated excess per alarmed entity
	rate   [Families][]float64 // smoothed excess per step
	age    [Families][]int
	steps  int // observed since the last reset
	alarms []Alarm

	lastExit []float64
	seen     []bool
	comp     []float64
	dpX      []float64
	ppX      []float64
	nodeLog  []float64
	tmp      []float64
}

func New(topo trace.Topology, cfg Config) *Localizer {
	tp, _, _ := topo.Members()
	n := topo.Ranks()
	l := &Localizer{
		cfg: cfg, topo: topo, tp: tp,
		gpu:   newFamily(n, cfg.Warmup),
		node:  newFamily(topo.Nodes(), cfg.Warmup),
		dp:    newFamily(topo.DPGroups(), cfg.Warmup),
		pp:    newFamily(topo.PPLinks(), cfg.Warmup),
		nicSc: make([]float64, n), lastExit: make([]float64, n), seen: make([]bool, n),
		comp: make([]float64, n), dpX: make([]float64, topo.DPGroups()), ppX: make([]float64, topo.PPLinks()),
		nodeLog: make([]float64, topo.Nodes()), tmp: make([]float64, max(topo.G, topo.D, topo.S, topo.PPLinks())),
	}
	for i, m := range [Families]int{n, topo.Nodes(), n} {
		l.loss[i], l.rate[i], l.age[i] = make([]float64, m), make([]float64, m), make([]int, m)
	}
	return l
}

// SetThresholds replaces the per-family thresholds.
func (l *Localizer) SetThresholds(h []float64) { copy(l.cfg.Thresholds[:], h) }

// Reset forgets everything, as after a restart onto different hardware.
func (l *Localizer) Reset() {
	for _, f := range []*family{l.gpu, l.node, l.dp, l.pp} {
		f.reset()
	}
	for i := range l.loss {
		clear(l.loss[i])
		clear(l.rate[i])
		clear(l.age[i])
	}
	clear(l.seen)
	clear(l.nicSc)
	l.alarms = l.alarms[:0]
}

// Observe takes one step's trace.
func (l *Localizer) Observe(st *trace.Step) {
	t := l.topo
	l.steps++
	clear(l.comp)
	clear(l.ppX)
	first := !l.seen[0]
	for i := range st.Ops {
		op := &st.Ops[i]
		xfer := math.Inf(1)
		for j, r := range op.Ranks {
			if l.seen[r] {
				l.comp[r] += op.Enter[j] - l.lastExit[r]
			}
			l.lastExit[r], l.seen[r] = op.Exit[j], true
			// The member who arrived last didn't wait, so its time inside
			// the operation is the transfer alone. Both ends of that
			// subtraction come from one clock.
			xfer = min(xfer, op.Exit[j]-op.Enter[j])
		}
		switch op.Kind {
		case trace.DP:
			l.dpX[op.Group] = xfer
		case trace.PP:
			l.ppX[op.Group] += xfer
		}
	}
	if first {
		// Compute before the first operation of the first step can't be
		// told apart from whatever ran before the trace started.
		return
	}

	// GPU: each rank's compute against its TP peers. They run the same
	// layers on the same data, so input-dependent variation cancels.
	for n, members := range l.tp {
		v := l.tmp[:len(members)]
		for j, r := range members {
			v[j] = math.Log(l.comp[r])
		}
		m := median(v)
		l.nodeLog[n] = m
		for _, r := range members {
			l.gpu.resid[r] = math.Log(l.comp[r]) - m
		}
	}
	if l.cfg.Without&PeerGroups != 0 {
		for r := range l.gpu.resid {
			l.gpu.resid[r] = math.Log(l.comp[r])
		}
		m := median(slices.Clone(l.gpu.resid))
		for r := range l.gpu.resid {
			l.gpu.resid[r] -= m
		}
	}
	l.gpu.update(&l.cfg)

	// Node: the median rank's compute per node, which one slow GPU doesn't
	// move. Each replica sees different data and each stage holds different
	// layers, so remove a replica effect and a stage effect first (one pass
	// of median polish) and look at what's left.
	copy(l.node.resid, l.nodeLog)
	if l.cfg.Without&Polish != 0 {
		m := median(slices.Clone(l.nodeLog))
		for n := range l.node.resid {
			l.node.resid[n] -= m
		}
	} else {
		l.polish()
	}
	l.node.update(&l.cfg)

	// DP groups: transfer time against the other groups at the same stage,
	// which move the same number of bytes.
	for s := range t.S {
		v := l.tmp[:t.G]
		for g := range t.G {
			v[g] = math.Log(l.dpX[t.DPGroup(s, g)])
		}
		m := median(v)
		for g := range t.G {
			i := t.DPGroup(s, g)
			l.dp.resid[i] = math.Log(l.dpX[i]) - m
		}
	}
	l.dp.update(&l.cfg)

	if t.S > 1 {
		v := l.tmp[:len(l.ppX)]
		for i, x := range l.ppX {
			v[i] = math.Log(x)
		}
		m := median(v)
		for i, x := range l.ppX {
			l.pp.resid[i] = math.Log(x) - m
		}
		l.pp.update(&l.cfg)
	}
	l.judge()
}

// polish removes a replica effect and a stage effect from the node
// residuals, one pass of median polish.
func (l *Localizer) polish() {
	t := l.topo
	for d := range t.D {
		row := l.node.resid[d*t.S : (d+1)*t.S]
		m := median(append(l.tmp[:0], row...))
		for s := range row {
			row[s] -= m
		}
	}
	for s := range t.S {
		v := l.tmp[:t.D]
		for d := range t.D {
			v[d] = l.node.resid[d*t.S+s]
		}
		m := median(v)
		for d := range t.D {
			l.node.resid[d*t.S+s] -= m
		}
	}
}

// judge turns CUSUM states into alarms and accumulates their cost.
func (l *Localizer) judge() {
	l.alarms = l.alarms[:0]
	if !l.gpu.ready(&l.cfg) {
		return
	}
	t, h := l.topo, l.cfg.Thresholds
	gpuSec := func(r int) float64 { return l.gpu.excessOf(r, l.comp[r]) }
	nodeSec := func(n int) float64 { return l.node.excessOf(n, math.Exp(l.nodeLog[n])) }
	nicSec := func(r int) float64 {
		d, s, g := t.Coords(r)
		x := l.dp.excessOf(t.DPGroup(s, g), l.dpX[t.DPGroup(s, g)])
		if s > 0 {
			i := t.PPLink(d, s-1, g)
			x += l.pp.excessOf(i, l.ppX[i])
		}
		if s < t.S-1 {
			i := t.PPLink(d, s, g)
			x += l.pp.excessOf(i, l.ppX[i])
		}
		return x
	}

	// A NIC is suspect only when every group it's part of is slow. A DP
	// group spans D NICs and a PP link two, and only the bad NIC sits in
	// all of its own groups.
	for r := range l.nicSc {
		d, s, g := t.Coords(r)
		sc := l.dp.s[t.DPGroup(s, g)]
		if l.cfg.Without&Intersect != 0 {
			l.nicSc[r] = sc
			continue
		}
		if s > 0 {
			sc = min(sc, l.pp.s[t.PPLink(d, s-1, g)])
		}
		if s < t.S-1 {
			sc = min(sc, l.pp.s[t.PPLink(d, s, g)])
		}
		l.nicSc[r] = sc
	}

	check := func(fam int, tg Target, scores []float64, node func(int) int, sec func(int) float64) {
		for i, sc := range scores {
			if sc < h[fam] {
				if sc == 0 {
					l.loss[fam][i], l.rate[fam][i], l.age[fam][i] = 0, 0, 0
				}
				continue
			}
			x := max(0, sec(i))
			l.loss[fam][i] += x
			l.age[fam][i]++
			// The cost per step follows only the last few steps. The
			// statistic takes a while to drain after a slowdown ends, and
			// the decision must see the slowdown gone before the alarm is.
			l.rate[fam][i] += (x - l.rate[fam][i]) / float64(min(l.age[fam][i], 5))
			l.alarms = append(l.alarms, Alarm{Target: tg, Where: i, Node: node(i), Score: sc / h[fam], Excess: l.rate[fam][i], Loss: l.loss[fam][i], Age: l.age[fam][i]})
		}
	}
	check(FamGPU, GPU, l.gpu.s, t.Node, gpuSec)
	check(FamNode, Node, l.node.s, func(n int) int { return n }, nodeSec)
	check(FamNIC, NIC, l.nicSc, t.Node, nicSec)
	slices.SortStableFunc(l.alarms, func(a, b Alarm) int {
		if a.Score > b.Score {
			return -1
		}
		if a.Score < b.Score {
			return 1
		}
		return 0
	})
}

// Alarms returns the components currently above threshold, most confident
// first. The slice is reused by the next Observe.
func (l *Localizer) Alarms() []Alarm { return l.alarms }

// Alarm returns the most confident current alarm.
func (l *Localizer) Alarm() (Alarm, bool) {
	if len(l.alarms) == 0 {
		return Alarm{}, false
	}
	return l.alarms[0], true
}

// Evict says which node to drain, if any alarm has now cost the job more
// than a restart would.
func (l *Localizer) Evict() (int, bool) {
	for _, a := range l.alarms {
		c := &l.cfg
		if c.Steps == 0 {
			if a.Loss >= c.Restart {
				return a.Node, true
			}
			continue
		}
		left := c.Steps - l.steps
		if a.Age >= c.Patience && a.Excess*float64(left) > c.Restart {
			return a.Node, true
		}
	}
	return -1, false
}

// Scores returns the highest statistic per family right now, for
// calibration.
func (l *Localizer) Scores() []float64 {
	if !l.gpu.ready(&l.cfg) {
		return []float64{0, 0, 0}
	}
	return []float64{slices.Max(l.gpu.s), slices.Max(l.node.s), slices.Max(l.nicSc)}
}

// median reorders v.
func median(v []float64) float64 {
	n := len(v)
	if n == 0 {
		return 0
	}
	slices.Sort(v)
	if n%2 == 1 {
		return v[n/2]
	}
	return (v[n/2-1] + v[n/2]) / 2
}
