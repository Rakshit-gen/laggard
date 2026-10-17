// Package sim runs a synthetic 3D-parallel training job, injects slow
// components into it, and scores detectors on what they make of the trace.
//
// The job is GPipe-style: every microbatch goes forward through the
// pipeline, then every microbatch comes back, then each DP group all-reduces
// its gradients. Inside a stage the TP group all-reduces after every chunk
// of compute. Point-to-point transfers block both ends. That's enough
// structure for lateness to spread the way it does in a real job: a slow
// GPU makes its TP peers wait, then the next stage, then the DP group.
package sim

import (
	"math"
	"math/rand/v2"

	"github.com/Rakshit-gen/laggard/trace"
)

// Config describes the job and its normal noise.
type Config struct {
	Topo   trace.Topology
	Micro  int // microbatches per step
	Chunks int // compute chunks (and TP all-reduces) per microbatch per direction

	Fwd, Bwd float64 // seconds of compute per chunk
	Optim    float64 // seconds of optimizer step per rank

	TPXfer, PPXfer, DPXfer float64 // seconds per transfer when healthy

	Jitter     float64 // sigma of the log of each compute chunk
	DataJitter float64 // sigma of the log of a replica's work in one step (sequence lengths vary)
	XferJitter float64 // sigma of the log of each transfer

	HiccupP    float64 // chance per rank per step of one long stall (GC, page faults, a noisy neighbour)
	HiccupMean float64 // seconds, exponentially distributed

	CongestP float64 // chance per DP group per step that cross traffic slows it 1.5 to 3x

	CheckpointEvery int     // steps
	CheckpointPause float64 // seconds, every rank

	Silicon float64 // sigma of the log of each GPU's built-in speed

	ClockOffset float64 // max seconds a host clock is off from true time
	ClockDrift  float64 // max fractional drift per host clock
}

// Default is a 128-GPU job: 4 replicas of a 4-stage pipeline with 8-way
// tensor parallelism. A step takes about 0.85 s.
func Default() Config {
	return Config{
		Topo: trace.Topology{D: 4, S: 4, G: 8}, Micro: 8, Chunks: 2,
		Fwd: 0.010, Bwd: 0.020, Optim: 0.015,
		TPXfer: 0.0003, PPXfer: 0.0015, DPXfer: 0.120,
		Jitter: 0.02, DataJitter: 0.03, XferJitter: 0.05,
		HiccupP: 0.003, HiccupMean: 0.060,
		CongestP:        0.01,
		CheckpointEvery: 500, CheckpointPause: 3,
		Silicon:     0.01,
		ClockOffset: 0.005, ClockDrift: 20e-6,
	}
}

// FaultKind is what's broken.
type FaultKind uint8

const (
	SlowGPU  FaultKind = iota // throttling, a failing HBM stack, a bad power cap
	SlowNIC                   // a link trained down, a flapping port, errors and retransmits
	SlowNode                  // every GPU on the host, e.g. a host-side power or cooling problem
)

func (k FaultKind) String() string { return [...]string{"GPU", "NIC", "node"}[k] }

// Fault slows one component by Factor from step Start. End is the first
// step it's gone by itself; 0 means it stays until the node is drained.
type Fault struct {
	Kind   FaultKind
	Where  int // rank for SlowGPU and SlowNIC, node for SlowNode
	Factor float64
	Start  int
	End    int
}

func (f Fault) Transient() bool { return f.End > 0 }

func (f Fault) Node(t trace.Topology) int {
	if f.Kind == SlowNode {
		return f.Where
	}
	return t.Node(f.Where)
}

// World is one running job.
type World struct {
	cfg        Config
	rng        *rand.Rand
	tp, pp, dp [][]int32

	clock   []float64 // true time per rank
	offset  []float64 // per node
	drift   []float64
	silicon []float64

	faults  []Fault
	cleared []bool
	step    int

	gpuF, nicF []float64
	data       []float64
	hicAt      []int
	hicAmt     []float64
	chunk      []int

	ops   []trace.Op
	times []float64
	nt    int
	out   trace.Step
}

// NewWorld builds a job. The seed fixes every random draw, and draws don't
// depend on faults or evictions, so two runs of one seed see the same
// noise whatever happens in them.
func NewWorld(cfg Config, seed uint64, faults []Fault) *World {
	t := cfg.Topo
	n := t.Ranks()
	w := &World{
		cfg: cfg, rng: rand.New(rand.NewPCG(seed, 0x6c6167)),
		clock: make([]float64, n), offset: make([]float64, t.Nodes()), drift: make([]float64, t.Nodes()),
		silicon: make([]float64, n), faults: faults, cleared: make([]bool, len(faults)),
		gpuF: make([]float64, n), nicF: make([]float64, n), data: make([]float64, t.D),
		hicAt: make([]int, n), hicAmt: make([]float64, n), chunk: make([]int, n),
	}
	w.tp, w.pp, w.dp = t.Members()
	for i := range w.offset {
		w.offset[i] = cfg.ClockOffset * (2*w.rng.Float64() - 1)
		w.drift[i] = cfg.ClockDrift * (2*w.rng.Float64() - 1)
	}
	for r := range w.silicon {
		w.silicon[r] = math.Exp(cfg.Silicon * w.rng.NormFloat64())
	}
	perStep := t.Nodes()*cfg.Micro*2*cfg.Chunks*t.G + len(w.pp)*cfg.Micro*2*2 + len(w.dp)*t.D
	w.times = make([]float64, 2*perStep)
	w.ops = make([]trace.Op, 0, t.Nodes()*cfg.Micro*2*cfg.Chunks+len(w.pp)*cfg.Micro*2+len(w.dp))
	return w
}

// local reads true time t on the clock of the host that runs rank r.
func (w *World) local(r int32, t float64) float64 {
	n := int(r) / w.cfg.Topo.G
	return t + w.offset[n] + w.drift[n]*t
}

func (w *World) record(k trace.Kind, group int, ranks []int32, enter []float64, exit float64) {
	m := len(ranks)
	en, ex := w.times[w.nt:w.nt+m:w.nt+m], w.times[w.nt+m:w.nt+2*m:w.nt+2*m]
	w.nt += 2 * m
	for j, r := range ranks {
		en[j] = w.local(r, enter[j])
		ex[j] = w.local(r, exit)
	}
	w.ops = append(w.ops, trace.Op{Kind: k, Group: int32(group), Ranks: ranks, Enter: en, Exit: ex})
}

func (w *World) active(i int) bool {
	f := w.faults[i]
	return !w.cleared[i] && w.step >= f.Start && (f.End == 0 || w.step < f.End)
}

// Step runs one training step and returns its trace, which stays valid
// until the next call.
func (w *World) Step() *trace.Step {
	c, t := &w.cfg, w.cfg.Topo
	w.ops, w.nt = w.ops[:0], 0

	for r := range w.gpuF {
		w.gpuF[r], w.nicF[r] = 1, 1
	}
	for i, f := range w.faults {
		if !w.active(i) {
			continue
		}
		switch f.Kind {
		case SlowGPU:
			w.gpuF[f.Where] *= f.Factor
		case SlowNIC:
			w.nicF[f.Where] *= f.Factor
		case SlowNode:
			for g := range t.G {
				w.gpuF[f.Where*t.G+g] *= f.Factor
			}
		}
	}
	for d := range w.data {
		w.data[d] = math.Exp(c.DataJitter * w.rng.NormFloat64())
	}
	perRank := c.Micro * 2 * c.Chunks
	for r := range w.hicAt {
		w.hicAt[r] = -1
		if w.rng.Float64() < c.HiccupP {
			w.hicAt[r] = w.rng.IntN(perRank)
			w.hicAmt[r] = c.HiccupMean * w.rng.ExpFloat64()
		}
		w.chunk[r] = 0
	}

	enter := make([]float64, max(t.G, t.D, 2))
	block := func(d, s int, base float64) {
		members := w.tp[t.TPGroup(d, s)]
		for range c.Chunks {
			last := 0.0
			for j, r := range members {
				dur := base * w.silicon[r] * w.gpuF[r] * w.data[d] * math.Exp(c.Jitter*w.rng.NormFloat64())
				if w.chunk[r] == w.hicAt[r] {
					dur += w.hicAmt[r]
				}
				w.chunk[r]++
				enter[j] = w.clock[r] + dur
				last = max(last, enter[j])
			}
			exit := last + c.TPXfer*math.Exp(c.XferJitter*w.rng.NormFloat64())
			for _, r := range members {
				w.clock[r] = exit
			}
			w.record(trace.TP, t.TPGroup(d, s), members, enter[:len(members)], exit)
		}
	}
	p2p := func(link int) {
		m := w.pp[link]
		a, b := m[0], m[1]
		enter[0], enter[1] = w.clock[a], w.clock[b]
		exit := max(enter[0], enter[1]) + c.PPXfer*max(w.nicF[a], w.nicF[b])*math.Exp(2*c.XferJitter*w.rng.NormFloat64())
		w.clock[a], w.clock[b] = exit, exit
		w.record(trace.PP, link, m, enter[:2], exit)
	}

	for range c.Micro {
		for s := range t.S {
			for d := range t.D {
				if s > 0 {
					for g := range t.G {
						p2p(t.PPLink(d, s-1, g))
					}
				}
				block(d, s, c.Fwd)
			}
		}
	}
	for range c.Micro {
		for s := t.S - 1; s >= 0; s-- {
			for d := range t.D {
				if s < t.S-1 {
					for g := range t.G {
						p2p(t.PPLink(d, s, g))
					}
				}
				block(d, s, c.Bwd)
			}
		}
	}
	for i, m := range w.dp {
		last, worst := 0.0, 1.0
		for j, r := range m {
			enter[j] = w.clock[r]
			last = max(last, enter[j])
			worst = max(worst, w.nicF[r])
		}
		x := c.DPXfer * worst * math.Exp(c.XferJitter*w.rng.NormFloat64())
		if w.rng.Float64() < c.CongestP {
			x *= 1.5 + 1.5*w.rng.Float64()
		}
		for _, r := range m {
			w.clock[r] = last + x
		}
		w.record(trace.DP, i, m, enter[:len(m)], last+x)
	}
	ckpt := c.CheckpointEvery > 0 && (w.step+1)%c.CheckpointEvery == 0
	for r := range w.clock {
		w.clock[r] += c.Optim * w.silicon[r] * w.gpuF[r] * math.Exp(c.Jitter*w.rng.NormFloat64())
		if ckpt {
			w.clock[r] += c.CheckpointPause
		}
	}

	w.out = trace.Step{Index: w.step, Ops: w.ops}
	w.step++
	return &w.out
}

// Now is the true time at which the slowest rank has finished.
func (w *World) Now() float64 {
	m := 0.0
	for _, x := range w.clock {
		m = max(m, x)
	}
	return m
}

// Evict drains a node and restarts the job from its last checkpoint on a
// healthy spare. Faults on that node are gone. Every rank resumes after
// cost seconds, which covers the restart and the work redone since the
// checkpoint.
func (w *World) Evict(node int, cost float64) {
	for i, f := range w.faults {
		if f.Node(w.cfg.Topo) == node {
			w.cleared[i] = true
		}
	}
	at := w.Now() + cost
	for r := range w.clock {
		w.clock[r] = at
	}
}
