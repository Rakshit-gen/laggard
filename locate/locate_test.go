package locate_test

import (
	"testing"

	"github.com/Rakshit-gen/laggard/locate"
	"github.com/Rakshit-gen/laggard/sim"
)

// watch runs a job with the given faults through a fresh Localizer and
// returns it after the last step.
func watch(t *testing.T, steps int, faults []sim.Fault, set func(*locate.Config)) *locate.Localizer {
	t.Helper()
	cfg := sim.Default()
	c := locate.DefaultConfig()
	if set != nil {
		set(&c)
	}
	l := locate.New(cfg.Topo, c)
	w := sim.NewWorld(cfg, 42, faults)
	for range steps {
		l.Observe(w.Step())
	}
	return l
}

func expect(t *testing.T, l *locate.Localizer, tg locate.Target, where, node int) {
	t.Helper()
	a, ok := l.Alarm()
	if !ok {
		t.Fatalf("no alarm, scores %v", l.Scores())
	}
	if a.Target != tg || a.Where != where || a.Node != node {
		t.Fatalf("blamed %s %d on node %d, want %s %d on node %d", a.Target, a.Where, a.Node, tg, where, node)
	}
}

func TestSlowGPU(t *testing.T) {
	top := sim.Default().Topo
	r := top.Rank(1, 2, 3)
	l := watch(t, 200, []sim.Fault{{Kind: sim.SlowGPU, Where: r, Factor: 1.1, Start: 150}}, nil)
	expect(t, l, locate.GPU, r, top.Node(r))
}

// A slow NIC makes its rank late to the next step's first TP all-reduce,
// which looks like a slow GPU to anyone counting who arrived last. It must
// come out as a NIC, and on the right node rather than the other DP
// members'.
func TestSlowNICIsNotAGPU(t *testing.T) {
	top := sim.Default().Topo
	r := top.Rank(3, 1, 6)
	l := watch(t, 200, []sim.Fault{{Kind: sim.SlowNIC, Where: r, Factor: 1.5, Start: 150}}, nil)
	expect(t, l, locate.NIC, r, top.Node(r))
	for _, a := range l.Alarms() {
		if a.Target == locate.GPU {
			t.Fatalf("also blamed GPU %d", a.Where)
		}
	}
}

// NICs on the first and last stage have one PP link instead of two.
func TestSlowNICAtPipelineEnds(t *testing.T) {
	top := sim.Default().Topo
	for _, s := range []int{0, top.S - 1} {
		r := top.Rank(0, s, 2)
		l := watch(t, 200, []sim.Fault{{Kind: sim.SlowNIC, Where: r, Factor: 2, Start: 150}}, nil)
		expect(t, l, locate.NIC, r, top.Node(r))
	}
}

// When every GPU on a host slows together, TP peers can't see it: they're
// all equally slow. The node comparison must.
func TestSlowNode(t *testing.T) {
	n := sim.Default().Topo.TPGroup(2, 3)
	l := watch(t, 220, []sim.Fault{{Kind: sim.SlowNode, Where: n, Factor: 1.1, Start: 150}}, nil)
	expect(t, l, locate.Node, n, n)
}

func TestHealthyStaysQuiet(t *testing.T) {
	l := watch(t, 1500, nil, nil)
	if a, ok := l.Alarm(); ok {
		t.Fatalf("alarm on a healthy job: %+v, scores %v", a, l.Scores())
	}
}

// Shifting one host's clock must change nothing: the localizer only ever
// subtracts two times read from the same clock.
func TestClockOffsetsDontMatter(t *testing.T) {
	cfg := sim.Default()
	top := cfg.Topo
	w := sim.NewWorld(cfg, 9, []sim.Fault{{Kind: sim.SlowNIC, Where: top.Rank(1, 1, 1), Factor: 1.3, Start: 120}})
	a := locate.New(top, locate.DefaultConfig())
	b := locate.New(top, locate.DefaultConfig())
	skewed := top.Node(top.Rank(1, 2, 0))
	for range 200 {
		st := w.Step()
		a.Observe(st)
		for i := range st.Ops {
			op := &st.Ops[i]
			for j, r := range op.Ranks {
				if top.Node(int(r)) == skewed {
					op.Enter[j] += 0.25
					op.Exit[j] += 0.25
				}
			}
		}
		b.Observe(st)
	}
	sa, sb := a.Scores(), b.Scores()
	for i := range sa {
		if d := sa[i] - sb[i]; d > 1e-6 || d < -1e-6 {
			t.Fatalf("scores differ with a skewed clock: %v vs %v", sa, sb)
		}
	}
}
