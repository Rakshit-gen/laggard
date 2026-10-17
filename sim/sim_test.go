package sim

import (
	"testing"

	"github.com/Rakshit-gen/laggard/locate"
)

func run(seed uint64, steps int, faults []Fault) *World {
	w := NewWorld(Default(), seed, faults)
	for range steps {
		w.Step()
	}
	return w
}

func TestDeterministic(t *testing.T) {
	a, b := run(3, 50, nil).Now(), run(3, 50, nil).Now()
	if a != b {
		t.Fatalf("same seed, different times: %v %v", a, b)
	}
	if c := run(4, 50, nil).Now(); c == a {
		t.Fatalf("different seeds, same time %v", c)
	}
}

// Faults must not change the random draws, or comparisons between
// policies on one seed would compare different noise.
func TestFaultsDontShiftTheNoise(t *testing.T) {
	f := []Fault{{Kind: SlowGPU, Where: 5, Factor: 2, Start: 40}}
	if a, b := run(3, 40, nil).Now(), run(3, 40, f).Now(); a != b {
		t.Fatalf("before the fault starts, times differ: %v %v", a, b)
	}
	if a, b := run(3, 60, nil).Now(), run(3, 60, f).Now(); b <= a {
		t.Fatalf("a 2x GPU didn't slow the job: %v vs %v", b, a)
	}
}

// A healthy step is about 0.96 s. Noise already puts some path on the
// critical path every step, so a GPU 10% slow mostly hides behind it,
// while one 50% slow sets the pace.
func TestSmallSlowdownsHideBehindNoise(t *testing.T) {
	healthy := run(8, 300, nil).Now() / 300
	if healthy < 0.9 || healthy > 1.0 {
		t.Fatalf("healthy step %.3f s", healthy)
	}
	small := run(8, 300, []Fault{{Kind: SlowGPU, Where: 70, Factor: 1.1}}).Now() / 300 / healthy
	big := run(8, 300, []Fault{{Kind: SlowGPU, Where: 70, Factor: 1.5}}).Now() / 300 / healthy
	if small > 1.01 || big < 1.12 || big > 1.22 {
		t.Fatalf("1.1x GPU made the job %.3fx, 1.5x made it %.3fx", small, big)
	}
}

func TestEvictClearsTheFaultAndCosts(t *testing.T) {
	cfg := Default()
	f := Fault{Kind: SlowNIC, Where: 77, Factor: 4}
	w := NewWorld(cfg, 2, []Fault{f})
	for range 20 {
		w.Step()
	}
	before := w.Now()
	w.Evict(f.Node(cfg.Topo), 60)
	if got := w.Now() - before; got != 60 {
		t.Fatalf("restart cost %v s, want 60", got)
	}
	ref := NewWorld(cfg, 2, nil)
	for range 20 {
		ref.Step()
	}
	t0, r0 := w.Now(), ref.Now()
	for range 20 {
		w.Step()
		ref.Step()
	}
	if a, b := w.Now()-t0, ref.Now()-r0; a > b*1.001 {
		t.Fatalf("after the drain 20 steps took %.2f s, healthy %.2f s", a, b)
	}
}

func TestJudge(t *testing.T) {
	cfg := Default()
	f := Fault{Kind: SlowNIC, Where: 77, Factor: 2, Start: 100}
	ep := Episode{Faults: []Fault{f}}
	node := f.Node(cfg.Topo)
	on := func(step, n int, tg locate.Target) Onset { return Onset{step, locate.Alarm{Target: tg, Node: n}} }
	v := Judge(cfg, ep, Outcome{Onsets: []Onset{on(40, 3, locate.GPU), on(103, node, locate.NIC)}})
	if v.FalseBefore != 1 || !v.Caught || v.Delay != 3 || !v.KindRight {
		t.Fatalf("%+v", v)
	}
	v = Judge(cfg, ep, Outcome{Onsets: []Onset{on(101, 0, locate.GPU), on(103, node, locate.NIC)}})
	if v.Caught || !v.WrongFirst {
		t.Fatalf("first blame on the wrong node should count against it: %+v", v)
	}
	tr := Episode{Faults: []Fault{{Kind: SlowGPU, Where: 0, Factor: 2, Start: 100, End: 110}}}
	v = Judge(cfg, tr, Outcome{Onsets: []Onset{on(200, 0, locate.GPU)}})
	if v.Caught {
		t.Fatalf("an alarm long after a transient ended counted as catching it: %+v", v)
	}
}
