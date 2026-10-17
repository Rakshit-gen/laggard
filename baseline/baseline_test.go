package baseline_test

import (
	"testing"

	"github.com/Rakshit-gen/laggard/baseline"
	"github.com/Rakshit-gen/laggard/locate"
	"github.com/Rakshit-gen/laggard/sim"
	"github.com/Rakshit-gen/laggard/trace"
)

type observer interface {
	Observe(*trace.Step)
	Alarm() (locate.Alarm, bool)
}

func feed(steps int, faults []sim.Fault, ds ...observer) {
	w := sim.NewWorld(sim.Default(), 21, faults)
	for range steps {
		st := w.Step()
		for _, d := range ds {
			d.Observe(st)
		}
	}
}

func TestBlameFindsASlowGPU(t *testing.T) {
	top := sim.Default().Topo
	b := baseline.NewBlame(top)
	feed(100, []sim.Fault{{Kind: sim.SlowGPU, Where: 70, Factor: 1.5}}, b)
	if a, ok := b.Alarm(); !ok || a.Where != 70 {
		t.Fatalf("alarm %+v %v, scores %v", a, ok, b.Scores())
	}
}

// A slow NIC makes its whole DP group late to the next step, so blame
// lands on some member of that group, not necessarily the NIC's own.
func TestBlameSpreadsANICOverItsDPGroup(t *testing.T) {
	top := sim.Default().Topo
	nic := top.Rank(1, 2, 3)
	b := baseline.NewBlame(top)
	feed(100, []sim.Fault{{Kind: sim.SlowNIC, Where: nic, Factor: 4}}, b)
	a, ok := b.Alarm()
	if !ok {
		t.Fatalf("no alarm, scores %v", b.Scores())
	}
	if _, s, g := top.Coords(a.Where); s != 2 || g != 3 {
		t.Fatalf("blamed rank %d, outside the slow NIC's DP group", a.Where)
	}
}

// At the 0.7 default a GPU must be over 1/0.7, about 1.43x, as slow as the
// median rank before it's reported, and only at a report boundary.
func TestNVRxDefaultsNeedABigSlowdown(t *testing.T) {
	top := sim.Default().Topo
	big := baseline.NewNVRx(top, 300, "")
	small := baseline.NewNVRx(top, 300, "")
	feed(330, []sim.Fault{{Kind: sim.SlowGPU, Where: 70, Factor: 1.5}}, big)
	feed(330, []sim.Fault{{Kind: sim.SlowGPU, Where: 70, Factor: 1.3}}, small)
	if a, ok := big.Alarm(); !ok || a.Where != 70 {
		t.Fatalf("1.5x GPU: alarm %+v %v", a, ok)
	}
	if a, ok := small.Alarm(); ok {
		t.Fatalf("1.3x GPU reported at the 0.7 default: %+v", a)
	}
}

func TestNVRxWaitsForTheReport(t *testing.T) {
	x := baseline.NewNVRx(sim.Default().Topo, 300, "")
	// With this GPU a step takes about 1.36 s, so 200 steps is ~270 s.
	feed(200, []sim.Fault{{Kind: sim.SlowGPU, Where: 70, Factor: 2}}, x)
	if _, ok := x.Alarm(); ok {
		t.Fatal("alarm before the first 300 s report")
	}
}

func TestStepTimeHasNoLocation(t *testing.T) {
	s := baseline.NewStepTime()
	w := sim.NewWorld(sim.Default(), 21, []sim.Fault{{Kind: sim.SlowGPU, Where: 9, Factor: 2, Start: 150}})
	for range 160 {
		s.Observe(w.Step())
	}
	a, ok := s.Alarm()
	if !ok || a.Target != locate.Job || a.Node != -1 {
		t.Fatalf("alarm %+v %v, scores %v", a, ok, s.Scores())
	}
	if _, ok := s.Evict(); ok {
		t.Fatal("step-time alert asked for a drain without knowing where")
	}
}
