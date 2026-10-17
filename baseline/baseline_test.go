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
