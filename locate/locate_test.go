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
