package locate_test

import (
	"testing"

	"github.com/Rakshit-gen/laggard/locate"
	"github.com/Rakshit-gen/laggard/sim"
	"github.com/Rakshit-gen/laggard/trace"
)

// steps returns n consecutive steps from a 128-GPU job, copied so they
// outlive the simulator's buffers.
func steps(n int, faults []sim.Fault) []trace.Step {
	w := sim.NewWorld(sim.Default(), 1, faults)
	out := make([]trace.Step, n)
	for i := range out {
		st := w.Step()
		ops := make([]trace.Op, len(st.Ops))
		for j, op := range st.Ops {
			op.Enter = append([]float64(nil), op.Enter...)
			op.Exit = append([]float64(nil), op.Exit...)
			ops[j] = op
		}
		out[i] = trace.Step{Index: st.Index, Ops: ops}
	}
	return out
}

// benchObserve times Observe on fresh steps. Replaying old ones would
// run time backwards, so every 200 calls it starts a new warmed-up
// localizer, off the clock.
func benchObserve(b *testing.B, faults []sim.Fault) {
	sts := steps(400, faults)
	var l *locate.Localizer
	for i := range b.N {
		if i%200 == 0 {
			b.StopTimer()
			l = locate.New(sim.Default().Topo, locate.DefaultConfig())
			for j := range 200 {
				l.Observe(&sts[j])
			}
			b.StartTimer()
		}
		l.Observe(&sts[200+i%200])
	}
}

func BenchmarkObserveHealthy(b *testing.B) { benchObserve(b, nil) }

// With an alarm up, each step also replays itself once per suspect.
func BenchmarkObserveAlarmed(b *testing.B) {
	benchObserve(b, []sim.Fault{{Kind: sim.SlowNIC, Where: 77, Factor: 2, Start: 150}})
}
