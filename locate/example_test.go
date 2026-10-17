package locate_test

import (
	"fmt"

	"github.com/Rakshit-gen/laggard/locate"
	"github.com/Rakshit-gen/laggard/sim"
)

// A 128-GPU job whose GPU 70 starts throttling at step 300.
func Example() {
	cfg := sim.Default()
	w := sim.NewWorld(cfg, 1, []sim.Fault{{Kind: sim.SlowGPU, Where: 70, Factor: 1.2, Start: 300}})

	c := locate.DefaultConfig()
	c.Restart = 120 // seconds a restart costs the job
	c.Steps = 5000  // steps left in the job
	c.Patience = 50 // wait out anything shorter
	l := locate.New(cfg.Topo, c)

	alarmed := false
	for range 1000 {
		l.Observe(w.Step())
		if a, ok := l.Alarm(); ok && !alarmed {
			alarmed = true
			fmt.Printf("%s %d on node %d\n", a.Target, a.Where, a.Node)
		}
		if n, ok := l.Evict(); ok {
			fmt.Printf("drain node %d\n", n)
			break
		}
	}
	// Output:
	// GPU 70 on node 8
	// drain node 8
}
