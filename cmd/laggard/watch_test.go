package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/Rakshit-gen/laggard/locate"
	"github.com/Rakshit-gen/laggard/sim"
	"github.com/Rakshit-gen/laggard/trace"
)

func TestWatchNamesTheSlowNIC(t *testing.T) {
	cfg := sim.Default()
	f, err := parseFault("nic:77:1.5:150", cfg.Topo)
	if err != nil {
		t.Fatal(err)
	}
	var in bytes.Buffer
	tw, _ := trace.NewWriter(&in, cfg.Topo)
	w := sim.NewWorld(cfg, 3, []sim.Fault{f})
	for range 260 {
		tw.Write(w.Step())
	}
	tw.Flush()

	c := locate.DefaultConfig()
	c.Restart, c.Steps, c.Patience = 120, 3000, 50
	var out bytes.Buffer
	if err := watch(&in, &out, c); err != nil {
		t.Fatal(err)
	}
	got := out.String()
	for _, want := range []string{"NIC 77 on node 9", "drain node 9"} {
		if !strings.Contains(got, want) {
			t.Errorf("output lacks %q:\n%s", want, got)
		}
	}
}

func TestParseFaultRejects(t *testing.T) {
	top := sim.Default().Topo
	for _, s := range []string{"gpu:1:1.5", "disk:1:1.5:0", "gpu:128:1.5:0", "node:16:1.5:0", "gpu:1:0.5:0", "gpu:x:1.5:0"} {
		if _, err := parseFault(s, top); err == nil {
			t.Errorf("accepted %q", s)
		}
	}
}
