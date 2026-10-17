package trace

import (
	"bytes"
	"io"
	"strings"
	"testing"
)

func TestJSONLRoundTrip(t *testing.T) {
	top := Topology{D: 2, S: 2, G: 2}
	_, pp, dp := top.Members()
	in := &Step{Index: 7, Ops: []Op{
		{Kind: PP, Group: 1, Ranks: pp[1], Enter: []float64{1, 1.5}, Exit: []float64{2, 2.5}},
		{Kind: DP, Group: 3, Ranks: dp[3], Enter: []float64{3, 3.25}, Exit: []float64{4, 4.25}},
	}}
	var buf bytes.Buffer
	w, err := NewWriter(&buf, top)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Write(in); err != nil {
		t.Fatal(err)
	}
	w.Flush()
	rd, err := NewReader(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if rd.Topo != top {
		t.Fatalf("topology %+v", rd.Topo)
	}
	out, err := rd.Next()
	if err != nil {
		t.Fatal(err)
	}
	if out.Index != 7 || len(out.Ops) != 2 || out.Ops[1].Kind != DP || out.Ops[1].Exit[1] != 4.25 {
		t.Fatalf("read back %+v", out)
	}
	if _, err := rd.Next(); err != io.EOF {
		t.Fatalf("want EOF, got %v", err)
	}
}

func TestJSONLRejectsBadOps(t *testing.T) {
	for _, line := range []string{
		`{"step":0,"ops":[{"kind":"XX","group":0,"ranks":[0],"enter":[0],"exit":[1]}]}`,
		`{"step":0,"ops":[{"kind":"DP","group":99,"ranks":[0],"enter":[0],"exit":[1]}]}`,
		`{"step":0,"ops":[{"kind":"TP","group":0,"ranks":[0,1],"enter":[0],"exit":[1,1]}]}`,
		`{"step":0,"ops":[{"kind":"TP","group":0,"ranks":[42],"enter":[0],"exit":[1]}]}`,
		`{"step":0,"ops":[{"kind":"TP","group":0,"ranks":[0],"enter":[2],"exit":[1]}]}`,
	} {
		rd, err := NewReader(strings.NewReader("{\"d\":2,\"s\":2,\"g\":2}\n" + line + "\n"))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := rd.Next(); err == nil {
			t.Errorf("accepted %s", line)
		}
	}
}
