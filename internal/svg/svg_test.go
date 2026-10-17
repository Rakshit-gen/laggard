package svg

import (
	"math"
	"strings"
	"testing"
)

func TestGapsBreakTheLine(t *testing.T) {
	c := &Chart{Title: "a < b", XMax: 4, YMax: 1, Series: []Series{{
		Name: "s", X: []float64{0, 1, 2, 3, 4}, Y: []float64{0.1, 0.2, math.NaN(), 0.4, 0.5},
	}}}
	var b strings.Builder
	if err := c.Write(&b); err != nil {
		t.Fatal(err)
	}
	out := b.String()
	// One path, two pen-downs: before and after the NaN.
	i := strings.Index(out, `<path d="`)
	if i < 0 {
		t.Fatal("no path written")
	}
	d := out[i+len(`<path d="`):]
	d = d[:strings.IndexByte(d, '"')]
	if got := strings.Count(d, "M"); got != 2 {
		t.Fatalf("path %q has %d moves, want 2", d, got)
	}
	if strings.Contains(out, "a < b") || !strings.Contains(out, "a &lt; b") {
		t.Fatal("title not escaped")
	}
}

func TestValuesAreClampedToTheAxis(t *testing.T) {
	c := &Chart{XMax: 1, YMax: 1, Series: []Series{{Name: "s", X: []float64{0, 1}, Y: []float64{-5, 5}}}}
	var b strings.Builder
	c.Write(&b)
	// The plot runs from y=top to y=height-bottom.
	if !strings.Contains(b.String(), "M64.0 312.0 L580.0 40.0") {
		t.Fatalf("out-of-range points not clamped:\n%s", b.String())
	}
}

func TestNiceStep(t *testing.T) {
	for raw, want := range map[float64]float64{0: 1, 0.3: 0.5, 7: 10, 45: 50, 120: 200} {
		if got := niceStep(raw); got != want {
			t.Errorf("niceStep(%g) = %g, want %g", raw, got, want)
		}
	}
}

func TestFractionalAxisLabels(t *testing.T) {
	c := Chart{XMin: 1, XMax: 1.5, YMin: 0, YMax: 1, Series: []Series{{Name: "a", X: []float64{1, 1.5}, Y: []float64{0, 1}}}}
	var b strings.Builder
	if err := c.Write(&b); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(b.String(), "0000000") {
		t.Fatal("float error leaked into an axis label")
	}
	for _, want := range []string{">1.1<", ">1.2<", ">1.5<"} {
		if !strings.Contains(b.String(), want) {
			t.Errorf("missing label %s", want)
		}
	}
}
