// Package svg draws the line charts in results/. It's deliberately plain:
// the charts live in a README on GitHub, which renders SVG as an image, so
// no scripts, no fonts beyond the system stack.
package svg

import (
	"fmt"
	"io"
	"math"
	"strings"
)

// Series is one line.
type Series struct {
	Name string
	X, Y []float64 // NaN in Y leaves a gap
	Dash bool
}

// Mark is a labelled vertical line, for events.
type Mark struct {
	X     float64
	Label string
}

// Chart is a line chart with a shared x axis.
type Chart struct {
	Title          string
	XLabel, YLabel string
	XMin, XMax     float64
	YMin, YMax     float64
	YPercent       bool
	Series         []Series
	Marks          []Mark
}

var palette = []string{"#1b6ca8", "#d1495b", "#2e8b57", "#e09f3e", "#6c4f9c", "#555555", "#8c564b"}

const (
	width, height = 760, 360
	left, right   = 64, 180
	top, bottom   = 40, 48
)

func esc(s string) string {
	return strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;").Replace(s)
}

// Write renders the chart.
func (c *Chart) Write(w io.Writer) error {
	pw, ph := float64(width-left-right), float64(height-top-bottom)
	sx := func(x float64) float64 { return left + (x-c.XMin)/(c.XMax-c.XMin)*pw }
	sy := func(y float64) float64 { return top + (1-(y-c.YMin)/(c.YMax-c.YMin))*ph }
	var b strings.Builder
	fmt.Fprintf(&b, `<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif" font-size="12">`+"\n", width, height, width, height)
	fmt.Fprintf(&b, `<rect width="100%%" height="100%%" fill="#ffffff"/>`+"\n")
	fmt.Fprintf(&b, `<text x="%d" y="22" font-size="14" font-weight="600" fill="#222">%s</text>`+"\n", left, esc(c.Title))

	ydec := max(0, int(-math.Floor(math.Log10((c.YMax-c.YMin)/5))))
	for i := 0; i <= 5; i++ {
		y := c.YMin + (c.YMax-c.YMin)*float64(i)/5
		label := fmt.Sprintf("%.*f", ydec, y)
		if c.YPercent {
			label = fmt.Sprintf("%.0f%%", 100*y)
		}
		fmt.Fprintf(&b, `<line x1="%d" x2="%.1f" y1="%.1f" y2="%.1f" stroke="#e5e5e5"/>`+"\n", left, left+pw, sy(y), sy(y))
		fmt.Fprintf(&b, `<text x="%d" y="%.1f" text-anchor="end" fill="#666">%s</text>`+"\n", left-6, sy(y)+4, label)
	}
	step := niceStep((c.XMax - c.XMin) / 8)
	// Enough decimals for the step and no more, so accumulated float error
	// never reaches a label.
	dec := max(0, int(-math.Floor(math.Log10(step))))
	for i := math.Ceil(c.XMin / step); i*step <= c.XMax+1e-9; i++ {
		x := i * step
		fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" text-anchor="middle" fill="#666">%.*f</text>`+"\n", sx(x), top+ph+16, dec, x)
	}
	fmt.Fprintf(&b, `<text x="%.1f" y="%d" text-anchor="middle" fill="#444">%s</text>`+"\n", left+pw/2, height-10, esc(c.XLabel))
	fmt.Fprintf(&b, `<text transform="translate(16 %.1f) rotate(-90)" text-anchor="middle" fill="#444">%s</text>`+"\n", top+ph/2, esc(c.YLabel))
	fmt.Fprintf(&b, `<rect x="%d" y="%d" width="%.0f" height="%.0f" fill="none" stroke="#bbb"/>`+"\n", left, top, pw, ph)

	for _, m := range c.Marks {
		x := sx(m.X)
		fmt.Fprintf(&b, `<line x1="%.1f" x2="%.1f" y1="%d" y2="%.1f" stroke="#999" stroke-dasharray="3 3"/>`+"\n", x, x, top, top+ph)
		fmt.Fprintf(&b, `<text x="%.1f" y="%d" fill="#777" font-size="11">%s</text>`+"\n", x+4, top+12, esc(m.Label))
	}

	for i, s := range c.Series {
		col := palette[i%len(palette)]
		dash := ""
		if s.Dash {
			dash = ` stroke-dasharray="6 4"`
		}
		var path strings.Builder
		pen := false
		for j := range s.X {
			y := s.Y[j]
			if math.IsNaN(y) || s.X[j] < c.XMin || s.X[j] > c.XMax {
				pen = false
				continue
			}
			y = math.Max(c.YMin, math.Min(c.YMax, y))
			cmd := "L"
			if !pen {
				cmd = "M"
			}
			fmt.Fprintf(&path, "%s%.1f %.1f ", cmd, sx(s.X[j]), sy(y))
			pen = true
		}
		fmt.Fprintf(&b, `<path d="%s" fill="none" stroke="%s" stroke-width="2"%s/>`+"\n", strings.TrimSpace(path.String()), col, dash)
		ly := top + 8 + 20*i
		fmt.Fprintf(&b, `<line x1="%d" x2="%d" y1="%d" y2="%d" stroke="%s" stroke-width="2"%s/>`+"\n", width-right+14, width-right+36, ly, ly, col, dash)
		fmt.Fprintf(&b, `<text x="%d" y="%d" fill="#333">%s</text>`+"\n", width-right+42, ly+4, esc(s.Name))
	}
	b.WriteString("</svg>\n")
	_, err := io.WriteString(w, b.String())
	return err
}

func niceStep(raw float64) float64 {
	if raw <= 0 {
		return 1
	}
	p := math.Pow(10, math.Floor(math.Log10(raw)))
	for _, m := range []float64{1, 2, 5, 10} {
		if raw <= m*p {
			return m * p
		}
	}
	return 10 * p
}
