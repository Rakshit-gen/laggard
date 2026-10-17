package trace

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// The file format is JSON lines. The first line is the topology:
//
//	{"d":4,"s":4,"g":8}
//
// Every line after it is one step:
//
//	{"step":12,"ops":[{"kind":"TP","group":3,"ranks":[24,25,...],"enter":[...],"exit":[...]}, ...]}
//
// Times are seconds on each member's own clock. Ops must be listed so that
// each rank's ops appear in the order that rank ran them.

type jsonOp struct {
	Kind  string    `json:"kind"`
	Group int32     `json:"group"`
	Ranks []int32   `json:"ranks"`
	Enter []float64 `json:"enter"`
	Exit  []float64 `json:"exit"`
}

type jsonStep struct {
	Step int      `json:"step"`
	Ops  []jsonOp `json:"ops"`
}

type jsonTopo struct {
	D int `json:"d"`
	S int `json:"s"`
	G int `json:"g"`
}

var kinds = map[string]Kind{"TP": TP, "PP": PP, "DP": DP}

// Reader reads a trace file.
type Reader struct {
	Topo Topology
	sc   *bufio.Scanner
	line int
}

// NewReader reads the topology line.
func NewReader(r io.Reader) (*Reader, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<30)
	rd := &Reader{sc: sc}
	if !sc.Scan() {
		if err := sc.Err(); err != nil {
			return nil, err
		}
		return nil, errors.New("trace: empty input")
	}
	rd.line = 1
	var t jsonTopo
	if err := json.Unmarshal(sc.Bytes(), &t); err != nil {
		return nil, fmt.Errorf("trace: line 1: %w", err)
	}
	if t.D < 1 || t.S < 1 || t.G < 1 {
		return nil, fmt.Errorf("trace: line 1: bad topology %+v", t)
	}
	rd.Topo = Topology{D: t.D, S: t.S, G: t.G}
	return rd, nil
}

// Next returns the next step, or io.EOF.
func (rd *Reader) Next() (*Step, error) {
	if !rd.sc.Scan() {
		if err := rd.sc.Err(); err != nil {
			return nil, err
		}
		return nil, io.EOF
	}
	rd.line++
	var js jsonStep
	if err := json.Unmarshal(rd.sc.Bytes(), &js); err != nil {
		return nil, fmt.Errorf("trace: line %d: %w", rd.line, err)
	}
	st := &Step{Index: js.Step, Ops: make([]Op, len(js.Ops))}
	for i, o := range js.Ops {
		op, err := rd.check(o)
		if err != nil {
			return nil, fmt.Errorf("trace: line %d, op %d: %w", rd.line, i, err)
		}
		st.Ops[i] = op
	}
	return st, nil
}

// check rejects anything the localizer would index out of range with.
func (rd *Reader) check(o jsonOp) (Op, error) {
	t := rd.Topo
	k, ok := kinds[o.Kind]
	if !ok {
		return Op{}, fmt.Errorf("unknown kind %q", o.Kind)
	}
	groups := map[Kind]int{TP: t.Nodes(), PP: t.PPLinks(), DP: t.DPGroups()}[k]
	if o.Group < 0 || int(o.Group) >= groups {
		return Op{}, fmt.Errorf("%s group %d out of range", o.Kind, o.Group)
	}
	if len(o.Ranks) == 0 || len(o.Enter) != len(o.Ranks) || len(o.Exit) != len(o.Ranks) {
		return Op{}, fmt.Errorf("%d ranks, %d enter times, %d exit times", len(o.Ranks), len(o.Enter), len(o.Exit))
	}
	for j, r := range o.Ranks {
		if r < 0 || int(r) >= t.Ranks() {
			return Op{}, fmt.Errorf("rank %d out of range", r)
		}
		if o.Exit[j] < o.Enter[j] {
			return Op{}, fmt.Errorf("rank %d exits before it enters", r)
		}
	}
	return Op{Kind: k, Group: o.Group, Ranks: o.Ranks, Enter: o.Enter, Exit: o.Exit}, nil
}

// Writer writes a trace file.
type Writer struct {
	w   *bufio.Writer
	enc *json.Encoder
}

// NewWriter writes the topology line.
func NewWriter(w io.Writer, t Topology) (*Writer, error) {
	bw := bufio.NewWriter(w)
	enc := json.NewEncoder(bw)
	if err := enc.Encode(jsonTopo{t.D, t.S, t.G}); err != nil {
		return nil, err
	}
	return &Writer{bw, enc}, nil
}

func (w *Writer) Write(st *Step) error {
	js := jsonStep{Step: st.Index, Ops: make([]jsonOp, len(st.Ops))}
	for i, o := range st.Ops {
		js.Ops[i] = jsonOp{o.Kind.String(), o.Group, o.Ranks, o.Enter, o.Exit}
	}
	return w.enc.Encode(js)
}

func (w *Writer) Flush() error { return w.w.Flush() }
