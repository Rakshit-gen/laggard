// Package trace describes what a training job reports about itself: one
// record per communication operation, with every member's own enter and
// exit time. That's roughly what NCCL's flight recorder or a profiler
// plugin can give you, and it's all the detectors in this repo get to see.
package trace

// Kind says which kind of parallelism an operation belongs to.
type Kind uint8

const (
	TP Kind = iota // tensor-parallel all-reduce, inside one node over NVLink
	PP             // pipeline send/recv between neighbouring stages, over the NIC
	DP             // data-parallel gradient all-reduce across replicas, over the NIC
)

func (k Kind) String() string {
	return [...]string{"TP", "PP", "DP"}[k]
}

// Op is one communication operation. Enter and Exit hold one time per
// member, in the same order as Ranks, each read from that member's own
// clock. Clocks on different hosts are not assumed to agree, so the only
// safe arithmetic is between two times from the same member.
type Op struct {
	Kind  Kind
	Group int32   // which TP group, PP link or DP group, see Topology
	Ranks []int32 // shared with the topology, don't modify
	Enter []float64
	Exit  []float64
}

// Step is every operation from one training step. Within the slice, each
// rank's operations appear in the order that rank ran them, which is what
// you get by concatenating per-rank logs and sorting each by its own clock.
type Step struct {
	Index int
	Ops   []Op
}

// Topology is a 3D-parallel job: D data-parallel replicas, each a pipeline
// of S stages, each stage a tensor-parallel group of G GPUs filling one
// node. Every GPU has its own NIC (a rail-optimized network), so a NIC is
// identified by the rank it belongs to.
type Topology struct {
	D, S, G int
}

func (t Topology) Ranks() int { return t.D * t.S * t.G }
func (t Topology) Nodes() int { return t.D * t.S }

// Rank numbers GPUs replica-major, then stage, then position in the node.
func (t Topology) Rank(d, s, g int) int { return (d*t.S+s)*t.G + g }

// Coords inverts Rank.
func (t Topology) Coords(r int) (d, s, g int) {
	g = r % t.G
	n := r / t.G
	return n / t.S, n % t.S, g
}

func (t Topology) Node(r int) int { return r / t.G }

// TPGroup is the node's index: one TP group per node.
func (t Topology) TPGroup(d, s int) int { return d*t.S + s }

// DPGroup holds the GPUs at stage s, position g, across all replicas.
func (t Topology) DPGroups() int     { return t.S * t.G }
func (t Topology) DPGroup(s, g int) int { return s*t.G + g }

// PPLink is the connection from stage s to s+1 at position g in replica d.
// Forward activations and backward gradients both use it.
func (t Topology) PPLinks() int          { return t.D * (t.S - 1) * t.G }
func (t Topology) PPLink(d, s, g int) int { return (d*(t.S-1)+s)*t.G + g }

// Members returns the ranks in each group, in a fixed order: TP by g,
// PP as (stage s, stage s+1), DP by replica.
func (t Topology) Members() (tp, pp, dp [][]int32) {
	tp = make([][]int32, t.Nodes())
	for d := range t.D {
		for s := range t.S {
			m := make([]int32, t.G)
			for g := range t.G {
				m[g] = int32(t.Rank(d, s, g))
			}
			tp[t.TPGroup(d, s)] = m
		}
	}
	pp = make([][]int32, t.PPLinks())
	for d := range t.D {
		for s := range t.S - 1 {
			for g := range t.G {
				pp[t.PPLink(d, s, g)] = []int32{int32(t.Rank(d, s, g)), int32(t.Rank(d, s+1, g))}
			}
		}
	}
	dp = make([][]int32, t.DPGroups())
	for s := range t.S {
		for g := range t.G {
			m := make([]int32, t.D)
			for d := range t.D {
				m[d] = int32(t.Rank(d, s, g))
			}
			dp[t.DPGroup(s, g)] = m
		}
	}
	return
}
