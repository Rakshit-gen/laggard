package trace

import (
	"slices"
	"testing"
)

func TestRankCoordsRoundTrip(t *testing.T) {
	top := Topology{D: 3, S: 4, G: 8}
	for r := range top.Ranks() {
		d, s, g := top.Coords(r)
		if top.Rank(d, s, g) != r {
			t.Fatalf("rank %d -> (%d,%d,%d) -> %d", r, d, s, g, top.Rank(d, s, g))
		}
		if top.Node(r) != top.TPGroup(d, s) {
			t.Fatalf("rank %d: node %d but TP group %d", r, top.Node(r), top.TPGroup(d, s))
		}
	}
}

func TestMembers(t *testing.T) {
	top := Topology{D: 3, S: 4, G: 8}
	tp, pp, dp := top.Members()
	if len(tp) != top.Nodes() || len(pp) != top.PPLinks() || len(dp) != top.DPGroups() {
		t.Fatalf("group counts %d %d %d", len(tp), len(pp), len(dp))
	}
	// Every rank is in exactly one TP group and one DP group.
	seenTP := make([]int, top.Ranks())
	seenDP := make([]int, top.Ranks())
	for _, m := range tp {
		for _, r := range m {
			seenTP[r]++
		}
	}
	for _, m := range dp {
		for _, r := range m {
			seenDP[r]++
		}
	}
	for r := range top.Ranks() {
		if seenTP[r] != 1 || seenDP[r] != 1 {
			t.Fatalf("rank %d in %d TP and %d DP groups", r, seenTP[r], seenDP[r])
		}
	}
	// A PP link joins neighbouring stages of one replica at one position.
	l := pp[top.PPLink(2, 1, 5)]
	if !slices.Equal(l, []int32{int32(top.Rank(2, 1, 5)), int32(top.Rank(2, 2, 5))}) {
		t.Fatalf("PP link members %v", l)
	}
	// A DP group is one stage and position across replicas.
	for _, r := range dp[top.DPGroup(3, 2)] {
		if _, s, g := top.Coords(int(r)); s != 3 || g != 2 {
			t.Fatalf("rank %d in DP group (3,2) is at stage %d position %d", r, s, g)
		}
	}
}
