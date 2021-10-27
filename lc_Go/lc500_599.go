package lc_Go

import (
	"math/rand"
	"sort"
)

// 528 - Random Pick with Weight - MEDIUM
// prefix sum + binary search
// seperate [1, total] in len(w) parts, each part has w[i] elements
type Solution struct {
	presum []int
}

func Constructor(w []int) Solution {
	for i := 1; i < len(w); i++ {
		w[i] += w[i-1]
	}
	return Solution{w}
}

func (so *Solution) PickIndex() int {
	x := rand.Intn(so.presum[len(so.presum)-1]) + 1
	return sort.SearchInts(so.presum, x)
}
