package main

import (
	"container/heap"
	"sort"
)

// Medium
// https://leetcode.cn/problems/minimum-operations-to-exceed-threshold-value-ii/
// https://leetcode.com/problems/minimum-operations-to-exceed-threshold-value-ii/

func minOperations(nums []int, k int) (ans int) {
	h := &hp{nums}
	heap.Init(h)
	for h.IntSlice[0] < k {
		x := heap.Pop(h).(int)
		h.IntSlice[0] += x * 2
		heap.Fix(h, 0)
		ans++
	}
	return
}

type hp struct{ sort.IntSlice }

func (hp) Push(any)    {}
func (h *hp) Pop() any { a := h.IntSlice; v := a[len(a)-1]; h.IntSlice = a[:len(a)-1]; return v }
