package main

import (
	"container/heap"
	"math"
)

// Medium
// https://leetcode.cn/problems/number-of-ways-to-arrive-at-destination/
// https://leetcode.com/problems/number-of-ways-to-arrive-at-destination/

func countPaths(n int, roads [][]int) int {
	type edge struct{ to, d int }
	g := make([][]edge, n)
	for _, r := range roads {
		x, y, d := r[0], r[1], r[2]
		g[x] = append(g[x], edge{y, d})
		g[y] = append(g[y], edge{x, d})
	}
	dis := make([]int, n)
	for i := 1; i < n; i++ {
		dis[i] = math.MaxInt
	}
	f := make([]int, n)
	f[0] = 1
	h := &hp{{}}
	for {
		p := heap.Pop(h).(pair)
		x := p.x
		if x == n-1 {
			return f[n-1]
		}
		if p.dis > dis[x] {
			continue
		}
		for _, e := range g[x] {
			y := e.to
			newDis := p.dis + e.d
			if newDis < dis[y] {
				dis[y] = newDis
				f[y] = f[x]
				heap.Push(h, pair{newDis, y})
			} else if newDis == dis[y] {
				f[y] = (f[y] + f[x]) % 1000000007
			}
		}
	}
}

type pair struct{ dis, x int }
type hp []pair

func (h hp) Len() int           { return len(h) }
func (h hp) Less(i, j int) bool { return h[i].dis < h[j].dis }
func (h hp) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *hp) Push(v any)        { *h = append(*h, v.(pair)) }
func (h *hp) Pop() (v any)      { a := *h; *h, v = a[:len(a)-1], a[len(a)-1]; return }
