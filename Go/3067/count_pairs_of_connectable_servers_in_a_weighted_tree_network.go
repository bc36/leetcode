package main

// Medium
// https://leetcode.cn/problems/count-pairs-of-connectable-servers-in-a-weighted-tree-network/
// https://leetcode.com/problems/count-pairs-of-connectable-servers-in-a-weighted-tree-network/

func countPairsOfConnectableServers(edges [][]int, signalSpeed int) []int {
	n := len(edges)
	type edge struct{ to, w int }
	g := make([][]edge, n+1)
	for _, e := range edges {
		x, y, w := e[0], e[1], e[2]
		g[x] = append(g[x], edge{y, w})
		g[y] = append(g[y], edge{x, w})
	}

	ans := make([]int, n+1)
	for i, gi := range g {
		var cnt int
		var dfs func(int, int, int)
		dfs = func(x, fa, d int) {
			if d%signalSpeed == 0 {
				cnt++
			}
			for _, e := range g[x] {
				if e.to != fa {
					dfs(e.to, x, d+e.w)
				}
			}
		}
		cur := 0
		for _, e := range gi {
			cnt = 0
			dfs(e.to, i, e.w)
			ans[i] += cnt * cur
			cur += cnt
		}
	}
	return ans
}
