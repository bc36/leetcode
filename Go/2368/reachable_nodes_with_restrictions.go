package main

// Medium
// https://leetcode.cn/problems/reachable-nodes-with-restrictions/
// https://leetcode.com/problems/reachable-nodes-with-restrictions/
func reachableNodes(n int, edges [][]int, restricted []int) int {
	r := make(map[int]bool, len(restricted))
	for _, x := range restricted {
		r[x] = true
	}
	g := make([][]int, n)
	for _, e := range edges {
		x, y := e[0], e[1]
		if !r[x] && !r[y] {
			g[x] = append(g[x], y)
			g[y] = append(g[y], x)
		}
	}
	var dfs func(int, int) int
	dfs = func(x, fa int) int {
		res := 1
		for _, y := range g[x] {
			if y != fa {
				res += dfs(y, x)
			}
		}
		return res
	}
	return dfs(0, -1)
}

func _(n int, edges [][]int, restricted []int) (ans int) {
	g := make([][]int, n)
	for _, e := range edges {
		x, y := e[0], e[1]
		g[x] = append(g[x], y)
		g[y] = append(g[y], x)
	}
	vis := make([]bool, n)
	for _, x := range restricted {
		vis[x] = true
	}
	q := []int{0}
	for vis[0] = true; len(q) > 0; ans++ {
		x := q[0]
		q = q[1:]
		for _, y := range g[x] {
			if !vis[y] {
				vis[y] = true
				q = append(q, y)
			}
		}
	}
	return ans
}
