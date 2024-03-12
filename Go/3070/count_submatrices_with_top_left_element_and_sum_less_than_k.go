package main

// Medium
// https://leetcode.cn/problems/count-submatrices-with-top-left-element-and-sum-less-than-k/
// https://leetcode.com/problems/count-submatrices-with-top-left-element-and-sum-less-than-k/

func countSubmatrices(grid [][]int, k int) (ans int) {
	m, n := len(grid), len(grid[0])
	sum := make([][]int, m+1)
	sum[0] = make([]int, n+1)
	for i, row := range grid {
		sum[i+1] = make([]int, n+1)
		for j, x := range row {
			sum[i+1][j+1] = sum[i+1][j] + sum[i][j+1] - sum[i][j] + x
			if sum[i+1][j+1] <= k {
				ans++
			}
		}
	}
	return
}
