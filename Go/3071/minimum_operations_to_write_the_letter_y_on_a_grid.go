package main

// Medium
// https://leetcode.cn/problems/minimum-operations-to-write-the-letter-y-on-a-grid/
// https://leetcode.com/problems/minimum-operations-to-write-the-letter-y-on-a-grid/

func minimumOperationsToWriteY(grid [][]int) int {
	var ys, other [3]int
	n := len(grid)
	m := n / 2
	for i, row := range grid[:m] {
		ys[row[i]]++
		ys[row[n-1-i]]++
		for j, x := range row {
			if j != i && j != n-1-i {
				other[x]++
			}
		}
	}
	for _, row := range grid[m:] {
		ys[row[m]]++
		for j, x := range row {
			if j != m {
				other[x]++
			}
		}
	}

	maxNotChange := 0
	for i, c1 := range ys {
		for j, c2 := range other {
			if i != j {
				maxNotChange = max(maxNotChange, c1+c2)
			}
		}
	}
	return n*n - maxNotChange
}
