package main

import "math"

// Hard
// https://leetcode.cn/problems/find-the-maximum-sum-of-node-values/
// https://leetcode.com/problems/find-the-maximum-sum-of-node-values/

func maximumValueSum(nums []int, k int, edges [][]int) int64 {
	f0, f1 := 0, math.MinInt
	for _, x := range nums {
		f0, f1 = max(f0+x, f1+(x^k)), max(f1+x, f0+(x^k))
	}
	return int64(f0)
}
