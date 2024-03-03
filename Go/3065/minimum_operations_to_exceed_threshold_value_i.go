package main

// Easy
// https://leetcode.cn/problems/minimum-operations-to-exceed-threshold-value-i/
// https://leetcode.com/problems/minimum-operations-to-exceed-threshold-value-i/

func minOperations(nums []int, k int) int {
    ans := 0
	for _, v := range nums {
		if v < k {
			ans++
		}
	}
	return ans
}
