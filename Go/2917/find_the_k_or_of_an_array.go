package main

// Easy
// https://leetcode.cn/problems/find-the-k-or-of-an-array/
// https://leetcode.com/problems/find-the-k-or-of-an-array/

func findKOr(nums []int, k int) int {
	ans := 0
	for i := 0; i < 31; i++ {
		cnt := 0
		for _, x := range nums {
			cnt += x >> i & 1
		}
		if cnt >= k {
			ans |= 1 << i
		}
	}
	return ans
}
