package main

// Easy
// https://leetcode.cn/problems/distribute-elements-into-two-arrays-i/
// https://leetcode.com/problems/distribute-elements-into-two-arrays-i/

func resultArray(nums []int) []int {
	a := nums[:1]
	b := []int{nums[1]}
	for _, x := range nums[2:] {
		if a[len(a)-1] > b[len(b)-1] {
			a = append(a, x)
		} else {
			b = append(b, x)
		}
	}
	return append(a, b...)
}
