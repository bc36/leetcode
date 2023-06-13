package lc_Go

// 918 - Maximum Sum Circular Subarray - MEDIAM
// Find maximum sub array and minimum arrary. Similar question: lc53
func maxSubarraySumCircular(nums []int) int {
	length, sum, dpi, ansmax := len(nums), nums[0], nums[0], nums[0]
	for i := 1; i < length; i++ {
		sum += nums[i]
		dpi = nums[i] + max(dpi, 0)
		ansmax = max(dpi, ansmax)
	}

	dpi = nums[0]
	ansmin := 0
	for i := 1; i < length; i++ {
		dpi = nums[i] + min(0, dpi)
		ansmin = min(ansmin, dpi)
	}

	if sum-ansmin == 0 { // indicate that all are negative number
		return ansmax
	}
	return max(ansmax, sum-ansmin)
}
func maxSubarraySumCircular2(nums []int) int {
	sum, ansmax, ansmin, premax, premin := 0, nums[0], nums[0], 0, 0
	for i := 0; i < len(nums); i++ {
		premax = max(premax+nums[i], nums[i])
		ansmax = max(ansmax, premax)
		premin = min(premin+nums[i], nums[i])
		ansmin = min(ansmin, premin)
		sum += nums[i]
	}

	if sum-ansmin == 0 { // all numbers are negative
		return ansmax
	}
	return max(ansmax, sum-ansmin)
}
