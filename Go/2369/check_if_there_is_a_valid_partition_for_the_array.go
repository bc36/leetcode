package main

// Medium
// https://leetcode.cn/problems/check-if-there-is-a-valid-partition-for-the-array/
// https://leetcode.com/problems/check-if-there-is-a-valid-partition-for-the-array/
func _(nums []int) bool {
	vis := make([]bool, len(nums))
	var dfs func(int) bool
	dfs = func(i int) bool {
		if i == -1 {
			return true
		}
		if vis[i] {
			// 不能 return true, 因为第一次遇到 i 如果没通过检查, 则后续直接返回 false 即可
			return false
		}
		vis[i] = true
		if i >= 1 && nums[i-1] == nums[i] && dfs(i-2) {
			return true
		}
		if i >= 2 && ((nums[i-2] == nums[i] && nums[i-1] == nums[i]) || (nums[i-2]+2 == nums[i] && nums[i-1]+1 == nums[i])) && dfs(i-3) {
			return true
		}
		return false
	}
	return dfs(len(nums) - 1)
}

func validPartition(nums []int) bool {
	memo := make(map[int]bool)
	// memo := map[int]bool{}
	var dfs func(int) bool
	dfs = func(i int) bool {
		if i == -1 {
			return true
		}
		if v, ok := memo[i]; ok {
			return v
		}
		res := false
		if i >= 1 && nums[i-1] == nums[i] {
			res = res || dfs(i-2)
		}
		if i >= 2 && ((nums[i-2] == nums[i] && nums[i-1] == nums[i]) || (nums[i-2]+2 == nums[i] && nums[i-1]+1 == nums[i])) {
			res = res || dfs(i-3)
		}
		memo[i] = res
		return res
	}
	return dfs(len(nums) - 1)
}

func _(nums []int) bool {
	n := len(nums)
	f := make([]bool, n+1)
	f[0] = true
	for i, x := range nums {
		if i >= 1 && f[i-1] && x == nums[i-1] || i >= 2 && f[i-2] && (x == nums[i-1] && x == nums[i-2] || x == nums[i-1]+1 && x == nums[i-2]+2) {
			f[i+1] = true
		}
	}
	return f[n]
}
