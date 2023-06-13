package lc_Go

// 1218 - Longest Arithmetic Subsequence of Given Difference - MEDIUM
func longestSubsequence(arr []int, difference int) (ans int) {
	dp := make(map[int]int)
	for _, v := range arr {
		dp[v] = dp[v-difference] + 1
		if dp[v] > ans {
			ans = dp[v]
		}
	}
	return ans
}
