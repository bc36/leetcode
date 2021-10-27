package lc_Go

// Dynamic Programming
// dp[i][0]: the i day that has max profit and with holding stock
// dp[i][1]: the i day that has max profit and without holding stock
func maxProfit6(prices []int, fee int) int {
	n := len(prices)
	dp := make([][2]int, n)
	dp[0][0] = -prices[0]
	for i := 1; i < n; i++ {
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]-prices[i])
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]+prices[i]-fee)
	}
	return dp[n-1][1]
}

// space optimized
func maxProfit61(prices []int, fee int) int {
	d0, d1 := -prices[0], 0
	for i := 1; i < len(prices); i++ {
		d0 = max(d0, d1-prices[i])
		d1 = max(d1, d0+prices[i]-fee)
	}
	return d1
}
