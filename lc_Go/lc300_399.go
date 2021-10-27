package lc_Go

// 309 - Best Time to Buy and Sell Stock with Cooldown - MEDIUM
// Dynamic Programming
// dp[i][0]: the i day that has max profit and with holding stock
// dp[i][1]: the i day that has max profit and without holding stock, having cooldown
// dp[i][2]: the i day that has max profit and without holding stock, not having cooldown
func maxProfit5(prices []int) int {
	if len(prices) < 2 {
		return 0
	}
	n := len(prices)
	dp := make([][3]int, n)
	dp[0][0] = -prices[0]
	for i := 1; i < n; i++ {
		dp[i][0] = max(dp[i-1][0], dp[i-1][2]-prices[i])
		dp[i][1] = dp[i-1][0] + prices[i]
		dp[i][2] = max(dp[i-1][1], dp[i-1][2])
	}
	return max(dp[n-1][1], dp[n-1][2])
}

// space optimized
// only keep dp[i-1][0], dp[i-1][1], dp[i-1][2]
func maxProfit51(prices []int) int {
	if len(prices) < 2 {
		return 0
	}
	d0, d1, d2 := -prices[0], 0, 0
	for i := 1; i < len(prices); i++ {
		newd0 := max(d0, d2-prices[i])
		newd1 := d0 + prices[i]
		newd2 := max(d1, d2)
		d0, d1, d2 = newd0, newd1, newd2
	}
	return max(d1, d2)
}
