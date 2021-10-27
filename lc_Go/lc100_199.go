package lc_Go

import (
	"fmt"
	"strconv"
)

// 121 - Best Time to Buy and Sell Stock -EASY
// Dynamic Programming / Space Optimized
func maxProfit(prices []int) (ans int) {
	hisLowPrice := prices[0]
	for i := 1; i < len(prices); i++ {
		hisLowPrice = min(hisLowPrice, prices[i-1])
		ans = max(ans, prices[i]-hisLowPrice)
	}
	return
}

// 122 - Best Time to Buy and Sell Stock II - MEDIUM
// Dynamic Programming
// dp[i][0]: the i day that has max profit and without holding stock
// dp[i][1]: the i day that has max profit and with holding stock
func maxProfit21(prices []int) (ans int) {
	if len(prices) < 2 {
		return 0
	}
	dp := make([][2]int, len(prices))
	dp[0][1] = -prices[0] // dp[0][0] = 0
	for i := 1; i < len(prices); i++ {
		dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
		dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i])
	}
	return dp[len(prices)-1][0]
}

// Greedy
// reduce call max() can speed up -> use if
func maxProfit22(prices []int) (ans int) {
	for i := 1; i < len(prices); i++ {
		if prices[i]-prices[i-1] > 0 {
			ans += prices[i] - prices[i-1]
		}
	}
	return
}

// 166 - Fraction to Recurring Decimal - MEDIUM
// a / b, max loop section b-1
func fractionToDecimal(x int, y int) string {
	hash, res := make(map[int]int), ""
	if x%y == 0 {
		res = strconv.Itoa(x / y)
		return res
	}
	if x*y < 0 {
		res += "-"
	}
	x, y = abs(x), abs(y)
	res += strconv.Itoa(x/y) + "."
	x %= y
	for x != 0 {
		hash[x] = len(res)
		x *= 10
		res += strconv.Itoa(x / y)
		x %= y
		fmt.Println(x, hash)
		if i, ok := hash[x]; ok {
			res = res[0:i] + "(" + res[i:] + ")"
			break
		}
	}
	return res
}
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
