package lc_Go

import (
	"fmt"
	"sort"
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

// 129 - Sum Root to Leaf Numbers - MEDIUM
func sumNumbers(root *TreeNode) int {
	var dfs func(root *TreeNode, pre int) int
	dfs = func(root *TreeNode, pre int) int {
		if root == nil {
			return 0
		}
		cur := pre*10 + root.Val
		if root.Left == nil && root.Right == nil {
			return cur
		}
		return dfs(root.Left, cur) + dfs(root.Right, cur)
	}
	return dfs(root, 0)
}

// 136 - Single Number - EASY
// XOR operation
func singleNumber(nums []int) int {
	ans := 0
	for _, e := range nums {
		ans ^= e
	}
	return ans
}

// 137 - Single Number II - MEDIUM
// sort, jump 3 element
// use HashMap also works
func singleNumber21(nums []int) int {
	sort.Ints(nums)
	for i := 0; i < len(nums)-1; i += 3 {
		if nums[i] != nums[i+1] {
			return nums[i]
		}
	}
	if len(nums) != 1 && nums[len(nums)-1] != nums[len(nums)-2] {
		return nums[len(nums)-1]
	}
	return nums[0]
}

// 166 - Fraction to Recurring Decimal - MEDIUM
// a / b, max loop section b-1
func fractionToDecimal(x int, y int) string {
	hash, ret := make(map[int]int), ""
	if x%y == 0 {
		ret = strconv.Itoa(x / y)
		return ret
	}
	if x*y < 0 {
		ret += "-"
	}
	x, y = abs(x), abs(y)
	ret += strconv.Itoa(x/y) + "."
	x %= y
	for x != 0 {
		hash[x] = len(ret)
		x *= 10
		ret += strconv.Itoa(x / y)
		x %= y
		fmt.Println(x, hash)
		if i, ok := hash[x]; ok {
			ret = ret[0:i] + "(" + ret[i:] + ")"
			break
		}
	}
	return ret
}
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
