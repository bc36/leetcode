package lc_Go

/*
All type define and basic functions are in the first '.go' file

bit opertion
&	bitwise AND
|	bitwise OR
^	bitwise XOR
&^	AND NOT
<<	left shift
>>	right shift

*/
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type ListNode struct {
	Val  int
	Next *ListNode
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

type Node struct {
	Val   int
	Prev  *Node
	Next  *Node
	Child *Node
}

// 1 - Two Sum - EASY
func twoSum(nums []int, target int) []int {
	m := map[int]int{}
	for i, v := range nums {
		if j, ok := m[target-v]; ok {
			return []int{i, j}
		}
		m[v] = i
	}
	return nil
}

// 3 - Longest Substring Without Repeating Characters - MEDIUM
func lengthOfLongestSubstring(s string) int {
	var exist [256]int // 0: not repeated / 1: repeated
	ans, l, r := 0, 0, 0
	for r < len(s) {
		if exist[s[r]-'a'] == 0 {
			exist[s[r]-'a']++
			r++
		} else {
			exist[s[l]-'a']--
			l++
		}
		if r-l > ans {
			ans = r - l
		}
	}
	return ans
}

func _(s string) int {
	l, r, ans := 0, 0, 0
	pre := make(map[byte]int, len(s))
	for r < len(s) {
		if p, ok := pre[s[r]]; ok && p >= l {
			l = p + 1
		}
		pre[s[r]] = r
		ans = max(ans, r-l+1)
		r++
	}
	return ans
}

func _(s string) int {
	m := map[byte]int{}
	ans, l, n := 0, 0, len(s)
	for r := 0; r < n; r++ {
		for m[s[r]] > 0 {
			m[s[l]]--
			l++
		}
		m[s[r]]++
		if r-l+1 > ans {
			ans = r - l + 1
		}
	}
	return ans
}

// 50 - Pow(x, n) - MEDIUM
// Iteration / let exponent: decimal -> binary
func myPow(x float64, n int) float64 {
	if n == 0 {
		return 1
	}
	if n < 0 {
		x = 1 / x
		n = -n
	}
	var pow float64 = 1
	for n > 0 {
		if n&1 == 1 {
			// binary lowest bit is 1
			pow *= x
		}
		x *= x
		n = n >> 1
	}
	return pow
}

// The contributing part of the exponent corresponds to each 1 in the binary representation of the original exponent.
func myPow2(x float64, n int) float64 {
	if n >= 0 {
		return quickMul(x, n)
	}
	return 1.0 / quickMul(x, -n)
}

func quickMul(x float64, N int) float64 {
	ans := 1.0
	// 贡献的初始值为 x
	x_contribute := x
	// 在对 N 进行二进制拆分的同时计算答案
	for N > 0 {
		if N%2 == 1 {
			// 如果 N 二进制表示的最低位为 1，那么需要计入贡献
			ans *= x_contribute
		}
		// 将贡献不断地平方
		x_contribute *= x_contribute
		// 舍弃 N 二进制表示的最低位，这样我们每次只要判断最低位即可
		N /= 2
	}
	return ans
}

// Recursion / from right to left is more easy / round down
// x -> x^2 -> x^4 -> x^9 -> x^19 -> x^38 -> x^77
func myPow3(x float64, n int) float64 {
	if n >= 0 {
		return quickMul3(x, n)
	}
	return 1.0 / quickMul3(x, -n)
}

func quickMul3(x float64, n int) float64 {
	if n == 0 {
		return 1
	}
	y := quickMul3(x, n/2)
	if n%2 == 0 {
		return y * y
	}
	return y * y * x
}

// 53 - Maximum Subarray - EASY
func maxSubArray(nums []int) int {
	length := len(nums)
	if length == 1 {
		return nums[0]
	}
	// dp[i]: Maximum subsequence that ends with the current index
	ans, dp := nums[0], make([]int, length)
	dp[0] = nums[0]
	for i := 1; i < length; i++ {
		dp[i] = max(dp[i-1]+nums[i], nums[i])
	}
	for i := range dp {
		if dp[i] > ans {
			ans = dp[i]
		}
	}
	return ans
}

// 66 - Plus One - EASY
// do not convert a int, cuz overflow
func plusOne(digits []int) []int {
	l := len(digits)
	for i := l - 1; i >= 0; i-- {
		if digits[i] != 9 {
			digits[i]++
			return digits
		} else {
			digits[i] = 0
		}
	}
	return append([]int{1}, digits...)
}
