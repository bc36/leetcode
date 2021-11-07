package lc_Go

import (
	"sort"
	"strconv"
)

// 229 - Majority Element II - MEDIUM
// similar question: 169 Majority Element
// Boyer–Moore majority vote algorithm / using map to count works too
func majorityElement(nums []int) (ans []int) {
	if len(nums) == 1 {
		return nums
	}
	sort.Ints(nums)
	time, n := 1, len(nums)/3
	for i := 1; i < len(nums); i++ {
		if nums[i] != nums[i-1] {
			if time > n {
				ans = append(ans, nums[i-1])
			}
			time = 1
		} else {
			time++
		}
		if i == len(nums)-1 {
			if time > n {
				ans = append(ans, nums[i])
			}
		}
	}
	return
}
func majorityElement2(nums []int) (ans []int) {
	element1, element2 := 0, 0
	vote1, vote2 := 0, 0
	for _, num := range nums {
		if vote1 > 0 && num == element1 { // 如果该元素为第一个元素，则计数加1
			vote1++
		} else if vote2 > 0 && num == element2 { // 如果该元素为第二个元素，则计数加1
			vote2++
		} else if vote1 == 0 { // 选择第一个元素
			element1 = num
			vote1++
		} else if vote2 == 0 { // 选择第二个元素
			element2 = num
			vote2++
		} else { // 如果三个元素均不相同，则相互抵消1次
			vote1--
			vote2--
		}
	}
	cnt1, cnt2 := 0, 0
	for _, num := range nums {
		if vote1 > 0 && num == element1 {
			cnt1++
		}
		if vote2 > 0 && num == element2 {
			cnt2++
		}
	}
	// 检测元素出现的次数是否满足要求
	if vote1 > 0 && cnt1 > len(nums)/3 {
		ans = append(ans, element1)
	}
	if vote2 > 0 && cnt2 > len(nums)/3 {
		ans = append(ans, element2)
	}
	return
}

// 230 - Kth Smallest Element in a BST - MEDIUM
// inorder / use a slice to save all values / O(n) + O(n)
func kthSmallest(root *TreeNode, k int) int {
	var res []int
	var dfs func(r *TreeNode)
	dfs = func(root *TreeNode) {
		if root != nil {
			dfs(root.Left)
			res = append(res, root.Val)
			dfs(root.Right)
		}
	}
	dfs(root)
	return res[k-1]
}

// 236 - Lowest Common Ancestor of a Binary Tree - MEDIUM
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil || root == p || root == q {
		return root
	}
	left := lowestCommonAncestor(root.Left, p, q)
	right := lowestCommonAncestor(root.Right, p, q)
	if left == nil {
		return right
	}
	if right == nil {
		return left
	}
	return root
}

// 237 - Delete Node in a Linked List - EASY
// copy the value of next to the node to be deleted
// jump through the next node since the value has been saved
func deleteNode(node *ListNode) {
	node.Val = node.Next.Val
	node.Next = node.Next.Next
}

func deleteNode2(node *ListNode) {
	*node = *node.Next
}

// 240 - Search a 2D Matrix II - MEDIUM
// search each row, then search each element from qualified row
func searchMatrix(matrix [][]int, target int) bool {
	for i := len(matrix) - 1; i >= 0; i-- {
		if matrix[i][0] <= target && target <= matrix[i][len(matrix[0])-1] {
			for j := 0; j < len(matrix[0]); j++ {
				if target == matrix[i][j] {
					return true
				}
			}
		}
	}
	return false
}

// Binary search
func searchMatrix2(matrix [][]int, target int) bool {
	for _, row := range matrix {
		i := sort.SearchInts(row, target)
		if i < len(row) && row[i] == target {
			return true
		}
	}
	return false
}

// Zigzag search
// from (0, n-1) to (n-1, 0) / remove one column or one row at each search
func searchMatrix3(matrix [][]int, target int) bool {
	m, n := len(matrix), len(matrix[0])
	x, y := 0, n-1
	for x < m && y >= 0 {
		if matrix[x][y] == target {
			return true
		}
		if matrix[x][y] > target {
			y--
		} else {
			x++
		}
	}
	return false
}

// 260 - Single Number III - MEDIUM
// "lsb" is the last 1 of its binary representation, means that two numbers are different in that bit
// split nums[] into two lists, one with that bit as 0 and the other with that bit as 1.
// separately perform XOR operation, find the number that appears once in each list.
// O(n) + O(1)
func singleNumber31(nums []int) []int {
	xorSum := 0
	for _, num := range nums {
		xorSum ^= num
	}
	lsb := xorSum & -xorSum
	// lsb := 1
	// for xorSum&lsb == 0 {
	// 	lsb <<= 1
	// }
	ans1, ans2 := 0, 0
	for _, num := range nums {
		if num&lsb > 0 {
			ans1 ^= num
		} else {
			ans2 ^= num
		}
	}
	return []int{ans1, ans2}
}

// Hash map, O(n) + O(n)
func singleNumber32(nums []int) (ans []int) {
	freq := map[int]int{}
	for _, num := range nums {
		freq[num]++
	}
	for num, occ := range freq {
		if occ == 1 {
			ans = append(ans, num)
		}
	}
	return
}

// 268 - Missing Number - EASY
// sort
func missingNumber(nums []int) int {
	sort.Ints(nums)
	for i, v := range nums {
		if i != v {
			return i
		}
	}
	return len(nums)
}

// XOR
func missingNumber2(nums []int) int {
	ans := len(nums)
	for i, v := range nums {
		ans = ans ^ i ^ v
	}
	return ans
}

// math
func missingNumber3(nums []int) int {
	n := len(nums)
	total := (0 + n) * (n + 1) / 2
	sum := 0
	for _, v := range nums {
		sum += v
	}
	return total - sum
}

// 299 - Bulls and Cows - MEDIUM
func getHint(secret, guess string) string {
	bulls := 0
	var cntS, cntG [10]int
	for i := range secret {
		if secret[i] == guess[i] {
			bulls++
		} else {
			cntS[secret[i]-'0']++
			cntG[guess[i]-'0']++
		}
	}
	cows := 0
	for i := 0; i < 10; i++ {
		cows += min(cntS[i], cntG[i])
	}
	// return fmt.Sprintf("%dA%dB", bulls, cows)
	return strconv.Itoa(bulls) + "A" + strconv.Itoa(cows) + "B"
}
