package main

import (
	"fmt"
	"math/rand"
	"sort"
)

func main() {
	// 23, 1249, 236, 371, 437, 146, 600, 47, 430, 994, 460,
	// 739 117 215 938 173 88 986
	// sort.Ints / strconv.Itoa / sort.SearchInts / rand.Intn / strings.ToLower / unicode.ToLower
	// [Go, Golang] R: 0ms/100% M: 2.2MB/69%

	// fmt.Println(maxSubArray([]int{-2, 1, -3, 4, -1, 2, 1, -5, 4})) //2
	// fmt.Println()
	// fmt.Println(([]int{2, 3, 1, 1, 4})) //2
	// fmt.Println()
	// fmt.Printf("%b,%b,%b\n", a, b, c)
	// fmt.Printf("%b, %d\n", a^b^c, a^b^c)
	// a := 5
	// b := -5
	// fmt.Printf("%b, %b\n", uint8(b), uint8(a))
	// fmt.Println(s[0:])
	// fmt.Println(s[1:])
	fmt.Printf("%b, %b, %b, %b\n", 128, 84, 202, 256)
	// d := a ^ a ^ a
}

// 139 - Word Break - MEDIUM
// dp[i] means s[:i] can be segmented
// split s[:i] to s[:j] + s[j+1:i]
func wordBreak(s string, wordDict []string) bool {
	dp, dic := make([]bool, len(s)+1), make(map[string]bool)
	dp[0] = true
	for _, i := range wordDict {
		dic[i] = true
	}
	for i := range s[1:] {
		for j := range s[:i] {
			if dp[j] && dic[s[j+1:i]] {
				dp[i] = true
				break
			}
		}
	}
	return false
}

type Solution struct {
	pre []int
}

func Constructor(w []int) Solution {
	for i := 1; i < len(w); i++ {
		w[i] += w[i-1]
	}
	return Solution{w}
}

func (this *Solution) PickIndex() int {
	x := rand.Intn(this.pre[len(this.pre)-1]) + 1
	return sort.SearchInts(this.pre, x)
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func min(a, b int) int {
	if a > b {
		return b
	}
	return a
}

func minElement(m map[int]struct{}) int {
	var keys []int
	for k := range m {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	return keys[0]
}

func maxSlidingWindow(nums []int, k int) []int {
	q := []int{}
	push := func(i int) {
		for len(q) > 0 && nums[i] <= nums[q[len(q)-1]] {
			q = q[:len(q)-1]
		}
		q = append(q, i)
	}

	for i := 0; i < k; i++ {
		push(i)
	}

	n := len(nums)
	ans := make([]int, 1, n-k+1)
	ans[0] = nums[q[0]]
	for i := k; i < n; i++ {
		push(i)
		for q[0] <= i-k {
			q = q[1:]
		}
		ans = append(ans, nums[q[0]])
	}
	sort.Ints(ans)
	return ans
}

func smallestK(arr []int, k int) []int {
	sort.Ints(arr)
	return arr[:k]
}

func minSteps(n int) int {
	if n == 1 {
		return 0
	}
	if n == 2 {
		return 2
	}
	if n == 3 {
		return 3
	}
	d := 0
	for n > 1 {
		n /= 2
		d++
	}
	return 2 * (d - 1)
}
func findLongestWord(s string, dictionary []string) string {
	// sort.Strings(dictionary)
	sort.Slice(dictionary, func(i, j int) bool {
		a, b := dictionary[i], dictionary[j]
		return len(a) > len(b) || len(a) == len(b) && a < b
	})
	// fmt.Println(dictionary)
	dm := make(map[rune]int)
	for _, i := range s {
		dm[i]++
	}
	// fmt.Println(dm)
	check := func(ds string, m map[rune]int) bool {
		fmt.Println(dm, m)
		for _, i := range ds {
			if value, ok := m[i]; ok {
				if value > 0 {
					m[i]--
					continue
				}
				return false
			}
			return false
		}
		return true
	}
	// fmt.Println(dictionary)
	for i := range dictionary {
		cpdm := make(map[rune]int)
		for i := range dm {
			cpdm[i] = dm[i]
		}
		if check(dictionary[i], cpdm) {
			return dictionary[i]
		}
	}
	return ""
}

func sortList(head *ListNode) *ListNode {
	m := make(map[*ListNode]int)
	for head != nil {
		m[head] = head.Val
		head = head.Next
	}
	if len(m) == 0 {
		return nil
	}
	l := mapvalue(m)
	h := &ListNode{}
	rt := h
	for i := 0; i < len(l); i++ {
		h.Next = &ListNode{Val: l[i].Value}
	}
	return rt.Next
}

// map key
func mapkey(m map[int]string) {
	var keys []int
	for k := range m {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	// To perform the opertion you want
	for _, k := range keys {
		fmt.Println("Key:", k, "Value:", m[k])
	}
}

// map value

type Pair struct {
	Key   *ListNode
	Value int
}
type PairList []Pair

func (p PairList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p PairList) Len() int           { return len(p) }
func (p PairList) Less(i, j int) bool { return p[i].Value < p[j].Value }
func mapvalue(m map[*ListNode]int) PairList {
	p := make(PairList, len(m))
	i := 0
	for k, v := range m {
		p[i] = Pair{Key: k, Value: v}
	}
	sort.Sort(p)
	return p
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

// reverse ...
func reverse(s string) string {
	a := []rune(s)
	for i, j := 0, len(a)-1; i < j; i, j = i+1, j-1 {
		a[i], a[j] = a[j], a[i]
	}
	return string(a)
}

func twoDimensionSlice() {
	boolMap := make([][]bool, 3) // 3 行
	for i := range boolMap {
		boolMap[i] = make([]bool, 4) // 4 列
	}
}
