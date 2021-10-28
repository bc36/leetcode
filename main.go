package main

import (
	"fmt"
	"math/rand"
	"sort"
)

func main() {
	// 23, 1249, 236, 371, 437, 146, 600, 47, 430, 994, 460,
	// 739 117 215 938 173 88 986
	// sort.Ints / strconv.Itoa / sort.SearchInts / rand.Intn
	// [Go, Golang] R: 100ms/51% M: 9.7MB/17%
	// oa2("abcde", 1, "10101111111111111111111111")   // 3
	// oa2("abcde", 2, "10101111111111111111111111")   // 5
	// oa2("giraffe", 2, "01111001111111111011111111") // 3 agfr
	// oa4(3, 4, [][]int{{0}, {1, 2}, {0}, {2, 1}, {0}, {1, 1}, {0}})
	// fmt.Println(maxSubArray([]int{-2, 1, -3, 4, -1, 2, 1, -5, 4})) //2
	// fmt.Println()
	// fmt.Println(([]int{2, 3, 1, 1, 4})) //2
	// fmt.Println()
	// fmt.Println(([]int{2, 3, 0, 1, 4})) //2
	// fmt.Println(reorderedPowerOf2([]int{1, 2, 5}, 0))
	// fmt.Println(reorderedPowerOf2(1))
	// fmt.Println(reorderedPowerOf2(24))
	// fmt.Println(reorderedPowerOf2(16))
	// fmt.Println(reorderedPowerOf2(10))
	// fmt.Println(reorderedPowerOf2()
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

func oa2(s string, k int, charValue string) (ret int) {
	alphabet := "abcdefghijklmnopqrstuvwxyz"
	specialMap := make(map[byte]struct{})
	for i := range charValue {
		if charValue[i] == '0' {
			specialMap[alphabet[i]] = struct{}{}
		}
	}
	// fmt.Println(specialMap)
	// first right position
	spNum, pos := 0, 0
	for spNum < k && pos < len(s) {
		if _, ok := specialMap[s[pos]]; ok {
			spNum++
		}
		pos++
	}
	fmt.Println(pos, "pos")
	ret = pos + 1
	for i, j := 0, pos+1; j < len(s) && i <= j; {
		if _, jIn := specialMap[s[j]]; !jIn {
			ret = max(ret, j-i+1)
			fmt.Println(s[i:j], i, j, "0.")
			j++
			continue
		}
		i++
		if _, iIn := specialMap[s[i]]; !iIn {
			ret = max(ret, j-i+1)
			fmt.Println(s[i:j], i, j, "0..")
			i++
			continue
		}
	}
	fmt.Println(ret, "ret")
	return ret
}

func oa4(n, m int, queries [][]int) (ret []int) {
	row := make(map[int]struct{})
	for i := 0; i < n; i++ {
		row[i] = struct{}{}
	}
	column := make(map[int]struct{})
	for i := 0; i < m; i++ {
		column[i] = struct{}{}
	}

	for i := range queries {
		if len(queries[i]) == 1 {
			ret = append(ret, (minElement(row)+1)*(minElement(column)+1))
			// fmt.Println("add: ", (minElement(row)+1)*(minElement(column)+1))
			continue
		}
		if queries[i][0] == 1 {
			// remove row
			delete(row, queries[i][1]-1)
			// fmt.Println(queries[i], "remove row", row)
			continue
		}
		// remove column
		delete(column, queries[i][1]-1)
		// fmt.Println(queries[i], "remove column", column)
		continue
	}
	// fmt.Println(ret)
	return ret
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
