package main

import (
	"slices"

	//lint:ignore ST1001 ¯\_(ツ)_/¯
	. "github.com/bc36/leetcode/Go/testutils"
)

// Easy
// https://leetcode.cn/problems/binary-tree-postorder-traversal/
// https://leetcode.com/problems/binary-tree-postorder-traversal/
func postorderTraversal(root *TreeNode) []int {
	if root == nil {
		return nil
	}
	var ans []int
	ans = append(ans, postorderTraversal(root.Left)...)
	ans = append(ans, postorderTraversal(root.Right)...)
	ans = append(ans, root.Val)
	return ans
}

func _(root *TreeNode) (ans []int) {
	var postorder func(*TreeNode)
	postorder = func(root *TreeNode) {
		if root == nil {
			return
		}
		postorder(root.Left)
		postorder(root.Right)
		ans = append(ans, root.Val)
	}
	postorder(root)
	return
}

func _(root *TreeNode) []int {
	st := []*TreeNode{}
	ans := []int{}
	for root != nil || len(st) > 0 {
		for root != nil {
			ans = append(ans, root.Val)
			st = append(st, root)
			root = root.Right
		}
		root = st[len(st)-1].Left
		st = st[:len(st)-1]
	}
	slices.Reverse(ans)
	return ans
}
