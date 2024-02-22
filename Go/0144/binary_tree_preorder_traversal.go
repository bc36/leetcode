package main

//lint:ignore ST1001 ¯\_(ツ)_/¯
import . "github.com/bc36/leetcode/Go/testutils"

// Easy
// https://leetcode.cn/problems/binary-tree-preorder-traversal/
// https://leetcode.com/problems/binary-tree-preorder-traversal/
func preorderTraversal(root *TreeNode) []int {
	if root == nil {
		return nil
	}
	var ans []int
	ans = append(ans, root.Val)
	ans = append(ans, preorderTraversal(root.Left)...)
	ans = append(ans, preorderTraversal(root.Right)...)
	return ans
}

func _(root *TreeNode) (ans []int) {
	var preorder func(*TreeNode)
	preorder = func(root *TreeNode) {
		if root == nil {
			return
		}
		ans = append(ans, root.Val)
		preorder(root.Left)
		preorder(root.Right)
	}
	preorder(root)
	return
}

func _(root *TreeNode) []int {
	st := []*TreeNode{}
	ans := []int{}
	for root != nil || len(st) > 0 {
		for root != nil {
			ans = append(ans, root.Val)
			st = append(st, root)
			root = root.Left
		}
		root = st[len(st)-1].Right
		st = st[:len(st)-1]
	}
	return ans
}

func _(root *TreeNode) []int {
	st := []*TreeNode{root}
	ans := []int{}
	for len(st) > 0 {
		x := st[len(st)-1]
		st = st[:len(st)-1]
		if x != nil {
			ans = append(ans, x.Val)
			st = append(st, x.Right, x.Left)
		}
	}
	return ans
}
