package main

//lint:ignore ST1001 ¯\_(ツ)_/¯
import . "github.com/bc36/leetcode/Go/testutils"

// Easy
// https://leetcode.cn/problems/binary-tree-inorder-traversal/
// https://leetcode.com/problems/binary-tree-inorder-traversal/
func inorderTraversal(root *TreeNode) []int {
	if root == nil {
		return nil
	}
	var ans []int
	ans = append(ans, inorderTraversal(root.Left)...)
	ans = append(ans, root.Val)
	ans = append(ans, inorderTraversal(root.Right)...)
	return ans
}

func _(root *TreeNode) (ans []int) {
	var inorder func(*TreeNode)
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inorder(node.Left)
		ans = append(ans, node.Val)
		inorder(node.Right)
	}
	inorder(root)
	return
}

func _(root *TreeNode) []int {
	st := []*TreeNode{}
	ans := []int{}
	for root != nil || len(st) > 0 {
		for root != nil {
			st = append(st, root)
			root = root.Left
		}
		root = st[len(st)-1]
		st = st[:len(st)-1]
		ans = append(ans, root.Val)
		root = root.Right
	}
	return ans
}
