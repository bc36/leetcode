package main

import (
	"os"
	"testing"

	"github.com/bc36/leetcode/Go/testutils"
)

func Test_binary_tree_inorder_traversal(t *testing.T) {
	targetCaseNum := 0 // -1
	if err := testutils.RunLeetCodeFuncWithFile(t, inorderTraversal, "/Users/"+os.Getenv("USER")+"/workspace/leetcode/Go/0094/binary_tree_inorder_traversal_test_cases.txt", targetCaseNum); err != nil {
		t.Fatal(err)
	}
}
