package main

import (
	"os"
	"testing"

	"github.com/bc36/leetcode/Go/testutils"
)

func Test_binary_tree_postorder_traversal(t *testing.T) {
	targetCaseNum := 0 // -1
	if err := testutils.RunLeetCodeFuncWithFile(t, postorderTraversal, "/Users/"+os.Getenv("USER")+"/workspace/leetcode/Go/0145/binary_tree_postorder_traversal_test_cases.txt", targetCaseNum); err != nil {
		t.Fatal(err)
	}
}
