package main

import (
	"os"
	"testing"

	"github.com/bc36/leetcode/Go/testutils"
)

func Test_reachable_nodes_with_restrictions(t *testing.T) {
	targetCaseNum := 0 // -1
	if err := testutils.RunLeetCodeFuncWithFile(t, reachableNodes, "/Users/"+os.Getenv("USER")+"/workspace/leetcode/Go/2368/reachable_nodes_with_restrictions_test_cases.txt", targetCaseNum); err != nil {
		t.Fatal(err)
	}
}
