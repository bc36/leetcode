package main

import (
	"os"
	"testing"

	"github.com/bc36/leetcode/Go/testutils"
)

func Test_count_number_of_possible_root_nodes(t *testing.T) {
	targetCaseNum := 0 // -1
	if err := testutils.RunLeetCodeFuncWithFile(t, rootCount, "/Users/"+os.Getenv("USER")+"/workspace/leetcode/Go/2581/count_number_of_possible_root_nodes_test_cases.txt", targetCaseNum); err != nil {
		t.Fatal(err)
	}
}
