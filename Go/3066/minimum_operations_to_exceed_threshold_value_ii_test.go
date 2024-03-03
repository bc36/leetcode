package main

import (
	"os"
	"testing"

	"github.com/bc36/leetcode/Go/testutils"
)

func Test_minimum_operations_to_exceed_threshold_value_ii(t *testing.T) {
	targetCaseNum := 0 // -1
	if err := testutils.RunLeetCodeFuncWithFile(t, minOperations, "/Users/"+os.Getenv("USER")+"/workspace/leetcode/Go/3066/minimum_operations_to_exceed_threshold_value_ii_test_cases.txt", targetCaseNum); err != nil {
		t.Fatal(err)
	}
}
