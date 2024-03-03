package main

import (
	"os"
	"testing"

	"github.com/bc36/leetcode/Go/testutils"
)

func Test_minimum_operations_to_exceed_threshold_value_i(t *testing.T) {
	targetCaseNum := 0 // -1
	if err := testutils.RunLeetCodeFuncWithFile(t, minOperations, "/Users/"+os.Getenv("USER")+"/workspace/leetcode/Go/3065/minimum_operations_to_exceed_threshold_value_i_test_cases.txt", targetCaseNum); err != nil {
		t.Fatal(err)
	}
}
