package main

import (
	"os"
	"testing"

	"github.com/bc36/leetcode/Go/testutils"
)

func Test_find_the_maximum_sum_of_node_values(t *testing.T) {
	targetCaseNum := 0 // -1
	if err := testutils.RunLeetCodeFuncWithFile(t, maximumValueSum, "/Users/"+os.Getenv("USER")+"/workspace/leetcode/Go/3068/find_the_maximum_sum_of_node_values_test_cases.txt", targetCaseNum); err != nil {
		t.Fatal(err)
	}
}
