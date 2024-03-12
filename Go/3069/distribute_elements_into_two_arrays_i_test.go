package main

import (
	"os"
	"testing"

	"github.com/bc36/leetcode/Go/testutils"
)

func Test_distribute_elements_into_two_arrays_i(t *testing.T) {
	targetCaseNum := 0 // -1
	if err := testutils.RunLeetCodeFuncWithFile(t, resultArray, "/Users/"+os.Getenv("USER")+"/workspace/leetcode/Go/3069/distribute_elements_into_two_arrays_i_test_cases.txt", targetCaseNum); err != nil {
		t.Fatal(err)
	}
}
