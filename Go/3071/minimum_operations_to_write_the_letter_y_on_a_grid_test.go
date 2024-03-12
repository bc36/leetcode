package main

import (
	"os"
	"testing"

	"github.com/bc36/leetcode/Go/testutils"
)

func Test_minimum_operations_to_write_the_letter_y_on_a_grid(t *testing.T) {
	targetCaseNum := 0 // -1
	if err := testutils.RunLeetCodeFuncWithFile(t, minimumOperationsToWriteY, "/Users/"+os.Getenv("USER")+"/workspace/leetcode/Go/3071/minimum_operations_to_write_the_letter_y_on_a_grid_test_cases.txt", targetCaseNum); err != nil {
		t.Fatal(err)
	}
}
