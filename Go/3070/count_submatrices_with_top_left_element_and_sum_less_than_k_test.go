package main

import (
	"os"
	"testing"

	"github.com/bc36/leetcode/Go/testutils"
)

func Test_count_submatrices_with_top_left_element_and_sum_less_than_k(t *testing.T) {
	targetCaseNum := 0 // -1
	if err := testutils.RunLeetCodeFuncWithFile(t, countSubmatrices, "/Users/"+os.Getenv("USER")+"/workspace/leetcode/Go/3070/count_submatrices_with_top_left_element_and_sum_less_than_k_test_cases.txt", targetCaseNum); err != nil {
		t.Fatal(err)
	}
}
