package main

import (
	"os"
	"testing"

	"github.com/bc36/leetcode/Go/testutils"
)

func Test_find_the_k_or_of_an_array(t *testing.T) {
	targetCaseNum := 0 // -1
	if err := testutils.RunLeetCodeFuncWithFile(t, findKOr, "/Users/"+os.Getenv("USER")+"/workspace/leetcode/Go/2917/find_the_k_or_of_an_array_test_cases.txt", targetCaseNum); err != nil {
		t.Fatal(err)
	}
}
