package main

import (
	"os"
	"testing"

	"github.com/bc36/leetcode/Go/testutils"
)

func Test_check_if_there_is_a_valid_partition_for_the_array(t *testing.T) {
	targetCaseNum := 0 // -1
	if err := testutils.RunLeetCodeFuncWithFile(t, validPartition, "/Users/"+os.Getenv("USER")+"/workspace/leetcode/Go/2369/check_if_there_is_a_valid_partition_for_the_array_test_cases.txt", targetCaseNum); err != nil {
		t.Fatal(err)
	}
}
