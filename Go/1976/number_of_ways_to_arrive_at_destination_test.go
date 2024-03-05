package main

import (
	"os"
	"testing"

	"github.com/bc36/leetcode/Go/testutils"
)

func Test_number_of_ways_to_arrive_at_destination(t *testing.T) {
	targetCaseNum := 0 // -1
	if err := testutils.RunLeetCodeFuncWithFile(t, countPaths, "/Users/"+os.Getenv("USER")+"/workspace/leetcode/Go/1976/number_of_ways_to_arrive_at_destination_test_cases.txt", targetCaseNum); err != nil {
		t.Fatal(err)
	}
}
