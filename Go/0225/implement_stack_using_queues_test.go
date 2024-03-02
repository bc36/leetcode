package main

import (
	"os"
	"testing"

	"github.com/bc36/leetcode/Go/testutils"
)

func Test_implement_stack_using_queues(t *testing.T) {
	t.Log("记得初始化所有全局变量")
	targetCaseNum := 0 // -1
	if err := testutils.RunLeetCodeClassWithFile(t, Constructor, "/Users/"+os.Getenv("USER")+"/workspace/leetcode/Go/0225/implement_stack_using_queues_test_cases.txt", targetCaseNum); err != nil {
		t.Fatal(err)
	}
}
