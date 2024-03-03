package main

import (
	"os"
	"testing"

	"github.com/bc36/leetcode/Go/testutils"
)

func Test_count_pairs_of_connectable_servers_in_a_weighted_tree_network(t *testing.T) {
	targetCaseNum := 0 // -1
	if err := testutils.RunLeetCodeFuncWithFile(t, countPairsOfConnectableServers, "/Users/"+os.Getenv("USER")+"/workspace/leetcode/Go/3067/count_pairs_of_connectable_servers_in_a_weighted_tree_network_test_cases.txt", targetCaseNum); err != nil {
		t.Fatal(err)
	}
}
