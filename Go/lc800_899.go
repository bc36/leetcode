package lc_Go

import (
	"sort"
	"strconv"
)

// 869 - Reordered Power of 2 - MEDIUM
// find the (pow) that has the same length of (n)
// split and add to list, sort.Ints and compare each element
func reorderedPowerOf2(n int) bool {
	ori := []int{}
	for n != 0 {
		ori = append(ori, n%10)
		n /= 10
	}
	pow := 1
	targetPow := []int{}
	for len(strconv.Itoa(pow)) <= len(ori) {
		if len(strconv.Itoa(pow)) == len(ori) {
			targetPow = append(targetPow, pow)
		}
		pow <<= 1
	}
	sort.Ints(ori)
	for _, tp := range targetPow {
		target := []int{}
		for tp != 0 {
			target = append(target, tp%10)
			tp /= 10
		}
		sort.Ints(target)
		for i := 0; i < len(target); i++ {
			if target[i] != ori[i] {
				break
			}
			if i == len(target)-1 {
				return true
			}
		}
	}
	return false
}
