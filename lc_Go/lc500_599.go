package lc_Go

import (
	"math/rand"
	"sort"
	"strings"
	"unicode"
)

// 500 - Keyboard Row - EASY
// convert char to the line where it is
// find the word in which index and determine if it is in the line
func findWords(words []string) (ans []string) {
	const rowIdx = "12210111011122000010020202"
	// const rowIdx = "abcdefghijklmnopqrstuvwxyz"
Label:
	for _, word := range words {
		idx := rowIdx[unicode.ToLower(rune(word[0]))-'a']
		for _, ch := range word[1:] {
			if rowIdx[unicode.ToLower(ch)-'a'] != idx {
				continue Label
			}
		}
		ans = append(ans, word)
	}
	return
}

// Three HashMap
func findWords2(words []string) (ans []string) {
	toMap := func(word string) map[rune]struct{} {
		m := make(map[rune]struct{})
		for _, i := range word {
			m[i] = struct{}{}
		}
		return m
	}
	firstLine := toMap("qwertyuiop")
	secondLine := toMap("asdfghjkl")
	thirdLine := toMap("zxcvbnm")
	var oriW []string
	for i := 0; i < len(words); i++ {
		oriW = append(oriW, words[i])
		words[i] = strings.ToLower(words[i])
	}
Label:
	for k, i := range words {
		var whichLine map[rune]struct{}
		if _, ok := firstLine[rune(i[0])]; ok {
			whichLine = firstLine
		}
		if _, ok := secondLine[rune(i[0])]; ok {
			whichLine = secondLine
		}
		if _, ok := thirdLine[rune(i[0])]; ok {
			whichLine = thirdLine
		}
		for j := 1; j < len(i); j++ {
			if _, ok := whichLine[rune(i[j])]; !ok {
				continue Label
			}
		}
		ans = append(ans, oriW[k])
	}
	return ans
}

// 528 - Random Pick with Weight - MEDIUM
// prefix sum + binary search
// seperate [1, total] in len(w) parts, each part has w[i] elements
type Solution struct {
	presum []int
}

// Calculate the prefix sum to generate a random number
// The coordinates of the distribution correspond to the size of the number
// 计算前缀和，这样可以生成一个随机数，根据数的大小对应分布的坐标
func Constructor(w []int) Solution {
	for i := 1; i < len(w); i++ {
		w[i] += w[i-1]
	}
	return Solution{w}
}

func (so *Solution) PickIndex() int {
	x := rand.Intn(so.presum[len(so.presum)-1]) + 1
	return sort.SearchInts(so.presum, x)
}
