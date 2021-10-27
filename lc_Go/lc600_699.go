package lc_Go

// 696 - Count Binary Substrings - EASY
// group consecutive '0' and '1', min(u, v), u and v are consecutive element in group
func countBinarySubstrings(s string) int {
	group, char, counter := []int{}, s[0], 1
	for i := 1; i < len(s); i++ {
		if char == s[i] {
			counter++
		} else {
			group = append(group, counter)
			counter = 1
			char = s[i]
		}
		if i == len(s)-1 {
			group = append(group, counter)
		}
	}
	ans := 0
	for i := 1; i < len(group); i++ {
		ans += min(group[i], group[i-1])
	}
	return ans
}

// Space optimization: saving an array's memory
func countBinarySubstrings2(s string) int {
	var ptr, last, ans int
	n := len(s)
	for ptr < n {
		c := s[ptr]
		count := 0
		for ptr < n && s[ptr] == c {
			ptr++
			count++
		}
		ans += min(count, last)
		last = count
	}

	return ans
}
