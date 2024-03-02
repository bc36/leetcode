package algo

func kmp(text, pattern string) (pos []int) {
	m := len(pattern)
	pi := make([]int, m)
	cnt := 0
	for i := 1; i < m; i++ {
		v := pattern[i]
		for cnt > 0 && pattern[cnt] != v {
			cnt = pi[cnt-1]
		}
		if pattern[cnt] == v {
			cnt++
		}
		pi[i] = cnt
	}

	cnt = 0
	for i, v := range text {
		for cnt > 0 && pattern[cnt] != byte(v) {
			cnt = pi[cnt-1]
		}
		if pattern[cnt] == byte(v) {
			cnt++
		}
		if cnt == m {
			pos = append(pos, i-m+1)
			cnt = pi[cnt-1]
		}
	}
	return
}

// 在 text 中查找 pattern, 返回所有成功匹配位置(pattern 首字母的下标)
func kmpSearch(text, pattern string) (pos []int) {
	calcPi := func(s string) []int {
		pi := make([]int, len(s))
		for i, cnt := 1, 0; i < len(s); i++ {
			v := s[i]
			for cnt > 0 && s[cnt] != v {
				cnt = pi[cnt-1]
			}
			if s[cnt] == v {
				cnt++
			}
			pi[i] = cnt
		}
		return pi
	}
	pi := calcPi(pattern)
	cnt := 0
	for i, v := range text {
		for cnt > 0 && pattern[cnt] != byte(v) {
			cnt = pi[cnt-1]
		}
		if pattern[cnt] == byte(v) {
			cnt++
		}
		if cnt == len(pattern) {
			pos = append(pos, i-len(pattern)+1)
			cnt = pi[cnt-1]
		}
	}
	return
}
