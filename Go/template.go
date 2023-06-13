package lc_Go

// set
func set() {
	s := map[int]bool{5: true, 2: true}
	if _, ok := s[6]; ok { // check for existence
		s[8] = true  // add element
		delete(s, 2) // remove element
	}
	s1 := map[int]bool{5: true, 2: true}
	s2 := map[int]bool{5: true, 2: true}
	// union
	s_union := map[int]bool{}
	for k := range s1 {
		s_union[k] = true
	}
	for k := range s2 {
		s_union[k] = true
	}
	// intersection
	s_intersection := map[int]bool{}
	if len(s1) > len(s2) {
		s1, s2 = s2, s1 // better to iterate over a shorter set
	}
	for k := range s1 {
		if s2[k] {
			s_intersection[k] = true
		}
	}
}

// union find / disjoint set
// 并查集模板（哈希表写法）
func uf(member []int) {
	fa := map[int]int{}
	size := map[int]int{}
	var find func(int) int
	find = func(x int) int {
		if fa[x] != x {
			fa[x] = find(fa[x])
		}
		return fa[x]
	}
	groups, maxSize := len(member), 0
	merge := func(x, y int) {
		if _, ok := fa[y]; !ok {
			return
		}
		x, y = find(x), find(y)
		if x == y {
			return
		}
		fa[x] = y
		size[y] += size[x]
		maxSize = max(maxSize, size[y]) // 维护答案
		groups--
	}
	merge(member[0], member[1])
}
