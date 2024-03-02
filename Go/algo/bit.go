package algo

type fenwick struct {
	tree []int
}

func newFenwickTree(n int) fenwick {
	return fenwick{make([]int, n+1)}
}
func (f fenwick) add(i int, val int) {
	//i++
	for ; i < len(f.tree); i += i & -i {
		f.tree[i] += val
	}
}
func (f fenwick) sum(i int) (res int) {
	//i++
	for ; i > 0; i &= i - 1 {
		res += f.tree[i]
	}
	return
}
func (f fenwick) query(l, r int) (res int) { // [l,r]
	return f.sum(r) - f.sum(l-1)
}

a := fenwick{
	tree
}