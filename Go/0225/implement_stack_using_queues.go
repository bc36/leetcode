package main

// Easy
// https://leetcode.cn/problems/implement-stack-using-queues/
// https://leetcode.com/problems/implement-stack-using-queues/
type MyStack struct {
	a, b []int
}

func Constructor() (s MyStack) {
	return
}

func (s *MyStack) Push(x int) {
	s.b = append(s.b, x)
	for len(s.a) > 0 {
		s.b = append(s.b, s.a[0])
		s.a = s.a[1:]
	}
	s.a, s.b = s.b, s.a
}

func (s *MyStack) Pop() int {
	v := s.a[0]
	s.a = s.a[1:]
	return v
}

func (s *MyStack) Top() int {
	return s.a[0]
}

func (s *MyStack) Empty() bool {
	return len(s.a) == 0
}
