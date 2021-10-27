package lc_Go

import "math"

// 492 - Construct the Rectangle - EASY
func constructRectangle(area int) (ans []int) {
	q := math.Sqrt(float64(area))
	intq := int(q)
	if q != float64(intq) {
		intq++
	}
	for intq != 0 {
		if area%intq == 0 {
			l := max(intq, area/intq)
			intq = area / l
			ans = append(ans, l, intq)
			return
		}
		intq--
	}
	return
}
func constructRectangle2(area int) []int {
	w := int(math.Sqrt(float64(area)))
	for area%w > 0 {
		w--
	}
	return []int{area / w, w}
}

// 496 - Next Greater Element I - EASY
// map. key: nums1's element, value: next greater element
func nextGreaterElement(nums1 []int, nums2 []int) (ans []int) {
	tmp := make(map[int]int, len(nums1))
	for _, v := range nums1 {
		tmp[v] = -1
	}
	for i := 0; i < len(nums2); i++ {
		if i == len(nums2)-1 {
			tmp[nums2[i]] = -1
		}
		if _, ok := tmp[nums2[i]]; ok {
			for j := i + 1; j < len(nums2); j++ {
				if nums2[j] > nums2[i] {
					tmp[nums2[i]] = nums2[j]
					break
				}
			}
		}
	}
	for i := range nums1 {
		ans = append(ans, tmp[nums1[i]])
	}
	return
}
