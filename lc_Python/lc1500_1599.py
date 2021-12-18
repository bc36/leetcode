from typing import List


# 1518 - Water Bottles - EASY
class Solution:
    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        ans, empty = 0, 0
        while numBottles:
            ans += numBottles
            numBottles, empty = divmod(empty + numBottles, numExchange)
        return ans


# 1570 - Dot Product of Two Sparse Vectors - MEDIUM
class SparseVector:
    def __init__(self, nums: List[int]):
        self.nums = {k: num for k, num in enumerate(nums) if num != 0}

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        ans = 0
        for key, value in self.nums.items():
            if key in vec.nums:
                ans += value * vec.nums[key]
        return ans


# Your SparseVector object will be instantiated and called as such:
# v1 = SparseVector(nums1)
# v2 = SparseVector(nums2)
# ans = v1.dotProduct(v2)