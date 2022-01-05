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

    def dotProduct(self, vec: 'SparseVector') -> int:
        ans = 0
        for key, value in self.nums.items():
            if key in vec.nums:
                ans += value * vec.nums[key]
        return ans


# 1576 - Replace All ?'s to Avoid Consecutive Repeating Characters - EASY
class Solution:
    def modifyString(self, s: str) -> str:
        alpha = 'abcdefghijklmnopqrstuvwxyz'
        s = ['#'] + list(s) + ['#']
        for i in range(1, len(s) - 1):
            if s[i] == '?':
                new = i
                while s[i - 1] == alpha[new % 26] or s[i + 1] == alpha[new %
                                                                       26]:
                    new += 1
                s[i] = alpha[new % 26]
        return ''.join(s[1:-1])

    def modifyString(self, s: str) -> str:
        alpha = 'abc'
        s = ['#'] + list(s) + ['#']
        for i in range(1, len(s) - 1):
            if s[i] == '?':
                for ch in alpha:
                    if s[i - 1] != ch and s[i + 1] != ch:
                        s[i] = ch
                        break
        return ''.join(s[1:-1])