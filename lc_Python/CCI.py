from typing import List

"""
Cracking the Coding Interview
"""


# https://leetcode.cn/problems/missing-two-lcci/
# 面试题 17.19. 消失的两个数字 - HARD
class Solution:
    # O(n) / O(n), 原地哈希, 改变出现的值所映射下标的符号, 作为标记
    def missingTwo(self, nums: List[int]) -> List[int]:
        n = len(nums) + 2
        nums.extend([1] * 2)
        for i in range(n - 2):
            nums[abs(nums[i]) - 1] *= -1
        return [i + 1 for i in range(n) if nums[i] > 0]
