"""Longest Increasing Subsequence (LIS)"""

import bisect
from typing import List


class LongestIncreasingSubsequence:
    def __init__(self):
        return

    @staticmethod
    def definitely_increase(nums: List[int]):
        """最长单调递增子序列(严格上升)"""
        dp = []
        for x in nums:
            i = bisect.bisect_left(dp, x)
            if 0 <= i < len(dp):
                dp[i] = x
            else:
                dp.append(x)
        return len(dp)

    @staticmethod
    def definitely_not_reduce(nums: List[int]):
        """最长单调不减子序列(不降)"""
        dp = []
        for x in nums:
            i = bisect.bisect_right(dp, x)
            if 0 <= i < len(dp):
                dp[i] = x
            else:
                dp.append(x)
        return len(dp)

    def definitely_reduce(self, nums: List[int]):
        """最长单调递减子序列(严格下降)"""
        nums = [-x for x in nums]
        return self.definitely_increase(nums)

    def definitely_not_increase(self, nums: List[int]):
        """最长单调不增子序列(不升)"""
        nums = [-x for x in nums]
        return self.definitely_not_reduce(nums)
