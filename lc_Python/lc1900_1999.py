import collections
from typing import List


# 1995 - Count Special Quadruplets - EASY
class Solution:
    # brutal force: O(n^4) + O(1)
    def countQuadruplets(self, nums: List[int]) -> int:
        ans = 0
        for i in range(len(nums) - 3):
            for j in range(i + 1, len(nums) - 2):
                for k in range(j + 1, len(nums) - 1):
                    for l in range(k + 1, len(nums)):
                        if nums[i] + nums[j] + nums[k] == nums[l]:
                            ans += 1
        return ans

    # O(n^2) + O(min(n,C)^2): 'C' is the range of elements in the array nums
    def countQuadruplets(self, nums: List[int]) -> int:
        ans, cnt = 0, collections.Counter()  # a + b = d - c, save (d-c) in cnt
        for b in range(len(nums) - 3, 0, -1):
            for d in range(b + 2, len(nums)):
                cnt[nums[d] - nums[b + 1]] += 1
            for a in range(b):
                if (total := nums[a] + nums[b]) in cnt:
                    ans += cnt[total]
        return ans