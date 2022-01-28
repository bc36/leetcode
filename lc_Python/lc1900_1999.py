import collections, bisect, itertools, functools, math, heapq
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


# 1996 - The Number of Weak Characters in the Game - MEDIUM
class Solution:
    def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
        properties.sort(key=lambda x: (-x[0], x[1]))
        ans = 0
        maxDf = 0
        for _, df in properties:
            if maxDf > df:
                ans += 1
            else:
                maxDf = max(maxDf, df)
        return ans

    def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
        properties.sort(key=lambda x: (x[0], -x[1]))
        stack = []
        ans = 0
        for _, d in properties:
            while stack and stack[-1] < d:
                stack.pop()
                ans += 1
            stack.append(d)
        return ans