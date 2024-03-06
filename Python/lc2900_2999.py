import bisect, collections, functools, heapq, itertools, math, operator, string, sys
from typing import List, Optional, Tuple
import sortedcontainers

# 2917 - Find the K-or of an Array - EASY
class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        ans = 0
        for i in range(31):
            cnt = 0
            for x in nums:
                cnt += 1 << i & x == 1 << i
            if cnt >= k:
                ans |= 1 << i
        return ans

    def findKOr(self, nums: List[int], k: int) -> int:
        ans = 0
        for i in range(31):
            ans |= (sum(x >> i & 1 for x in nums) >= k) << i
        return ans

    def findKOr(self, nums: List[int], k: int) -> int:
        return functools.reduce(
            operator.or_, ((sum(x >> i & 1 for x in nums) >= k) << i for i in range(31))
        )
