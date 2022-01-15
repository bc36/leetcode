from typing import List
import collections, heapq, functools, math, random


# https://leetcode-cn.com/problems/na-ying-bi/
# LCP 06. 拿硬币
class Solution:
    def minCount(self, coins: List[int]) -> int:
        ans = 0
        for c in coins:
            if c & 1:
                ans += c // 2 + 1
            else:
                ans += c // 2
        return ans

    def minCount(self, coins: List[int]) -> int:
        return sum([(x + 1) // 2 for x in coins])
