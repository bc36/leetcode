import collections
from typing import List


# 2006 - Count Number of Pairs With Absolute Difference K - EASY
class Solution:
    def countKDifference(self, nums: List[int], k: int) -> int:
        seen, counter = collections.defaultdict(int), 0
        for num in nums:
            counter += seen[num - k] + seen[num + k]
            seen[num] += 1
        return counter


# 2022 - Convert 1D Array Into 2D Array - EASY
class Solution:
    def construct2DArray(self, original: List[int], m: int,
                         n: int) -> List[List[int]]:
        if len(original) != m * n: return []
        ans = []
        for i in range(0, len(original), n):
            ans.append([x for x in original[i:i + n]])
        return ans

    def construct2DArray(self, original: List[int], m: int,
                         n: int) -> List[List[int]]:
        return [original[i:i + n] for i in range(0, len(original), n)
                ] if len(original) == m * n else []


# 2029 - Stone Game IX - MEDIUM
class Solution:
    def stoneGameIX(self, stones: List[int]) -> bool:
        d = [0, 0, 0]
        for n in stones:
            d[n % 3] += 1
        if d[0] % 2 == 0:
            return d[1] != 0 and d[2] != 0
        return d[2] > d[1] + 2 or d[1] > d[2] + 2