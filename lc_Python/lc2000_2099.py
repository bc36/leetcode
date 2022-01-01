import collections
from typing import List


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
