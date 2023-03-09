import bisect, collections, functools, heapq, itertools, math, operator, string
from typing import List, Optional, Tuple
import sortedcontainers


# https://leetcode.cn/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/
# 剑指 Offer 09. 用两个栈实现队列 - EASY
class CQueue:
    def __init__(self):
        self.inn = []
        self.out = []

    def appendTail(self, value: int) -> None:
        self.inn.append(value)
        return

    def deleteHead(self) -> int:
        if not self.out:
            if not self.inn:
                return -1
            while self.inn:
                self.out.append(self.inn.pop())
        return self.out.pop()


# https://leetcode.cn/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/
# 剑指 Offer 10. II. 青蛙跳台阶问题 - EASY
class Solution:
    def numWays(self, n: int) -> int:
        a = b = 1
        mod = 10**9 + 7
        for _ in range(n):
            a, b = b, (a + b) % mod
        return a


# 计算放在 global 加速
f = [0] * 101
f[0] = 1
f[1] = 1
mod = 10**9 + 7
for i in range(2, 101):
    f[i] = (f[i - 1] + f[i - 2]) % mod


class Solution:
    def numWays(self, n: int) -> int:
        return f[n]


# https://leetcode.cn/problems/li-wu-de-zui-da-jie-zhi-lcof/
# 剑指 Offer 47. 礼物的最大价值 - MEDIUM
class Solution:
    # O(mn) / O(mn)
    def maxValue(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        @functools.lru_cache(None)
        def dfs(i: int, j: int) -> int:
            if i < 0 or j < 0:
                return 0
            return max(dfs(i - 1, j), dfs(i, j - 1)) + grid[i][j]

        return dfs(m - 1, n - 1)

    def maxValue(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        f = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                f[i][j] = max(f[i - 1][j], f[i][j - 1]) + grid[i - 1][j - 1]
        return f[m][n]

    # O(mn) / O(n)
    def maxValue(self, grid: List[List[int]]) -> int:
        n = len(grid[0])
        f = [0] * (n + 1)
        for row in grid:
            for j, v in enumerate(row):
                f[j + 1] = max(f[j], f[j + 1]) + v
        return f[n]
