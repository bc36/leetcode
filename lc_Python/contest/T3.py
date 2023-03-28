import bisect, collections, copy, functools, heapq, itertools, math, random, string
from operator import xor
from typing import List, Optional, Tuple
from heapq import heappushpop, heapreplace

from sortedcontainers import SortedList, SortedDict, SortedSet

from test_tool import ListNode, TreeNode, null, tree2array, parseInput


"""
Some hint

10**6   = 1000000
1 << 20 = 1048576
10**9   = 1000000000
1 << 30 = 1073741824
"""


# https://leetcode.cn/contest/weekly-contest-328/problems/difference-between-maximum-and-minimum-price-sum/
class Solution:
    def solve(self, n: int, edges: List[List[int]], price: List[int]) -> int:
        g = [[] for _ in range(n)]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)
        ans = 0

        def dfs(x: int, fa: int) -> Tuple[int, int]:  # 返回带叶子的最大路径和, 不带叶子的最大路径和
            nonlocal ans
            mx_s1 = p = price[x]
            mx_s2 = 0
            for y in g[x]:
                if y == fa:
                    continue
                s1, s2 = dfs(y, x)
                # 已遍历过的最大带叶子的路径和(s1) + 当前不带叶子的路径和
                # 已遍历过的最大不带叶子的路径和(s2) + 当前带叶子的路径和
                ans = max(ans, mx_s1 + s2, mx_s2 + s1)
                mx_s1 = max(mx_s1, s1 + p)
                mx_s2 = max(mx_s2, s2 + p)  # 这里加上 p 是因为 x 必然不是叶子
            return mx_s1, mx_s2

        dfs(0, -1)
        return ans


# print(Solution().solve())
# return

testcase = """
n = 6, edges = [[0,1],[1,2],[1,3],[3,4],[3,5]], price = [9,8,7,6,10,5]
n = 3, edges = [[0,1],[1,2]], price = [1,1,1]
"""

obj = Solution()
for i, args in enumerate(parseInput(testcase)):
    print(f"\nTestcase {i}: {args}\n")
    print(obj.solve(*args))
