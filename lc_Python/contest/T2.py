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


# https://leetcode.cn/contest/weekly-contest-329/problems/apply-bitwise-operations-to-make-strings-equal/
class Solution:
    def solve(self, s: str, target: str) -> bool:
        a = collections.Counter(s)
        b = collections.Counter(target)
        if b["0"] == 0:  # 全是 1
            return a["1"] > 0
        if b["1"] == 0:  # 全是 0
            return a["1"] == 0
        return a["1"] > 0


# print(Solution().solve())
# return


testcase = """
s = "1010", target = "0110"
s = "11", target = "00"
"""

obj = Solution()
for i, args in enumerate(parseInput(testcase)):
    print(f"Testcase {i}: ")
    print(args)
    print(obj.solve(*args))
