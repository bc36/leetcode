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


class Solution:
    def solve(self) -> None:
        return


# print(Solution().solve())
# return


testcase = """
"""

obj = Solution()
for i, args in enumerate(parseInput(testcase)):
    print(f"Testcase {i}: ")
    print(obj.solve(*args))
