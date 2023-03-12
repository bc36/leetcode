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


# https://leetcode.cn/contest/weekly-contest-329/problems/sort-the-students-by-their-kth-score/
class Solution:
    def solve(self, score: List[List[int]], k: int) -> List[List[int]]:
        return sorted(score, key=lambda x: x[k], reverse=True)


# print(Solution().solve())
# return


testcase = """
score = [[10,6,9,1],[7,5,11,2],[4,8,3,15]], k = 2
score = [[3,4],[5,6]], k = 0
"""

obj = Solution()
for i, args in enumerate(parseInput(testcase)):
    print(f"Testcase {i}: ")
    print(obj.solve(*args))
