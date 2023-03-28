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


# https://leetcode.cn/contest/weekly-contest-326/problems/count-the-digits-that-divide-a-number/
class Solution:
    def solve(self, num: int) -> int:
        cnt = collections.defaultdict(int)
        x = num
        while x:
            cnt[x % 10] += 1
            x //= 10
        ans = 0
        for k, v in cnt.items():
            if num % k == 0:
                ans += v
        return ans


# print(Solution().solve())
# return


testcase = """
num = 7
num = 121
num = 1248
"""

obj = Solution()
for i, args in enumerate(parseInput(testcase)):
    print(f"\nTestcase {i}: {args}\n")
    print(obj.solve(*args))
