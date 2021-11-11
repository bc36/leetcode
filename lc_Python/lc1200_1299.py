from operator import le
from typing import List
import collections


# 1218 - Longest Arithmetic Subsequence of Given Difference - MEDIUM
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        dp = collections.defaultdict(int)
        for i in arr:
            dp[i] = dp[i - difference] + 1
        return max(dp.values())


# 1249 - Minimum Remove to Make Valid Parentheses - MEDUIM
# left must less than right
# make the invalid parentheses as special character "*"
# remove the extra "(" and "*"
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        s = list(s)
        left = 0
        for i in range(len(s)):
            if s[i] == ")":
                if left <= 0:
                    s[i] = "*"
                else:
                    left -= 1
            if s[i] == "(":
                left += 1
        # remove "*"
        for i in range(len(s)):
            if s[i] == "*":
                s[i] = ""
        # remove extra "("
        i = len(s) - 1
        while left > 0 and i >= 0:
            if s[i] == "(":
                s[i] = ""
                left -= 1
            i -= 1
        return "".join(s)


# better
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        left = []
        s = list(s)
        for i in range(0, len(s)):
            # record the index of each "("
            if s[i] == '(':
                left.append(i)
            elif s[i] == ')':
                # no "(" to be paired, remove invalid ")"
                if not left:
                    s[i] = ''
                # has extra "(" to be paired, remove one "("
                # pop will remove the rightmost "("
                else:
                    left.pop()
        for i in left:
            s[i] = ''
        return ''.join(s)


# better
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        arr = list(s)
        stack = []
        for i, ch in enumerate(s):
            if ch == "(":
                stack.append(i)
            elif ch == ')':
                if stack:
                    stack.pop()
                else:
                    arr[i] = ""
        while stack:
            arr[stack.pop()] = ""
        return ''.join(arr)
