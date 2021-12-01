from sys import int_info
from typing import List
import collections

# 1428 - Leftmost Column with at Least a One - MEDIUM


# 1446 - Consecutive Characters - EASY
class Solution:
    def maxPower(self, s: str) -> int:
        tmp, ans = 1, 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                tmp += 1
                ans = max(tmp, ans)
            else:
                tmp = 1
        return ans

class Solution:
    def maxPower(self, s: str) -> int:
        i, ans = 0, 1
        while i < len(s):
            j = i
            while j < len(s) and s[i] == s[j]:
                j += 1
            ans = max(ans, j - i)
            i = j
        return ans
