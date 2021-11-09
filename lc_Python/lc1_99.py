import bisect, collections, functools, random, operator, math
from typing import Iterable
'''
Function usually used

bit operation
&   bitwise AND
|   bitwise OR
^   bitwise XOR
~   bitwise NOT Inverts all the bits (~x = -x-1)
<<  left shift
>>  right shift
'''

# For viewing definitions
bisect.bisect_left()
collections.Counter(dict)
collections.deque(Iterable)
random.randint()
functools.reduce()
operator.xor()


# 3 - Longest Substring Without Repeating Characters - MEDIUM
# sliding window + hashmap
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        left, right, ans = 0, 0, 0
        dic = {}
        while right < len(s):
            if s[right] in dic and dic[s[right]] >= left:
                left = dic[s[right]] + 1
            dic[s[right]] = right
            right += 1
            ans = max(ans, right - left)
        return ans


# ord(), chr() / byte -> position
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        exist = [0 for _ in range(256)]
        left, right, ans = 0, 0, 0
        while right < len(s):
            if exist[ord(s[right]) - 97] == 0:
                exist[ord(s[right]) - 97] += 1
                right += 1
            else:
                exist[ord(s[right]) - 97] -= 1
                left += 1

            ans = max(ans, right - left)
        return ans


# 43 - Multiply Strings - MEDIUM
class Solution:
    def multiply(self, num1, num2):
        res = [0] * (len(num1) + len(num2))
        for i, e1 in enumerate(reversed(num1)):
            for j, e2 in enumerate(reversed(num2)):
                res[i + j] += int(e1) * int(e2)
                res[i + j + 1] += res[i + j] // 10
                res[i + j] %= 10
        # reverse, prepare to output
        while len(res) > 1 and res[-1] == 0:
            res.pop()
        return ''.join(map(str, res[::-1]))


# 62 - Unique Paths - MEDIUM
# dp[i][j] peresent the maximum value of paths that can reach this point
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
        dp = [[0 for _ in range(n)] for j in range(m)]
        # initialize
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]


# combination
# To make the machine get to the corner, the number of steps to the right and the number of steps to the left are fixed
# m - 1 down && n - 1 right -> m + n - 1 times movement
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        return math.comb(m + n - 2, n - 1)


# 96 - Unique Binary Search Trees - MEDIUM
