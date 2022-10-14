import collections, itertools, functools, math, re
from typing import List

# 1800 - Maximum Ascending Subarray Sum - EASY
class Solution:
    def maxAscendingSum(self, nums: List[int]) -> int:
        ans = cur = p = 0
        for v in nums:
            if p < v:
                cur += v
            else:
                cur = v
            p = v
            ans = max(ans, cur)
        return ans


# 1805 - Number of Different Integers in a String - EASY
class Solution:
    def numDifferentIntegers(self, word: str) -> int:
        word = word + "a"
        s = set()
        num = ""
        for c in word:
            if c.isdigit():
                num += c
            else:
                if num != "":
                    s.add(int(num))
                num = ""
        return len(s)

    def numDifferentIntegers(self, word: str) -> int:
        return len(set(map(int, re.findall("\d+", word))))


# 1816 - Truncate Sentence - EASY
class Solution:
    def truncateSentence(self, s: str, k: int) -> str:
        return " ".join(s.split()[:k])


# 1822 - Sign of the Product of an Array - EASY
class Solution:
    def arraySign(self, nums: List[int]) -> int:
        a = 0
        for n in nums:
            if n == 0:
                return 0
            a += 1 if n < 0 else 0
        return -1 if a & 1 else 1

    def arraySign(self, nums: List[int]) -> int:
        a = 1
        for n in nums:
            if n == 0:
                return 0
            a *= 1 if n > 0 else -1
        return a


# 1823 - Find the Winner of the Circular Game - MEDIUM
class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        nxt = (0,)
        arr = [i for i in range(1, n + 1)]
        while len(arr) > 1:
            lost = (nxt + k - 1) % len(arr)
            nxt = lost if (lost != len(arr) - 1) else 0
            del arr[lost]
        return arr[0]

    # Josephus problem
    def findTheWinner(self, n: int, k: int) -> int:
        p = 0
        for i in range(2, n + 1):
            p = (p + k) % i
        return p + 1

    def findTheWinner(self, n: int, k: int) -> int:
        f = [0] * (n + 1)
        f[1] = 1
        for i in range(2, n + 1):
            f[i] = (f[i - 1] + k - 1) % i + 1
        return f[n]

    def findTheWinner(self, n: int, k: int) -> int:
        dp = 1
        for i in range(2, n + 1):
            dp = (dp + k - 1) % i + 1
        return dp
