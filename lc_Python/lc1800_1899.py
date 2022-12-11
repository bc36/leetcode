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
        s = set()
        x = ""
        for c in word + "#":
            if c.isdigit():
                x += c
            else:
                if x:
                    s.add(int(x))
                x = ""
        return len(s)

    def numDifferentIntegers(self, word: str) -> int:
        return len(set(map(int, re.findall("\d+", word))))


# 1812 - Determine Color of a Chessboard Square - EASY
class Solution:
    def squareIsWhite(self, coordinates: str) -> bool:
        return (ord(coordinates[0]) - 97 + int(coordinates[1])) % 2 == 0


# 1813 - Sentence Similarity III - MEDIUM
class Solution:
    def areSentencesSimilar(self, sentence1: str, sentence2: str) -> bool:
        d1 = collections.deque(sentence1.split())
        d2 = collections.deque(sentence2.split())
        if len(d1) < len(d2):
            return self.areSentencesSimilar(sentence2, sentence1)
        while d2 and d1[0] == d2[0]:
            d1.popleft()
            d2.popleft()
        while d2 and d1[-1] == d2[-1]:
            d1.pop()
            d2.pop()
        return len(d2) == 0

    def areSentencesSimilar(self, sentence1: str, sentence2: str) -> bool:
        sentence1 = sentence1.split()
        sentence2 = sentence2.split()
        m = len(sentence1)
        n = len(sentence2)
        i = j = 0
        while i < m and i < n and sentence1[i] == sentence2[i]:
            i += 1
        while (
            m - 1 - j > -1
            and n - 1 - j > -1
            and sentence1[m - 1 - j] == sentence2[n - 1 - j]
        ):
            j += 1
        return i + j >= min(m, n)


# 1814 - Count Nice Pairs in an Array - MEDIUM
class Solution:
    def countNicePairs(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        cnt = collections.defaultdict(int)
        ans = 0
        for v in nums:
            rev = int(str(v)[::-1])
            ans = (ans + cnt[v - rev]) % mod
            cnt[v - rev] += 1
        return ans

    def countNicePairs(self, nums: List[int]) -> int:
        return sum(
            map(
                lambda x: x * (x - 1) // 2,
                collections.Counter([v - int(str(v)[::-1]) for v in nums]).values(),
            )
        ) % (10**9 + 7)


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


# 1827 - Minimum Operations to Make the Array Increasing - EASY
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        ans = 0
        for i in range(1, len(nums)):
            if nums[i - 1] >= nums[i]:
                ans += nums[i - 1] - nums[i] + 1
                nums[i] = nums[i - 1] + 1
        return ans

    def minOperations(self, nums: List[int]) -> int:
        ans = mx = 0
        for v in nums:
            ans += max(0, mx - v + 1)
            mx = max(mx + 1, v)
        return ans
