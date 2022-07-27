import bisect, functools, collections, itertools, math
from typing import List

# 1108 - Defanging an IP Address - EASY
class Solution:
    def defangIPaddr(self, address: str) -> str:
        return address.replace(".", "[.]")


# 1122 - Relative Sort Array - EASY
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        s = set(arr1)
        cnt = collections.Counter(arr1)
        ans = []
        for v in arr2:
            if v in s:
                ans.extend([v] * cnt[v])
                del cnt[v]
        for k, v in sorted(cnt.items()):
            ans.extend([k] * v)
        return ans

    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        freq = [0] * 1001
        for v in arr1:
            freq[v] += 1
        ans = []
        for v in arr2:
            ans.extend([v] * freq[v])
            freq[v] = 0
        for v in range(1001):
            if freq[v] > 0:
                ans.extend([v] * freq[v])
        return ans


# 1128 - Number of Equivalent Domino Pairs - EASY
class Solution:
    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        cnt = collections.Counter(tuple(sorted(d)) for d in dominoes)
        return sum(v * (v - 1) // 2 for v in cnt.values())

    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        cnt = [0] * 100
        ans = 0
        for x, y in dominoes:
            v = (x * 10 + y if x <= y else y * 10 + x)
            ans += cnt[v]
            cnt[v] += 1
        return ans


# 1137 - N-th Tribonacci Number - EASY
class Solution:
    @functools.lru_cache(None)
    def tribonacci(self, n: int) -> int:
        if n == 0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 1
        return self.tribonacci(n - 1) + self.tribonacci(n - 2) + self.tribonacci(n - 3)


class Solution:
    def __init__(self):
        self.cache = {0: 0, 1: 1, 2: 1}

    def tribonacci(self, n: int) -> int:
        if n in self.cache:
            return self.cache[n]
        self.cache[n] = (
            self.tribonacci(n - 1) + self.tribonacci(n - 2) + self.tribonacci(n - 3)
        )
        return self.cache[n]


class Solution:
    def tribonacci(self, n: int) -> int:
        if n == 0:
            return 0
        elif n < 3:
            return 1
        one, two, three, ans = 0, 1, 1, 0
        for _ in range(2, n):
            ans = one + two + three
            one, two, three = two, three, ans
        return ans


# 1143 - Longest Common Subsequence - MEDIUM
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        len1, len2 = len(text1) + 1, len(text2) + 1
        dp = [[0 for _ in range(len2)] for _ in range(len1)]
        for i in range(1, len1):
            for j in range(1, len2):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]


# 1146 - Snapshot Array - MEDIUM
# only update the change of each element, rather than record the whole arr
class SnapshotArray:
    def __init__(self, length: int):
        self.arr = [{0: 0} for _ in range(length)]
        self.snap_id = 0

    def set(self, index: int, val: int) -> None:
        self.arr[index][self.snap_id] = val
        return

    def snap(self) -> int:
        self.snap_id += 1
        return self.snap_id - 1

    def get(self, index: int, snap_id: int) -> int:
        d = self.arr[index]
        if snap_id in d:
            return d[snap_id]
        k = list(d.keys())
        i = bisect.bisect_left(k, snap_id)
        return d[k[i - 1]]


# 1154 - Day of the Year - EASY
class Solution:
    def dayOfYear(self, date: str) -> int:
        m = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        year, month, day = [int(x) for x in date.split("-")]
        if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):
            m[1] += 1
        return sum(m[: month - 1]) + day


# 1175 - Prime Arrangements - EASY
class Solution:
    def numPrimeArrangements(self, n: int) -> int:
        def isPrime(n: int) -> int:
            if n == 1:
                return 0
            for i in range(2, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    return 0
            return 1

        def factorial(n: int) -> int:
            res = 1
            for i in range(1, n + 1):
                res *= i
                res %= mod
            return res

        mod = 10**9 + 7
        primes = sum(isPrime(i) for i in range(1, n + 1))
        return factorial(primes) * factorial(n - primes) % mod


# 1178 - Number of Valid Words for Each Puzzle - HARD
class Solution:
    # TLE
    def findNumOfValidWords(self, words: List[str], puzzles: List[str]) -> List[int]:
        puzzleSet = [set(p) for p in puzzles]
        wordSet = [set(w) for w in words]
        # firstLetters = set([p[0] for p in puzzles])
        ans = []
        for i, puzzle in enumerate(puzzles):
            num = 0
            for j in range(len(words)):
                # contain the first letter of puzzle
                if puzzle[0] in wordSet[j]:
                    # every letter is in puzzle
                    if wordSet[j] <= puzzleSet[i]:
                        num += 1
            ans.append(num)

        return ans


# 1184 - Distance Between Bus Stops - EASY
class Solution:
    def distanceBetweenBusStops(
        self, distance: List[int], start: int, destination: int
    ) -> int:
        a = min(destination, start)
        b = max(destination, start)
        return min(sum(distance[a:b]), sum(distance[b:] + distance[:a]))


# 1189 - Maximum Number of Balloons - EASY
class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        cnt = collections.Counter(text)
        return min(cnt["b"], cnt["a"], cnt["l"] // 2, cnt["o"] // 2, cnt["n"])
