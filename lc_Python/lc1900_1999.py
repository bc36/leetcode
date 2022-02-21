import collections, bisect, itertools, functools, math, heapq
from typing import List


# 1984 - Minimum Difference Between Highest and Lowest of K Scores - EASY
class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        ans = math.inf
        nums.sort()
        for i in range(len(nums) - k + 1):
            ans = min(ans, nums[i + k - 1] - nums[i])
        return ans


# 1994 - The Number of Good Subsets - HARD
class Solution:
    def numberOfGoodSubsets(self, nums: List[int]) -> int:
        cnt, mod = collections.Counter(nums), 10**9 + 7
        d = collections.defaultdict(int)
        d[1] = (1 << cnt[1]) % mod
        for p in [
                2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29,
                30
        ]:
            for x in list(d):
                # RuntimeError: dictionary changed size during iteration
                #     for x in d:
                if math.gcd(p, x) == 1:
                    d[p * x] += cnt[p] * d[x]
                    d[p * x] %= mod
        return (sum(d.values()) - d[1]) % mod

    def numberOfGoodSubsets(self, nums: List[int]) -> int:
        cnt = collections.Counter(nums)
        mod = 10**9 + 7
        prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        m = len(prime)
        dp = [0] * (1 << m)
        dp[0] = 1
        for num in cnt:
            if num == 1: continue
            if any(num % (p * p) == 0 for p in prime):
                continue
            mask = 0
            for i in range(m):
                if num % prime[i] == 0:
                    mask |= 1 << i
            for state in range(1 << m):
                if state & mask == 0:
                    dp[state | mask] += dp[state] * cnt[num]
                    dp[state | mask] %= mod
        return (1 << cnt[1]) * (sum(dp) - 1) % mod

    # O(2^10 * 30)
    def numberOfGoodSubsets(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        dp = [1] + [0] * (1 << 10)
        cnt = collections.Counter(nums)
        for n in cnt:
            if n == 1: continue
            if n % 4 == 0 or n % 9 == 0 or n == 25: continue
            # mask == a set of primes where primes[i] is included if bitmask[i] == 1
            mask = sum(1 << i for i, p in enumerate(primes) if n % p == 0)
            for i in range(1 << 10):
                if i & mask: continue
                dp[i | mask] = (dp[i | mask] + cnt[n] * dp[i]) % mod
        return (1 << cnt[1]) * (sum(dp) - 1) % mod

    def numberOfGoodSubsets(self, nums: List[int]) -> int:
        p = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        cnt = collections.Counter(nums)
        bm = [
            sum(1 << i for i, p in enumerate(p) if x % p == 0)
            for x in range(31)
        ]
        bad = set([4, 8, 9, 12, 16, 18, 20, 24, 25, 27, 28])
        m = 10**9 + 7

        @functools.lru_cache(None)
        def dp(mask, num):
            if num == 1: return 1
            ans = dp(mask, num - 1)
            if num not in bad and mask | bm[num] == mask:
                ans += dp(mask ^ bm[num], num - 1) * cnt[num]
            return ans % m

        return ((dp(1023, 30) - 1) * pow(2, cnt[1], m)) % m

    # enumeration
    def numberOfGoodSubsets(self, nums: List[int]) -> int:
        cnt = collections.Counter(nums)
        mod = 10**9 + 7
        ans = 0
        # prime factor
        a, p = 1, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        for i in p:
            a = a * (cnt[i] + 1) % mod
        ans = (ans + a + mod - 1) % mod
        # composite number 6 10 14 15 21 22 26 30
        # 6
        a, p = 1, [5, 7, 11, 13, 17, 19, 23, 29]
        for i in p:
            a = a * (cnt[i] + 1) % mod
        ans = (ans + a * cnt[6]) % mod
        # 10
        a, p = 1, [3, 7, 11, 13, 17, 19, 23, 29]
        for i in p:
            a = a * (cnt[i] + 1) % mod
        ans = (ans + a * cnt[10]) % mod
        # 14
        a, p = 1, [3, 5, 11, 13, 17, 19, 23, 29]
        for i in p:
            a = a * (cnt[i] + 1) % mod
        ans = (ans + a * cnt[14]) % mod
        # 15
        a, p = 1, [2, 7, 11, 13, 17, 19, 23, 29]
        for i in p:
            a = a * (cnt[i] + 1) % mod
        ans = (ans + a * cnt[15]) % mod
        # 21
        a, p = 1, [2, 5, 11, 13, 17, 19, 23, 29]
        for i in p:
            a = a * (cnt[i] + 1) % mod
        ans = (ans + a * cnt[21]) % mod
        # 22
        a, p = 1, [3, 5, 7, 13, 17, 19, 23, 29]
        for i in p:
            a = a * (cnt[i] + 1) % mod
        ans = (ans + a * cnt[22]) % mod
        # 26
        a, p = 1, [3, 5, 7, 11, 17, 19, 23, 29]
        for i in p:
            a = a * (cnt[i] + 1) % mod
        ans = (ans + a * cnt[26]) % mod
        # 30
        a, p = 1, [7, 11, 13, 17, 19, 23, 29]
        for i in p:
            a = a * (cnt[i] + 1) % mod
        ans = (ans + a * cnt[30]) % mod
        # composite number combination 10 21 / 14 15 / 15 22 / 15 26 / 21 22 / 21 26
        # 10 21
        a, p = 1, [11, 13, 17, 19, 23, 29]
        for i in p:
            a = a * (cnt[i] + 1) % mod
        a = a * cnt[10] * cnt[21] % mod
        ans = (ans + a) % mod
        # 14 15
        a, p = 1, [11, 13, 17, 19, 23, 29]
        for i in p:
            a = a * (cnt[i] + 1) % mod
        a = a * cnt[14] * cnt[15] % mod
        ans = (ans + a) % mod
        # 15 22
        a, p = 1, [7, 13, 17, 19, 23, 29]
        for i in p:
            a = a * (cnt[i] + 1) % mod
        a = a * cnt[15] * cnt[22] % mod
        ans = (ans + a) % mod
        # 15 26
        a, p = 1, [7, 11, 17, 19, 23, 29]
        for i in p:
            a = a * (cnt[i] + 1) % mod
        a = a * cnt[15] * cnt[26] % mod
        ans = (ans + a) % mod
        # 21 22
        a, p = 1, [5, 13, 17, 19, 23, 29]
        for i in p:
            a = a * (cnt[i] + 1) % mod
        a = a * cnt[21] * cnt[22] % mod
        ans = (ans + a) % mod
        # 21 26
        a, p = 1, [5, 11, 17, 19, 23, 29]
        for i in p:
            a = a * (cnt[i] + 1) % mod
        a = a * cnt[21] * cnt[26] % mod
        ans = (ans + a) % mod

        for i in range(cnt[1]):
            ans = 2 * ans % mod
        return ans


# 1995 - Count Special Quadruplets - EASY
class Solution:
    # brutal force: O(n^4) + O(1)
    def countQuadruplets(self, nums: List[int]) -> int:
        ans = 0
        for i in range(len(nums) - 3):
            for j in range(i + 1, len(nums) - 2):
                for k in range(j + 1, len(nums) - 1):
                    for l in range(k + 1, len(nums)):
                        if nums[i] + nums[j] + nums[k] == nums[l]:
                            ans += 1
        return ans

    # O(n^2) + O(min(n,C)^2): 'C' is the range of elements in the array nums
    def countQuadruplets(self, nums: List[int]) -> int:
        ans, cnt = 0, collections.Counter()  # a + b = d - c, save (d-c) in cnt
        for b in range(len(nums) - 3, 0, -1):
            for d in range(b + 2, len(nums)):
                cnt[nums[d] - nums[b + 1]] += 1
            for a in range(b):
                if (total := nums[a] + nums[b]) in cnt:
                    ans += cnt[total]
        return ans


# 1996 - The Number of Weak Characters in the Game - MEDIUM
class Solution:
    def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
        properties.sort(key=lambda x: (-x[0], x[1]))
        ans = 0
        maxDf = 0
        for _, df in properties:
            if maxDf > df:
                ans += 1
            else:
                maxDf = max(maxDf, df)
        return ans

    def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
        properties.sort(key=lambda x: (x[0], -x[1]))
        stack = []
        ans = 0
        for _, d in properties:
            while stack and stack[-1] < d:
                stack.pop()
                ans += 1
            stack.append(d)
        return ans