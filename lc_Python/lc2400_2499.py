import bisect, collections, functools, math, itertools, heapq, string, operator, sortedcontainers
from typing import List, Optional

# 2400 - Number of Ways to Reach a Position After Exactly k Steps - MEDIUM
class Solution:
    # 有负数, 写记忆化, 数组麻烦
    def numberOfWays(self, startPos: int, endPos: int, k: int) -> int:
        d = abs(startPos - endPos)
        if d > k or (k - d) & 1:
            return 0
        mod = 10**9 + 7

        @functools.lru_cache(None)
        def f(p: int, k: int) -> int:
            """位置 p, 剩余步数 k, 从 p 往 0 走"""
            if abs(p) > k:
                return 0
            if k == 0 and p == 0:
                return 1
            return (f(p - 1, k - 1) + f(p + 1, k - 1)) % mod

        return f(d, k)

    def numberOfWays(self, startPos: int, endPos: int, k: int) -> int:
        d = abs(startPos - endPos)
        if d > k or d % 2 != k % 2:
            return 0
        mod = 10**9 + 7

        @functools.lru_cache(None)
        def f(p: int, k: int) -> int:
            """位置 p, 剩余步数 k"""
            if abs(p - endPos) > k:
                return 0
            if k == 0:
                return 1
            return (f(p - 1, k - 1) + f(p + 1, k - 1)) % mod

        return f(startPos, k)

    # 组合数学
    def numberOfWays(self, startPos: int, endPos: int, k: int) -> int:
        """
        向终点走 x 步, 向反方向走 k - x 步
        -> + x - (k - x) = d
        -> x = (d + k) // 2
        -> comb(k, x), k 步挑 x 步
        """
        d = abs(startPos - endPos)
        if d > k or (k - d) & 1:
            return 0
        mod = 10**9 + 7
        return math.comb(k, (d + k) // 2) % mod


# 2401 - Longest Nice Subarray - MEDIUM
class Solution:
    # O(930n) / O(n), 30n + 均摊复杂度, 每次移动一次 l 指针, 移动一次 900, 移动 n 次
    # 9.3e7 -> 1e8, 4s, 快 TLE 了
    def longestNiceSubarray(self, nums: List[int]) -> int:
        """
        双指针, 对每一位求 1 的位置, 某一位 1 有 2 个, 就移动左边指针
        我们需要选出最长的区间，使得区间中每个二进制位最多出现一个 1
        1 <= nums[i] <= 1e9, 30位
        """
        l = 0
        ans = 1
        cnt = collections.defaultdict(int)
        z = nums[0]
        i = 0
        while z:  # O(30)
            if z & 1:
                cnt[i] += 1
            i += 1
            z //= 2
        for r in range(1, len(nums)):  # O(n)
            z = nums[r]
            i = 0
            while z:  # O(30)
                if z & 1:
                    cnt[i] += 1
                i += 1
                z //= 2
            while l < len(nums) and any(v >= 2 for v in cnt.values()):  # O(30)
                z = nums[l]
                i = 0
                while z:  # O(30)
                    if z & 1:
                        cnt[i] -= 1
                    i += 1
                    z //= 2
                l += 1
            ans = max(ans, r - l + 1)
        return ans

    # python 位运算好慢
    def longestNiceSubarray(self, nums: List[int]) -> int:
        ans = l = 0
        cnt = collections.defaultdict(int)
        for r in range(len(nums)):  # O(n)
            for k in range(30):
                cnt[k] += nums[r] >> k & 1
            while l < len(nums) and any(v >= 2 for v in cnt.values()):  # O(30)
                for k in range(30):
                    cnt[k] -= nums[l] >> k & 1
                l += 1
            ans = max(ans, r - l + 1)
        return ans

    # O(n * log(max(nums))) -> O(30n) / O(1)
    def longestNiceSubarray(self, nums: List[int]) -> int:
        """
        由于所有元素对按位与均为 0, 在优雅子数组中的从低到高的第 i 个位上, 至多有一个 1, 其余均为 0
        因此在本题数据范围下, 优雅子数组的长度不会超过 30
        """
        ans = 0
        for i, v in enumerate(nums):
            j = i - 1
            while j >= 0 and (v & nums[j]) == 0:
                v |= nums[j]
                j -= 1
            ans = max(ans, i - j)
        return ans

    # O(n) / O(1)
    def longestNiceSubarray(self, nums: List[int]) -> int:
        """由于优雅子数组的所有元素按位与均为 0, 可以理解成这些二进制数对应的集合没有交集, 所以可以用 xor 把它去掉"""
        ans = l = orr = 0
        for r, v in enumerate(nums):
            while orr & v:
                orr ^= nums[l]
                l += 1
            orr |= v
            ans = max(ans, r - l + 1)
        return ans


# 2402 - Meeting Rooms III - HARD
class Solution:
    # O(n + m(logm + logn)) / O(n)
    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        meetings.sort()
        cnt = [0] * n
        idle = list(range(n))
        using = []
        for s, e in meetings:
            while using and using[0][0] <= s:
                _, i = heapq.heappop(using)
                heapq.heappush(idle, i)
            if idle:
                i = heapq.heappop(idle)
                heapq.heappush(using, (e, i))
            else:
                end, i = heapq.heappop(using)
                heapq.heappush(using, (end + e - s, i))
            cnt[i] += 1
        mx = max(cnt)
        for i, v in enumerate(cnt):
            if v == mx:
                return i
        return -1

    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        meetings.sort()
        cnt = [0] * n
        idle = list(range(n))
        using = []
        for s, e in meetings:
            while using and using[0][0] <= s:
                heapq.heappush(idle, heapq.heappop(using)[1])
            if len(idle) == 0:
                end, i = heapq.heappop(using)
                e += end - s
            else:
                i = heapq.heappop(idle)
            cnt[i] += 1
            heapq.heappush(using, (e, i))
        ans = 0
        for i, c in enumerate(cnt):
            if c > cnt[ans]:
                ans = i
        return ans

    # 数据范围小, n <= 100 , 暴力
    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        cnt = [0] * n
        t = [0] * n
        for s, e in sorted(meetings):
            t = list(map(lambda x: max(x, s), t))
            choice = t.index(min(t))
            t[choice] += e - s
            cnt[choice] += 1
        return cnt.index(max(cnt))
