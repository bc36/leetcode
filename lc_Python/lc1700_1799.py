from typing import List
import heapq, collections, itertools, functools, math


# 1705 - Maximum Number of Eaten Apples - MEDIUM
class Solution:
    def eatenApples(self, apples: List[int], days: List[int]) -> int:
        pq, i, ans = [], 0, 0
        while i < len(apples) or pq:
            while pq and pq[0][0] <= i:
                heapq.heappop(pq)
            if i < len(apples) and apples[i]:
                heapq.heappush(pq, [i + days[i], apples[i]])
            if pq:
                pq[0][1] -= 1
                ans += 1
                if not pq[0][1]:
                    heapq.heappop(pq)
            i += 1
        return ans

    def eatenApples(self, apples: List[int], days: List[int]) -> int:
        pq, i, ans = [], 0, 0
        while i < len(apples):
            while pq and pq[0][0] <= i:
                heapq.heappop(pq)
            if apples[i]:
                heapq.heappush(pq, [i + days[i], apples[i]])
            if pq:
                pq[0][1] -= 1
                if not pq[0][1]:
                    heapq.heappop(pq)
                ans += 1
            i += 1
        while pq:
            cur = heapq.heappop(pq)
            d = min(cur[0] - i, cur[1])
            i += d
            ans += d
            while pq and pq[0][0] <= i:
                heapq.heappop(pq)
        return ans


# 1716 - Calculate Money in Leetcode Bank - EASY
class Solution:
    def totalMoney(self, n: int) -> int:
        div, mod = divmod(n, 7)
        ans = 0
        for i in range(mod):
            ans += i + 1 + div
        while div:
            ans += 28 + (div - 1) * 7
            div -= 1
        return ans

    def totalMoney(self, n: int) -> int:
        div, mod = divmod(n, 7)
        ans = 0
        for i in range(mod):
            ans += i + 1 + div
        ans += div * 28 + (div - 1) * div * 7 // 2 if div else 0
        return ans


# 1725 - Number Of Rectangles That Can Form The Largest Square - EASY
class Solution:
    def countGoodRectangles(self, rectangles: List[List[int]]) -> int:
        count = pre = 0
        for l, w in rectangles:
            sq = min(l, w)
            if sq > pre:
                count = 1
                pre = sq
            elif sq == pre:
                count += 1
        return count


# 1748 - Sum of Unique Elements - EASY
class Solution:
    def sumOfUnique(self, nums: List[int]) -> int:
        freq = [0] * 100
        for n in nums:
            freq[n - 1] += 1
        ans = 0
        for i in range(len(freq)):
            ans += i + 1 if freq[i] == 1 else 0
        return ans

    def sumOfUnique(self, nums: List[int]) -> int:
        return sum(k for k, v in collections.Counter(nums).items() if v == 1)


# 1762 - Buildings With an Ocean View - MEDIUM
class Solution:
    def findBuildings(self, heights: List[int]) -> List[int]:
        ans, maxH = [], 0
        for i in range(len(heights) - 1, -1, -1):
            if heights[i] > maxH:
                ans.append(i)
                maxH = heights[i]
        ans.reverse()
        return ans


# 1763 - Longest Nice Substring - EASY
class Solution:
    def longestNiceSubstring(self, s: str) -> str:
        if not s: return ""
        ss = set(s)
        for i, c in enumerate(s):
            if c.swapcase() not in ss:
                s0 = self.longestNiceSubstring(s[:i])
                s1 = self.longestNiceSubstring(s[i + 1:])
                return max(s0, s1, key=len)
        return s


# 1765 - Map of Highest Peak - MEDIUM
class Solution:
    def highestPeak(self, isWater: List[List[int]]) -> List[List[int]]:
        dq = collections.deque([])
        m, n = len(isWater), len(isWater[0])
        for i in range(m):
            for j in range(n):
                if isWater[i][j] == 1:
                    dq.append((i, j))
                isWater[i][j] -= 1
        while dq:
            x, y = dq.popleft()
            for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if 0 <= nx < m and 0 <= ny < n and isWater[nx][ny] == -1:
                    dq.append((nx, ny))
                    isWater[nx][ny] = isWater[x][y] + 1
        return isWater