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
