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


# 1706 - Where Will the Ball Fall - MEDIUM
class Solution:
    def findBall(self, grid: List[List[int]]) -> List[int]:
        ans = []
        m, n = len(grid), len(grid[0])
        for b in range(n):
            i = 0
            j = b
            succ = True
            while i < m:
                if grid[i][j] == 1 and j + 1 < n and grid[i][j + 1] == 1:
                    j += 1
                    i += 1
                elif grid[i][j] == -1 and j - 1 >= 0 and grid[i][j - 1] == -1:
                    j -= 1
                    i += 1
                else:
                    succ = False
                    break
            if succ:
                ans.append(j)
            else:
                ans.append(-1)
        return ans

    # for-else
    def findBall(self, grid: List[List[int]]) -> List[int]:
        n = len(grid[0])
        ans = [-1] * n
        for j in range(n):
            col = j
            for row in grid:
                dir = row[col]
                col += dir
                if col < 0 or col == n or row[col] != dir:
                    break
            else:
                ans[j] = col
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


# 1768 - Merge Strings Alternately - EASY
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        return ''.join(
            a + b
            for a, b in itertools.zip_longest(word1, word2, fillvalue=''))


# 1779 - Find Nearest Point That Has the Same X or Y Coordinate - EASY
class Solution:
    def nearestValidPoint(self, x: int, y: int,
                          points: List[List[int]]) -> int:
        ans = -1
        mi = float('inf')
        for i, (xx, yy) in enumerate(points):
            if xx == x or yy == y:
                d = abs(xx - x) + abs(yy - y)
                if d < mi:
                    ans = i
                    mi = d
        return ans


# 1790 - Check if One String Swap Can Make Strings Equal - EASY
class Solution:
    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        cnt1 = collections.Counter(s1)
        cnt2 = collections.Counter(s2)
        if cnt1 != cnt2:
            return False
        cnt = 0
        for c1, c2 in zip(s1, s2):
            if c1 != c2:
                cnt += 1
        if cnt == 0 or cnt == 2:
            return True
        else:
            return False


# 1791 - Find Center of Star Graph - EASY
class Solution:
    def findCenter(self, edges: List[List[int]]) -> int:
        n = len(edges)
        d = [0] * (n + 1)
        for o, i in edges:
            d[o - 1] += 1
            d[i - 1] += 1
        for i in range(len(d)):
            if d[i] == n:
                return i + 1
        return -1

    def findCenter(self, edges: List[List[int]]) -> int:
        if edges[0][0] in edges[1]:
            return edges[0][0]
        else:
            return edges[0][1]