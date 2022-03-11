import collections, bisect, itertools, functools, math, heapq, re
from typing import List


# 2000 - Reverse Prefix of Word - EASY
class Solution:
    def reversePrefix(self, word: str, ch: str) -> str:
        idx = -1
        for i in range(len(word)):
            if word[i] == ch:
                idx = i
                break
        return word[:idx + 1][::-1] + word[idx + 1:]

    def reversePrefix(self, word: str, ch: str) -> str:
        i = word.find(ch) + 1
        return word[:i][::-1] + word[i:]


# 2006 - Count Number of Pairs With Absolute Difference K - EASY
class Solution:
    def countKDifference(self, nums: List[int], k: int) -> int:
        seen, counter = collections.defaultdict(int), 0
        for num in nums:
            counter += seen[num - k] + seen[num + k]
            seen[num] += 1
        return counter

    def countKDifference(self, nums: List[int], k: int) -> int:
        ans, dic = 0, {}
        for n in nums:
            if n - k in dic:
                ans += dic[n - k]
            if n + k in dic:
                ans += dic[n + k]
            dic[n] = dic.get(n, 0) + 1
        return ans


# 2013 - Detect Squares - MEDIUM
class DetectSquares:
    def __init__(self):
        self.samex = collections.defaultdict(collections.Counter)

    def add(self, point: List[int]) -> None:
        x, y = point
        self.samex[x][y] += 1
        return

    def count(self, point: List[int]) -> int:
        x, y = point
        ans = 0

        if x not in self.samex:
            return 0
        samex = self.samex[x]
        for k, diffy in self.samex.items():
            if k != x:
                d = k - x
                ans += diffy[y] * samex[y + d] * diffy[y + d]
                ans += diffy[y] * samex[y - d] * diffy[y - d]
        return ans


# 2016 - Maximum Difference Between Increasing Elements - EASY
class Solution:
    def maximumDifference(self, nums: List[int]) -> int:
        ans, mi = 0, math.inf
        for n in nums:
            mi = min(mi, n)
            ans = max(ans, n - mi)
        return ans if ans > 0 else -1

    def maximumDifference(self, nums: List[int]) -> int:
        ans, premin = -1, nums[0]
        for i in range(1, len(nums)):
            if nums[i] > premin:
                ans = max(ans, nums[i] - premin)
            else:
                premin = nums[i]
        return ans


# 2022 - Convert 1D Array Into 2D Array - EASY
class Solution:
    def construct2DArray(self, original: List[int], m: int,
                         n: int) -> List[List[int]]:
        if len(original) != m * n: return []
        ans = []
        for i in range(0, len(original), n):
            ans.append([x for x in original[i:i + n]])
        return ans

    def construct2DArray(self, original: List[int], m: int,
                         n: int) -> List[List[int]]:
        return [original[i:i + n] for i in range(0, len(original), n)
                ] if len(original) == m * n else []


# 2029 - Stone Game IX - MEDIUM
class Solution:
    def stoneGameIX(self, stones: List[int]) -> bool:
        d = [0, 0, 0]
        for n in stones:
            d[n % 3] += 1
        if d[0] % 2 == 0:
            return d[1] != 0 and d[2] != 0
        return d[2] > d[1] + 2 or d[1] > d[2] + 2


# 2034 - Stock Price Fluctuation - MEDIUM
class StockPrice:
    def __init__(self):
        self.maxPrice = []
        self.minPrice = []
        self.timePrice = {}
        self.maxTimestamp = 0

    def update(self, timestamp: int, price: int) -> None:
        heapq.heappush(self.maxPrice, (-price, timestamp))
        heapq.heappush(self.minPrice, (price, timestamp))
        self.timePrice[timestamp] = price
        self.maxTimestamp = max(self.maxTimestamp, timestamp)

    def current(self) -> int:
        return self.timePrice[self.maxTimestamp]

    def maximum(self) -> int:
        while True:
            price, timestamp = self.maxPrice[0]
            if -price == self.timePrice[timestamp]:
                return -price
            heapq.heappop(self.maxPrice)

    def minimum(self) -> int:
        while True:
            price, timestamp = self.minPrice[0]
            if price == self.timePrice[timestamp]:
                return price
            heapq.heappop(self.minPrice)


# 2045 - Second Minimum Time to Reach Destination - HARD
class Solution:
    def secondMinimum(self, n: int, edges: List[List[int]], time: int,
                      change: int) -> int:
        g = [[] for _ in range(n + 1)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        dist = [-1 for _ in range(n + 1)]
        dist[1] = 0
        dq = collections.deque([1])
        while dq:
            u = dq.popleft()
            for v in g[u]:
                if dist[v] == -1:
                    dq.append(v)
                    dist[v] = dist[u] + 1
        exist = False
        dq.append(n)
        while not exist and dq:
            u = dq.popleft()
            for v in g[u]:
                if dist[v] == dist[u]:
                    exist = True
                    break
                elif dist[v] == dist[u] - 1:
                    dq.append(v)
        d = dist[n] + 1 if exist else dist[n] + 2
        ans = 0
        for i in range(d):
            ans += time
            if (ans // change) % 2 == 1 and i != d - 1:
                ans += change - ans % change
        return ans

    # TODO
    def secondMinimum(self, n: int, edges: List[List[int]], time: int,
                      change: int) -> int:
        dis, dis2 = [float("inf")] * (n + 1), [float("inf")] * (n + 1)
        graph = collections.defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        dis[1] = 0
        dq = collections.deque([(0, 1)])
        while dq:
            cost, node = dq.popleft()
            for nei in graph[node]:
                new_cost = cost + time
                if (cost // change) % 2 == 1:
                    new_cost += change - (cost % change)
                if dis[nei] > new_cost:
                    dis2[nei], dis[nei] = dis[nei], new_cost
                    dq.append((new_cost, nei))
                elif dis[nei] < new_cost < dis2[nei]:
                    dis2[nei] = new_cost
                    dq.append((new_cost, nei))
        return dis2[n]


# 2047 - Number of Valid Words in a Sentence - EASY
class Solution:
    def countValidWords(self, sentence: str) -> int:
        return sum(
            bool(re.match(r'[a-z]*([a-z]-[a-z]+)?[!.,]?$', w))
            for w in sentence.split())

    def countValidWords(self, sentence: str) -> int:
        def valid(s: str) -> bool:
            hasHyphens = False
            for i, ch in enumerate(s):
                if ch.isdigit() or ch in "!.," and i < len(s) - 1:
                    return False
                if ch == '-':
                    if hasHyphens or i == 0 or i == len(s) - 1 or not s[
                            i - 1].islower() or not s[i + 1].islower():
                        return False
                    hasHyphens = True
            return True

        return sum(valid(s) for s in sentence.split())

# 2049 - Count Nodes With the Highest Score - MEDIUM


# 2055 - Plates Between Candles - MEDIUM
