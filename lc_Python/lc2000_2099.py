import collections, bisect, itertools, functools, math, heapq
from typing import List


# 2006 - Count Number of Pairs With Absolute Difference K - EASY
class Solution:
    def countKDifference(self, nums: List[int], k: int) -> int:
        seen, counter = collections.defaultdict(int), 0
        for num in nums:
            counter += seen[num - k] + seen[num + k]
            seen[num] += 1
        return counter


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