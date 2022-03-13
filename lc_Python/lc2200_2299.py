import bisect, collections, functools, math, itertools, heapq
from typing import List, Optional


# 2200 - Find All K-Distant Indices in an Array - EASY
class Solution:
    def findKDistantIndices(self, nums: List[int], key: int,
                            k: int) -> List[int]:
        ans = []
        n = len(nums)
        for i in range(n):
            for j in range(i - k, i + k + 1):
                if 0 <= j < n and nums[j] == key:
                    ans.append(i)
                    break
        return ans


# 2201 - Count Artifacts That Can Be Extracted - MEDIUM
class Solution:
    def digArtifacts(self, n: int, artifacts: List[List[int]],
                     dig: List[List[int]]) -> int:
        s = set((i, j) for i, j in dig)
        ans = 0
        for r1, c1, r2, c2 in artifacts:
            have = True
            f = False
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    if (r, c) not in s:
                        have = False
                        f = True
                        break
                if f:
                    break
            if have:
                ans += 1
        return ans


# 2202 - Maximize the Topmost Element After K Moves - MEDIUM
class Solution:
    def maximumTop(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 1 or k == 0:
            if k & 1:
                return -1
            else:
                return nums[0]

        f = max(nums[:k - 1]) if k > 1 else 0
        s = nums[k] if k < n else 0
        return max(f, s)


# 2203 - Minimum Weighted Subgraph With the Required Paths - HARD
class Solution:
    def minimumWeight(self, n: int, edges: List[List[int]], src1: int,
                      src2: int, dest: int) -> int:
        def dijkstra(g: List[List[tuple]], start: int) -> List[int]:
            dis = [math.inf] * n
            dis[start] = 0
            pq = [(0, start)]
            while pq:
                d, x = heapq.heappop(pq)
                if dis[x] < d:
                    continue
                for y, wt in g[x]:
                    new_d = dis[x] + wt
                    if new_d < dis[y]:
                        dis[y] = new_d
                        heapq.heappush(pq, (new_d, y))
            return dis

        g = [[] for _ in range(n)]
        rg = [[] for _ in range(n)]
        for x, y, wt in edges:
            g[x].append((y, wt))
            rg[y].append((x, wt))

        d1 = dijkstra(g, src1)
        d2 = dijkstra(g, src2)
        d3 = dijkstra(rg, dest)

        ans = min(sum(d) for d in zip(d1, d2, d3))
        return ans if ans < math.inf else -1

    def minimumWeight(self, n: int, edges: List[List[int]], src1: int,
                      src2: int, dest: int) -> int:
        g = collections.defaultdict(list)
        reverse_g = collections.defaultdict(list)
        for i, j, w in edges:
            g[i].append((j, w))
            reverse_g[j].append((i, w))

        def dijkstra(src: int, G: collections.defaultdict):
            dis = [math.inf] * n
            pq = [(0, src)]
            while pq:
                w, node = heapq.heappop(pq)
                if dis[node] <= w:  # see the different symbols between here and solution above
                    continue
                dis[node] = w
                for nxt, wt in G[node]:
                    if w + wt < dis[nxt]:  # and here
                        heapq.heappush(pq, (w + wt, nxt))
            return dis

        l1 = dijkstra(src1, g)
        l2 = dijkstra(src2, g)
        l3 = dijkstra(dest, reverse_g)
        ans = math.inf
        for i in range(n):
            ans = min(ans, l1[i] + l2[i] + l3[i])
        return ans if ans != math.inf else -1

    def minimumWeight(self, n: int, edges: List[List[int]], src1: int,
                      src2: int, dest: int) -> int:
        G1 = collections.defaultdict(list)
        G2 = collections.defaultdict(list)
        for a, b, w in edges:
            G1[a].append((b, w))
            G2[b].append((a, w))

        def dijkstra(graph: collections.defaultdict, src: int):
            pq = [(0, src)]
            t = {}
            while pq:
                time, node = heapq.heappop(pq)
                if node not in t:
                    t[node] = time
                    for v, w in graph[node]:
                        heapq.heappush(pq, (time + w, v))
            return [t.get(i, float("inf")) for i in range(n)]

        arr1 = dijkstra(G1, src1)
        arr2 = dijkstra(G1, src2)
        arr3 = dijkstra(G2, dest)

        ans = float("inf")
        for i in range(n):
            ans = min(ans, arr1[i] + arr2[i] + arr3[i])
        return ans if ans != float("inf") else -1
