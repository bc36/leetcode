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


# 2206 - Divide Array Into Equal Pairs - EASY
class Solution:
    def divideArray(self, nums: List[int]) -> bool:
        # cnt = collections.Counter(nums)
        # for n in cnt:
        #     if cnt[n] & 1:
        #         return False
        # return True
        return all(v % 2 == 0 for _, v in collections.Counter(nums).items())
        return not any(v & 1 for _, v in collections.Counter(nums).items())


# 2207 - Maximize Number of Subsequences in a String - MEDIUM
class Solution:
    def maximumSubsequenceCount(self, t: str, p: str) -> int:
        a = ans = 0
        b = t.count(p[1])
        if p[1] == p[0]:
            ans = b * (b + 1) // 2
        else:
            for ch in t:
                if ch == p[0]:
                    ans += b
                    a += 1
                elif ch == p[1]:
                    # ans += a
                    b -= 1
            # ans //= 2
            ans += max(a, t.count(p[1]))
        return ans

    # O(n) / O(1)
    def maximumSubsequenceCount(self, t: str, p: str) -> int:
        ans = c1 = c2 = 0
        for ch in t:
            if ch == p[1]:
                ans += c1
                c2 += 1
            if ch == p[0]:  # two 'if' to handle the case where p[0] == p[1]
                c1 += 1
        return ans + max(c1, c2)


# 2208 - Minimum Operations to Halve Array Sum -  MEDIUM
class Solution:
    # O(n * logn + m * logn) / O(n), where m is the number of operations
    def halveArray(self, nums: List[int]) -> int:
        t = sum(nums)
        half = t / 2
        hp = [-n for n in nums]
        heapq.heapify(hp)  # nlogn
        ans = 0
        while t > half:
            n = -heapq.heappop(hp) / 2
            t -= n
            heapq.heappush(hp, -n)
            ans += 1
        return ans


# 2209 - Minimum White Tiles After Covering With Carpets - HARD
class Solution:
    # dp[i][j], means that consider the previous floor with length 'j',
    # 'i' carpets are used, the minimum remaining white tiles.
    # i == 0, not use carpet, dp[0][j] is the number of white tiles before index j
    # i != 0, dp[i][j] = min(dp[i][j-1] + isWhite[j], dp[i-1][j-carpetLen])
    # dp[i][j-1] + isWhite[j]: not use carpet in 'floor[j]'
    # dp[i-1][j-carpetLen]: use carpet in 'floor[j]'
    # O(nm) / O(nm), n = len(floor), m = numCarpets
    def minimumWhiteTiles(self, floor: str, ncp: int, l: int) -> int:
        n = len(floor)
        dp = [[0] * n for _ in range(ncp + 1)]
        dp[0][0] = 1 if floor[0] == '1' else 0
        isWhite = [0] * n
        for i in range(1, n):
            dp[0][i] = dp[0][i - 1]
            if floor[i] == '1':
                dp[0][i] += 1
                isWhite[i] = 1
        for i in range(1, ncp + 1):
            for j in range(n):
                # less than 'carpetLen' bricks will end up with 0 white bricks left after using the carpet
                if j < l:
                    dp[i][j] = 0
                else:
                    dp[i][j] = min(dp[i][j - 1] + isWhite[j], dp[i - 1][j - l])
        return dp[ncp][n - 1]

    def minimumWhiteTiles(self, floor: str, numCarpets: int,
                          carpetLen: int) -> int:
        # define dp[i][numCarpet]
        # choose use or not use
        # if use: dp[i][use] = dp[i+carpetLen][use-1]
        @functools.lru_cache(None)
        def dfs(i, num):
            if i >= len(floor): return 0
            res = float('inf')
            # use
            if num:
                res = dfs(i + carpetLen, num - 1)
            # not use
            res = min(res, (floor[i] == '1') + dfs(i + 1, num))
            return res

        return dfs(0, numCarpets)

    def minimumWhiteTiles(self, floor: str, numCarpets: int,
                          carpetLen: int) -> int:
        pre = [0]
        n = len(floor)
        for i in range(n):
            if floor[i] == '1':
                pre.append(pre[-1] + 1)
            else:
                pre.append(pre[-1])

        @functools.lru_cache(None)
        def dp(i, j):
            if i < 0:
                return 0
            if j == 0:
                return pre[i + 1]
            return min(
                dp(i - 1, j) + (1 if floor[i] == '1' else 0),
                dp(i - carpetLen, j - 1))

        return dp(n - 1, numCarpets)


# 2210 - Count Hills and Valleys in an Array - EASY
class Solution:
    def countHillValley(self, nums: List[int]) -> int:
        nums.append(math.inf)
        a = []
        for i in range(len(nums) - 1):
            if nums[i] != nums[i + 1]:
                a.append(nums[i])
        ans = 0
        for i in range(1, len(a) - 1):
            if a[i - 1] > a[i] < a[i + 1] or a[i - 1] < a[i] > a[i + 1]:
                ans += 1
        return ans


# 2211 - Count Collisions on a Road - MEDIUM
class Solution:
    def countCollisions(self, directions: str) -> int:
        s = []
        co = 0
        for ch in directions:
            if ch == 'L':
                if not s:
                    continue
                elif s[-1] == 'S':
                    co += 1
                elif s[-1] == 'R':
                    while s and s[-1] == 'R':
                        s.pop()
                        co += 1
                    co += 1
                    s.append('S')
            elif ch == 'R':
                s.append(ch)
            else:
                while s and s[-1] == 'R':
                    s.pop()
                    co += 1
                s.append('S')
        return co

    # All the cars that move to the middle will eventually collide
    def countCollisions(self, directions: str) -> int:
        return sum(d != 'S' for d in directions.lstrip('L').rstrip('R'))

# 2212 - Maximum Points in an Archery Competition - MEDIUM