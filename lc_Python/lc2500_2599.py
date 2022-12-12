import bisect, collections, functools, math, itertools, heapq, string, operator
from typing import List, Optional
import sortedcontainers


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 2500 - Delete Greatest Value in Each Row - EASY
class Solution:
    def deleteGreatestValue(self, grid: List[List[int]]) -> int:
        for row in grid:
            row.sort()
        ans = 0
        for col in zip(*grid):
            ans += max(col)
        return ans


# 2501 - Longest Square Streak in an Array - MEDIUM
class Solution:
    # O(nlogn) / O(n)
    def longestSquareStreak(self, nums: List[int]) -> int:
        d = {}
        for v in sorted(set(nums), reverse=True):
            d[v] = d.get(v * v, 0) + 1
        ans = max(d.values())
        return -1 if ans == 1 else ans

    # O(nlog(logU)) / O(n),  U = max(nums)
    # 其实 2, 4, 8 ... 2**32 > 10**5, while 就终止了
    def longestSquareStreak(self, nums: List[int]) -> int:
        ans = 0
        s = set(nums)
        for v in s:
            cnt = 0
            while v in s:
                cnt += 1
                v *= v
            ans = max(ans, cnt)
        return ans if ans > 1 else -1


# 2502 - Design Memory Allocator - MEDIUM
class Allocator:
    # 1 <= n, size, mID <= 1000 / call (allocate + free) 1000 times
    # 10**6 -> 暴力
    # O(nq) / O(n), q = query times
    def __init__(self, n: int):
        self.a = [0] * n

    def allocate(self, size: int, mID: int) -> int:
        cnt = 0
        for i, v in enumerate(self.a):
            if v:
                cnt = 0
            else:
                cnt += 1
            if cnt == size:
                self.a[i - size + 1 : i + 1] = [mID] * size
                return i - size + 1
        return -1

    def free(self, mID: int) -> int:
        cnt = 0
        for i, v in enumerate(self.a):
            if v == mID:
                cnt += 1
                self.a[i] = 0
        return cnt


# 2503 - Maximum Number of Points From Grid Queries - HARD
class Solution:
    # O(mnlog(mn) + qlogq) / O(mn + q), 1200 ms
    def maxPoints(self, grid: List[List[int]], queries: List[int]) -> List[int]:
        m = len(grid)
        n = len(grid[0])

        class UnionFind:
            def __init__(self, n: int) -> None:
                self.p = [i for i in range(n)]
                self.sz = [1] * n

            def find(self, x: int) -> int:
                """path compression"""
                if self.p[x] != x:
                    self.p[x] = self.find(self.p[x])
                return self.p[x]

            def union(self, x: int, y: int) -> None:
                """x's root -> y"""
                px = self.find(x)
                py = self.find(y)
                if px == py:
                    return
                self.p[px] = py
                self.sz[py] += self.sz[px]
                return

        # 点权转换成边权 -> 边权 = 两个点的点权最大值
        # 每次合并 < q 的边, 答案为左上角 (0) 的连通块的大小
        edges = []
        for i, row in enumerate(grid):
            for j, v in enumerate(row):  # 向左, 向上合并边, 防止重复 / 坐标转换
                if i:
                    edges.append((max(v, grid[i - 1][j]), i * n + j, (i - 1) * n + j))
                if j:
                    edges.append((max(v, grid[i][j - 1]), i * n + j, i * n + j - 1))
        edges.sort(key=lambda x: x[0])  # 排了第一个就不排了, 快一点点
        uf = UnionFind(m * n)
        ans = [0] * len(queries)
        j = 0
        for i, q in sorted(enumerate(queries), key=lambda x: x[1]):
            while j < len(edges) and edges[j][0] < q:
                uf.union(edges[j][1], edges[j][2])
                j += 1
            if grid[0][0] < q:
                ans[i] = uf.sz[uf.find(0)]

        # 矩阵排序 + 询问排序 + 双指针遍历
        # uf = UnionFind(m * n)
        # arr = sorted((v, i, j) for i, row in enumerate(grid) for j, v in enumerate(row))
        # ans = [0] * len(queries)
        # j = 0
        # for i, q in sorted(enumerate(queries), key=lambda x: x[1]):
        #     while j < m * n and arr[j][0] < q:
        #         _, x, y = arr[j]
        #         for nx, ny in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
        #             if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] < q:
        #                 uf.union(x * n + y, nx * n + ny)
        #         j += 1
        #     if grid[0][0] < q:
        #         ans[i] = uf.sz[uf.find(0)]

        return ans

    # O(mnlog(mn) + qlogq) / O(mn + k), 530 ms
    def maxPoints(self, grid: List[List[int]], queries: List[int]) -> List[int]:
        m = len(grid)
        n = len(grid[0])
        ans = [0] * len(queries)
        h = [(grid[0][0], 0, 0)]
        grid[0][0] = 0
        cnt = 0
        for i, q in sorted(enumerate(queries), key=lambda p: p[1]):
            while h and h[0][0] < q:
                cnt += 1
                _, x, y = heapq.heappop(h)
                for nx, ny in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
                    if 0 <= nx < m and 0 <= ny < n and grid[nx][ny]:
                        heapq.heappush(h, (grid[nx][ny], nx, ny))
                        grid[nx][ny] = 0
            ans[i] = cnt
        return ans
