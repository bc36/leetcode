import bisect, collections, functools, math, itertools, heapq, string, operator
from typing import List, Optional, Tuple
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


# 2506 - Count Pairs Of Similar Strings - EASY
class Solution:
    def similarPairs(self, words: List[str]) -> int:
        d = collections.defaultdict(int)
        ans = 0
        for w in words:
            m = 0
            for c in w:
                m |= 1 << (ord(c) - ord("a"))
            ans += d[m]
            d[m] += 1
        return ans

    def similarPairs(self, words: List[str]) -> int:
        d = collections.defaultdict(int)
        ans = 0
        for w in words:
            s = "".join(sorted(list(set(w))))
            ans += d[s]
            d[s] += 1
        return ans


# 2507 - Smallest Value After Replacing With Sum of Prime Factors - MEDIUM
class Solution:
    # O(n**0.5) / O(1)
    def smallestValue(self, n: int) -> int:
        while True:
            ori = n
            s = 0
            i = 2
            while i * i <= n:
                while n % i == 0:
                    s += i
                    n //= i
                i += 1
            if n > 1:
                s += n
            if ori == s:
                return ori
            n = s


# TODO 线性欧拉筛
n = 100000
min_prime, primes = [*range(n + 1)], []
for x in range(2, n):
    if min_prime[x] == x:
        primes.append(x)
    for p in primes:
        if p > min_prime[x] or x * p > n:
            break
        min_prime[x * p] = p


def calc(n: int, res=0) -> int:
    while n > 1:
        res += min_prime[n]
        n //= min_prime[n]
    return res


def gen(x):
    while True:
        yield x
        x = calc(x)


class Solution:
    def smallestValue(self, n: int) -> int:
        return next(a for a, b in pairwise(gen(n)) if a == b)


# 2508 - Add Edges to Make Degrees of All Nodes Even - HARD
class Solution:
    def isPossible(self, n: int, edges: List[List[int]]) -> bool:
        g = collections.defaultdict(set)
        for x, y in edges:
            g[x].add(y)
            g[y].add(x)
        arr = list(k for k, v in g.items() if len(v) & 1)
        if len(arr) == 0:
            return True
        elif len(arr) == 2:
            a, b = arr
            if b not in g[a]:
                return True
            for i in range(1, n + 1):
                if i != a and i != b and a not in g[i] and b not in g[i]:
                    return True
        elif len(arr) == 4:
            a, b, c, d = arr
            if a not in g[b] and c not in g[d]:
                return True
            if a not in g[c] and b not in g[d]:
                return True
            if a not in g[d] and b not in g[c]:
                return True
        return False

    # require Python >= 3.10
    # def isPossible(self, n: int, edges: List[List[int]]) -> bool:
    #     g = [set() for _ in range(n + 1)]
    #     for x, y in edges:
    #         g[x].add(y)
    #         g[y].add(x)
    #     arr = [v for v in range(1, n + 1) if len(g[v]) & 1]
    #     match len(arr):
    #         case 0:
    #             return True
    #         case 2:
    #             if arr[0] not in g[arr[1]]:
    #                 return True
    #             else:
    #                 cannot = g[arr[0]] | g[arr[1]] | {*arr}
    #                 return any(v not in cannot for v in range(1, n + 1))
    #         case 4: return any(a not in g[b] and c not in g[d] for a, b, c, d in itertools.permutations(arr))
    #         case _: return False


# 2509 - Cycle Length Queries in a Tree - HARD
class Solution:
    # O(qn) / O(q)
    def cycleLengthQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        def lca(x: int, y: int):
            t = 0
            while x != y:
                if x > y:
                    x //= 2
                else:
                    y //= 2
                t += 1
            return t + 1

        return [lca(x, y) for x, y in queries]

    # 去除节点编号的二进制表示相同的部分
    def cycleLengthQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        def calc(q: Tuple[int, int]):
            a, b = map(lambda x: [x >> i for i in range(x.biq_length())], q)
            while a and b and a[-1] == b[-1]:
                a.pop()
                b.pop()
            return len(a) + len(b) + 1

        return list(map(calc, queries))

    # O(q) / O(1), 节点编号的二进制的长度恰好等于节点深度
    def cycleLengthQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        # x <= y, d = x y深度之差
        # y 上跳之后 -> x y 在同一层了
        #   x == y: d + 1
        #   x != y: d + 2 * 深度 L (即二进制长度) + 1
        def calc(q: Tuple[int, int]):
            x, y = q
            if x > y:
                x, y = y, x
            d = y.bit_length() - x.bit_length()
            return d + (x ^ (y >> d)).bit_length() * 2 + 1

        return list(map(calc, queries))
