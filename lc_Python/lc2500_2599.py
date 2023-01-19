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


# 2511 - Maximum Enemy Forts That Can Be Captured - EASY
class Solution:
    def captureForts(self, forts: List[int]) -> int:
        p = ans = t = 0
        for v in forts:
            if v == 0:
                t += 1
            elif v == 1:
                if p < 0:
                    ans = max(ans, t)
                p = 1
                t = 0
            else:
                if p > 0:
                    ans = max(ans, t)
                p = -1
                t = 0
        return ans


# 2512 - Reward Top K Students - MEDIUM
class Solution:
    def topStudents(
        self,
        positive_feedback: List[str],
        negative_feedback: List[str],
        report: List[str],
        student_id: List[int],
        k: int,
    ) -> List[int]:
        p = set(positive_feedback)
        n = set(negative_feedback)
        arr = []
        for i, r in enumerate(report):
            s = 0
            for w in r.split():
                if w in p:
                    s += 3
                if w in n:
                    s -= 1
            arr.append((s, -student_id[i]))
        return [-i for _, i in sorted(arr, reverse=True)[:k]]

    def topStudents(
        self,
        positive_feedback: List[str],
        negative_feedback: List[str],
        report: List[str],
        student_id: List[int],
        k: int,
    ) -> List[int]:
        p = set(positive_feedback)
        n = set(negative_feedback)
        s = collections.defaultdict(int)
        for i, r in zip(student_id, report):
            for w in r.split():
                if w in p:
                    s[i] += 3
                if w in n:
                    s[i] -= 1
        return sorted(s.keys(), key=lambda x: (-s[x], x))[:k]


# 2513 - Minimize the Maximum of Two Arrays - MEDIUM
class Solution:
    def minimizeSet(
        self, divisor1: int, divisor2: int, uniqueCnt1: int, uniqueCnt2: int
    ) -> int:
        def check(m: int):
            u1 = uniqueCnt1
            u2 = uniqueCnt2
            x = m // divisor1
            y = m // divisor2
            z = m // lcm
            arr2 = x - z  # 只有 1 不可用, 加入 arr2
            arr1 = y - z  # 只有 2 不可用, 加入 arr1
            can = m - x - y + z  # 都可用
            u1 = max(u1 - arr1, 0)
            u2 = max(u2 - arr2, 0)
            return u1 + u2 <= can

        l = 1
        r = 2 * 10**9
        lcm = divisor1 * divisor2 // math.gcd(divisor1, divisor2)
        while l < r:
            m = (l + r) // 2
            if check(m):
                r = m
            else:
                l = m + 1
        return l

    def minimizeSet(
        self, divisor1: int, divisor2: int, uniqueCnt1: int, uniqueCnt2: int
    ) -> int:
        lcm = math.lcm(divisor1, divisor2)

        def check(m: int) -> bool:
            arr1 = max(uniqueCnt1 - m // divisor2 + m // lcm, 0)
            arr2 = max(uniqueCnt2 - m // divisor1 + m // lcm, 0)
            common = m - m // divisor1 - m // divisor2 + m // lcm
            return common >= arr1 + arr2

        return bisect.bisect_left(range((uniqueCnt1 + uniqueCnt2) * 2), True, key=check)

    def minimizeSet(
        self, divisor1: int, divisor2: int, uniqueCnt1: int, uniqueCnt2: int
    ) -> int:
        arr = []
        p = uniqueCnt1 // (divisor1 - 1)
        if uniqueCnt1 % (divisor1 - 1) == 0:
            v = divisor1 * p - 1
        else:
            v = divisor1 * p + uniqueCnt1 % (divisor1 - 1)
        arr.append(v)

        p = uniqueCnt2 // (divisor2 - 1)
        if uniqueCnt2 % (divisor2 - 1) == 0:
            v = divisor2 * p - 1
        else:
            v = divisor2 * p + uniqueCnt2 % (divisor2 - 1)
        arr.append(v)

        div = math.lcm(divisor1, divisor2)
        cnt = uniqueCnt1 + uniqueCnt2
        p = cnt // (div - 1)
        if cnt % (div - 1) == 0:
            v = div * p - 1
        else:
            v = div * p + cnt % (div - 1)
        arr.append(v)

        return max(arr)

    def minimizeSet(
        self, divisor1: int, divisor2: int, uniqueCnt1: int, uniqueCnt2: int
    ) -> int:
        a = (uniqueCnt1 - 1) * divisor1 // (divisor1 - 1) + 1
        b = (uniqueCnt2 - 1) * divisor2 // (divisor2 - 1) + 1
        lcm = math.lcm(divisor1, divisor2)
        c = lcm * (uniqueCnt1 + uniqueCnt2 - 1) // (lcm - 1) + 1
        return max(a, b, c)


# 2514 - Count Anagrams - HARD
class Solution:
    def countAnagrams(self, s: str) -> int:
        def perm_count_with_duplicate(s: str) -> int:
            """
            return 含重复元素的列表 s, 全排列的种类
            假设长度 n, 含 x 种元素, 分别计数为[c1, c2, c3 ... cx]
            则答案是C(n, c1) * C(n-c1, c2) * C(n-c1-c2, c3) * ... * C(cx, cx)
            """
            n = len(s)
            ans = 1
            for v in collections.Counter(s).values():
                ans = ans * math.comb(n, v) % mod
                n -= v
            return ans

        mod = 10**9 + 7
        ans = 1
        for w in s.split():
            ans = (ans * perm_count_with_duplicate(w)) % mod
        return ans


mod = 10**9 + 7
fac = [1]
for i in range(1, 10**5 + 1):
    fac.append(fac[-1] * i % mod)


class Solution:
    # 我们只需要考虑一个最终可行的方案会重复计数多少次即可
    # 对有相同字母的位置进行排序, 不改变单词本身
    # 因此实际上每一个最终可行的方案会被重复计数 cnta! * cntb! * ... * cntz! 次 (考虑每种字母的不改变单词本身的排列)
    # 总方案数为 len(s)! // (cnta! * cntb! * ... * cntz!)

    # 如果有 cntC 个位置同时变成了某一未出现过的字符 C, 那么这些位置在排列中的顺序就无法区分了
    # 答案会变成原先的 1 // cntC!, 故答案为 n! // (cntA! * cntB! * ...)
    def countAnagrams(self, s: str) -> int:
        ans = 1
        for w in s.split():
            cnt = collections.Counter(w)
            ans *= fac[len(w)]
            ans %= mod
            for v in cnt.values():
                ans *= pow(fac[v], -1, mod)
                ans %= mod
        return ans


# 2520 - Count the Digits That Divide a Number - EASY
class Solution:
    def countDigits(self, num: int) -> int:
        ans = 0
        x = num
        while x:
            if num % (x % 10) == 0:
                ans += 1
            x //= 10
        return ans


# 2521 - Distinct Prime Factors of Product of Array - MEDIUM
class Solution:
    # O(nU) / O(U/logU), 空间复杂度: 素数分布, 1200 ms
    def distinctPrimeFactors(self, nums: List[int]) -> int:
        s = set()
        for v in nums:
            i = 2
            while i <= v:
                while v % i == 0:
                    v //= i
                    s.add(i)
                i += 1
        return len(s)

    # O(n * sqrt(U)) / O(U/logU), 130 ms
    def distinctPrimeFactors(self, nums: List[int]) -> int:
        s = set()
        for v in nums:
            i = 2
            while i * i <= v:
                if v % i == 0:
                    s.add(i)
                    while v % i == 0:
                        v //= i
                i += 1
            if v > 1:
                s.add(v)
        return len(s)


# 2522 - Partition String Into Substrings With Values at Most K - MEDIUM
class Solution:
    def minimumPartition(self, s: str, k: int) -> int:
        ans = 0
        p = ""
        for c in s:
            if int(c) > k:
                return -1
            if int(p + c) <= k:
                p += c
            else:
                ans += 1
                p = c
        return ans + 1

    def minimumPartition(self, s: str, k: int) -> int:
        ans = 1
        x = 0
        for v in map(int, s):
            if v > k:
                return -1
            x = x * 10 + v
            if x > k:
                ans += 1
                x = v
        return ans


# 2523 - Closest Prime Numbers in Range - MEDIUM
# O(n * loglogn) / O(n / logn), [2, n] 范围内素数个数
def eratosthenes(n: int) -> List[int]:
    primes = []
    is_prime = [True] * (n + 1)
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return primes


primes = eratosthenes(10**6 + 1)


class Solution:
    def closestPrimes(self, left: int, right: int) -> List[int]:
        # O(log(n / logn)) / O(n / logn)
        i = bisect.bisect_left(primes, left)
        if i == len(primes):
            return [-1, -1]
        x = y = -1
        # O(r / logr - l / logl) / O(1)
        while i + 1 < len(primes) and primes[i + 1] <= right:
            if x < 0 or primes[i + 1] - primes[i] < y - x:
                x = primes[i]
                y = primes[i + 1]
            i += 1
        return [x, y]


# O(n) / O(n / logn)
def euler(n: int) -> List[int]:
    primes = []
    is_prime = [True] * (n + 1)
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
        for p in primes:
            if i * p >= n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                break
    return primes


primes = euler(10**6 + 1)


class Solution:
    def closestPrimes(self, left: int, right: int) -> List[int]:
        i = bisect.bisect_left(primes, left)
        if i == len(primes):
            return [-1, -1]
        x = y = -1
        while i + 1 < len(primes) and primes[i + 1] <= right:
            if x < 0 or primes[i + 1] - primes[i] < y - x:
                x = primes[i]
                y = primes[i + 1]
            i += 1
        return [x, y]
