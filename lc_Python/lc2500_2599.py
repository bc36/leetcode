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


# 2515 - Shortest Distance to Target String in a Circular Array - EASY
class Solution:
    def closetTarget(self, words: List[str], target: str, startIndex: int) -> int:
        ans = n = len(words)
        for i, w in enumerate(words):
            if w == target:
                ans = min(ans, abs(i - startIndex), n - abs(i - startIndex))
        return ans if ans < n else -1


# 2516 - Take K of Each Character From Left and Right - MEDIUM
class Solution:
    # O(nlogn) / O(1)
    def takeCharacters(self, s: str, k: int) -> int:
        if k == 0:
            return 0
        cnt = collections.Counter(s)
        if cnt["a"] < k or cnt["b"] < k or cnt["c"] < k:
            return -1

        def pick(m: int) -> bool:
            d = collections.defaultdict(int)
            for i in range(m):
                d[s[i]] += 1
            if all(d[c] >= k for c in "abc"):
                return True
            for i in range(m):
                d[s[-1 - i]] += 1
                d[s[m - i - 1]] -= 1
                if all(d[c] >= k for c in "abc"):
                    return True
            return False

        l = 1
        r = len(s)
        while l < r:
            m = (l + r) // 2
            if pick(m):
                r = m
            else:
                l = m + 1
        return l

    # O(n) / O(1), 每当 l 增加, 即从左边多取一个, 尽量减少从右边取走的
    def takeCharacters(self, s: str, k: int) -> int:
        r = n = len(s)
        cnt = collections.Counter()
        # while any(v < k for v in cnt.values()):
        # 注意这种情况下, cnt 内若没有元素, 则会返回 False 退出 while 循环
        # any(): If the iterable is empty, return False
        while cnt["a"] < k or cnt["b"] < k or cnt["c"] < k:
            if r == 0:
                return -1
            r -= 1
            cnt[s[r]] += 1
        ans = n - r
        for l, ch in enumerate(s):
            cnt[ch] += 1
            while r < n and cnt[s[r]] > k:
                cnt[s[r]] -= 1
                r += 1
            ans = min(ans, l + 1 + n - r)
            if r == n:
                break
        return ans

    # O(n) / O(1), 每当 r 增加, 即从右边少取一个, 移动 l 直至满足条件
    def takeCharacters(self, s: str, k: int) -> int:
        cnt = collections.Counter(s)
        if any(cnt[v] < k for v in "abc"):
            return -1
        ans = n = len(s)
        l = 0
        for r, ch in enumerate(s):
            cnt[ch] -= 1
            while cnt[ch] < k:
                cnt[s[l]] += 1
                l += 1
            ans = min(ans, n - (r - l + 1))
        return ans


# 2517 - Maximum Tastiness of Candy Basket - MEDIUM
class Solution:
    # O(nlogn + nlogU) / O(1), U = max(price)
    def maximumTastiness(self, price: List[int], k: int) -> int:
        price.sort()
        l = 0
        r = max(price)
        while l < r:
            m = l + r + 1 >> 1
            t = 1
            p = price[0]
            for i in range(1, len(price)):
                if price[i] - p >= m:
                    p = price[i]
                    t += 1
            if t >= k:
                l = m
            else:
                r = m - 1
        return l


# 2518 - Number of Great Partitions - HARD
class Solution:
    # O(nk) / O(nk)
    def countPartitions(self, nums: List[int], k: int) -> int:
        if sum(nums) < k * 2:
            return 0
        mod = 10**9 + 7
        n = len(nums)
        # f[i][j]: 表示从前 i 个数中选择若干元素, 和为 j 的方案数
        f = [[0] * k for _ in range(n + 1)]
        f[0][0] = 1
        for i in range(1, n + 1):
            for j in range(k):
                f[i][j] = f[i - 1][j]
                if j >= nums[i - 1]:
                    f[i][j] = (f[i][j] + f[i - 1][j - nums[i - 1]]) % mod

        return (pow(2, n, mod) - sum(f[-1]) * 2) % mod

    # https://leetcode.cn/problems/number-of-great-partitions/solution/by-tsreaper-69ff/
    def countPartitions(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        n = len(nums)
        f = [[0] * k for _ in range(n + 1)]
        f[0][0] = 1
        for i in range(1, n + 1):
            for j in range(k):
                f[i][j] = f[i - 1][j]
                if j >= nums[i - 1]:
                    f[i][j] = (f[i][j] + f[i - 1][j - nums[i - 1]]) % mod

        summ = sum(nums)
        # total amount of cases
        ans = 1
        for _ in range(n):
            ans = ans * 2 % mod
        # subtract the number of bad cases
        for j in range(k):
            d = summ - j
            if d >= k:
                # only one group which sum is less than K
                ans = (ans - f[n][j] * 2 % mod) % mod
            else:
                # both groups have a sum less than K
                ans = (ans - f[n][j]) % mod
        return ans

    # O(nk) / O(k)
    def countPartitions(self, nums: List[int], k: int) -> int:
        if sum(nums) < k * 2:
            return 0
        mod = 10**9 + 7
        f = [0] * k
        f[0] = 1
        for v in nums:
            for j in range(k - 1, v - 1, -1):
                f[j] = (f[j] + f[j - v]) % mod
        return (pow(2, len(nums), mod) - sum(f) * 2) % mod


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


# 2525 - Categorize Box According to Criteria - EASY
class Solution:
    def categorizeBox(self, length: int, width: int, height: int, mass: int) -> str:
        b = (
            length >= 1e4
            or width >= 1e4
            or height >= 1e4
            or mass >= 1e4
            or length * width * height >= 1e9
        )
        h = mass >= 100
        if b and h:
            return "Both"
        if b:
            return "Bulky"
        if h:
            return "Heavy"
        return "Neither"


# 2526 - Find Consecutive Integers from a Data Stream - MEDIUM
class DataStream:
    def __init__(self, value: int, k: int):
        self.cnt = 0
        self.v = value
        self.k = k

    def consec(self, num: int) -> bool:
        if num != self.v:
            self.cnt = -1
        self.cnt += 1
        return self.cnt >= self.k


# 2527 - Find Xor-Beauty of Array - MEDIUM
class Solution:
    # 方法一:
    # 因为不同二进制位之间互不影响, 单独考虑每个比特
    # 看有多少个 1 -> (a | b) & c = 1 -> 需要 a | b = 1 (a b 不能都是 0) 并且 c = 1
    # 设这个比特有 x 个 0, y 个 1, 则 x + y = n
    # 有多少个 1: ones = (n * n - x * x) * y
    # 因为问题是 ones 的异或结果, 所以只需要考虑 ones 的奇偶性
    # ones = (n^2 - x^2) * y
    #      = (n^2 - (n - y)^2) * y
    #      = (2n - y) * y * y
    #      = 2ny^2 - y^3
    # 进而考察 y 的奇偶性, 可以看成是 这个比特位 0 和 1 异或的结果
    # 进而转换为 nums 中每个 num 的每个 bit 的异或结果

    # 方法二:
    # 三元组 (a, b, c), (b, a, c) 异或和为 0
    # 在剩余的形式为 (a, a, b) 的三元组中, 可以化为 a & b, 又 (a, a, b) ^ (b, b, a) 异或和为 0
    # 所以此时只剩余 (a, a, a) 形式的三元组, 即为 nums[i] 的异或结果
    # O(n) / O(1)
    def xorBeauty(self, nums: List[int]) -> int:
        return functools.reduce(operator.xor, nums)

    # 常用位运算技巧: 不同二进制位之间互不影响, 单独考虑每个二进制位的答案
    # nums[i], nums[j] 其中一个是 1, nums[k] 是 1 时, 三元组值才为 1
    # cnt[0] 表示 nums[i] == 0 的 i 有几个
    # cnt[1] 表示 nums[i] == 1 的 i 有几个
    # 有效值是 1 的三元组的数量即为 (n * n - cnt[0] * cnt[0]) * cnt[1]
    #                          前两个数至少一个 1 的选法 * 第三个数是 1 的选法
    # O(nlogU) / O(1)
    def xorBeauty(self, nums: List[int]) -> int:
        ans = 0
        n = len(nums)
        for k in range(31):
            cnt = [0, 0]
            for v in nums:
                cnt[v >> k & 1] += 1
            f = (n * n - cnt[0] * cnt[0]) * cnt[1]
            if f & 1:
                ans |= 1 << k
        return ans


# 2528 - Maximize the Minimum Powered City - HARD
class Solution:
    # 从左到右维护每个城市 i 的当前电量, [i - r, i + r] 区间和
    # 尽可能靠右新建变电站
    # O(nlog(nU + k)) / O(n), U = max(stations)
    def maxPower(self, stations: List[int], r: int, k: int) -> int:
        def check(mid: int) -> bool:
            p = stations.copy()
            cur = need = j = 0
            for i in range(len(p)):
                # 移动窗口右端点
                while j < len(p) and j <= i + r:
                    cur += p[j]
                    j += 1
                # 移动窗口左端点
                if i - r - 1 >= 0:
                    cur -= p[i - r - 1]
                if cur < mid:
                    p[j - 1] += mid - cur
                    need += mid - cur
                    cur = mid  # 即 cur += mid - cur
                    if need > k:
                        return False
            return True

        low = 0
        high = 10**15
        while low < high:
            # 左侧最后一个满足条件的值 的模版
            mid = low + high + 1 >> 1
            if check(mid):
                low = mid
            else:
                high = mid - 1
        return low

        # or
        # while low < high:
        #     # 左侧最后一个满足条件的值 的模版
        #     mid = low + high >> 1
        #     if check(mid):
        #         low = mid + 1
        #     else:
        #         high = mid
        # return high - 1

    # O(nlog(nU + k)) / O(n), U = max(stations)
    def maxPower(self, stations: List[int], r: int, k: int) -> int:
        def check(mid: int) -> bool:
            p = stations[::]
            cur = sum(p[: min(r + 1, n)])  # 初始的滑动窗口和
            left = need = 0
            right = r  # 维护窗口 [left, right]
            for i in range(n):
                if cur < mid:
                    d = mid - cur
                    need += d
                    if need > k:
                        return False
                    # 新建在窗口最右边
                    p[right] += d
                    cur += d  # 即 cur = mid
                # 窗口向前移动一个城市
                if i >= r:
                    cur -= p[left]
                    left += 1
                if right != n - 1:
                    cur += p[right + 1]
                    right += 1
            return True

        n = len(stations)
        low = 0
        high = 10**15
        while low < high:
            mid = low + high + 1 >> 1
            if check(mid):
                low = mid
            else:
                high = mid - 1
        return low

    # 用 前缀和 + 差分数组 更新
    # O(nlogk) / O(n)
    def maxPower(self, stations: List[int], r: int, k: int) -> int:
        def check(mid: int) -> bool:
            diff = [0] * n
            sumd = need = 0
            for i, v in enumerate(has):
                sumd += diff[i]  # 累加差分值
                d = mid - v - sumd
                if d > 0:  # 需要 d 个供电站
                    need += d
                    if need > k:
                        return False
                    sumd += d  # 差分更新
                    if i + r * 2 + 1 < n:
                        diff[i + r * 2 + 1] -= d  # 差分更新
            return True

        n = len(stations)
        p = list(itertools.accumulate(stations, initial=0))  # len(p) = n + 1
        has = [p[min(i + r + 1, n)] - p[max(i - r, 0)] for i in range(n)]
        left = min(has)
        right = left + k + 1
        while left < right:
            mid = left + right + 1 >> 1
            if check(mid):
                left = mid
            else:
                right = mid - 1
        return left

    # 直接调 bisect 库, 遇到找 左侧的最后一个满足条件值 时, 需要特殊更改, 总体来说使用感觉一般
    # 1. 改 check 中的 mid, 反转 check 中的 True False
    # 2. 改 range 区间, 反转 check 中的 True False

    # 原因:
    # 现正常思路为 TTTTT FFFF, 想找最后一个 T 的下标
    #                ^
    # 但是标准库提供的二分只支持左边 F 找右边第一个 T (即左侧 < x or <= x, 右侧 >= x or > x, 以保证序列有序)
    # 所以把 check 反过来, 反转 True False
    # 类似 FFFFF TTTT
    #           ^
    def maxPower(self, stations: List[int], r: int, k: int) -> int:
        def check(mid: int) -> bool:
            mid += 1
            diff = [0] * n
            sumd = need = 0
            for i, v in enumerate(has):
                sumd += diff[i]
                d = mid - v - sumd
                if d > 0:
                    need += d
                    if need > k:
                        return True
                    sumd += d
                    if i + r * 2 + 1 < n:
                        diff[i + r * 2 + 1] -= d
            return False

        n = len(stations)
        p = list(itertools.accumulate(stations, initial=0))  # len(p) = n + 1
        has = [p[min(i + r + 1, n)] - p[max(i - r, 0)] for i in range(n)]
        return bisect.bisect_left(range(p[n] + k), x=True, key=check)

    def maxPower(self, stations: List[int], r: int, k: int) -> int:
        def check(mid: int) -> bool:
            diff = [0] * n
            sumd = need = 0
            for i, v in enumerate(has):
                sumd += diff[i]
                d = mid - v - sumd
                if d > 0:
                    need += d
                    if need > k:
                        return True
                    sumd += d
                    if i + r * 2 + 1 < n:
                        diff[i + r * 2 + 1] -= d
            return False

        n = len(stations)
        p = list(itertools.accumulate(stations, initial=0))  # len(p) = n + 1
        has = [p[min(i + r + 1, n)] - p[max(i - r, 0)] for i in range(n)]
        return bisect.bisect_left(range(p[n] + k + 1), x=True, key=check) - 1


# 2529 - Maximum Count of Positive Integer and Negative Integer - EASY
class Solution:
    # O(n) / O(1)
    def maximumCount(self, nums: List[int]) -> int:
        return max(sum(v > 0 for v in nums), sum(v < 0 for v in nums))

    # O(logn) / O(1)
    def maximumCount(self, nums: List[int]) -> int:
        a = bisect.bisect_left(nums, 0)
        b = bisect.bisect_right(nums, 0)
        return max(a, len(nums) - b)


# 2530 - Maximal Score After Applying K Operations - MEDIUM
class Solution:
    # O(klogn) / O(1)
    def maxKelements(self, nums: List[int], k: int) -> int:
        nums = [-v for v in nums]
        heapq.heapify(nums)
        ans = 0
        for _ in range(k):
            ans -= heapq.heapreplace(nums, nums[0] // 3)  # nums[0] is negetive
        return ans


# 2531 - Make Number of Distinct Characters Equal - MEDIUM
class Solution:
    # O(n + m + 26**3) / O(26)
    def isItPossible(self, word1: str, word2: str) -> bool:
        c1 = [0] * 26
        c2 = [0] * 26
        for c in word1:
            c1[ord(c) - 97] += 1
        for c in word2:
            c2[ord(c) - 97] += 1
        for i in range(26):
            if c1[i]:
                for j in range(26):
                    if c2[j]:
                        # change
                        c1[i] -= 1
                        c1[j] += 1
                        c2[i] += 1
                        c2[j] -= 1
                        # count
                        n = m = 0
                        for k in range(26):
                            n += c1[k] > 0
                            m += c2[k] > 0
                        if n == m:
                            return True
                        # undo
                        c1[i] += 1
                        c1[j] -= 1
                        c2[i] -= 1
                        c2[j] += 1
        return False

    # O(n + m + 26**2) / O(26)
    def isItPossible(self, word1: str, word2: str) -> bool:
        c1 = collections.Counter(word1)
        c2 = collections.Counter(word2)
        for x, n in c1.items():
            for y, m in c2.items():
                if y == x:
                    if len(c1) == len(c2):
                        return True
                elif len(c1) - (n == 1) + (y not in c1) == len(c2) - (m == 1) + (
                    x not in c2
                ):
                    return True
        return False


# 2532 - Time to Cross a Bridge - HARD
class Solution:
    # while loop:
    #   pop worker from workL to waitL
    #   pop worker from workR to waitR
    #   if waitR:
    #       pop worker from waitR, go through the bridge, then put it into workL(record finishing time)
    #   elif waitL:
    #       pop worker from waitL, go through the bridge, then put it into workR(record finishing time), n--
    #   else:
    #       can do nothing, waiting for workers to finish, update time
    # the last one go through is the answer

    # O(nlogk) / O(k)
    def findCrossingTime(self, n: int, k: int, time: List[List[int]]) -> int:
        time.sort(key=lambda t: t[0] + t[2])
        cur = 0
        workL = []
        waitL = [-i for i in range(k)[::-1]]
        workR = []  # (finishing time, -index)
        waitR = []  # (-index, finishing time)
        while n:
            while workL and workL[0][0] <= cur:
                _, i = heapq.heappop(workL)
                heapq.heappush(waitL, i)  # put boxes on the left
            while workR and workR[0][0] <= cur:
                _, i = heapq.heappop(workR)
                heapq.heappush(waitR, i)  # pick up boxes on the right
            if waitR:
                i = heapq.heappop(waitR)
                cur += time[-i][2]
                heapq.heappush(workL, (cur + time[-i][3], i))  # put
            elif waitL:
                i = heapq.heappop(waitL)
                cur += time[-i][0]
                heapq.heappush(workR, (cur + time[-i][1], i))  # pick up
                n -= 1
            # find the earliest time a worker has finished
            elif len(workL) == 0:  # all workers on the right
                cur = workR[0][0]
            elif len(workR) == 0:  # all workers on the left
                cur = workL[0][0]
            else:
                cur = min(workL[0][0], workR[0][0])
        while workR:
            # wait until all workers on the right move to the left
            t, i = heapq.heappop(workR)
            cur = max(t, cur) + time[-i][2]
        return cur


# 2544 - Alternating Digit Sum - EASY
class Solution:
    # O(logn) / O(n)
    def alternateDigitSum(self, n: int) -> int:
        ans = 0
        f = 1
        for c in str(n):
            ans += f * int(c)
            f *= -1
        return ans

    # O(logn) / O(1)
    def alternateDigitSum(self, n: int) -> int:
        ans = 0
        f = 1
        while n:
            ans += n % 10 * f
            n //= 10
            f *= -1
        return ans * -f


# 2545 - Sort the Students by Their Kth Score - MEDIUM
class Solution:
    def sortTheStudents(self, score: List[List[int]], k: int) -> List[List[int]]:
        # sorted 会复制一份, return a new sorted list
        return sorted(score, key=lambda x: x[k], reverse=True)

    def sortTheStudents(self, score: List[List[int]], k: int) -> List[List[int]]:
        score.sort(key=lambda x: -x[k])
        return score


# 2546 - Apply Bitwise Operations to Make Strings Equal - MEDIUM
class Solution:
    # 0 0 -> 0 0
    # 0 1 -> 1 1
    # 1 0 -> 1 1
    # 1 1 -> 1 0
    # 观察, 只要有一个 1, 即可随意变换到 (最少一个 1 + 任意组合), 即只有全 0 是一个在有 1 的情况下达不到的变换
    def makeStringsEqual(self, s: str, target: str) -> bool:
        a = collections.Counter(s)
        b = collections.Counter(target)
        if b["0"] == 0:  # 全是 1
            return a["1"] > 0
        if b["1"] == 0:  # 全是 0
            return a["1"] == 0
        return a["1"] > 0

    # 观察变换
    # 当有 1 时, 可以把 1 边 0, 0 变 1
    # 都有 1 -> True
    # 都没有 1 / 全 0 -> True
    # 一个有 1, 同时另一个没有 1 -> False
    def makeStringsEqual(self, s: str, target: str) -> bool:
        return ("1" in s) == ("1" in target)


# 2547 - Minimum Cost to Split an Array - HARD
class Solution:
    # f[i + 1] = min(k + cost(j, i) + f[j] for j in range(i))
    def minCost(self, nums: List[int], k: int) -> int:
        n = len(nums)
        f = [0] * (n + 1)
        for i in range(n):
            cnt = [0] * n
            t = 0
            mi = math.inf
            for j in range(i, -1, -1):
                x = nums[j]
                cnt[x] += 1
                if cnt[x] == 2:
                    t += 2
                elif cnt[x] > 2:
                    t += 1
                mi = min(mi, f[j] + t)
            f[i + 1] = k + mi
        return f[n]

    # f[i + 1] = min(k + i - j + 1 - unique_j + f[j])
    #          = i + 1 + k + min(f[j] - j - unique_j)
    def minCost(self, nums: List[int], k: int) -> int:
        n = len(nums)
        f = [0] * (n + 1)
        for i in range(n):
            # 子数组长度(i - j + 1) - unique
            cnt = [0] * n
            u = 0  # unique
            mi = math.inf
            for j in range(i, -1, -1):
                x = nums[j]
                cnt[x] += 1
                if cnt[x] == 1:
                    u += 1
                elif cnt[x] == 2:
                    u -= 1
                mi = min(mi, f[j] - j - u)  # 美服 TLE, 得拆开 min
            f[i + 1] = i + 1 + k + mi
        return f[n]

    # 转移方程如上
    # f[i + 1] = i + 1 + k + min(f[j] - j - unique_j)
    # f[i + 1] - (i + 1) = k + min(f[j] - j - unique_j)
    # 令 g[i] = f[i] - i
    # g[i + 1] = k + min(g[j] - unique_j)
    # g[n] = f[n] + n
    def minCost(self, nums: List[int], k: int) -> int:
        n = len(nums)
        f = [0] * (n + 1)
        for i in range(n):
            cnt = [0] * n
            u = 0  # unique
            mi = math.inf
            for j in range(i, -1, -1):
                x = nums[j]
                cnt[x] += 1
                if cnt[x] == 1:
                    u += 1
                elif cnt[x] == 2:
                    u -= 1
                mi = min(mi, f[j] - u)
            f[i + 1] = k + mi
        return f[n] + n

    def minCost(self, nums: List[int], k: int) -> int:
        n = len(nums)
        f = [0] * (n + 1)
        f[n] = 0
        for i in range(n - 1, -1, -1):
            cnt = collections.Counter()
            v = 0
            cur = math.inf
            for j in range(i, n):
                if nums[j] not in cnt:
                    v += 1
                elif cnt[nums[j]] == 1:
                    v -= 1
                cnt[nums[j]] += 1
                tmp = (j - i + 1 - v) + k + f[j + 1]
                if tmp < cur:
                    cur = tmp
            f[i] = cur
        return f[0]

    def minCost(self, nums: List[int], k: int) -> int:
        @functools.lru_cache(None)
        def dfs(i: int) -> int:
            if i == len(nums):
                return 0
            ans = math.inf
            once = set()
            many = set()
            for j, v in enumerate(nums[i:], start=i):
                if v not in many:
                    if v in once:
                        once.remove(v)
                        many.add(v)
                    else:
                        once.add(v)
                ans = min(ans, k + j - i + 1 - len(once) + dfs(j + 1))
            return ans

        return dfs(0)


# TODO
class Solution:
    def minCost(self, nums: List[int], k: int) -> int:
        # Lazy 线段树模板(区间加, 查询区间最小)
        n = len(nums)
        mn = [0] * (4 * n)
        todo = [0] * (4 * n)

        def do(o: int, v: int) -> None:
            mn[o] += v
            todo[o] += v

        def spread(o: int) -> None:
            v = todo[o]
            if v:
                do(o * 2, v)
                do(o * 2 + 1, v)
                todo[o] = 0

        # 区间 [L,R] 内的数都加上 v   o,l,r=1,1,n
        def update(o: int, l: int, r: int, L: int, R: int, v: int) -> None:
            if L <= l and r <= R:
                do(o, v)
                return
            spread(o)
            m = (l + r) // 2
            if m >= L:
                update(o * 2, l, m, L, R, v)
            if m < R:
                update(o * 2 + 1, m + 1, r, L, R, v)
            mn[o] = min(mn[o * 2], mn[o * 2 + 1])

        # 查询区间 [L,R] 的最小值   o,l,r=1,1,n
        def query(o: int, l: int, r: int, L: int, R: int) -> int:
            if L <= l and r <= R:
                return mn[o]
            spread(o)
            m = (l + r) // 2
            if m >= R:
                return query(o * 2, l, m, L, R)
            if m < L:
                return query(o * 2 + 1, m + 1, r, L, R)
            return min(query(o * 2, l, m, L, R), query(o * 2 + 1, m + 1, r, L, R))

        ans = 0
        last = [0] * n
        last2 = [0] * n
        for i, x in enumerate(nums, 1):
            update(1, 1, n, i, i, ans)  # 相当于设置 f[i+1] 的值
            update(1, 1, n, last[x] + 1, i, -1)
            if last[x]:
                update(1, 1, n, last2[x] + 1, last[x], 1)
            ans = k + query(1, 1, n, 1, i)
            last2[x] = last[x]
            last[x] = i
        return ans + n


# 2540 - Minimum Common Value - EASY
class Solution:
    # O(m + n) / O(m + n)
    def getCommon(self, nums1: List[int], nums2: List[int]) -> int:
        return min(set(nums1) & set(nums2), default=-1)

    # O(m + n) / O(n)
    def getCommon(self, nums1: List[int], nums2: List[int]) -> int:
        s = set(nums2)
        for v in nums1:
            if v in s:
                return v
        return -1

    # O(m + n) / O(1)
    def getCommon(self, nums1: List[int], nums2: List[int]) -> int:
        i = 0
        for v in nums2:
            while i < len(nums1) and nums1[i] < v:
                i += 1
            if i < len(nums1) and nums1[i] == v:
                return v
        return -1


# 2541 - Minimum Operations to Make Array Equal II - MEDIUM
class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int], k: int) -> int:
        if k == 0:
            return 0 if nums1 == nums2 else -1
        p = n = 0
        for x, y in zip(nums1, nums2):
            d = x - y
            if abs(d) % k != 0:
                return -1
            if d > 0:
                p += d // k
            if d < 0:
                n += -d // k
        return p if n == p else -1

    def minOperations(self, nums1: List[int], nums2: List[int], k: int) -> int:
        ans = summ = 0
        for x, y in zip(nums1, nums2):
            d = x - y
            if k:
                if d % k:
                    return -1
                summ += d // k
                if d > 0:
                    ans += d // k
            elif d:
                return -1
        return -1 if summ else ans


# 2542 - Maximum Subsequence Score - MEDIUM
class Solution:
    # 子序列 -> 排序?
    # min   -> 枚举最小值
    # 如何排序 -> 每次枚举的 nums2[i] 都是最小值 -> 从大到小排序 nums2 (nums1 顺序无所谓)
    # 选择 nums1[0...j] 个中最大的 k 个 -> 小根堆维护当前最大的 k 个数
    # O(nlogn) / O(n)
    def maxScore(self, nums1: List[int], nums2: List[int], k: int) -> int:
        arr = sorted(zip(nums1, nums2), key=lambda x: -x[1])
        h = [x for x, _ in arr[:k]]
        heapq.heapify(h)
        summ = sum(h)
        ans = summ * arr[k - 1][1]
        for x, y in arr[k:]:
            # 不用 if 也可以通过, 在 x 小于 h[0] 时, summ 变小
            # 但是这个换进来的'小'值仍然在堆顶, 会被后续插入的'大'值弹出
            # 所以不影响正确答案
            # summ += x - heapq.heapreplace(h, x)
            # ans = max(ans, summ * y)

            # 这个更容易理解, x 小于 h[0] 时, summ 不变
            summ += x - heapq.heappushpop(h, x)
            ans = max(ans, summ * y)

            # 或者加入判断
            if x > h[0]:
                summ += x - heapq.heapreplace(h, x)
                ans = max(ans, summ * y)
        return ans


# 2543 - Check if Point Is Reachable - HARD
class Solution:
    # (x, y - x)
    # (x - y, y)
    # -> 更相减损术 / 辗转相除法
    # -> 不会更改 gcd
    #
    # (2 * x, y)
    # (x, 2 * y)
    # -> gcd * 2^k, gcd 乘 2 的幂次
    def isReachable(self, targetX: int, targetY: int) -> bool:
        g = math.gcd(targetX, targetY)
        return (g & (g - 1)) == 0  # 位运算, 判断是否为 2 的幂次
        return math.gcd(targetX, targetY).bit_count() == 1

    # 反过来思考 (x, y) 到 (1, 1)
    # 若 x 和 y 都是偶数, 则将它们同时除以 2, 这不改变它们的 gcd 是 2 的若干次方的性质
    # 若 x 和 y 其中之一是偶数, 则将偶数除以 2, 因为另一个数是奇数, 肯定不能被 2 整除, 这不改变它们的 gcd
    # 若 x 和 y 都是奇数
    #   x != y 时, x + y 是偶数, 设 x > y, 可以通过 +y, /2 操作 变为 (x + y) / 2,
    #       根据性质 gcd(x + y) = gcd(x + y, y), 以及 y 是奇数, 不改变 gcd, 还能让坐标变小
    #   x = y 时, 任何操作都不能让坐标变小, break, 验证 x = y = 1
    def isReachable(self, targetX: int, targetY: int) -> bool:
        while targetX % 2 == 0:
            targetX //= 2
        while targetY % 2 == 0:
            targetY //= 2
        return math.gcd(targetX, targetY) == 1

    # 反过来思考 (x, y) 到 (1, 1)
    # 如果两个数中有偶数, 将偶数除以 2
    # 如果两个数都是奇数, 保留较小的数, 较大的数加上较小的数
    def isReachable(self, targetX: int, targetY: int) -> bool:
        while (targetX, targetY) != (1, 1):
            if targetX % 2 == 0:
                targetX //= 2
                continue
            if targetY % 2 == 0:
                targetY //= 2
                continue
            if targetX > targetY:
                targetX += targetY
            elif targetX < targetY:
                targetY += targetX
            else:
                break
        return targetX == targetY == 1


# 2558 - Take Gifts From the Richest Pile - EASY
class Solution:
    # O(klogn) / O(n)
    def pickGifts(self, gifts: List[int], k: int) -> int:
        h = [-v for v in gifts]
        heapq.heapify(h)
        for _ in range(k):
            heapq.heapreplace(h, -int((-h[0]) ** 0.5))  # truncates towards zero
        return -sum(h)


# 2559 - Count Vowel Strings in Ranges - MEDIUM
class Solution:
    def vowelStrings(self, words: List[str], queries: List[List[int]]) -> List[int]:
        s = set("aeiou")
        p = [0]
        for w in words:
            p.append(p[-1] + (1 if w[0] in s and w[-1] in s else 0))
        return [p[r + 1] - p[l] for l, r in queries]


# 2560 - House Robber IV - MEDIUM
class Solution:
    # O(nlogU) / O(1), U = max(nums)
    def minCapability(self, nums: List[int], k: int) -> int:
        l = min(nums)
        r = max(nums)
        while l < r:
            m = l + r >> 1
            t = 0
            p = -2  # an index not adjacent to index zero
            for i, v in enumerate(nums):
                if v <= m and p != i - 1:
                    t += 1
                    p = i
            if t >= k:
                r = m
            else:
                l = m + 1
        return l

    def minCapability(self, nums: List[int], k: int) -> int:
        def check(m: int) -> bool:
            t = 0
            p = -2
            for i, v in enumerate(nums):
                if v <= m and p != i - 1:
                    t += 1
                    p = i
            return t >= k

        return bisect.bisect_left(
            range(0, max(nums) + 1), True, min(nums), max(nums) + 1, key=check
        )

    def minCapability(self, nums: List[int], k: int) -> int:
        n = len(nums)

        def check(m: int) -> int:
            # f[i] 表示偷前 i 个房子, 最多可以偷多少个房子(房子金额不大于 m)
            # f[i] = max(f[i-1], f[i-2] + 1)
            # f[0] = max(f[-1], f[-2] + 1)
            # 防止越界, 下标加 2
            # f[i+2] = max(f[i+1], f[i] + 1)
            f = [0] * (n + 2)
            for i, v in enumerate(nums):
                f[i + 2] = f[i + 1]
                if v <= m:
                    f[i + 2] = max(f[i + 2], f[i] + 1)
            return f[n + 1]

            # 滚动数组优化空间, f2 = max(f0, f1 + 1)
            f0 = f1 = 0
            for v in nums:
                if v > m:
                    f0 = f1
                else:
                    f0, f1 = f1, max(f1, f0 + 1)
            return f1

        return bisect.bisect_left(range(max(nums) + 1), k, key=check)


# 2561 - Rearranging Fruits - HARD
class Solution:
    # 1. not exchange two big values directly, use another small value as a "tool" to exchange two big values
    # 2. the small value is exactly the minimum value in the basket and does not need to care where it is
    # 3. (optimal) one list is enough

    # O(nlogn) / O(n)
    def minCost(self, basket1: List[int], basket2: List[int]) -> int:
        c1 = collections.Counter(basket1)
        c2 = collections.Counter(basket2)
        cnt = c1 + c2
        if any(v & 1 for v in cnt.values()):
            return -1
        target = collections.Counter({k: v // 2 for k, v in cnt.items()})
        d1 = sorted((target - c1).elements())  # will omit negative values
        d2 = sorted((c1 - target).elements(), reverse=True)
        mi = min(cnt)
        return sum(min(a, b, mi * 2) for a, b in zip(d1, d2))

    def minCost(self, basket1: List[int], basket2: List[int]) -> int:
        cnt = collections.Counter()
        for x, y in zip(basket1, basket2):
            cnt[x] += 1
            cnt[y] -= 1
        mi = min(cnt)
        arr = []
        for k, v in cnt.items():
            if v % 2:
                return -1
            arr.extend([k] * (abs(v) // 2))
        arr.sort()
        return sum(min(v, mi * 2) for v in arr[: len(arr) // 2])
