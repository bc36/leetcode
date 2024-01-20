import bisect, collections, functools, heapq, itertools, math, operator, string
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


# 2706 - Buy Two Chocolates - EASY
class Solution:
    # O(nlogn) / O(1)
    def buyChoco(self, prices: List[int], money: int) -> int:
        prices.sort()
        return money if prices[0] + prices[1] > money else money - prices[0] - prices[1]

    # O(nlogn) / O(1)
    def buyChoco(self, prices: List[int], money: int) -> int:
        v = sum(heapq.nsmallest(2, prices))
        return money if v > money else money - v


# 2707 - Extra Characters in a String - MEDIUM
class Solution:
    # O(n^3) / O(n)
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        d = set(dictionary)

        @functools.lru_cache(None)
        def dfs(i: int) -> int:
            if i < 0:
                return 0
            cur = dfs(i - 1) + 1
            for j in range(i + 1):
                if s[j : i + 1] in d:
                    cur = min(cur, dfs(j - 1))
            return cur

        return dfs(len(s) - 1)

    # O(n^3) / O(n)
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dictionary = set(dictionary)
        f = [0] + [n] * n
        for i in range(n):
            f[i + 1] = f[i] + 1
            for j in range(i + 1):
                if s[j : i + 1] in dictionary:
                    f[i + 1] = min(f[i + 1], f[j])
        return f[n]


# 2708 - Maximum Strength of a Group - MEDIUM
class Solution:
    # O(n) / O(n)
    def maxStrength(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        pos = [v for v in nums if v > 0]
        neg = [v for v in nums if v < 0]
        # 这个表达式整合了奇数(不小于3)和偶数(不小于2)个负数, 0 或 1 个负数则不选负数
        # 注意只要数组元素不止1个, 答案最小也是0
        negP = (
            functools.reduce(operator.mul, neg) // (max(neg) ** (len(neg) % 2))
            if len(neg) >= 2
            else 0
        )
        if not pos:
            return negP
        posP = functools.reduce(operator.mul, pos)
        return posP * negP if negP else posP

    # O(n) / O(n), dp
    def maxStrength(self, nums: List[int]) -> int:
        mx = mi = nums[0]
        for x in nums[1:]:
            tmp = mx
            mx = max(mx, x, mx * x, mi * x)
            mi = min(mi, x, tmp * x, mi * x)
        return mx

    # O(2**n * n) / O(1)
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        ans = -math.inf
        for i in range(1, 1 << n):
            p = 1
            for j in range(n):
                if (i >> j) & 1:
                    p *= nums[j]
            ans = max(ans, p)
        return ans


# 2709 - Greatest Common Divisor Traversal - HARD
@functools.lru_cache(None)
def get_prime_factor(n: int) -> set:
    if n == 1:
        return set()
    ans = set()
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            n //= i
            ans.add(i)
            ans = ans.union(get_prime_factor(n))
            return ans
    ans.add(n)
    return ans


class Solution:
    # O(nlogU) / O(n), 边很少, log(U)
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        if 1 in nums:
            return False

        class UnionFind:
            def __init__(self, n: int) -> None:
                self.p = [i for i in range(n)]
                self.sz = [1] * n

            def find(self, x: int) -> int:
                if self.p[x] != x:
                    self.p[x] = self.find(self.p[x])
                return self.p[x]

            def union(self, x: int, y: int) -> None:
                """x's root = y"""
                px = self.find(x)
                py = self.find(y)
                if px == py:
                    return
                self.p[px] = py
                return

        uf = UnionFind(len(nums) + max(nums) + 1)
        for x in nums:
            for y in get_prime_factor(x):
                uf.union(x, y)
        return len(set(uf.find(x) for x in nums)) == 1
        root = uf.find(nums[0])
        return all(uf.find(x) == root for x in nums)


# 2719 - Count of Integers - HARD
class Solution:
    # O(nmD) / O(nm), n = len(nums2), m = min(9n, max_sum), D = 10
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        @functools.cache
        def dfs(i: int, cur: int, is_limit: bool, is_num: bool, s: str) -> int:
            if i == len(s):
                return min_sum <= cur + d <= max_sum
                return int(is_num)
            ans = 0
            if not is_num:
                ans = dfs(i + 1, cur, False, False, s)
            bound = int(s[i]) if is_limit else 9
            for d in range(0 if is_num else 1, bound + 1):
                # if min_sum <= cur + d <= max_sum:  #  放这里的话, 遇到大的 num1 num2, 都无法进入 if 循环
                ans += dfs(i + 1, cur + d, is_limit and d == bound, True, s)
            return ans % 1000000007

        return (
            dfs(0, 0, True, False, num2)
            - dfs(0, 0, True, False, str(int(num1) - 1))
            + 1000000007
        ) % 1000000007

    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        """前导 0 对数位和没有影响可以省略"""

        @functools.cache
        def dfs(i: int, cur: int, is_limit: bool, s: str) -> int:
            if cur > max_sum:
                return 0
            if i == len(s):
                return cur >= min_sum
            ans = 0
            bound = int(s[i]) if is_limit else 9
            for d in range(bound + 1):
                ans += dfs(i + 1, cur + d, is_limit and d == bound, s)
            return ans % 1000000007

        return (
            dfs(0, 0, True, num2) - dfs(0, 0, True, str(int(num1) - 1)) + 1000000007
        ) % 1000000007

    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        def calc(s: str):
            @functools.cache
            def dfs(i: int, cur: int, is_limit: bool) -> int:
                if cur > max_sum:
                    return 0
                if i == len(s):
                    return cur >= min_sum
                ans = 0
                bound = int(s[i]) if is_limit else 9
                for d in range(bound + 1):
                    ans += dfs(i + 1, cur + d, is_limit and d == bound)
                return ans % 1000000007

            return dfs(0, 0, True)

        return (calc(num2) - calc(str(int(num1) - 1))) % 1000000007

    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        """数位dp 2.0模版 - https://leetcode.cn/problems/count-of-integers/solutions/2296043/shu-wei-dp-tong-yong-mo-ban-pythonjavacg-9tuc/"""
        n = len(num2)
        num1 = "0" * (n - len(num1)) + num1  # 补前导零

        @functools.cache
        def dfs(i: int, s: int, limit_low: bool, limit_high: bool) -> int:
            if s > max_sum:
                return 0
            if i == n:
                return s >= min_sum
            lo = int(num1[i]) if limit_low else 0
            hi = int(num2[i]) if limit_high else 9
            res = 0
            for d in range(lo, hi + 1):  # 枚举当前数位填 d
                res += dfs(i + 1, s + d, limit_low and d == lo, limit_high and d == hi)
            return res

        return dfs(0, 0, True, True) % 1000000007


# 2729 - Check if The Number is Fascinating - EASY
class Solution:
    def isFascinating(self, n: int) -> bool:
        s = str(n) + str(n * 2)[-3:] + str(n * 3)[-3:]
        cnt = collections.Counter(s)
        return len(cnt) == 9 and "0" not in cnt

    def isFascinating(self, n: int) -> bool:
        if n < 123 or n > 329:
            return False
        s = str(n) + str(n * 2) + str(n * 3)
        return "0" not in s and len(set(s)) == 9

    def isFascinating(self, n: int) -> bool:
        return n in (192, 219, 273, 327)


# 2730 - Find the Longest Semi-Repetitive Substring - MEDIUM
class Solution:
    # O(n) / O(1)
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        i = has = 0
        ans = 1
        for j in range(1, len(s)):
            has += s[j] == s[j - 1]
            if has < 2:
                ans = max(ans, j - i + 1)
            while has == 2:
                i += 1
                has -= s[i] == s[i - 1]
        return ans


# 2731 - Movement of Robots - MEDIUM
class Solution:
    # O(nlogn) / O(n)
    def sumDistance(self, nums: List[int], s: str, d: int) -> int:
        mod = 10**9 + 7
        arr = sorted(x + d if y == "R" else x - d for x, y in zip(nums, s))
        ans = pre = 0
        for i, x in enumerate(arr):
            ans = (ans + i * x - pre) % mod
            pre += x
        return ans

    # O(nlogn) / O(1)
    def sumDistance(self, nums: List[int], s: str, d: int) -> int:
        mod = 10**9 + 7
        for i, c in enumerate(s):
            nums[i] += d if c == "R" else -d
        nums.sort()
        ans = pre = 0
        for i, x in enumerate(nums):
            # ans = (ans + i * x - pre) % mod
            ans += i * x - pre
            pre += x
        return ans % mod  # 可能 ans 不是很大, 最后再进行取模计算甚至运行更快


# 2732 - Find a Good Subset of the Matrix - HARD
class Solution:
    def goodSubsetofBinaryMatrix(self, grid: List[List[int]]) -> List[int]:
        d = collections.defaultdict(list)
        for i, row in enumerate(grid):
            mask = 0
            for x in row:
                mask *= 2
                mask += x
            d[mask].append(i)
        k = list(d.keys())
        if 0 in k:
            return [d[0][0]]
        for i in range(len(k)):
            for j in range(i + 1, len(k)):
                if k[i] & k[j] == 0:
                    return [d[k[i]][0], d[k[j]][0]]
        return []

    def goodSubsetofBinaryMatrix(self, grid: List[List[int]]) -> List[int]:
        d = {}
        for i, row in enumerate(grid):
            mask = 0
            for j, x in enumerate(row):
                mask |= x << j
            d[mask] = i
        if 0 in d:
            return [d[0]]
        for x, i in d.items():
            for y, j in d.items():
                if (x & y) == 0:
                    return sorted((i, j))
        return []


# 2733 - Neither Minimum nor Maximum - EASY
class Solution:
    # O(n) / O(1)
    def findNonMinOrMax(self, nums: List[int]) -> int:
        mi = min(nums)
        mx = max(nums)
        for v in nums:
            if v != mi and v != mx:
                return v
        return -1

    # O(1) / O(1)
    def findNonMinOrMax(self, nums: List[int]) -> int:
        return sorted(nums[:3])[1] if len(nums) > 2 else -1


# 2734. Lexicographically Smallest String After Substring Operation - MEDIUM
class Solution:
    def smallestString(self, s: str) -> str:
        t = list(s)
        for i, c in enumerate(t):
            if c != "a":
                for j in range(i, len(t)):
                    if t[j] == "a":
                        break
                    t[j] = chr(ord(t[j]) - 1)
                return "".join(t)
        t[-1] = "z"
        return "".join(t)


# 2735. Collecting Chocolates - MEDIUM
class Solution:
    # 如果不操作, 第 i 个巧克力必须花费 nums[i] 收集, 总成本为所有 nums[i] 之和
    # 如果操作一次, 第 i 个巧克力可以花费 min(nums[i], nums[(i + 1) % n]) 收集
    # 如果操作两次, 第 i 个巧克力可以花费 min(nums[i], nums[(i + 1) % n], nums[(i + 2) % n]) 收集
    # ... 暴力枚举
    # O(n^3) / O(n)
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        ans = sum(nums)
        mi = [x for x in nums]
        for i in range(1, n):
            for j in range(n):
                mi[j] = min(mi[j], nums[(j + n - i) % n])
            ans = min(ans, sum(mi) + x * i)
        return ans

    # 枚举旋转的次数
    def minCost(self, nums: List[int], x: int) -> int:
        ans = sum(nums)
        tmp = nums[:]
        for i in range(1, len(nums)):
            tmp = tmp[1:] + [tmp[0]]
            nums = [min(x, y) for x, y in zip(nums, tmp)]
            ans = min(ans, sum(nums) + x * i)
        return ans

    # O(n^2) / O(n), s[i] 对应操作 i 次的总成本
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        s = list(range(0, n * x, x))
        for i, mi in enumerate(nums):
            for j in range(i, n + i):
                mi = min(mi, nums[j % n])
                s[j - i] += mi  # 累加操作 j-i 次的成本
        return min(s)

    # 或者 预处理 f[i][j] 表示进行 j 次类型修改操作后, 类型为 i 的巧克力的最小代价


# 2736. Maximum Sum Queries - HARD
class Solution:
    # 先把 nums1 和 询问 queries 按照 xi 排序
    # 可以按照 x 从大到小, nums1[j] 从大到小的顺序处理, 同时增量地维护满足 nums1[j] >= xi 的 nums2[j]
    # 分类讨论 nums2[j] 和之前遍历过的 nums2[k] 的大小关系, 注意 nums1 是从大到小遍历的
    # O(n + qlogn) / O(n)
    def maximumSumQueries(
        self, nums1: List[int], nums2: List[int], queries: List[List[int]]
    ) -> List[int]:
        ans = [-1] * len(queries)
        st = []
        arr = sorted((a, b) for a, b in zip(nums1, nums2))
        idx = len(arr) - 1
        for qid, (x, y) in sorted(enumerate(queries), key=lambda p: -p[1][0]):
            while idx >= 0 and arr[idx][0] >= x:
                ax, ay = arr[idx]
                while st and st[-1][1] <= ax + ay:  # ay >= st[-1][0]
                    st.pop()
                if not st or st[-1][0] < ay:
                    st.append((ay, ax + ay))
                idx -= 1
            p = bisect.bisect_left(st, (y,))
            if p < len(st):
                ans[qid] = st[p][1]
        return ans

    def maximumSumQueries(
        self, nums1: List[int], nums2: List[int], queries: List[List[int]]
    ) -> List[int]:
        ans = [-1] * len(queries)
        st = []
        arr = sorted(zip(nums1, nums2))
        for i in range(len(queries)):
            queries[i].append(i)
        queries.sort(key=lambda x: -x[0])  # 按 x 从大到小排序
        idx = len(nums1) - 1
        for x, y, qid in queries:
            while idx > -1 and arr[idx][0] >= x:  # 是不是可以加入这个区间
                ax, ay = arr[idx]
                while st and st[-1][1] <= ax + ay:  # 要么是一个空的栈 要么就是 无用的数据
                    st.pop()
                if not st or ay > st[-1][0]:
                    st.append((ay, ax + ay))
                idx -= 1
            # ax + ay 从低到顶增加 ay 从低到顶减少
            p = bisect.bisect_left(st, (y,))
            if p != len(st):
                ans[qid] = st[p][1]
        return ans

    # 线段树, 树状数组, ST表


# 2739 - Total Distance Traveled - EASY
class Solution:
    # O(mainTank) / O(1)
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        ans = 0
        while mainTank >= 5:
            mainTank -= 5
            ans += 50
            if additionalTank:
                additionalTank -= 1
                mainTank += 1
        return ans + mainTank * 10

    # O(1) / O(1)
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        return (mainTank + min(additionalTank, (mainTank - 1) // 4)) * 10


# 2740 - Find the Value of the Partition - MEDIUM
class Solution:
    # O(nlogn) / O(1)
    def findValueOfPartition(self, nums: List[int]) -> int:
        nums.sort()
        return min(y - x for x, y in pairwise(nums))


# 2741 - Special Permutations - MEDIUM
class Solution:
    # m: 2^n 种, j: n 种
    # 时间复杂度 = O(状态个数) * O(单个状态的计算时间)
    #          = O(n * 2^n) * O(n)
    #          = O(n^2 * 2^n)
    # 空间复杂度 = O(状态个数)
    #          = O(n * 2^n)
    # O(n^2 * 2^n) / O(n * 2^n)
    def specialPerm(self, nums: List[int]) -> int:
        @functools.lru_cache(None)
        def dfs(m: int, j: int) -> int:
            """m 表示当前可以选的下标集合(状态), j 表示上一个选的数的下标是 j"""
            if m == 0:
                return 1
            res = 0
            for k, x in enumerate(nums):
                if m >> k & 1 and (nums[j] % x == 0 or x % nums[j] == 0):
                    res += dfs(m ^ (1 << k), k)
            return res

        u = (1 << len(nums)) - 1  # 全集
        return sum(dfs(u ^ (1 << i), i) for i in range(len(nums))) % 1000000007

    def specialPerm(self, nums: List[int]) -> int:
        @functools.lru_cache(None)
        def dfs(state: int, last: int) -> int:
            """state 表示当前集合状态, last 表示上一个选的数"""
            if state == (1 << len(nums)) - 1:
                return 1
            res = 0
            for i in range(len(nums)):
                if (1 << i) & state:  # 这一位(的数字)已经选过了, 跳过
                    continue
                if nums[i] % last == 0 or last % nums[i] == 0:
                    res += dfs(state | (1 << i), nums[i])
            return res % 1000000007

        return dfs(0, 1)


# 2742 - Painting the Walls - HARD
class Solution:
    # 1. 选或不选
    # 2. 枚举选哪个
    # 是 dp 经常需要思考的问题
    # 时间复杂度 = O(状态个数) * O(单个状态的计算时间) = O(n^2) * O(1)
    # 空间复杂度 = O(状态个数) = O(n^2)
    # O(n^2) / O(n^2)
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        @functools.lru_cache(None)
        def dfs(i: int, j: int) -> int:
            """刷完 0 ~ i 的墙, 且当前累计付费时间为 j 的最小开销"""
            if j > i:  # 剩余的墙都可以免费刷
                return 0
            if i < 0:
                return math.inf
            return min(dfs(i - 1, j + time[i]) + cost[i], dfs(i - 1, j - 1))  # 付费 / 不付费

        return dfs(len(cost) - 1, 0)

    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        @functools.lru_cache(None)
        def dfs(i: int, free: int) -> int:
            if free >= len(cost):  # 攒够足以 cover 所有墙的 free
                return 0
            if i == len(cost):  # i 已经到了最后, 但是还没攒够 free
                return math.inf
            return min(dfs(i + 1, free + time[i] + 1) + cost[i], dfs(i + 1, free))

        return dfs(0, 0)

    # O(n^2) / O(n), [至少装满型] 0/1 背包
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        f = [0] + [math.inf] * n  # f[i][0] 表示 j<=0 的状态
        for c, t in zip(cost, time):
            for j in range(n, 0, -1):
                f[j] = min(f[j], f[max(j - t - 1, 0)] + c)
        return f[n]

    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        f = [0] + [math.inf] * n
        for c, t in zip(cost, time):
            g = f.copy()
            for j in range(0, n + 1):
                g[min(j + 1 + t, n)] = min(g[min(j + 1 + t, n)], f[j] + c)
            f = g
        return f[n]


# 2744 - Find Maximum Number of String Pairs - EASY
class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        ans = 0
        vis = set()
        for s in words:
            if s[::-1] in vis:
                ans += 1
            else:
                vis.add(s)
        return ans


# 2745 - Construct the Longest New String - MEDIUM
class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        return 2 * (min(x, y) * 2 + (x != y) + z)


# 2746 - Decremental String Concatenation - MEDIUM
class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        @functools.lru_cache(None)
        def dfs(i: int, start: str, end: str) -> int:
            if i == len(words):
                return 0
            w = words[i]
            return len(w) + min(
                dfs(i + 1, w[0], end) - (w[-1] == start),  # 接在开头
                dfs(i + 1, start, w[-1]) - (w[0] == end),  # 接在末尾
            )

        return len(words[0]) + dfs(1, words[0][0], words[0][-1])


# 2747 - Count Zero Request Servers - HARD
class Solution:
    # 维护一个 window, 收到请求的 server
    def countServers(
        self, n: int, logs: List[List[int]], x: int, queries: List[int]
    ) -> List[int]:
        logs.sort(key=lambda x: x[1])
        ans = [0] * len(queries)
        l = r = 0
        vis = dict()
        for qi, q in sorted(enumerate(queries), key=lambda x: x[1]):
            while r < len(logs) and logs[r][1] <= q:  # 进入 window
                vis[logs[r][0]] = vis.get(logs[r][0], 0) + 1
                r += 1
            while l < len(logs) and logs[l][1] < q - x:  # 退出 window
                # Q: no need to determine if key is in dict, why?
                # A: dict[key] always >= 0, since it is accumulated from "right" and reduced from "left"
                vis[logs[l][0]] -= 1
                if vis[logs[l][0]] == 0:
                    del vis[logs[l][0]]
                l += 1
            ans[qi] = n - len(vis)  # 总数 - 当前在 window 内的
        return ans

    def countServers(
        self, n: int, logs: List[List[int]], x: int, queries: List[int]
    ) -> List[int]:
        logs.sort(key=lambda x: x[1])
        ans = [0] * len(queries)
        l = r = 0
        cnt = [0] * (n + 1)
        outOfRange = n
        for qi, q in sorted(enumerate(queries), key=lambda x: x[1]):
            while r < len(logs) and logs[r][1] <= q:
                if cnt[logs[r][0]] == 0:
                    outOfRange -= 1
                cnt[logs[r][0]] += 1
                r += 1
            while l < len(logs) and logs[l][1] < q - x:
                cnt[logs[l][0]] -= 1
                if cnt[logs[l][0]] == 0:
                    outOfRange += 1
                l += 1
            ans[qi] = outOfRange
        return ans


# 2748 - Number of Beautiful Pairs - EASY
class Solution:
    # O(n^2 * logU) / O(1), U = max(nums)
    def countBeautifulPairs(self, nums: List[int]) -> int:
        ans = 0
        for i in range(len(nums) - 1):
            for j in range(i + 1, len(nums)):
                while nums[i] >= 10:
                    nums[i] //= 10
                if math.gcd(nums[i], nums[j] % 10) == 1:
                    ans += 1
        return ans

    # O(n * (10 + logU)) / O(10)
    def countBeautifulPairs(self, nums: List[int]) -> int:
        ans = 0
        cnt = [0] * 10
        for x in nums:
            for y in range(1, 10):
                if cnt[y] and math.gcd(x % 10, y) == 1:
                    ans += cnt[y]
            while x >= 10:
                x //= 10
            cnt[x] += 1
        return ans


# 2749 - Minimum Operations to Make the Integer Zero - MEDIUM
class Solution:
    # 难点: 2^i + num2 有正有负
    # 问题转换: 操作 k 次后, num1 = k (2^i + num2)
    # 即 x = num1 - num2 * k 能否拆分成 k 个 2^i ?
    # k 的范围 [x.bit_count(), x], 仔细计算会发现 k 不会超过 36
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        # for k in range(1, 64):
        for k in itertools.count(1):
            x = num1 - num2 * k
            if x < k:
                return -1
            if k >= x.bit_count():
                return k


# 2750 - Ways to Split Array Into Good Subarrays - MEDIUM
class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        ans = 1
        pre = -1
        for i, x in enumerate(nums):
            if x == 0:
                continue
            if pre >= 0:
                ans = ans * (i - pre) % mod
            pre = i
        return 0 if pre < 0 else ans

    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        indexs = []
        for i in range(len(nums)):
            if nums[i] == 1:
                indexs.append(i)
        if len(indexs) == 0:
            return 0
        ans = 1
        for i in range(1, len(indexs)):
            ans *= indexs[i] - indexs[i - 1]
            ans %= mod
        return ans


# 2751 - Robot Collisions - HARD
class Solution:
    def survivedRobotsHealths(
        self, positions: List[int], healths: List[int], directions: str
    ) -> List[int]:
        st = []
        arr = sorted(zip(positions, healths, directions))
        toLeft = []
        for p, h, d in arr:
            if d == "R":
                st.append((p, h))
            else:
                while st and h:
                    if st[-1][1] < h:
                        st.pop()
                        h -= 1
                    elif st[-1][1] == h:
                        st.pop()
                        h = 0
                    else:
                        p, r = st.pop()
                        r -= 1
                        if r:
                            st.append((p, r))
                        h = 0
                if h:
                    toLeft.append((p, h))
        idx = {v: i for i, v in enumerate(positions)}
        return list(h for _, h in sorted(toLeft + st, key=lambda x: idx[x[0]]))

    def survivedRobotsHealths(
        self, positions: List[int], healths: List[int], directions: str
    ) -> List[int]:
        st = []
        arr = sorted(
            zip(range(len(positions)), positions, healths, directions),
            key=lambda p: p[1],
        )
        toLeft = []
        for i, _, h, d in arr:
            if d == "R":
                st.append([i, h])
                continue
            while st:
                top = st[-1]
                if top[1] > h:
                    top[1] -= 1
                    break
                if top[1] == h:
                    st.pop()
                    break
                h -= 1
                st.pop()
            else:  # while 循环没有 break, 说明当前机器人把栈中的全部撞掉
                toLeft.append([i, h])
        toLeft += st
        toLeft.sort(key=lambda p: p[0])
        return [h for _, h in toLeft]


# 2760 - Longest Even Odd Subarray With Threshold - EASY
class Solution:
    # O(n^3) / O(1)
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        for l in range(n, 0, -1):  # l: length
            for i in range(n - l + 1):
                if (
                    nums[i] % 2 == 0
                    and all(nums[j] % 2 != nums[j + 1] % 2 for j in range(i, i + l - 1))
                    and all(nums[j] <= threshold for j in range(i, i + l))
                ):
                    return l

        return 0

    # O(n) / O(1), 题目的约束实际上把数组划分成了若干段, 每段都满足要求, 且互不相交
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        ans = j = 0
        while j < n:
            if nums[j] % 2 or nums[j] > threshold:
                j += 1
            else:
                i = j
                j += 1
                while j < n and nums[j] <= threshold and nums[j] % 2 != nums[j - 1] % 2:
                    j += 1
                ans = max(ans, j - i)
        return ans


# 2761 - Prime Pairs With Target Sum - MEDIUM
def eratosthenes(n: int) -> List[int]:
    """[2, x] 内的质数"""
    primes = []
    isPrime = [True] * (n + 1)
    for i in range(2, n + 1):
        if isPrime[i]:
            primes.append(i)
            for j in range(i * i, n + 1, i):  # 注意是 *, 不是 +, 比 i 小的 i 的倍数已经被枚举过了
                isPrime[j] = False
    return primes


primes = eratosthenes(10**6)
sp = set(primes)


class Solution:
    def findPrimePairs(self, n: int) -> List[List[int]]:
        ans = []
        for i in range(len(primes)):
            x = primes[i]
            if x > n // 2:
                break
            if n - x in sp:
                ans.append([x, n - x])
        return ans


# 2762 - Continuous Subarrays - MEDIUM
class Solution:
    # O(nlogn) / O(n)
    def continuousSubarrays(self, nums: List[int]) -> int:
        sl = sortedcontainers.SortedList()
        ans = l = 0
        for r, v in enumerate(nums):
            sl.add(v)
            while sl and sl[-1] - sl[0] > 2:
                sl.remove(nums[l])
                l += 1
            ans += r - l + 1
        return ans

    # O(n) / O(n)
    def continuousSubarrays(self, nums: List[int]) -> int:
        ans = l = 0
        cnt = collections.Counter()
        for r, x in enumerate(nums):
            cnt[x] += 1
            while max(cnt) - min(cnt) > 2:
                y = nums[l]
                cnt[y] -= 1
                if cnt[y] == 0:
                    del cnt[y]
                l += 1
            ans += r - l + 1
        return ans


# 2763 - Sum of Imbalance Numbers of All Subarrays - HARD
class Solution:
    # O(n^2 * logn) / O(n)
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        ans = 0
        for r, x in enumerate(nums):
            sl = sortedcontainers.SortedList([x])
            cnt = 0
            for l in range(r - 1, -1, -1):
                p = sl.bisect_left(nums[l])
                if p > 0:
                    cnt += nums[l] - sl[p - 1] > 1
                if p < len(sl):
                    cnt += sl[p] - nums[l] > 1
                if 0 < p < len(sl):
                    cnt -= sl[p] - sl[p - 1] > 1
                sl.add(nums[l])
                ans += cnt
        return ans

    # O(n^2) / O(n)
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        ans = 0
        for i, x in enumerate(nums):
            vis = [False] * (len(nums) + 2)
            vis[x] = True
            cnt = 0
            for j in range(i + 1, len(nums)):
                x = nums[j]
                if not vis[x]:
                    cnt += 1 - vis[x - 1] - vis[x + 1]
                    vis[x] = True
                ans += cnt
        return ans

    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        ans = 0
        vis = [-1] * (len(nums) + 2)  # 避免反复创建 vis 数组
        for i, x in enumerate(nums):
            vis[x] = i
            cnt = 0
            for j in range(i + 1, len(nums)):
                x = nums[j]
                if vis[x] != i:
                    cnt += 1 - (vis[x - 1] == i) - (vis[x + 1] == i)
                    vis[x] = i
                ans += cnt
        return ans

    # O(n) / O(n), 左边可以有 x, 右边没有 x, 且整个子数组都不包含 x - 1 的子数组的个数
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        right = [0] * n  # right[i] 表示 x = nums[i] 右侧最近的 x 或 x - 1 的位置的最小值
        idx = [n] * (n + 1)
        for i in range(n - 1, -1, -1):
            x = nums[i]
            right[i] = min(idx[x], idx[x - 1])
            idx[x] = i
        ans = 0
        idx = [-1] * (n + 1)
        for i in range(n):
            left = idx[nums[i] - 1]
            ans += (i - left) * (right[i] - i)
            idx[nums[i]] = i
        return ans - n * (n + 1) // 2

    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        right = [0] * n  # nums[i] 右侧的 x 和 x-1 的最近下标(不存在时为 n)
        idx = [n] * (n + 1)
        for i in range(n - 1, -1, -1):
            x = nums[i]
            right[i] = min(idx[x], idx[x - 1])
            idx[x] = i
        ans = 0
        idx = [-1] * (n + 1)
        for i, (x, r) in enumerate(zip(nums, right)):
            ans += (i - idx[x - 1]) * (r - i)  # 子数组左端点个数 * 子数组右端点个数
            idx[x] = i
        # 上面计算的时候, 每个子数组的最小值必然可以作为贡献, 而这是不合法的
        # 所以每个子数组都多算了 1 个不合法的贡献
        return ans - n * (n + 1) // 2

    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        # 默认固定 x 在没有 x + 1 的时候有一个贡献, 如果存在 x 的时候不重复算贡献
        left = [0] * n
        idx = [-1] * (n + 2)
        for i in range(n):
            x = nums[i]
            left[i] = idx[x + 1]
            idx[x] = i
        right = [n] * n
        idx = [n] * (n + 2)
        for i in range(n - 1, -1, -1):
            x = nums[i]
            right[i] = min(idx[x + 1], idx[x])
            idx[x] = i
        ans = 0
        for i in range(n):
            ans += (i - left[i]) * (right[i] - i)
        return ans - n * (n + 1) // 2


# 2769 - Find the Maximum Achievable Number - EASY
class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        return num + 2 * t


# 2770 - Maximum Number of Jumps to Reach the Last Index - MEDIUM
class Solution:
    # O(n^2) / O(n)
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        f = [-1] * n
        f[0] = 0
        for j in range(1, n):
            for i in range(j):
                if f[i] >= 0 and -target <= nums[j] - nums[i] <= target:
                    f[j] = max(f[j], f[i] + 1)
        return f[-1]

    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        f = [-math.inf] * n
        f[0] = 0
        for j in range(1, n):
            for i in range(j):
                if -target <= nums[j] - nums[i] <= target:
                    f[j] = max(f[j], f[i] + 1)
        return -1 if f[-1] < 0 else f[-1]


# 2771 - Longest Non-decreasing Subarray From Two Arrays - MEDIUM
class Solution:
    # O(n) / O(n)
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        # f[i][0] 表示以 nums1[i] 结尾的最长非递减子数组长度
        # f[i][1] 表示以 nums2[i] 结尾的最长非递减子数组长度
        f = [[1] * 2 for _ in range(n)]
        for i in range(1, n):
            if nums1[i - 1] <= nums1[i]:
                f[i][0] = max(f[i][0], f[i - 1][0] + 1)
            if nums1[i - 1] <= nums2[i]:
                f[i][1] = max(f[i][1], f[i - 1][0] + 1)
            if nums2[i - 1] <= nums1[i]:
                f[i][0] = max(f[i][0], f[i - 1][1] + 1)
            if nums2[i - 1] <= nums2[i]:
                f[i][1] = max(f[i][1], f[i - 1][1] + 1)
        return max(max(x) for x in f)

    # O(n) / O(1)
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        ans = f0 = f1 = 1
        for (x0, y0), (x1, y1) in pairwise(zip(nums1, nums2)):
            g0 = g1 = 1
            if x0 <= x1:
                g0 = f0 + 1
            if y0 <= x1:
                g0 = max(g0, f1 + 1)
            if x0 <= y1:
                g1 = f0 + 1
            if y0 <= y1:
                g1 = max(g1, f1 + 1)
            f0, f1 = g0, g1
            ans = max(ans, f0, f1)
        return ans


# 2772 - Apply Operations to Make All Array Elements Equal to Zero - MEDIUM
class Solution:
    # 一边遍历原数组, 一边累加标记
    # O(n) / O(n)
    def checkArray(self, nums: List[int], k: int) -> bool:
        d = [0] * (len(nums) + 1)
        cur = 0
        for i, v in enumerate(nums):
            cur += d[i]
            v += cur
            if v == 0:
                continue
            if v < 0 or i + k > len(nums):
                return False
            cur -= v
            d[i + k] = v
        return True

    def checkArray(self, nums: List[int], k: int) -> bool:
        d = [0] * (len(nums) + 1)
        for i, v in enumerate(nums):
            d[i] += d[i - 1]
            v += d[i]
            if v < 0 or v > 0 and i + k > len(nums):
                return False
            d[i] -= v
            d[min(i + k, len(nums))] += v
        return True


# 2778 - Sum of Squares of Special Elements - EASY
class Solution:
    def sumOfSquares(self, nums: List[int]) -> int:
        return sum(v * v for i, v in enumerate(nums, start=1) if len(nums) % i == 0)


# 2779 - Maximum Beauty of an Array After Applying Operation - MEDIUM
class Solution:
    # O(nlogn) / O(1), 排序, 双指针
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        nums.sort()
        mx = l = r = 0
        while r < len(nums):
            while nums[l] + k < nums[r] - k:
                l += 1
            mx = max(mx, r - l + 1)
            r += 1
        return mx

    def maximumBeauty(self, nums: List[int], k: int) -> int:
        nums.sort()
        mx = l = 0
        for r, x in enumerate(nums):
            while x - nums[l] > k * 2:
                l += 1
            mx = max(mx, r - l + 1)
        return mx

    # O(n + U) / O(U), U = max(nums) + k + 2, 离散化差分, 注意值域不要开太大, 以免 TLE
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        cap = max(nums) + k + 2
        d = [0] * cap
        for x in nums:
            d[max(x - k, 0)] += 1
            d[min(x + k, cap) + 1] -= 1
        return max(itertools.accumulate(d))


# 2780 - Minimum Index of a Valid Split - MEDIUM
class Solution:
    # O(n) / O(n)
    def minimumIndex(self, nums: List[int]) -> int:
        most, val = collections.Counter(nums).most_common(1)[0]
        freq = 0
        for i, x in enumerate(nums):
            freq += x == most
            if freq * 2 > i + 1 and (val - freq) * 2 > len(nums) - i - 1:
                return i
        return -1

    def minimumIndex(self, nums: List[int]) -> int:
        cnt = collections.Counter(nums)
        max_freq = max(cnt.values())
        for x in nums:
            if cnt[x] == max_freq:
                most = x
        c1, c2 = 0, cnt[most]
        for i in range(len(nums) - 1):
            if nums[i] == most:
                c1 += 1
                c2 -= 1
            if c1 * 2 > i + 1 and c2 * 2 > len(nums) - 1 - i:
                return i
        return -1


# 2781 - Length of the Longest Valid Substring - HARD
class Solution:
    # O(m + 100n) / O(m), n = len(word), m = len(forbidden), 双指针, 每次尝试向右移动右指针
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        s = set(forbidden)
        ans = l = 0
        for r in range(len(word)):
            # for k in range(10):
            #     if r - k < 0 or r - k < l:
            #         break
            for k in range(min(10, r + 1, r - l + 1)):
                if word[r - k : r + 1] in s:
                    l = r - k + 1
                    break
            ans = max(ans, r - l + 1)
        return ans

    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        s = set(forbidden)
        ans = l = 0
        for r in range(len(word)):
            for i in range(r, max(r - 10, l - 1), -1):
                if word[i : r + 1] in s:
                    l = i + 1
                    break
            ans = max(ans, r - l + 1)
        return ans

    # 字典树预处理, 可以优化查询的过程. 外层从右到左枚举, 内层从左到右(即字典树内顺序)
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        TRIE = lambda: collections.defaultdict(TRIE)
        trie = TRIE()
        for w in forbidden:
            functools.reduce(dict.__getitem__, w, trie)["#"] = True
        r = len(word)
        ans = 0
        for l in range(len(word) - 1, -1, -1):
            t = trie
            for i in range(l, min(l + 10, r)):
                if word[i] not in t:
                    break
                t = t[word[i]]
                if "#" in t:
                    r = i
                    break
            ans = max(ans, r - l)
        return ans


# 2788 - Split Strings by Separator - EASY
class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        return [x for w in words for x in w.split(separator) if x]
        return [s for s in separator.join(words).split(separator) if s]


# 2798 - Number of Employees Who Met the Target - EASY
class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        return sum(v >= target for v in hours)


# 2799 - Count Complete Subarrays in an Array - MEDIUM
class Solution:
    # O(n^2) / O(n)
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        m = len(set(nums))
        ans = 0
        for i in range(len(nums)):
            vis = set()
            for j in range(i, len(nums)):
                vis.add(nums[j])
                if len(vis) == m:
                    ans += len(nums) - j
                    break
        return ans

    def countCompleteSubarrays(self, nums: List[int]) -> int:
        m = len(set(nums))
        ans = 0
        for i in range(len(nums)):
            vis = set()
            for x in nums[i:]:
                vis.add(x)
                ans += len(vis) // m
        return ans

    # 子数组中 不同 元素的数目恰好等于 m 的数目
    # 当右端点固定时, 不断循环直到 [left, right] 不满足要求
    # 合法左端点个数即 left
    # O(n) / O(n)
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        m = len(set(nums))
        cnt = dict()
        ans = l = 0
        for v in nums:
            cnt[v] = cnt.get(v, 0) + 1
            while len(cnt) == m:
                cnt[nums[l]] -= 1
                if cnt[nums[l]] == 0:
                    del cnt[nums[l]]
                l += 1
            ans += l
        return ans
