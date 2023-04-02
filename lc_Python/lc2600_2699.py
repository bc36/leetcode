import bisect, collections, functools, heapq, itertools, math, operator, string
from typing import List, Optional, Tuple
import sortedcontainers

# 2600 - K Items With the Maximum Sum - EASY
class Solution:
    # O(k) / O(1)
    def kItemsWithMaximumSum(
        self, numOnes: int, numZeros: int, numNegOnes: int, k: int
    ) -> int:
        ans = 0
        for _ in range(k):
            if numOnes:
                numOnes -= 1
                ans += 1
            elif numZeros:
                numZeros -= 1
            elif numNegOnes:
                numNegOnes -= 1
                ans -= 1
            k -= 1
        return

    # O(1) / O(1)
    def kItemsWithMaximumSum(
        self, numOnes: int, numZeros: int, numNegOnes: int, k: int
    ) -> int:
        if k <= numOnes + numZeros:
            return min(k, numOnes)
        return numOnes - (k - (numOnes + numZeros))


# 2601 - Prime Subtraction Operation - MEDIUM
def eratosthenes(n: int) -> List[int]:
    primes = []
    is_prime = [True] * (n + 1)
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, n + 1, i):  # 注意是 *, 不是 +, 比 i 小的 i 的倍数已经被枚举过了
                is_prime[j] = False
    return primes


primes = [0] + eratosthenes(1000)


class Solution:
    # O(nlogU) / O(1), U = len(primes)
    def primeSubOperation(self, nums: List[int]) -> bool:
        p = 0
        for x in nums:
            if x <= p:
                return False
            p = x - primes[bisect.bisect_left(primes, x - p) - 1]  # 减去小于 x - p 的最大质数
            # p = x - primes[bisect.bisect_right(primes, x - p - 1) - 1]
        return True


# 2602 - Minimum Operations to Make All Array Elements Equal - MEDIUM
class Solution:
    def minOperations(self, nums: List[int], queries: List[int]) -> List[int]:
        nums.sort()
        p = list(itertools.accumulate(nums, initial=0))
        ans = []
        for q in queries:
            i = bisect.bisect_left(nums, q)
            left = q * i - p[i]
            right = p[-1] - p[i] - q * (len(nums) - i)
            ans.append(left + right)
        return ans


# 2603 - Collect Coins in a Tree - HARD
class Solution:
    # 以树上任意一点为根的欧拉回路长度 = 边数 * 2
    # 1. 先删掉叶节点为0的边, 使得每个树叶都为 1
    # 2. 再删两层边
    # 3. 答案就是 剩下的边数 * 2
    # O(n) / O(n)
    def collectTheCoins(self, coins: List[int], edges: List[List[int]]) -> int:
        n = len(coins)
        ind = [0] * n
        g = [[] for _ in range(n)]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)
            ind[x] += 1
            ind[y] += 1
        useless = collections.deque(i for i in range(n) if ind[i] == 1 and not coins[i])
        removePoint = 0
        while useless:
            x = useless.popleft()
            ind[x] = 0
            for y in g[x]:
                if ind[y] == 0:
                    continue
                removePoint += 1
                ind[y] -= 1
                if ind[y] == 1 and coins[y] == 0:
                    useless.append(y)
        q = collections.deque(i for i in range(n) if ind[i] == 1)
        dist = [0] * n  # distance from the leaf
        while q:
            x = q.popleft()
            ind[x] = 0
            for y in g[x]:
                if ind[y] == 0:
                    continue
                ind[y] -= 1
                removePoint += 1
                dist[y] = max(dist[y], dist[x] + 1)
                if ind[y] == 1 and dist[y] < 2:
                    q.append(y)
        return (n - 1 - removePoint) * 2

    def collectTheCoins(self, coins: List[int], edges: List[List[int]]) -> int:
        n = len(coins)
        g = [[] for _ in range(n)]
        ind = [0] * n
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)
            ind[x] += 1
            ind[y] += 1
        e = n - 1
        useless = collections.deque(i for i in range(n) if ind[i] == 1 and not coins[i])
        while useless:
            x = useless.pop()
            ind[x] -= 1
            e -= 1
            for y in g[x]:
                ind[y] -= 1
                if ind[y] == 1 and coins[y] == 0:
                    useless.append(y)
        useless = collections.deque(i for i in range(n) if ind[i] == 1)
        e -= len(useless)

        # coins = [0,0], edges = [[0,1]]
        # coins = [1,1], edges = [[0,1]]
        # if e <= 2:
        #     return 0

        for x in useless:
            for y in g[x]:
                ind[y] -= 1
                if ind[y] == 1:
                    e -= 1

        # return e * 2
        return max(e * 2, 0)

    def collectTheCoins(self, coins: List[int], edges: List[List[int]]) -> int:
        n = len(coins)
        g = [set() for _ in range(n)]
        for x, y in edges:
            g[x].add(y)
            g[y].add(x)
        # 1. 删除所有的无金币的叶子节点, 直到树中所有的叶子节点都是有金币的, 类似拓扑排序s
        q = {i for i in range(n) if len(g[i]) == 1 and coins[i] == 0}
        while q:
            x = q.pop()
            for y in g[x]:
                g[y].remove(x)
                if len(g[y]) == 1 and coins[y] == 0:
                    q.add(y)
            g[x].clear()
        # 2. 删除树中所有叶子节点, 及其相邻边, 删两次
        for _ in range(2):
            q = [i for i in range(n) if len(g[i]) == 1]
            for x in q:
                for y in g[x]:
                    g[y].remove(x)
                g[x].clear()
        # 3. 答案就是剩下的树中的边数
        # return max(sum(len(g[i]) > 0 for i in range(n)) - 1, 0) * 2
        return sum(2 for x, y in edges if len(g[x]) and len(g[y]))

    # 如果不是 2, 要求任意距离 q 的话
    # cnt = [0] * n
    # for x, y in edges:
    #     cnt[min(dist[x], dist[y])] += 1
    # for i in range(n - 2, -1, -1):
    #     cnt[i] += cnt[i + 1]
    # cnt[q] 即是对应距离 q 所需要遍历的边数
    def collectTheCoins(self, coins: List[int], edges: List[List[int]]) -> int:
        n = len(coins)
        g = [[] for _ in range(n)]
        ind = [0] * n
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)
            ind[x] += 1
            ind[y] += 1
        q = collections.deque(i for i in range(n) if ind[i] == 1 and not coins[i])
        while q:
            for y in g[q.popleft()]:
                ind[y] -= 1
                if ind[y] == 1 and coins[y] == 0:
                    q.append(y)
        q = collections.deque(i for i in range(n) if ind[i] == 1 and coins[i])
        if len(q) <= 1:
            return 0
        dist = [0] * n
        while q:
            x = q.popleft()
            for y in g[x]:
                ind[y] -= 1
                if ind[y] == 1:
                    dist[y] = dist[x] + 1
                    q.append(y)
        return sum(2 for x, y in edges if dist[x] >= 2 and dist[y] >= 2)


# 2605 - Form Smallest Number From Two Digit Arrays - EASY
class Solution:
    def minNumber(self, nums1: List[int], nums2: List[int]) -> int:
        ans = 100
        for x in nums1:
            for y in nums2:
                if x == y:
                    ans = min(ans, x)
                else:
                    ans = min(ans, x * 10 + y, y * 10 + x)
        return ans

    def minNumber(self, nums1: List[int], nums2: List[int]) -> int:
        s = set(nums1) & set(nums2)
        if len(s) > 0:
            return min(s)
        x = min(nums1)
        y = min(nums2)
        return min(x * 10 + y, y * 10 + x)


# 2606 - Find the Substring With Maximum Cost - MEDIUM
class Solution:
    # O(n) / O(n)
    def maximumCostSubstring(self, s: str, chars: str, vals: List[int]) -> int:
        d = dict(zip(chars, vals))
        for i in range(0, 26):
            if chr(i + ord("a")) not in d:
                d[chr(i + ord("a"))] = i + 1
        f = [0] * (len(s) + 1)
        for i in range(1, len(s) + 1):
            f[i] = max(f[i - 1] + d[s[i - 1]], d[s[i - 1]])
        return max(f)

    # O(n) / O(26)
    def maximumCostSubstring(self, s: str, chars: str, vals: List[int]) -> int:
        ans = cur = 0
        d = {c: v for c, v in zip(string.ascii_lowercase, range(1, 27))}
        d.update(zip(chars, vals))
        for c in s:
            cur += d[c]
            if cur < 0:
                cur = 0
            ans = max(ans, cur)
        return ans


# 2607 - Make K-Subarray Sums Equal - MEDIUM
class Solution:
    # O(nlogn) / O(n)
    def makeSubKSumEqual(self, arr: List[int], k: int) -> int:
        n = len(arr)
        vis = [False] * n
        ans = 0
        for i in range(n):
            if not vis[i]:
                x = i
                l = []
                while not vis[x]:
                    vis[x] = True
                    l.append(arr[x])
                    x = (x + k) % n
                mid = sorted(l)[len(l) // 2]

                # 注意, 最大值和最小值移动到 target 的和是一定的
                # 不断去除不影响结果的 最大/小值 之后, 若剩余偶数, 任选一个即可
                # mid = sorted(l)[(len(l) - 1) // 2]

                ans += sum(abs(x - mid) for x in l)
        return ans

    # 按照 i mod k 的结果将 arr 分组, 让每组 l 的所有元素相等的最少运算次数 之和
    # 另有结论: 一个循环数组如果既有周期 n 又有周期 k 那么必然有周期 g = gcd(n, k)
    # 由裴蜀定理可证明: arr[i] = arr[i + nx + ky] = a[i + g]
    # 从而转换成不是循环数组的情况
    def makeSubKSumEqual(self, arr: List[int], k: int) -> int:
        g = math.gcd(k, len(arr))
        ans = 0
        for i in range(g):
            l = sorted(arr[i::g])
            mid = l[len(l) // 2]
            ans += sum(abs(x - mid) for x in l)
        return ans


# 2608 - Shortest Cycle in a Graph - HARD
class Solution:
    # 最小环模板题
    # 枚举所有边, 每次把一条边 u - v 从图中删掉, 然后求从 u 出发, 不经过 u - v 到达 v 的最短路, 这条最短路, 加上被删掉的边 u - v 就是一个环
    # 而且由于我们求了从 u 到 v 的最短路, 这个环就是包含 u - v 的最小环
    # 因此, 实际上就是在枚举, 真正的最小环包含图中的哪一条边
    # 取所有答案的最小值, 就是真正的最小环
    # 因为边长都是 1 只要通过 BFS 即可求最短路 / 边长不是 1, dijkstra
    # O(ne) / O(n + e), e = len(edges)
    def findShortestCycle(self, n: int, edges: List[List[int]]) -> int:
        g = [[] for _ in range(n)]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)
        ans = math.inf
        for x in range(n):
            dist = {}
            q = collections.deque([(x, -1, 0)])
            while q:
                x, p, d = q.popleft()
                if x in dist:  # 第二次遇到, 由于是 BFS, 后面不会遇到更短的环
                    ans = min(ans, d + dist[x])
                    break
                dist[x] = d
                for y in g[x]:
                    if y != p:
                        q.append((y, x, d + 1))
        return ans if ans < math.inf else -1

    def findShortestCycle(self, n: int, edges: List[List[int]]) -> int:
        g = [[] for _ in range(n)]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)

        def bfs(start: int) -> int:
            dist = [-1] * n  # dist[i] 表示从 start 到 i 的最短路长度
            dist[start] = 0
            q = collections.deque([(start, -1)])
            while q:
                x, fa = q.popleft()
                for y in g[x]:
                    if dist[y] < 0:
                        dist[y] = dist[x] + 1
                        q.append((y, x))
                    elif y != fa:
                        return dist[x] + dist[y] + 1
            return math.inf

        def bfs2(start: int) -> int:
            dist = [-1] * n
            dist[start] = 0
            q = collections.deque([start])
            while q:
                x = q.popleft()
                for y in g[x]:
                    if dist[y] < 0:
                        dist[y] = dist[x] + 1
                        q.append(y)
                    # 由于是 bfs, 可以用距离判断是不是父节点
                    elif dist[y] + 1 != dist[x]:
                        return dist[x] + dist[y] + 1
            return math.inf

        ans = min(bfs(i) for i in range(n))
        return ans if ans < math.inf else -1

    # 枚举点, 在BFS的过程中, 如果发现有两条路径都到达了 v, 更新答案 ans = min(ans, 两条路径之和)

    # 1 -  2 - 3 - 5
    #      |     /
    #      4  --
    #
    # 1 —> 2 —> 3 —> 5,  1 —> 2 —> 4 —> 5, 这两条路径其实没有环, 但是会在 BFS(2) 时排除

    # O(n * (n + e)) / O(n + e), e = len(edges)
    def findShortestCycle(self, n: int, edges: List[List[int]]) -> int:
        g = [[] for _ in range(n)]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)
        ans = math.inf
        for x in range(n):
            vis = [-1] * n
            pre = [-1] * n
            vis[x] = 0
            q = collections.deque([x])
            while q:
                x = q.popleft()
                for y in g[x]:
                    if vis[y] == -1:
                        vis[y] = vis[x] + 1
                        pre[y] = x
                        q.append(y)
                    elif y != pre[x]:
                        l = vis[x] + vis[y] + 1
                        ans = min(ans, l)
        return -1 if ans == math.inf else ans
