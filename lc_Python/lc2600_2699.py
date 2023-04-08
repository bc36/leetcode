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
        # d = dict(zip(string.ascii_lowercase, range(1, 27))) | dict(zip(chars, vals))
        d = dict(zip(string.ascii_lowercase, range(1, 27)))
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


# 2609 - Find the Longest Balanced Substring of a Binary String - EASY
class Solution:
    def findTheLongestBalancedSubstring(self, s: str) -> int:
        pre0 = ans = 0
        for k, g in itertools.groupby(s):
            l = len(list(g))
            if k == "0":
                pre0 = l
            else:
                ans = max(ans, min(l, pre0) * 2)
                pre0 = 0
        return ans

    def findTheLongestBalancedSubstring(self, s: str) -> int:
        ans = pre = cur = 0
        for i, c in enumerate(s):
            cur += 1
            if i == len(s) - 1 or c != s[i + 1]:  # 找分界线套路
                if c == "1":
                    ans = max(ans, min(pre, cur) * 2)
                pre = cur
                cur = 0
        return ans

    def findTheLongestBalancedSubstring(self, s: str) -> int:
        for i in range(25, 0, -1):
            if ("0" * i) + ("1" * i) in s:
                return 2 * i
        return 0


# 2610 - Convert an Array Into a 2D Array With Conditions - MEDIUM
class Solution:
    def findMatrix(self, nums: List[int]) -> List[List[int]]:
        cnt = collections.Counter(nums)
        ans = []
        for _ in range(max(cnt.values())):
            l = []
            for k, v in list(cnt.items()):
                l.append(k)
                cnt[k] = v - 1
                if v - 1 == 0:
                    del cnt[k]
            ans.append(l)
        return ans

    def findMatrix(self, nums: List[int]) -> List[List[int]]:
        ans = []
        cnt = collections.Counter(nums)
        for k in cnt:
            for i in range(cnt[k]):
                if i == len(ans):
                    ans.append([])
                ans[i].append(k)
        return ans

    def findMatrix(self, nums: List[int]) -> List[List[int]]:
        ans = []
        cnt = collections.Counter(nums)
        while cnt:
            row = list(cnt)
            ans.append(row)
            for k in row:
                cnt[k] -= 1
                if cnt[k] == 0:
                    del cnt[k]
        return ans


# 2611 - Mice and Cheese - MEDIUM
class Solution:
    # 任意 i, j
    # a[i] + b[j] 选法 1
    # a[j] + b[i] 选法 2
    #
    # 若选法 1 优于 法2
    # -> a[i] + b[j] > a[j] + b[i]
    # -> a[i] - a[j] > b[i] - b[j]
    # -> d[i] > d[j]
    # 说明 d[i] 越大, 越选 a 内的值, 反之选 b 内的值
    # 贪心排序套路: 按差值排序
    def solve(self, reward1: List[int], reward2: List[int], k: int) -> int:
        arr = sorted(
            [(x - y, x, y) for x, y in zip(reward1, reward2)], key=lambda x: -x[0]
        )
        return sum(x for _, x, _ in arr[:k]) + sum(y for _, _, y in arr[k:])

    def miceAndCheese(self, reward1: List[int], reward2: List[int], k: int) -> int:
        arr = sorted([x - y for x, y in zip(reward1, reward2)], reverse=True)
        return sum(reward2) + sum(arr[:k])


# 2612 - Minimum Reverse Operations - HARD
class Solution:
    # 难点一
    #   求反转长度为 k 的子数组, 从位置 i 可以到达的左右边界闭区间
    # 难点二
    #   一个边数为 O(n * k) 的无权图最短路问题

    # 容易想到一个 BFS 的方法
    # 当位置 i 从 BFS 队列中出队时, 枚举翻转哪个子数组, 就能算出位置 i 能跳到哪些位置.
    # 这样做的复杂度是 O(n(n−k)) 的, 因为共有 n 个位置, 且共有 (n−k+1) 个子数组可以翻转.
    # 上述 BFS 做法复杂度较高的原因是, 我们检查了很多没用的"跳跃"(目标位置已经被跳过了, 不需要再检查一次)
    # 所以需要更多性质来优化这个做法
    # 位置 i 可以"跳跃"到的位置是"连续"的(事实上, 是和奇偶性有关的连续)
    # 既然可以跳到的位置是"连续"的, 那么我们可以考虑用两个 有序集合 来保存还没有被跳过的位置(分奇偶)
    # 因为 有序集合 里保存的都是没有被跳过的位置, 我们就不会重复枚举一个已经被跳过的位置,
    # 这样枚举的次数只有 O(n) 次. 总体复杂度 O(nlogn)

    # 翻转后的所有位置组成了一个公差为 2 的等差数列
    # 注意当 i 在数组边界 0 或 n - 1 附近时, 有些位置是无法翻转到的
    # 不考虑边界, [i - k + 1, i + k - 1]
    # 在左边附近, 0 + (k - 1) - i = k - i - 1
    # 在右边附近, (n - k) + (n - 1) - i = 2 * n - k - i - 1
    # 实际范围, [max(i - k + 1, k - i - 1), min(i + k - 1, 2 * n - k - i - 1)]
    def minReverseOperations(
        self, n: int, p: int, banned: List[int], k: int
    ) -> List[int]:
        s = set(banned)
        arr = [i for i in range(n) if i != p and i not in s]
        odd = sortedcontainers.SortedList([x for x in arr if x % 2])
        even = sortedcontainers.SortedList([x for x in arr if x % 2 == 0])
        dist = {p: 0}
        q = collections.deque([p])
        while q:
            x = q.popleft()
            left = max(0, x - k + 1)
            right = min(n - 1, x + k - 1) - k + 1
            left = 2 * left + k - 1 - x
            right = 2 * right + k - 1 - x
            tree = odd if left % 2 else even
            i = tree.bisect_left(left)
            j = tree.bisect_right(right)
            for it in range(i, j):
                y = tree[it]
                dist[y] = dist[x] + 1
                q.append(y)
            for it in range(i, j):
                tree.pop(i)
        return [dist.get(i, -1) for i in range(n)]

    def minReverseOperations(
        self, n: int, p: int, banned: List[int], k: int
    ) -> List[int]:
        kend = k - 1
        s = set(banned) | {p}
        choice = [
            sortedcontainers.SortedList(set(range(0, n, 2)) - s),
            sortedcontainers.SortedList(set(range(1, n, 2)) - s),
        ]

        def rotate(p: int) -> int:
            left = max(p - kend, 0) * 2 + kend - p
            right = min(p + 1, n - kend) * 2 + kend - p
            cur = choice[left % 2]
            res = list(cur.irange(left, right - 1))
            for i in res:
                cur.discard(i)
            return res

        bfs = [p]
        dist = {p: 0}
        for i in bfs:
            d = dist[i] + 1
            for j in rotate(i):
                dist[j] = d
                bfs.append(j)
        return [dist.get(i, -1) for i in range(n)]

    def minReverseOperations(
        self, n: int, p: int, banned: List[int], k: int
    ) -> List[int]:
        ban = set(banned)
        # 把除了 p 和 banned 的所有位置, 按奇偶性放进两个 set 里
        # 这些就是我们还没被跳到的位置
        st = [sortedcontainers.SortedList() for _ in range(2)]
        for i in range(n):
            if i != p and i not in ban:
                st[i % 2].add(i)
        q = collections.deque()
        q.append(p)
        ans = [-1] * n
        ans[p] = 0
        while q:
            cur = q.popleft()
            # 计算可以跳的范围
            left = max(-(k - 1), k - 1 - cur * 2)
            right = min(k - 1, -(k - 1) + (n - cur - 1) * 2)
            # 寻找第一个大于等于 cur + left 的位置, 并开始枚举后面连续的位置
            x = (cur + (k - 1)) % 2
            idx = st[x].bisect_left(cur + left)
            while idx != len(st[x]):
                # 遇到了第一个大于 cur + right 的位置, 结束枚举
                if st[x][idx] > cur + right:
                    break
                # 这个位置还没被跳过, 但是可以从 cur 一步跳过来
                # 更新答案, 并从 set<int> 里去掉
                ans[st[x][idx]] = ans[cur] + 1
                q.append(st[x][idx])
                st[x].remove(st[x][idx])
        return ans

    # 并查集的思路是, 如果要删除一个元素, 那么把它的下标 j 和 j+1 合并, 这样后面删除的时候就会自动跳过已删除的元素
    def minReverseOperations(
        self, n: int, p: int, banned: List[int], k: int
    ) -> List[int]:
        s = set(banned) | {p}
        not_banned = [[], []]
        for i in range(n):
            if i not in s:
                not_banned[i % 2].append(i)
        not_banned[0].append(n)
        not_banned[1].append(n)  # 哨兵

        fa = [list(range(len(not_banned[0]))), list(range(len(not_banned[1])))]

        def find(i: int, x: int) -> int:
            f = fa[i]
            if f[x] != x:
                f[x] = find(i, f[x])
            return f[x]

        def merge(i: int, from_: int, to: int) -> None:
            x, y = find(i, from_), find(i, to)
            fa[i][x] = y

        ans = [-1] * n
        q = [p]
        step = 0
        while q:
            tmp = q
            q = []
            for i in tmp:
                ans[i] = step
                # 从 mn 到 mx 的所有位置都可以翻转到
                mn = max(i - k + 1, k - i - 1)
                mx = min(i + k - 1, n * 2 - k - i - 1)
                a = not_banned[mn % 2]
                j = find(mn % 2, bisect.bisect_left(a, mn))
                while a[j] <= mx:
                    q.append(a[j])
                    merge(mn % 2, j, j + 1)  # 删除 j
                    j = find(mn % 2, j + 1)
            step += 1
        return ans
