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


# 2614 - Prime In Diagonal - EASY
class Solution:
    # 注意 1 不是质数
    # 数据范围 4 * 10^6, nlogn 暴力预处理筛质数会 TLE
    # O(n * sqrt(U)) / O(1), U = max(对角线上最大值)
    def diagonalPrime(self, nums: List[List[int]]) -> int:
        def isPrime(x: int):
            if x == 1:
                return False
            for i in range(2, int(x**0.5) + 1):
                if x % i == 0:
                    return False
            return True

        ans = 0
        for i, row in enumerate(nums):
            for x in row[i], row[-1 - i]:
                if x > ans and isPrime(x):
                    ans = x
        return ans


# 2615 - Sum of Distances - MEDIUM
class Solution:
    def distance(self, nums: List[int]) -> List[int]:
        d = {}
        n = len(nums)
        left = [0] * n
        for i, v in enumerate(nums):
            j, t, pre = d.get(v, (0, 0, 0))
            left[i] += t * (i - j) + pre
            d[v] = (i, t + 1, left[i])
        d.clear()
        right = [0] * n
        for i in range(n - 1, -1, -1):
            j, t, pre = d.get(nums[i], (n - 1, 0, 0))
            right[i] = t * (j - i) + pre
            d[nums[i]] = (i, t + 1, right[i])
        return [a + b for a, b in zip(left, right)]

    def distance(self, nums: List[int]) -> List[int]:
        groups = collections.defaultdict(list)
        for i, x in enumerate(nums):
            groups[x].append(i)
        ans = [0] * len(nums)
        for a in groups.values():
            n = len(a)
            s = list(itertools.accumulate(a, initial=0))
            for j, target in enumerate(a):
                left = target * j - s[j]
                right = s[n] - s[j] - target * (n - j)
                ans[target] = left + right
        return ans

    def distance(self, nums: List[int]) -> List[int]:
        groups = collections.defaultdict(list)
        for i, x in enumerate(nums):
            groups[x].append(i)
        ans = [0] * len(nums)
        for a in groups.values():
            n = len(a)
            s = sum(x - a[0] for x in a)  # a[0] 到其它下标的距离之和
            ans[a[0]] = s
            for i in range(1, n):
                # 从计算 a[i-1] 到计算 a[i], 考虑 s 增加了多少
                s += (i * 2 - n) * (a[i] - a[i - 1])
                ans[a[i]] = s
        return ans


# 2616 - Minimize the Maximum Difference of Pairs - MEDIUM
class Solution:
    # O(nlogn + nlogU) / O(1), U = max(nums) - min(nums)
    def minimizeMax(self, nums: List[int], p: int) -> int:
        nums.sort()

        def check(m: int) -> bool:
            t = i = 0
            while i < n - 1:
                if nums[i + 1] - nums[i] <= m:
                    t += 1
                    i += 1
                i += 1
            return t >= p

        n = len(nums)
        l = 0
        r = max(nums)
        while l < r:
            m = l + r >> 1
            if check(m):
                r = m
            else:
                l = m + 1
        return l

        return bisect.bisect_left(range(nums[-1] - nums[0]), True, key=check)
        return bisect.bisect_left(range(nums[-1] - nums[0]), 1, key=check)

    def minimizeMax(self, nums: List[int], p: int) -> int:
        def check(m: int) -> int:
            t = i = 0
            while i < len(nums) - 1:
                if nums[i + 1] - nums[i] <= m:
                    t += 1
                    i += 1
                i += 1
            return t

        nums.sort()
        return bisect.bisect_left(range(nums[-1] - nums[0]), p, key=check)


# 2617 - Minimum Number of Visited Cells in a Grid - HARD
class Solution:
    # m + n 个优先队列
    # 假如往下走, 前一个位置 (ii, j), ii < i, ii 还需要满足:
    # 1. (ii, j) 能到 (i, j)
    # 2. 到达 (ii, j) 移动次数最少
    # 由条件二可以想到优先队列」来维护所有的 ii, 堆顶为移动次数最少的位置
    # 由条件一可得若堆顶的 ii 不满足要求, 就可以将它永久从优先队列中移除,
    # 因为之后共享同一列 j 的位置 i 只会更大, 更不可能走到
    # O(n * m * log(nm)) / O(nm)
    def minimumVisitedCells(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        f = [[-1] * m for _ in range(n)]  # distance
        f[0][0] = 1
        row = [[] for _ in range(n)]
        col = [[] for _ in range(m)]

        def update(x: int, y: int) -> int:
            return y if x == -1 or x > y else x

        for i in range(n):
            for j in range(m):
                while row[i] and row[i][0][1] + grid[i][row[i][0][1]] < j:
                    heapq.heappop(row[i])
                if row[i]:
                    f[i][j] = update(f[i][j], f[i][row[i][0][1]] + 1)

                while col[j] and col[j][0][1] + grid[col[j][0][1]][j] < i:
                    heapq.heappop(col[j])
                if col[j]:
                    f[i][j] = update(f[i][j], f[col[j][0][1]][j] + 1)

                if f[i][j] != -1:
                    heapq.heappush(row[i], (f[i][j], j))
                    heapq.heappush(col[j], (f[i][j], i))

        return f[n - 1][m - 1]

    def minimumVisitedCells(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        a = [[] for _ in range(m)]

        f = [[-1] * m for _ in range(n)]
        f[0][0] = 1
        for i in range(n):
            q = []
            for j in range(m):
                while len(a[j]) > 0 and a[j][0][1] < i:
                    heapq.heappop(a[j])
                if len(a[j]) > 0:
                    f[i][j] = a[j][0][0] + 1
                while len(q) > 0 and q[0][1] < j:
                    heapq.heappop(q)
                if len(q) > 0 and (f[i][j] == -1 or f[i][j] > q[0][0] + 1):
                    f[i][j] = q[0][0] + 1
                if f[i][j] != -1:
                    heapq.heappush(q, (f[i][j], j + grid[i][j]))
                    heapq.heappush(a[j], (f[i][j], i + grid[i][j]))
        return f[-1][-1]

    def minimumVisitedCells(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        f = [[-1] * n for _ in range(m)]
        f[0][0] = 1
        rows = [sortedcontainers.SortedList(range(0, n)) for _ in range(m)]
        cols = [sortedcontainers.SortedList(range(0, m)) for _ in range(n)]
        q = [[0, 0]]
        index = 0
        while index < len(q):
            x, y = q[index]
            index += 1
            sl = rows[x]
            while True:
                pos = sl.bisect_left(y + 1)
                if pos == len(sl):
                    break
                ny = sl[pos]
                if ny > y + grid[x][y]:
                    break
                f[x][ny] = f[x][y] + 1
                q.append([x, ny])
                del sl[pos]
                cols[ny].remove(x)
            sl = cols[y]
            while True:
                pos = sl.bisect_left(x + 1)
                if pos == len(sl):
                    break
                nx = sl[pos]
                if nx > x + grid[x][y]:
                    break
                f[nx][y] = f[x][y] + 1
                q.append([nx, y])
                del sl[pos]
                rows[nx].remove(y)
        return f[-1][-1]

    def minimumVisitedCells(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        q = collections.deque([(0, 0)])
        row = [sortedcontainers.SortedList(range(n)) for _ in range(m)]
        col = [sortedcontainers.SortedList(range(m)) for _ in range(n)]
        visit = [[False] * n for _ in range(m)]
        visit[0][0] = True
        cnt = 1
        while q:
            for _ in range(len(q)):
                x, y = q.popleft()
                if x == m - 1 and y == n - 1:
                    return cnt
                index1 = row[x].bisect_right(y + grid[x][y])
                index2 = row[x].bisect_left(y)
                for _ in range(index1 - index2):
                    t = row[x].pop(index2)
                    if visit[x][t] == False:
                        visit[x][t] = True
                        q.append((x, t))

                index3 = col[y].bisect_right(x + grid[x][y])
                index4 = col[y].bisect_left(x)
                for _ in range(index3 - index4):
                    t = col[y].pop(index4)
                    if visit[t][y] == False:
                        visit[t][y] = True
                        q.append((t, y))
            cnt += 1
        return -1


# 2639 - Find the Width of Columns of a Grid - EASY
class Solution:
    # O(nm) / O(1)
    def findColumnWidth(self, grid: List[List[int]]) -> List[int]:
        m = len(grid)
        n = len(grid[0])
        return [max(len(str(grid[i][j])) for i in range(m)) for j in range(n)]

    def findColumnWidth(self, grid: List[List[int]]) -> List[int]:
        return [max(len(str(v)) for v in col) for col in zip(*grid)]


# 2640 - Find the Score of All Prefixes of an Array - MEDIUM
class Solution:
    # O(n) / O(1)
    def findPrefixScore(self, nums: List[int]) -> List[int]:
        mx = 0
        n = len(nums)
        cover = [0] * n
        for i, v in enumerate(nums):
            mx = max(mx, v)
            cover[i] = v + mx
        return list(itertools.accumulate(cover))

    def findPrefixScore(self, nums: List[int]) -> List[int]:
        n = len(nums)
        preMax = list(itertools.accumulate(nums, max))
        cover = [nums[i] + preMax[i] for i in range(n)]
        return list(itertools.accumulate(cover))
        return list(itertools.accumulate(cover, operator.add))  # same

    def findPrefixScore(self, nums: List[int]) -> List[int]:
        ans = []
        mx = summ = 0
        for v in nums:
            mx = max(mx, v)
            summ += v + mx
            ans.append(summ)
        return ans


# 2641 - Cousins in Binary Tree II - MEDIUM
class Solution:
    # O(n) / O(n)
    def replaceValueInTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        q = [(root, -1)]
        while q:
            new = []
            d = dict()
            summ = 0
            for x, fa in q:
                d[fa] = d.get(fa, 0) + x.val
                summ += x.val
                if x.left:
                    new.append((x.left, x))
                if x.right:
                    new.append((x.right, x))
            # summ = sum(d.values())
            for x, fa in q:
                x.val = summ - d.get(fa)
            q = new
        return root

    def replaceValueInTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        root.val = 0
        q = [root]
        while q:
            new = []
            summ = 0
            for x in q:
                if x.left:
                    new.append(x.left)
                    summ += x.left.val
                if x.right:
                    new.append(x.right)
                    summ += x.right.val
            for x in q:
                cur = 0
                if x.left:
                    cur += x.left.val
                if x.right:
                    cur += x.right.val
                if x.left:
                    x.left.val = summ - cur
                if x.right:
                    x.right.val = summ - cur
            q = new
        return root

    def replaceValueInTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        d = collections.defaultdict(int)

        def dfs(root: TreeNode, lv: int) -> None:
            if not root:
                return
            dfs(root.left, lv + 1)
            dfs(root.right, lv + 1)
            d[lv] += root.val
            return

        dfs(root, 0)

        def dfs2(root: TreeNode, pre: int, lv: int) -> None:
            if not root:
                return
            root.val = pre
            nxt = d[lv + 1]
            if root.left:
                nxt -= root.left.val
            if root.right:
                nxt -= root.right.val
            dfs2(root.left, nxt, lv + 1)
            dfs2(root.right, nxt, lv + 1)
            return

        dfs2(root, 0, 0)
        return root


# 2642 - Design Graph With Shortest Path Calculator - HARD
# 可以看出 Dijkstra 适合加边多的情况, Floyd 适合查询多的情况
class Graph:
    # Dijkstra 堆实现
    # O(qmlogm) / O(m), q = shortestPath 调用次数, 稀疏图中 m = O(n), 稠密图中 m = (n^2)
    def __init__(self, n: int, edges: List[List[int]]):
        self.g = [[] for _ in range(n)]
        for x, y, w in edges:
            self.g[x].append((y, w))

    def addEdge(self, edge: List[int]) -> None:
        x, y, w = edge
        self.g[x].append((y, w))
        return

    # O(1)
    def shortestPath(self, node1: int, node2: int) -> int:
        arr = self.dijkstra(self.g, node1)
        return -1 if arr[node2] == math.inf else arr[node2]

    def dijkstra(self, g: List[List[Tuple[int]]], start: int) -> List[int]:
        dist = [math.inf] * len(g)
        dist[start] = 0
        q = [(0, start)]
        while q:
            cur, x = heapq.heappop(q)
            if cur > dist[x]:
                continue
            for y, w in g[x]:
                new = dist[x] + w
                if new < dist[y]:
                    dist[y] = new
                    heapq.heappush(q, (new, y))
        return dist


class Graph:
    # Dijkstra 朴素实现
    # O(qn^2) / O(n^2), q = shortestPath 调用次数
    def __init__(self, n: int, edges: List[List[int]]):
        g = [[math.inf] * n for _ in range(n)]
        for x, y, w in edges:
            g[x][y] = w
        self.g = g

    # O(1)
    def addEdge(self, e: List[int]) -> None:
        self.g[e[0]][e[1]] = e[2]

    # 朴素 Dijkstra 算法每次求最短路的时间复杂度为 O(n^2)
    # 在本题的输入下 (稠密图), 这比堆的实现要快, (实际测试还是稀疏图多...)
    def shortestPath(self, start: int, end: int) -> int:
        n = len(self.g)
        dist = [math.inf] * n
        dist[start] = 0
        vis = [False] * n
        while True:  # 最多 n 次
            x = -1
            for i in range(n):
                if not vis[i] and (x < 0 or dist[i] < dist[x]):
                    x = i
            if x < 0 or dist[x] == math.inf:
                return -1
            if x == end:
                return dist[x]
            vis[x] = True
            for y, w in enumerate(self.g[x]):
                if dist[x] + w < dist[y]:
                    dist[y] = dist[x] + w


class Graph:
    # Floyd, 本质是动态规划

    # 定义 f[k][i][j] 表示从 i 到 j 的最短路长度, 并且从 i 到 j 的路径上的中间节点(不含 i 和 j)的编号至多为 k.
    # 分类讨论:
    #   1. 从 i 到 j 的最短路没有 k, 那么按照定义 f[k][i][j]= f[k - 1][i][j].
    #   2. 从 i 到 j 的最短路有 k, 说明 k 一定是中间节点, 那么可以视作先从 i 到 k, 再从 k 到 j
    #      所以有 f[k][i][j] = f[k - 1][i][k] + f[k - 1][k][j].
    # 取最小值, 得 f[k][i][j] = min(f[k - 1][i][j], f[k - 1][i][k] + f[k - 1][k][j])
    #
    # k 从 k - 1转移而来, 一般可以优化掉 (倒序更新)
    # f[i][j] = min(f[i][j], f[i][k] + f[k][j])
    #
    # 但是注意 f[k - 1][i][k] 和 f[k - 1][k][j] 同时包含 k - 1 和 k, 优化后转移方程不就错了?
    # f[k][i][k] + f[k][k][j]
    # 为什么状态被覆盖还能算出来正确答案?
    # 由状态定义得: f[k][i][k] 的一端点为 k, 中间又包含 k 个节点(其实这是矛盾的), f[k][i][k] 等于 f[k - 1][i][k]
    # 所以可以直接压缩状态
    # 算法导论也有解释

    # O(n^3 + qn^2) / O(n^2), q = shortestPath 调用次数
    def __init__(self, n: int, edges: List[List[int]]):
        g = [[math.inf] * n for _ in range(n)]
        for i in range(n):
            g[i][i] = 0
        for x, y, w in edges:
            g[x][y] = w
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    g[i][j] = min(g[i][j], g[i][k] + g[k][j])
        self.g = g

    def addEdge(self, e: List[int]) -> None:
        g = self.g
        n = len(g)
        x, y, w = e
        if w >= g[x][y]:
            return
        # 只有一种加边 (增加 x 到 y) 情况可能会更新距离 a 到 b:
        # a ------------- b
        #   \           /
        #    x  ----- y
        for i in range(n):
            for j in range(n):
                g[i][j] = min(g[i][j], g[i][x] + w + g[y][j])

    # O(1)
    def shortestPath(self, start: int, end: int) -> int:
        ans = self.g[start][end]
        return ans if ans < math.inf else -1


# 2643 - Row With Maximum Ones - EASY
class Solution:
    def rowAndMaximumOnes(self, mat: List[List[int]]) -> List[int]:
        idx = summ = -1
        for i, row in enumerate(mat):
            v = sum(row)
            if v > summ:
                idx, summ = i, v
        return [idx, summ]


# 2644 - Find the Maximum Divisibility Score - EASY
class Solution:
    def maxDivScore(self, nums: List[int], divisors: List[int]) -> int:
        n = len(divisors)
        arr = [0] * n
        cnt = collections.Counter(nums)
        for i, x in enumerate(divisors):
            for k, v in cnt.items():
                if k % x == 0:
                    arr[i] += v
        ans = 0
        mx = -1
        for i in range(n):
            if arr[i] > mx:
                ans = divisors[i]
                mx = arr[i]
            elif arr[i] == mx:
                ans = min(ans, divisors[i])
        return ans

    def maxDivScore(self, nums: List[int], divisors: List[int]) -> int:
        n = len(divisors)
        arr = [0] * n
        for num in nums:
            for i, v in enumerate(divisors):
                if num % v == 0:
                    arr[i] += 1
        mx = max(arr)
        ans = math.inf
        for i in range(n):
            if arr[i] == mx:
                ans = min(ans, divisors[i])
        return ans

    def maxDivScore(self, nums: List[int], divisors: List[int]) -> int:
        ans = mx = -1
        for d in divisors:
            cnt = sum(x % d == 0 for x in nums)
            if cnt > mx or (cnt == mx and d < ans):
                ans = d
                mx = cnt
        return ans


# 2645 - Minimum Additions to Make Valid String - MEDIUM
class Solution:
    def addMinimum(self, s: str) -> int:
        n = len(s)
        i = ans = 0
        while i < n:
            for c in "abc":
                if i < n and c == s[i]:
                    i += 1
                else:
                    ans += 1
        return ans

    # 若 s[i + 1] < s[i], 说明 s[i] 和 s[i + 1] 属于两个不同的 abc
    # 假设 s[i + 1] < s[i] 的下标 i 有 k 个
    # -> 至少有 (k + 1) 个 abc
    # -> 答案就是 3 * (k + 1) - n
    def addMinimum(self, word: str) -> int:
        return (1 + sum(a >= b for a, b in itertools.pairwise(word))) * 3 - len(word)

    # 考虑相邻字母, 细节不太好想
    # (y - x - 1 + 3) % 3
    # 再考虑特例
    #   第一个 s[0] - 'a',
    #   最后一个 'c' - s[n - 1]
    # 得 s[0] - s[n - 1] + 'c' - 'a'
    def addMinimum(self, s: str) -> int:
        ans = ord(s[0]) - ord(s[-1]) + 2
        for x, y in pairwise(map(ord, s)):
            ans += (y - x + 2) % 3
        return ans

    # 符合上述公式 y - x + 2 的哨兵, 更难想
    def addMinimum(self, s: str) -> int:
        s = "c" + s + "a"
        return sum((y - x + 2) % 3 for x, y in pairwise(map(ord, s)))

    def addMinimum(self, word: str) -> int:
        word = (
            word.replace("abc", " ")
            .replace("ab", "1")
            .replace("bc", "1")
            .replace("ac", "1")
            .replace("a", "2")
            .replace("b", "2")
            .replace("c", "2")
        )
        return word.count("1") + word.count("2") * 2


# 2646 - Minimize the Total Price of the Trips - HARD
class Solution:
    # 树形dp
    # 注意: 树 只有唯一的一条简单路径!
    # 思路:
    #   1. dfs求每个点被访问多少次
    #   2. 类似 打家劫舍3, dp计算答案
    # O(nm) / O(n), m = len(trips)
    def minimumTotalPrice(
        self, n: int, edges: List[List[int]], price: List[int], trips: List[List[int]]
    ) -> int:
        cnt = [0] * n
        g = [[] for _ in range(n)]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)

        def dfs(x: int, fa: int, end: int) -> False:
            if x == end:
                cnt[x] += 1
                return True
            for y in g[x]:
                if y != fa and dfs(y, x, end):
                    cnt[x] += 1
                    return True
            return False

        for start, end in trips:
            dfs(start, -1, end)

        def dfs2(x: int, fa: int) -> Tuple[int, int]:  # (ori, half)
            ori = price[x] * cnt[x]
            half = ori // 2
            for y in g[x]:
                if y != fa:
                    o, h = dfs2(y, x)
                    ori += min(o, h)  # 原价可选邻近的 原价或半价
                    half += o  # 半价只能选邻近原价
            return ori, half

        return min(dfs2(0, -1))  # 随便挑一个起点都可以


# 2651 - Calculate Delayed Arrival Time - EASY
class Solution:
    def findDelayedArrivalTime(self, arrivalTime: int, delayedTime: int) -> int:
        return (arrivalTime + delayedTime) % 24


# 2652 - Sum Multiples - EASY
class Solution:
    def sumOfMultiples(self, n: int) -> int:
        return sum(
            x for x in range(1, n + 1) if (x % 3 == 0 or x % 5 == 0 or x % 7 == 0)
        )

    # 容斥原理 + 等差数列求和
    def sumOfMultiples(self, n: int) -> int:
        def f(m: int) -> int:
            return (1 + n // m) * (n // m) // 2 * m

        return f(3) + f(5) + f(7) - f(15) - f(21) - f(35) + f(105)


# 2653 - Sliding Subarray Beauty - MEDIUM
class Solution:
    # O(nlogn) / O(U), U = 50
    def getSubarrayBeauty(self, nums: List[int], k: int, x: int) -> List[int]:
        sl = sortedcontainers.SortedList(nums[: k - 1])
        ans = []
        for i in range(k - 1, len(nums)):
            sl.add(nums[i])
            ans.append(sl[x - 1] if sl[x - 1] < 0 else 0)
            sl.remove(nums[i - (k - 1)])
        return ans

    def getSubarrayBeauty(self, nums: List[int], k: int, x: int) -> List[int]:
        sl = sortedcontainers.SortedList(nums[: k - 1])
        ans = []
        for i, v in enumerate(nums[k - 1 :]):
            sl.add(v)
            ans.append(min(sl[x - 1], 0))
            sl.discard(nums[i])
        return ans

    # O(nU) / O(U), U = 50
    def getSubarrayBeauty(self, nums: List[int], k: int, x: int) -> List[int]:
        cnt = [0] * 101
        for v in nums[: k - 1]:
            cnt[v] += 1
        ans = [0] * (len(nums) - k + 1)
        for i, (inn, out) in enumerate(zip(nums[k - 1 :], nums)):
            cnt[inn] += 1
            left = x
            for j in range(-50, 0):
                left -= cnt[j]
                if left <= 0:
                    ans[i] = j
                    break
            cnt[out] -= 1
        return ans

    def getSubarrayBeauty(self, nums: List[int], k: int, x: int) -> List[int]:
        U = 50
        cnt = [0] * (101)

        def f() -> int:
            s = 0
            for i in range(U):
                s += cnt[i]
                if s >= x:
                    return i - U
            return 0

        ans = []
        for i in range(k):
            cnt[nums[i] + U] += 1
        ans.append(f())
        for i in range(k, len(nums)):
            cnt[nums[i] + U] += 1
            cnt[nums[i - k] + U] -= 1
            ans.append(f())
        return ans


# 2654 - Minimum Number of Operations to Make All Array Elements Equal to 1 - MEDIUM
class Solution:
    # O(n * (n + logU)) / O(1)
    def minOperations(self, nums: List[int]) -> int:
        ones = sum(x == 1 for x in nums)
        if ones > 0:
            return len(nums) - ones
        if math.gcd(*nums) > 1:
            return -1

        # 求区间 gcd
        n = len(nums)
        ans = math.inf
        for i in range(n - 1):
            x = nums[i]
            for j in range(i + 1, n):
                x = math.gcd(x, nums[j])
                if x == 1:
                    ans = min(ans, n + (j - i) - 1)
        return -1 if ans == math.inf else ans

        mi = n = len(nums)
        for i in range(n):
            g = 0
            for j in range(i, n):
                g = math.gcd(g, nums[j])
                if g == 1:
                    mi = min(mi, j - i)
                    break
        return mi + n - 1

        for l in range(2, len(nums) + 1):
            f = False
            for t in range(len(nums) - l + 1):
                if math.gcd(*nums[t : t + l]) == 1:
                    f = True
            if f:
                return len(nums) - 1 + l - 1


class Solution:
    # 线段树 区间运算问题
    def queryGCD(self, k, l, r, x, y):
        if l == x and r == y:
            return self.tree[k]
        mid = l + r >> 1
        if y <= mid:
            g = self.queryGCD(k << 1, l, mid, x, y)
        else:
            if x > mid:
                g = self.queryGCD(k << 1 | 1, mid + 1, r, x, y)
            else:
                g = math.gcd(
                    self.queryGCD(k << 1, l, mid, x, mid),
                    self.queryGCD(k << 1 | 1, mid + 1, r, mid + 1, y),
                )
        return g

    def build(self, k, l, r):
        if l == r:
            self.tree[k] = self.nums[l]
            if self.tree[k] == 1:
                self.m += 1
            return
        mid = l + r >> 1
        self.build(k << 1, l, mid)
        self.build(k << 1 | 1, mid + 1, r)
        self.tree[k] = math.gcd(self.tree[k << 1], self.tree[k << 1 | 1])

    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        self.m = 0
        self.nums = [0] + nums
        self.tree = [0] * (4 * n)
        self.build(1, 1, n)
        if self.m > 0:
            return n - self.m
        else:
            self.m, i = math.inf, 1
            for j in range(1, n + 1):
                while i < j and self.queryGCD(1, 1, n, i + 1, j) == 1:
                    i += 1
                if self.queryGCD(1, 1, n, i, j) == 1:
                    self.m = min(j - i, self.m)
            return n + self.m - 1 if self.m < n else -1


class SparseTable:
    def __init__(self, data: list, func=operator.or_):
        # ST表 稀疏表, O(nlgn) 预处理, O(1)查询区间最值/或和/gcd
        # 下标从 0 开始
        self.func = func
        self.st = st = [list(data)]
        i, N = 1, len(st[0])
        while 2 * i <= N + 1:
            pre = st[-1]
            st.append([func(pre[j], pre[j + i]) for j in range(N - 2 * i + 1)])
            i <<= 1

    def query(self, begin: int, end: int):  # 查询闭区间[begin, end]的最大值
        lg = (end - begin + 1).bit_length() - 1
        return self.func(self.st[lg][begin], self.st[lg][end - (1 << lg) + 1])


class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if math.gcd(*nums) != 1:
            return -1
        ones = nums.count(1)
        if ones:
            return n - ones
        st = SparseTable(nums, math.gcd)
        ans = math.inf
        for i in range(n):

            def query(k):
                return st.query(k, i)

            pos = bisect.bisect_right(range(i), 1, lo=0, key=query)
            if pos and query(pos - 1) == 1:
                ans = min(ans, i - pos + 1)
        return ans + n - 1


# 2660 - Determine the Winner of a Bowling Game - EASY
class Solution:
    def isWinner(self, player1: List[int], player2: List[int]) -> int:
        f = lambda arr: sum(
            x * 2 if i > 0 and arr[i - 1] == 10 or i > 1 and arr[i - 2] == 10 else x
            for i, x in enumerate(arr)
        )
        s1, s2 = f(player1), f(player2)
        return 0 if s1 == s2 else 1 if s1 > s2 else 2


# 2661 - First Completely Painted Row or Column - MEDIUM
class Solution:
    def firstCompleteIndex(self, arr: List[int], mat: List[List[int]]) -> int:
        m = len(mat)
        n = len(mat[0])
        row = [0] * m
        col = [0] * n
        d = {v: (x, y) for x, row in enumerate(mat) for y, v in enumerate(row)}
        for i, v in enumerate(arr):
            x, y = d[v]
            row[x] += 1
            col[y] += 1
            if row[x] == n or col[y] == m:
                return i


# 2670 - Find the Distinct Difference Array - EASY
class Solution:
    # O(n^2) / O(n)
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        return [
            len(set(nums[: i + 1])) - len(set(nums[i + 1 :])) for i in range(len(nums))
        ]

    # O(n) / O(n)
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        ans = [0] * len(nums)
        suf = collections.Counter(nums)
        pre = collections.defaultdict(int)
        for i, x in enumerate(nums):
            pre[x] += 1
            suf[x] -= 1
            if suf[x] == 0:
                del suf[x]
            ans[i] = len(pre) - len(suf)
        return ans

    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        cnt = collections.Counter(nums)
        suf = len(cnt)
        s = set()
        ans = []
        for v in nums:
            cnt[v] -= 1
            if cnt[v] == 0:
                suf -= 1
            s.add(v)
            ans.append(len(s) - suf)
        return ans

    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        suf = [0] * (n + 1)
        s = set()
        for i in range(n - 1, 0, -1):
            s.add(nums[i])
            suf[i] = len(s)
        s.clear()
        ans = [0] * n
        for i, x in enumerate(nums):
            s.add(x)
            ans[i] = len(s) - suf[i + 1]
        return ans


# 2671 - Frequency Tracker - MEDIUM
class FrequencyTracker:
    def __init__(self):
        self.freq = collections.defaultdict(int)
        self.d = collections.defaultdict(int)

    def add(self, number: int) -> None:
        if number in self.d:
            self.freq[self.d[number]] -= 1
        self.d[number] += 1
        self.freq[self.d[number]] += 1
        return

    def deleteOne(self, number: int) -> None:
        if self.d[number] > 0:  # not "if number in self.d" !!!
            self.freq[self.d[number]] -= 1
            self.d[number] -= 1
            self.freq[self.d[number]] += 1
        return

    def hasFrequency(self, frequency: int) -> bool:
        return self.freq[frequency] > 0


# 2672 - Number of Adjacent Elements With the Same Color - MEDIUM
class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        cnt = 0
        ans = []
        c = [0] * n
        for i, col in queries:
            if i >= 1:
                if 0 != c[i - 1] and c[i - 1] == c[i] and col != c[i]:
                    cnt -= 1
                if c[i - 1] != c[i] and col == c[i - 1]:
                    cnt += 1
            if i <= n - 2:
                if 0 != c[i] and c[i] == c[i + 1] and col != c[i + 1]:
                    cnt -= 1
                if c[i] != c[i + 1] and col == c[i + 1]:
                    cnt += 1
            ans.append(cnt)
            c[i] = col
        return ans

    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        c = [0 for _ in range(n)]
        ans = []
        cnt = 0
        for i, col in queries:
            if i != 0 and c[i - 1] == c[i] and c[i]:
                cnt -= 1
            if i != n - 1 and c[i + 1] == c[i] and c[i]:
                cnt -= 1
            c[i] = col
            if i != 0 and c[i - 1] == c[i] and c[i]:
                cnt += 1
            if i != n - 1 and c[i + 1] == c[i] and c[i]:
                cnt += 1
            ans.append(cnt)
        return ans

    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        c = [0] * n
        cnt = 0
        ans = []
        for i, col in queries:
            if c[i]:
                if i:
                    cnt -= c[i] == c[i - 1]
                if i + 1 < n:
                    cnt -= c[i] == c[i + 1]
            c[i] = col
            if i:
                cnt += c[i] == c[i - 1]
            if i + 1 < n:
                cnt += c[i] == c[i + 1]
            ans.append(cnt)
        return ans


# 2673 - Make Costs of Paths Equal in a Binary Tree - MEDIUM
class Solution:
    # O(n) / O(1)
    def minIncrements(self, n: int, cost: List[int]) -> int:
        ans = 0
        for i in range(n // 2, 0, -1):  # 从最后一个非叶节点开始算
            ans += abs(cost[i * 2 - 1] - cost[i * 2])
            cost[i - 1] += max(cost[i * 2 - 1], cost[i * 2])  # 累加路径和
        return ans

    # O(n) / O(n)
    def minIncrements(self, n: int, cost: List[int]) -> int:
        ans = 0

        def dfs(i):  # 表示节点 i 以下的最长路径
            nonlocal ans
            if i > n:
                return 0
            l = dfs(2 * i)
            r = dfs(2 * i + 1)
            ans += abs(l - r)
            return max(l, r) + cost[i - 1]

        dfs(1)
        return ans

    # O(n) / O(n)
    def minIncrements(self, n: int, cost: List[int]) -> int:
        f = [0] * (n + 1)
        g = [0] * (n + 1)
        for i in reversed(range(1, n + 1)):
            if i * 2 > n:
                f[i] = cost[i - 1]
            else:
                f[i] = max(f[i * 2], f[i * 2 + 1]) + cost[i - 1]
                g[i] = g[i * 2] + g[i * 2 + 1] + abs(f[i * 2] - f[i * 2 + 1])
        return g[1]


# 2678 - Number of Senior Citizens - EASY
class Solution:
    def countSeniors(self, details: List[str]) -> int:
        return sum(s[11:13] > "60" for s in details)


# 2679 - Sum in a Matrix - MEDIUM
class Solution:
    # O(mnlogn + logn * m * m) / O(1), m = len(nums), n = len(nums[0])
    def matrixSum(self, nums: List[List[int]]) -> int:
        for row in nums:
            row.sort()
        return sum(max(heapq.heappop(r) for r in nums) for _ in range(len(nums[0])))

    # O(mnlogn + mn) / O(1), 既然每次选最大的数, 那么干脆对每行排序, 这样每次就选的是一列的最大值
    def matrixSum(self, nums: List[List[int]]) -> int:
        for row in nums:
            row.sort()
        return sum(map(max, zip(*nums)))  # zip(*nums) 枚举每一列


# 2680 - Maximum OR - MEDIUM
class Solution:
    # O(nlogn) / O(n)
    def maximumOr(self, nums: List[int], k: int) -> int:
        cnt = collections.defaultdict(int)
        for x in nums:
            s = bin(x)[2:]
            n = len(s)
            for i, v in enumerate(s):
                cnt[n - i - 1] += v == "1"
        ans = mx = 0
        for x in nums:
            s = bin(x)[2:]
            n = len(s)
            if n < mx:
                continue
            mx = max(mx, n)
            for i, v in enumerate(s):
                cnt[n - i - 1] -= v == "1"

            p = x << k
            for a, b in cnt.items():
                if b > 0:
                    p |= 1 << a
            ans = max(ans, p)

            for i, v in enumerate(s):
                cnt[n - i - 1] += v == "1"
        return ans

    def maximumOr(self, nums: List[int], k: int) -> int:
        cnt = collections.defaultdict(int)
        for x in nums:
            y = 0
            while (1 << y) <= x:
                if x & (1 << y):
                    cnt[y] += 1
                y += 1
        ans = mx = 0
        for x in nums:
            y = 0
            while (1 << y) <= x:
                if x & (1 << y):
                    cnt[y] -= 1
                y += 1

            # if y < mx:
            #     y = 0
            #     while (1 << y) <= x:
            #         if x & (1 << y):
            #             cnt[y] += 1
            #         y += 1
            #     continue
            # mx = max(mx, y)

            p = x << k
            for a, b in cnt.items():
                if b > 0:
                    p |= 1 << a
            ans = max(ans, p)

            y = 0
            while (1 << y) <= x:
                if x & (1 << y):
                    cnt[y] += 1
                y += 1
        return ans

    # O(nlogn) / O(n)
    def maximumOr(self, nums: List[int], k: int) -> int:
        cnt = collections.defaultdict(int)
        for x in nums:
            s = bin(x)[2:]
            n = len(s)
            for i, v in enumerate(s):
                cnt[n - i - 1] += v == "1"
        ans = mx = 0
        for x in nums:
            s = bin(x)[2:]
            n = len(s)
            if n < mx:
                continue
            mx = max(mx, n)
            for i, v in enumerate(s):
                cnt[n - i - 1] -= v == "1"

            p = x << k
            for a, b in cnt.items():
                if b > 0:
                    p |= 1 << a
            ans = max(ans, p)

            for i, v in enumerate(s):
                cnt[n - i - 1] += v == "1"
        return ans

    # O(n) / O(n), 预处理右侧元素的或值, 左侧的或值可以一边枚举一边计算
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        suf = [0] * (n + 1)
        for i in range(n - 1, 0, -1):
            suf[i] = suf[i + 1] | nums[i]
        ans = pre = 0
        for i, x in enumerate(nums):
            ans = max(ans, pre | (x << k) | suf[i + 1])
            pre |= x
        return ans


# 2681 - Power of Heroes - HARD
class Solution:
    # 1. 元素的顺序不影响答案 -> 先排序
    # 2. mx ^ 2 * mi, mi 可以合并为一个 presum
    # 3. 不取余大数运算非常慢
    # O(nlogn) / O(1)
    def sumOfPower(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        ans = minSum = 0
        for v in sorted(nums):
            ans = (ans + (minSum + v) * v * v) % mod
            minSum = (minSum * 2 + v) % mod
            # minSum = minSum * 2 + v  # super slow
        return ans


# 2682 - Find the Losers of the Circular Game - EASY
class Solution:
    # O(n) / O(n)
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        l = [False] * n
        i = 0
        nk = k
        while not l[i]:
            l[i] = True
            i = (i + nk) % n
            nk += k
        return [i for i, v in enumerate(l, 1) if not v]


# 2683 - Neighboring Bitwise XOR - MEDIUM
class Solution:
    # O(n) / O(n)
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        a = [0] * n
        b = [1] * n
        for i in range(n - 1):
            a[i + 1] = a[i] ^ derived[i]
            b[i + 1] = b[i] ^ derived[i]
        return (a[-1] == derived[-1] ^ a[0]) or (b[-1] == derived[-1] ^ b[0])

    def doesValidArrayExist(self, derived: List[int]) -> bool:
        n = len(derived)
        a = [0] * n
        for i in range(n - 1):
            a[i + 1] = a[i] ^ derived[i]
        return a[-1] == derived[-1] ^ a[0]

    # O(n) / O(1)
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        return functools.reduce(operator.xor, derived) == 0


# 2684 - Maximum Number of Moves in a Grid - MEDIUM
class Solution:
    # O(mn) / O(mn)
    def maxMoves(self, grid: List[List[int]]) -> int:
        @functools.lru_cache(None)
        def dfs(i: int, j: int) -> int:
            cur = 0
            if i > 0 and j < n - 1 and grid[i][j] < grid[i - 1][j + 1]:
                cur = max(cur, 1 + dfs(i - 1, j + 1))
            if j < n - 1 and grid[i][j] < grid[i][j + 1]:
                cur = max(cur, 1 + dfs(i, j + 1))
            if i < m - 1 and j < n - 1 and grid[i][j] < grid[i + 1][j + 1]:
                cur = max(cur, 1 + dfs(i + 1, j + 1))
            return cur

        m, n = len(grid), len(grid[0])
        return max(dfs(i, 0) for i in range(m))

    def maxMoves(self, grid: List[List[int]]) -> int:
        @functools.lru_cache(None)
        def dfs(i: int, j: int) -> int:
            if j == n - 1:
                return 0
            cur = 0
            for k in i - 1, i, i + 1:
                if 0 <= k < m and grid[i][j] < grid[k][j + 1]:
                    cur = max(cur, 1 + dfs(k, j + 1))
            return cur

        m, n = len(grid), len(grid[0])
        return max(dfs(i, 0) for i in range(m))


# 2685 - Count the Number of Complete Components - MEDIUM
class Solution:
    # O(n + m + n^2) / O(n + m), m = len(edges)
    def countCompleteComponents(self, n: int, edges: List[List[int]]) -> int:
        g = collections.defaultdict(set)
        for x, y in edges:
            g[x].add(y)
            g[y].add(x)
        ans = 0
        s = set()
        vis = [False] * n
        for i in range(n):
            if vis[i]:
                continue
            q = [i]
            while q:
                new = []
                for x in q:
                    s.add(x)
                    vis[x] = True
                    for y in g[x]:
                        if not vis[y]:
                            vis[y] = True
                            s.add(y)
                            new.append(y)
                q = new
            f = True
            for x in s:
                for y in s:
                    if x == y:
                        continue
                    if y not in g[x]:
                        f = False
                        break
                if not f:
                    break
            ans += f

            # for x in s:
            #     for y in s:
            #         if x == y:
            #             continue
            #         if y not in g[x]:
            #             break
            #     else:
            #         continue
            #     break
            # else:
            #     ans += 1

            # v = len(s)
            # e = sum(len(g[x]) for x in s)
            # ans += e == v * (v - 1)

            s.clear()
        return ans

    # O(n + m) / O(n + m), m = len(edges)
    def countCompleteComponents(self, n: int, edges: List[List[int]]) -> int:
        g = [[] for _ in range(n)]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)
        vis = [False] * n

        def dfs(x: int) -> None:
            """统计当前连通块的点数 v 和边数 e"""
            vis[x] = True
            nonlocal v, e
            v += 1
            e += len(g[x])
            for y in g[x]:
                if not vis[y]:
                    dfs(y)
            return

        ans = 0
        for i, b in enumerate(vis):
            if not b:
                v = e = 0
                dfs(i)
                ans += e == v * (v - 1)
        return ans

    def countCompleteComponents(self, n: int, edges: List[List[int]]) -> int:
        class UnionFind:
            def __init__(self, n: int) -> None:
                self.p = [i for i in range(n)]
                self.e = [0] * n  # 每个连通块内边数量
                self.v = [1] * n  # 每个连通块内顶点数量

            def find(self, x: int) -> int:
                """path compression"""
                if self.p[x] != x:
                    self.p[x] = self.find(self.p[x])
                return self.p[x]

            def union(self, x: int, y: int) -> None:
                """x's root = y"""
                px = self.find(x)
                py = self.find(y)
                if px == py:
                    self.e[px] += 1
                    return
                # if self.v[px] > self.v[py]:
                #     px, py = py, px

                self.p[px] = py
                self.e[py] += self.e[px] + 1
                self.v[py] += self.v[px]
                return

        uf = UnionFind(n)
        for x, y in edges:
            uf.union(x, y)
        return sum(
            uf.e[i] == (uf.v[i] * (uf.v[i] - 1)) // 2 for i in range(n) if i == uf.p[i]
        )
        return sum(uf.e[i] == math.comb(uf.v[i], 2) for i in range(n) if i == uf.p[i])

    def countCompleteComponents(self, n: int, edges: List[List[int]]) -> int:
        class UnionFind:
            def __init__(self, n: int) -> None:
                self.p = [i for i in range(n)]

            def find(self, x: int) -> int:
                """path compression"""
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

        uf = UnionFind(n)
        cnt = [0] * n
        for x, y in edges:
            uf.union(x, y)
            cnt[x] += 1
            cnt[y] += 1
        v = [0] * n
        e = [0] * n
        for i in range(n):
            p = uf.find(i)
            v[p] += 1
            e[p] += cnt[i]
        return sum(e[i] == v[i] * (v[i] - 1) for i in range(n) if i == uf.p[i])


# 2696 - Minimum String Length After Removing Substrings - EASY
class Solution:
    # O(n^2) / O(n)
    def minLength(self, s: str) -> int:
        while "AB" in s or "CD" in s:
            s = s.replace("AB", "").replace("CD", "")
        return len(s)

    # O(n) / O(n)
    def minLength(self, s: str) -> int:
        st = []
        for c in s:
            if st and (c == "B" and st[-1] == "A" or c == "D" and st[-1] == "C"):
                st.pop()
            else:
                st.append(c)
        return len(st)


# 2697 - Lexicographically Smallest Palindrome - EASY
class Solution:
    def makeSmallestPalindrome(self, s: str) -> str:
        n = len(s)
        s = list(s)
        for i in range(n // 2):
            if s[i] != s[n - 1 - i]:
                s[i] = s[n - 1 - i] = min(s[i], s[n - 1 - i])
        return "".join(s)


# 2698 - Find the Punishment Number of an Integer - MEDIUM
def check(t: int, s: str) -> bool:
    if not s or t < 0:
        return False
    if t == int(s):
        return True
    for i in range(len(s)):
        if check(t - int(s[: i + 1]), s[i + 1 :]):
            return True
    return False


arr = [0] * 1001
for i in range(1, 1001):
    if check(i, str(i * i)):
        arr[i] += i * i
    arr[i] += arr[i - 1]


class Solution:
    def punishmentNumber(self, n: int) -> int:
        return arr[n]


def dfs(s: str, target: int, p: int, summ: int) -> bool:
    if p == len(s):
        return summ == target
    x = 0
    for j in range(p, len(s)):  # 从 s[p] 到 s[j] 组成的子串
        x = x * 10 + int(s[j])
        if dfs(s, target, j + 1, summ + x):
            return True
    return False


arr = [0] * 1001
for i in range(1, 1001):
    if dfs(str(i * i), i, 0, 0):
        arr[i] += i * i
    arr[i] += arr[i - 1]


class Solution:
    def punishmentNumber(self, n: int) -> int:
        return arr[n]


# https://oeis.org/A038206
# 因为 i ^ 2 无论如何切割相加, 得到结果除以 9 的余数都和 i ^ 2 除以 9 的余数相同,
# 因此要让切割后求和的结果等于 i, 必要条件是 i ^ 2 === i mod 9: 即 i * (i - 1) === 0 mod 9,
# 所以 i 只能是 9 的倍数或者除以 9 余 1 的数
arr = [
    0,
    1,
    9,
    10,
    36,
    45,
    55,
    82,
    91,
    99,
    100,
    235,
    297,
    369,
    370,
    379,
    414,
    657,
    675,
    703,
    756,
    792,
    909,
    918,
    945,
    964,
    990,
    991,
    999,
    1000,
]


class Solution:
    def punishmentNumber(self, n: int) -> int:
        return sum(v * v for v in arr if v <= n)


# 2699 - Modify Graph Edge Weights - HARD
class Solution:
    # https://codeforces.com/problemset/problem/715/B

    # Dijkstra

    # 错误思路:
    # 没经过-1 达到 大于 空()
    # 没经过-1 达到 正好, 返回 1
    # 没经过-1 达到 小于 空()
    # 经过-1 达到 大于 空()
    # 经过-1 达到 正好, 返回 1
    # 经过-1 达到 小于 可以调整, 但是不能简单地把路径差值赋值给某一个负边,
    # 因为在增加边权后, 最短路可能就走别的边了, 不走刚才修改的这条边了
    # 卡在如何给负边赋值上了...

    # 1. 按照什么顺序增大边, 怎么pick?
    # 2. 每个边增大多少比较合适?

    # 思路一
    # 首先注意到 target 的范围, 与边权最大值范围, 先将所有的 -1 改为 2 * 10^9.
    # 计算初始的最短路, 如果小于 target 则无论怎么修改肯定无解, 如果等于 target 则直接返回.
    # 如果大于 target, 则尝试依次将每条边权 2 * 10^9 改为 1,
    # 计算此时最短路 dist, 如果此时 dist 小于等于 target, 则有解,
    # 并将此时这条边权增加 target - dist, 输出即可.
    def modifiedGraphEdges(
        self, n: int, edges: List[List[int]], source: int, destination: int, target: int
    ) -> List[List[int]]:
        def dijkstra(g: List[List[Tuple[int]]], start: int, end: int) -> int:
            dist = [math.inf] * len(g)
            dist[start] = 0
            q = [(0, start)]
            while q:
                cur, x = heapq.heappop(q)
                if cur > dist[x]:
                    continue
                if x == end:
                    return cur
                for y in g[x]:
                    new = cur + g[x][y]
                    if new < dist[y]:
                        dist[y] = new
                        heapq.heappush(q, (new, y))
            return math.inf

        g = [dict() for _ in range(n)]
        negEdge = []
        for i, j, w in edges:
            if w == -1:
                negEdge.append([i, j, w])
                w = 2 * 10**9  # 为什么不能用 math.inf?
            g[i][j] = g[j][i] = w

        def check() -> List[List[int]]:
            ans = []
            vis = set()
            for i in range(n):
                for j in g[i]:
                    if (i, j) not in vis:
                        ans.append([i, j, g[i][j]])
                        vis.add((i, j))
                        vis.add((j, i))
            return ans

        # 第一, 二步
        dist = dijkstra(g, source, destination)
        if dist == target:
            return check()
        if dist < target:
            return []
        # 第三步
        for x, y, w in negEdge:
            g[x][y] = g[y][x] = 1
            dist = dijkstra(g, source, destination)
            if dist <= target:
                gap = target - dist
                g[x][y] = g[y][x] = g[x][y] + gap
                return check()
        return []

    # 思路二, 两次 Dijkstra(稠密图下是线性做法)
    # 先把 -1 都修改成 1, 然后跑第一遍 Dijkstra, 找到 dist: List[int]
    # 再跑一遍 Dijkstra, 由于 Dijkstra 算法保证每次拿到的点的最短路就是最终的最短路,
    # 所以按照 Dijkstra 算法遍历点/边的顺序去修改, 就不会对已确定的最短路产生影响

    # 对于一条可以修改的边 x - y, 假设要把它的边权改为 w, 那么
    # source −> x −> y −> destination 这条路径由三部分组成: (0 / 1 仅用于区分是第几次 dji)
    # 1. 从 source 到 x 的最短路, 这是第二遍 Dijkstra 算出来的, 即 dX1
    # 2. 从 x 到 y, 即 w.
    # 3. 从 y 到 destination 的最短路, 由于后面的边还没有修改, 这个最短路是第一遍 Dijkstra 算出来的, 即 dDestination0 - dY0
    #    注意这个式子仅当 y 在从 source 到 destination 的最短路上才成立.
    #    不过, 如果 y 不在最短路上, 修改 x − y 并不会对最短路产生影响, 所以代码中并没有判断 y 是否在最短路上.
    # 三部分之和为 target = dX1 + w + dDestination0 - dY0
    # 得 w = target - dDestination0 + dY0 - dX1
    #    注意 target - dDestination0 是个定值 (delta)

    # O(n^2) / O(m), 时间复杂度与边的数量 m = O(n^2) 成正比
    def modifiedGraphEdges(
        self, n: int, edges: List[List[int]], source: int, destination: int, target: int
    ) -> List[List[int]]:
        g = [[] for _ in range(n)]
        for i, (x, y, _) in enumerate(edges):
            g[x].append((y, i))
            g[y].append((x, i))  # 额外保存边的编号
        dist = [[math.inf, math.inf] for _ in range(n)]
        dist[source] = [0, 0]

        def dijkstra(k: int) -> None:  # 这里 k 表示第一次/第二次
            vis = [False] * n
            while True:
                # 找到当前最短路, 去更新它的邻居的最短路, dis[x][k] 一定是最短路长度
                x = -1
                for y, (b, d) in enumerate(zip(vis, dist)):
                    if not b and (x < 0 or d[k] < dist[x][k]):
                        x = y
                if x == destination:
                    return
                vis[x] = True
                for y, eid in g[x]:
                    wt = edges[eid][2]
                    if wt == -1:
                        wt = 1  # 第一次 dji 改成 1
                    if k == 1 and edges[eid][2] == -1:
                        # 第二次 dji, 由上述公式改成 w
                        w = delta + dist[y][0] - dist[x][1]
                        if w > wt:
                            edges[eid][2] = wt = w  # 直接在 edges 上修改
                    dist[y][k] = min(dist[y][k], dist[x][k] + wt)

        dijkstra(0)
        delta = target - dist[destination][0]
        if delta < 0:  # 全为 1 时, 最短路比 target 还大
            return []

        dijkstra(1)
        if dist[destination][1] < target:  # 最短路无法再变大, 无法达到 target
            return []

        for e in edges:
            if e[2] == -1:  # 剩余没修改的边全部改成 1
                e[2] = 1
        return edges
