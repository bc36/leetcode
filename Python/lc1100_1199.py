import bisect, collections, datetime, functools, heapq, itertools, math, queue, operator, string, threading
from typing import Callable, List, Optional, Tuple
import sortedcontainers


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 1103 - Distribute Candies to People - EASY
class Solution:
    def distributeCandies(self, candies: int, num_people: int) -> List[int]:
        ans = [0] * num_people
        i = 0
        x = 1
        while candies > x:
            ans[i] += x
            candies -= x
            x += 1
            i = (i + 1) % num_people
        ans[i] += candies
        return ans


# 1104 - Path In Zigzag Labelled Binary Tree - MEDIUM
class Solution:
    # 翻转前后的标号之和 = 2^(i-1) + 2^i - 1
    def pathInZigZagTree(self, label: int) -> List[int]:
        d = 1
        first = 1
        while first * 2 <= label:
            first *= 2
            d += 1
        ans = []
        if not d & 1:
            label = (1 << d - 1) + (1 << d) - 1 - label
        while d:
            if d & 1:
                ans.append(label)
            else:
                ans.append((1 << d - 1) + (1 << d) - 1 - label)
            d -= 1
            label >>= 1
        return ans[::-1]


# 1105 - Filling Bookcase Shelves - MEDIUM
class Solution:
    # O(n ^ 2) / O(n), 递归需要 O(n) 的栈空间
    def minHeightShelves(self, books: List[List[int]], shelf_width: int) -> int:
        @functools.lru_cache(None)
        def dfs(i: int) -> int:
            """定义 dfs(i) 表示把 books[0] 到 books[i] 按顺序摆放后的最小高度"""
            if i < 0:
                return 0  # 没有书了, 高度是 0
            res = math.inf
            mx = 0
            can = shelf_width
            for j in range(i, -1, -1):
                can -= books[j][0]
                if can < 0:
                    break  # 空间不足, 无法放书
                mx = max(mx, books[j][1])  # 从 j 到 i 的最大高度
                res = min(res, dfs(j - 1) + mx)
            return res

        return dfs(len(books) - 1)

    def minHeightShelves(self, books: List[List[int]], shelf_width: int) -> int:
        n = len(books)
        f = [0] + [math.inf] * n  # 在前面插入一个状态表示 dfs(-1)=0
        for i in range(n):
            mx = 0
            can = shelf_width
            for j in range(i, -1, -1):
                can -= books[j][0]
                if can < 0:
                    break
                mx = max(mx, books[j][1])
                f[i + 1] = min(f[i + 1], f[j] + mx)
        return f[n]


# 1106 - Parsing A Boolean Expression - HARD
class Solution:
    def parseBoolExpr(self, expression: str) -> bool:
        st = [[0, 0]]
        for c in expression:
            if c in "({":
                st.append([0, 0])
            elif c in ")}":
                f, t = st.pop()
                a = False
                sign = st.pop()
                if sign == "|":
                    a = True if t else False
                elif sign == "&":
                    a = False if f else True
                elif sign == "!":
                    a = True if f else False
                if a:
                    st[-1][1] += 1
                else:
                    st[-1][0] += 1
            elif c == "t":
                st[-1][1] += 1
            elif c == "f":
                st[-1][0] += 1
            elif c in "&|!":
                st.append(c)
        return True if st[0][1] else False


# 1108 - Defanging an IP Address - EASY
class Solution:
    def defangIPaddr(self, address: str) -> str:
        return address.replace(".", "[.]")


# 1109 - Corporate Flight Bookings - MEDIUM
class Solution:
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        st = SegmentTree(n)
        for f, l, s in bookings:
            st.range_update(1, 1, n, f, l, s)
        return [st.query(1, 1, n, i, i) for i in range(1, n + 1)]


class SegmentTree:
    def __init__(self, n: int):
        self.t = [0] * (4 * n)
        self.lazy = [0] * (4 * n)

    def build(self, nums: List[int], o: int, l: int, r: int) -> None:
        if l == r:
            self.t[o] = nums[l - 1]
            return
        m = l + r >> 1
        self.build(nums, o << 1, l, m)
        self.build(nums, o << 1 | 1, m + 1, r)
        self.t[o] = self.t[o << 1] + self.t[o << 1 | 1]
        return

    def pushdown(self, o: int, cnt: int) -> None:
        if self.lazy[o] != 0:
            self.t[o << 1] += self.lazy[o] * (cnt - cnt // 2)
            self.t[o << 1 | 1] += self.lazy[o] * (cnt // 2)
            self.lazy[o << 1] += self.lazy[o]
            self.lazy[o << 1 | 1] += self.lazy[o]
            self.lazy[o] = 0
        return

    def range_update(self, o: int, l: int, r: int, L: int, R: int, val: int) -> None:
        if L <= l and r <= R:
            self.t[o] += val * (r - l + 1)
            self.lazy[o] += val
            return
        self.pushdown(o, r - l + 1)
        m = l + r >> 1
        if L <= m:
            self.range_update(o << 1, l, m, L, R, val)
        if m < R:
            self.range_update(o << 1 | 1, m + 1, r, L, R, val)
        self.t[o] = self.t[o << 1] + self.t[o << 1 | 1]
        return

    def query(self, o: int, l: int, r: int, L: int, R: int) -> int:
        if L <= l and r <= R:
            return self.t[o]
        self.pushdown(o, r - l + 1)
        res = 0
        m = l + r >> 1
        if L <= m:
            res += self.query(o << 1, l, m, L, R)
        if m < R:
            res += self.query(o << 1 | 1, m + 1, r, L, R)
        return res


# 1114 - Print in Order - EASY
class Foo:
    def __init__(self):
        self.f = 0
        pass

    def first(self, printFirst: "Callable[[], None]") -> None:
        # printFirst() outputs "first". Do not change or remove this line.
        printFirst()
        self.f = 1

    def second(self, printSecond: "Callable[[], None]") -> None:
        while self.f != 1:
            continue
        # printSecond() outputs "second". Do not change or remove this line.
        printSecond()
        self.f = 2

    def third(self, printThird: "Callable[[], None]") -> None:
        while self.f != 2:
            continue
        # printThird() outputs "third". Do not change or remove this line.
        printThird()


class Foo:
    def __init__(self):
        self.c = threading.Condition()
        self.f = 0

    def first(self, printFirst: "Callable[[], None]") -> None:
        self.wrap(0, printFirst)

    def second(self, printSecond: "Callable[[], None]") -> None:
        self.wrap(1, printSecond)

    def third(self, printThird: "Callable[[], None]") -> None:
        self.wrap(2, printThird)

    def wrap(self, val: int, func: "Callable[[], None]") -> None:
        with self.c:
            self.c.wait_for(lambda: val == self.f)
            func()
            self.f += 1
            self.c.notify_all()
        return


class Foo:
    def __init__(self):
        self.l1 = threading.Lock()
        self.l1.acquire()
        self.l2 = threading.Lock()
        self.l2.acquire()

    def first(self, printFirst: "Callable[[], None]") -> None:
        printFirst()
        self.l1.release()

    def second(self, printSecond: "Callable[[], None]") -> None:
        self.l1.acquire()
        printSecond()
        self.l2.release()

    def third(self, printThird: "Callable[[], None]") -> None:
        self.l2.acquire()
        printThird()


class Foo:
    def __init__(self):
        self.s1 = threading.Semaphore(0)
        self.s2 = threading.Semaphore(0)

    def first(self, printFirst: "Callable[[], None]") -> None:
        printFirst()
        self.s1.release()

    def second(self, printSecond: "Callable[[], None]") -> None:
        self.s1.acquire()
        printSecond()
        self.s2.release()

    def third(self, printThird: "Callable[[], None]") -> None:
        self.s2.acquire()
        printThird()


class Foo:
    def __init__(self):
        self.e1 = threading.Event()
        self.e2 = threading.Event()

    def first(self, printFirst: "Callable[[], None]") -> None:
        printFirst()
        self.e1.set()

    def second(self, printSecond: "Callable[[], None]") -> None:
        self.e1.wait()
        printSecond()
        self.e2.set()

    def third(self, printThird: "Callable[[], None]") -> None:
        self.e2.wait()
        printThird()


class Foo:
    def __init__(self):
        self.q1 = queue.Queue()
        self.q2 = queue.Queue()

    def first(self, printFirst: "Callable[[], None]") -> None:
        printFirst()
        self.q1.put(0)

    def second(self, printSecond: "Callable[[], None]") -> None:
        self.q1.get()
        printSecond()
        self.q2.put(0)

    def third(self, printThird: "Callable[[], None]") -> None:
        self.q2.get()
        printThird()


class Foo:
    def __init__(self):
        self.q1 = queue.Queue(maxsize=1)
        self.q1.put(0)
        self.q2 = queue.Queue(maxsize=1)
        self.q2.put(0)

    def first(self, printFirst: "Callable[[], None]") -> None:
        printFirst()
        self.q1.get()

    def second(self, printSecond: "Callable[[], None]") -> None:
        self.q1.put(0)
        printSecond()
        self.q2.get()

    def third(self, printThird: "Callable[[], None]") -> None:
        self.q2.put(0)
        printThird()


# 1122 - Relative Sort Array - EASY
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        s = set(arr1)
        cnt = collections.Counter(arr1)
        ans = []
        for v in arr2:
            if v in s:
                ans.extend([v] * cnt[v])
                del cnt[v]
        for k, v in sorted(cnt.items()):
            ans.extend([k] * v)
        return ans

    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        freq = [0] * 1001
        for v in arr1:
            freq[v] += 1
        ans = []
        for v in arr2:
            ans.extend([v] * freq[v])
            freq[v] = 0
        for v in range(1001):
            if freq[v] > 0:
                ans.extend([v] * freq[v])
        return ans


# 1123 - Lowest Common Ancestor of Deepest Leaves - MEDIUM
class Solution:
    # O(n) / O(n)
    def lcaDeepestLeaves(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(node: Optional[TreeNode]) -> (Optional[TreeNode], int):
            if not node:
                return None, 0
            l, dl = dfs(node.left)
            r, dr = dfs(node.right)
            if dl > dr:
                return l, dl + 1
            elif dl < dr:
                return r, dr + 1
            return node, dl + 1

        return dfs(root)[0]


# 1124 - Longest Well-Performing Interval - MEDIUM
class Solution:
    # O(n) / O(n)
    def longestWPI(self, hours: List[int]) -> int:
        ans = s = 0
        before = {}
        for i, v in enumerate(hours):
            s += 1 if v > 8 else -1
            if s > 0:
                ans = i + 1
            # 为什么是 s - 1? 可以从以下几个方面理解:
            # 此时 s <= 0, 想象一个 v 型图案
            # 题目要求差值大于 0 而不是等于 0, 说明 [j+1,i] 这一段表现良好, 才能使得 s 从 s-1 变成 s
            # 因为 s <= 0, 所以 s-1 比 s-2, s-3 更早出现, 所以考察 s-1 的位置
            elif s - 1 in before:
                ans = max(ans, i - before[i - 1])
            if s not in before:
                before[s] = i
        return ans

    # 问题转换
    # 1. 劳累天数大于不劳累天数
    # -> 劳累天数减去不劳累天数大于 0
    # 2. 令劳累的一天视作 nums[i] = 1, 不劳累的一天 nums[i] = 1
    # -> 计算 nums 的最长子数组, 其元素和大于 0

    # 单调递减栈, 栈中保存可能的左端点
    # 倒序遍历, 考察左右最长距离
    def longestWPI(self, hours: List[int]) -> int:
        p = [0] * (len(hours) + 1)
        st = [0]
        for i, v in enumerate(hours, start=1):
            p[i] = p[i - 1] + (1 if v > 8 else -1)
            if p[i] < p[st[-1]]:
                st.append(i)
        ans = 0
        for i in range(len(hours), 0, -1):
            while st and p[st[-1]] < p[i]:
                ans = max(ans, i - st.pop())  # [st[-1],i) 可能是最长子数组
        return ans

    def longestWPI(self, hours: List[int]) -> int:
        p = [0] * (len(hours) + 1)
        st = [-1]
        for i, v in enumerate(hours):
            p[i + 1] = p[i] + (1 if v > 8 else -1)
            if p[i + 1] < p[st[-1]]:
                st.append(i)
        ans = 0
        for i in range(len(hours), -1, -1):
            while st and p[st[-1] + 1] < p[i]:
                ans = max(ans, i - (st.pop() + 1))
        return ans


# 1128 - Number of Equivalent Domino Pairs - EASY
class Solution:
    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        cnt = collections.Counter(tuple(sorted(d)) for d in dominoes)
        return sum(v * (v - 1) // 2 for v in cnt.values())

    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        cnt = [0] * 100
        ans = 0
        for x, y in dominoes:
            v = x * 10 + y if x <= y else y * 10 + x
            ans += cnt[v]
            cnt[v] += 1
        return ans


# 1129 - Shortest Path with Alternating Colors - MEDIUM
class Solution:
    # O(n + m) / O(n + m), n 节点数, m 边数
    def shortestAlternatingPaths(
        self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]
    ) -> List[int]:
        g = [collections.defaultdict(list), collections.defaultdict(list)]
        for x, y in redEdges:
            g[0][x].append(y)
        for x, y in blueEdges:
            g[1][x].append(y)
        ans = [-1] * n
        vis = set()
        q = [(0, 0), (0, 1)]
        d = 0
        while q:
            new = []
            for i, c in q:  # c -> color
                if ans[i] == -1:
                    ans[i] = d
                vis.add((i, c))
                c = 1 - c
                for j in g[c][i]:
                    if (j, c) not in vis:
                        new.append((j, c))
            d += 1
            q = new
        return ans

    def shortestAlternatingPaths(
        self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]
    ) -> List[int]:
        g = [[] for _ in range(n)]
        for x, y in redEdges:
            g[x].append((y, 0))
        for x, y in blueEdges:
            g[x].append((y, 1))

        ans = [-1] * n
        vis = {(0, 0), (0, 1)}
        q = [(0, 0), (0, 1)]
        d = 0
        while q:
            new = []
            for x, color in q:
                if ans[x] == -1:
                    ans[x] = d
                for y in g[x]:
                    if y[1] != color and y not in vis:
                        vis.add(y)
                        new.append(y)
            q = new
            d += 1
        return ans


# 1130 - Minimum Cost Tree From Leaf Values - MEDIUM
class Solution:
    # O(n^3) / O(n^2)
    def mctFromLeafValues(self, arr: List[int]) -> int:
        n = len(arr)
        f = [[0] * n for _ in range(n)]
        g = [[0] * n for _ in range(n)]
        for i in range(n):
            g[i][i] = arr[i]
        for i in range(n):
            for j in range(i + 1, n):
                g[i][j] = max(g[i][j - 1], arr[j])
                f[i][j] = math.inf
        for l in range(1, n):  # 长度
            for i in range(n - l):  # 起点
                for k in range(i, i + l):  # 枚举分割点(根)
                    f[i][i + l] = min(
                        f[i][i + l],
                        f[i][k] + f[k + 1][i + l] + g[i][k] * g[k + 1][i + l],
                    )
        return f[0][n - 1]

    def mctFromLeafValues(self, arr: List[int]) -> int:
        n = len(arr)
        f = [[0] * n for _ in range(n)]
        g = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            g[i][i] = arr[i]
            for j in range(i + 1, n):
                g[i][j] = max(g[i][j - 1], arr[j])
                f[i][j] = min(
                    f[i][k] + f[k + 1][j] + g[i][k] * g[k + 1][j] for k in range(i, j)
                )
        return f[0][n - 1]

    # O(n) / O(n)
    def mctFromLeafValues(self, arr: List[int]) -> int:
        ans = 0
        st = [math.inf]
        for x in arr:
            while st[-1] <= x:
                ans += st.pop() * min(x, st[-1])  # st[-1] * min(x, st[-2])
            st.append(x)
        while len(st) > 2:
            ans += st.pop() * st[-1]
        return ans


# 1137 - N-th Tribonacci Number - EASY
class Solution:
    @functools.lru_cache(None)
    def tribonacci(self, n: int) -> int:
        if n == 0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 1
        return self.tribonacci(n - 1) + self.tribonacci(n - 2) + self.tribonacci(n - 3)


class Solution:
    def __init__(self):
        self.cache = {0: 0, 1: 1, 2: 1}

    def tribonacci(self, n: int) -> int:
        if n in self.cache:
            return self.cache[n]
        self.cache[n] = (
            self.tribonacci(n - 1) + self.tribonacci(n - 2) + self.tribonacci(n - 3)
        )
        return self.cache[n]


class Solution:
    def tribonacci(self, n: int) -> int:
        if n == 0:
            return 0
        elif n < 3:
            return 1
        one, two, three, ans = 0, 1, 1, 0
        for _ in range(2, n):
            ans = one + two + three
            one, two, three = two, three, ans
        return ans


# 1138 - Alphabet Board Path - MEDIUM
class Solution:
    def alphabetBoardPath(self, target: str) -> str:
        board = ["abcde", "fghij", "klmno", "pqrst", "uvwxy", "z"]
        d = {}
        for i, r in enumerate(board):
            for j, c in enumerate(r):
                d[c] = (i, j)
        ans = ""
        p = (0, 0)
        for c in target:
            dx = d[c][0] - p[0]
            dy = d[c][1] - p[1]
            # due of the position of 'z', we should move to 'U' and 'L' first to prevent crossing the boundary
            if dx < 0:
                ans += "U" * (-dx)
            if dy < 0:
                ans += "L" * (-dy)
            if dx > 0:
                ans += "D" * dx
            if dy > 0:
                ans += "R" * dy
            ans += "!"
            p = d[c]
        return ans

    def alphabetBoardPath(self, target: str) -> str:
        ans = ""
        x = y = 0
        for c in target:
            nx, ny = divmod(ord(c) - ord("a"), 5)
            v = "UD"[nx > x] * abs(nx - x)
            h = "LR"[ny > y] * abs(ny - y)
            ans += (v + h if c != "z" else h + v) + "!"
            x, y = nx, ny
        return ans


# 1139 - Largest 1-Bordered Square - MEDIUM
class Solution:
    # O(mn * min(m, n)) / O(mn)
    def largest1BorderedSquare(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        pr = [list(itertools.accumulate(row, initial=0)) for row in grid]
        pc = [list(itertools.accumulate(col, initial=0)) for col in zip(*grid)]
        for d in range(min(n, m), 0, -1):  # 枚举边长 d
            for i in range(n - d + 1):
                for j in range(m - d + 1):  # 枚举左上角坐标 (i, j)
                    # 四条边 1 的个数均为 d
                    if (
                        pr[i][j + d] - pr[i][j] == d  # 上
                        and pc[j][i + d] - pc[j][i] == d  # 左
                        and pr[i + d - 1][j + d] - pr[i + d - 1][j] == d  # 下
                        and pc[j + d - 1][i + d] - pc[j + d - 1][i] == d  # 右
                    ):
                        return d * d
        return 0


# 1140 - Stone Game II - MEDIUM
class Solution:
    # O(n^3) / O(n^2)
    # dfs(i, M) indicates given M, the maximum number of stones that can be obtained by taking stones from piles[i]
    def stoneGameII(self, piles: List[int]) -> int:
        @functools.lru_cache(None)
        def dfs(i: int, m: int) -> int:
            if n - i <= 2 * m:
                return p[n] - p[i]
            return max(p[n] - p[i] - dfs(i + x, max(x, m)) for x in range(1, 2 * m + 1))

        n = len(piles)
        p = list(itertools.accumulate(piles, initial=0))
        return dfs(0, 1)

    def stoneGameII(self, piles: List[int]) -> int:
        @functools.lru_cache(None)
        def dfs(i: int, m: int) -> int:
            if i + m * 2 >= n:
                return p[n] - p[i]
            return p[n] - p[i] - min(dfs(i + x, max(m, x)) for x in range(1, 2 * m + 1))

        n = len(piles)
        p = list(itertools.accumulate(piles, initial=0))
        return dfs(0, 1)


# 1143 - Longest Common Subsequence - MEDIUM
class Solution:
    # O(nm) / O(nm), 1400ms
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n, m = len(text1), len(text2)

        @functools.lru_cache(None)
        def dfs(i: int, j: int) -> int:
            if i < 0 or j < 0:
                return 0
            if text1[i] == text2[j]:
                return dfs(i - 1, j - 1) + 1
            return max(dfs(i - 1, j), dfs(i, j - 1))

        return dfs(n - 1, m - 1)

    # O(nm) / O(nm), 900ms
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n, m = len(text1) + 1, len(text2) + 1
        f = [[0 for _ in range(m)] for _ in range(n)]
        for i in range(1, n):
            for j in range(1, m):
                if text1[i - 1] == text2[j - 1]:
                    f[i][j] = f[i - 1][j - 1] + 1
                else:
                    f[i][j] = max(f[i - 1][j], f[i][j - 1])
        return f[-1][-1]

    # O(nm) / O(min(n, m)), 660ms
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        f = [0] * (len(text2) + 1)
        for x in text1:
            pre = 0  # f[0]
            for j, y in enumerate(text2):
                tmp = f[j + 1]
                f[j + 1] = pre + 1 if x == y else max(f[j + 1], f[j])
                pre = tmp
        return f[-1]

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        def LCS(s: str, t: str) -> int:
            """requirement: len(s) <= len(t)"""
            d = collections.defaultdict(list)
            for i in reversed(range(len(t))):
                d[t[i]].append(i)
            nums = []
            for c in s:
                if c in d:
                    nums.extend(d[c])
            return LIS(nums)

        def LIS(nums: List[int]) -> int:
            """O(nlogn), Longest Increasing Subsequence"""
            stack = []
            for x in nums:
                idx = bisect.bisect_left(stack, x)
                if idx < len(stack):
                    stack[idx] = x
                else:
                    stack.append(x)
            return len(stack)

        if len(text1) > len(text2):
            return self.longestCommonSubsequence(text2, text1)
        if text1 in text2:
            return len(text1)  # huge speed up
        return LCS(text1, text2)

    # 36ms code of leetcode, only replaced with simple var-name
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        if len(text1) < len(text2):
            text1, text2 = text2, text1
        if text2 in text1:
            return len(text2)
        d = collections.defaultdict(list)
        e = []  # e: ends
        for i, c in enumerate(text2):
            d[c].append(i)
        for i in itertools.chain.from_iterable(reversed(d[c]) for c in text1 if c in d):
            l = bisect.bisect_left(e, i)
            if l == len(e):
                e.append(i)
            else:
                e[l] = i
        return len(e)


# 1144 - Decrease Elements To Make Array Zigzag - MEDIUM
class Solution:
    def movesToMakeZigzag(self, nums: List[int]) -> int:
        o = e = 0
        for i, v in enumerate(nums):
            if i % 2 == 0:
                l = v - nums[i - 1] + 1 if i > 0 and v >= nums[i - 1] else 0
                r = v - nums[i + 1] + 1 if i < len(nums) - 1 and v >= nums[i + 1] else 0
                o += max(l, r)
            else:
                l = v - nums[i - 1] + 1 if v >= nums[i - 1] else 0
                r = v - nums[i + 1] + 1 if i < len(nums) - 1 and v >= nums[i + 1] else 0
                e += max(l, r)
        return min(o, e)

    def movesToMakeZigzag(self, nums):
        s = [0] * 2
        for i, v in enumerate(nums):
            l = nums[i - 1] if i > 0 else math.inf
            r = nums[i + 1] if i < len(nums) - 1 else math.inf
            s[i % 2] += max(v - min(l, r) + 1, 0)
        return min(s)


# 1145 - Binary Tree Coloring Game - MEDIUM
class Solution:
    # 整颗树被分成三块: 父, 左, 右 -> 是否有一块大于总数的一半
    def btreeGameWinningMove(self, root: Optional[TreeNode], n: int, x: int) -> bool:
        def find(root: TreeNode) -> TreeNode:
            if not root:
                return None
            if root.val == x:
                return root
            return find(root.left) or find(root.right)

        p = find(root)

        def count(root: TreeNode) -> int:
            if not root:
                return 0
            return 1 + count(root.left) + count(root.right)

        l = count(p.left)
        r = count(p.right)
        choose = max(l, r, n - 1 - l - r)
        return choose > n - choose


#################
# 2022.10.17 VO #
#################
# 1146 - Snapshot Array - MEDIUM
# only update the change of each element, rather than record the whole arr
class SnapshotArray:
    def __init__(self, length: int):
        self.arr = [{0: 0} for _ in range(length)]
        self.snap_id = 0

    def set(self, index: int, val: int) -> None:
        self.arr[index][self.snap_id] = val
        return

    def snap(self) -> int:
        self.snap_id += 1
        return self.snap_id - 1

    def get(self, index: int, snap_id: int) -> int:
        d = self.arr[index]
        if snap_id in d:
            return d[snap_id]
        k = list(d.keys())
        i = bisect.bisect_left(k, snap_id)
        return d[k[i - 1]]  # [4, 6] 查找 5, 指向下标 1


# 1147 - Longest Chunked Palindrome Decomposition - HARD
class Solution:
    # O(n^2) / O(n)
    def longestDecomposition(self, text: str) -> int:
        n = len(text)
        ans = 0
        pre, suf = [], []
        for i in range(n // 2):
            pre.append(text[i])
            suf.append(text[-(i + 1)])
            if pre == suf[::-1]:
                ans += 2
                pre = []
                suf = []
        return ans + 1 if n % 2 or len(pre) else ans

    def longestDecomposition(self, s: str) -> int:
        if s == "":
            return 0
        for i in range(1, len(s) // 2 + 1):
            if s[:i] == s[-i:]:
                return 2 + self.longestDecomposition(s[i:-i])
        return 1

    def longestDecomposition(self, s: str) -> int:
        ans = 0
        while s:
            i = 1
            while i <= len(s) // 2 and s[:i] != s[-i:]:
                i += 1
            if i > len(s) // 2:
                ans += 1
                break
            ans += 2
            s = s[i:-i]
        return ans

    # O(n) / O(n)
    def longestDecomposition(self, text: str) -> int:
        def get(l, r):
            return (h[r] - h[l - 1] * p[r - l + 1]) % mod

        n = len(text)
        base = 131
        mod = int(1e9) + 7
        h = [0] * (n + 10)
        p = [1] * (n + 10)
        for i, c in enumerate(text):
            t = ord(c) - ord("a") + 1
            h[i + 1] = (h[i] * base) % mod + t
            p[i + 1] = (p[i] * base) % mod
        ans = 0
        i, j = 0, n - 1
        while i <= j:
            k = 1
            ok = False
            while i + k - 1 < j - k + 1:
                if get(i + 1, i + k) == get(j - k + 2, j + 1):
                    ans += 2
                    i += k
                    j -= k
                    ok = True
                    break
                k += 1
            if not ok:
                ans += 1
                break
        return ans


# 1154 - Day of the Year - EASY
class Solution:
    def dayOfYear(self, date: str) -> int:
        m = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        year, month, day = [int(x) for x in date.split("-")]
        if year % 400 == 0 or (year % 4 == 0 and year % 100 != 0):
            m[1] += 1
        return sum(m[: month - 1]) + day


# 1155 - Number of Dice Rolls With Target Sum - MEDIUM
class Solution:
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        @functools.lru_cache(None)
        def dfs(n: int, target: int) -> int:
            if n == 1:
                return int(target <= k)
            cur = 0
            for i in range(1, k + 1):
                if target > i:
                    cur = (cur + dfs(n - 1, target - i)) % mod
            return cur % mod

        mod = 1000000007
        return dfs(n, target) % mod


# 1156 - Swap For Longest Repeated Character Substring - MEDIUM
class Solution:
    def maxRepOpt1(self, text: str) -> int:
        cnt = collections.Counter(text)
        n = len(text)
        ans = i = 0
        while i < n:
            j = i
            while j < n and text[i] == text[j]:
                j += 1
            l = j - i
            k = j + 1
            while k < n and text[i] == text[k]:
                k += 1
            r = k - j - 1
            ans = max(ans, min(l + r + 1, cnt[text[i]]))
            i = j
        return ans


# 1160 - Find Words That Can Be Formed by Characters - EASY
class Solution:
    def countCharacters(self, words: List[str], chars: str) -> int:
        cnt = collections.Counter(chars)
        return sum(len(w) for w in words if collections.Counter(w) <= cnt)  # py3.10


# 1161 - Maximum Level Sum of a Binary Tree - MEDIUM
class Solution:
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        ans = lv = 1
        mx = total = root.val
        dq = collections.deque([root])
        while dq:
            total = 0
            for _ in range(len(dq)):
                n = dq.popleft()
                total += n.val
                if n.left:
                    dq.append(n.left)
                if n.right:
                    dq.append(n.right)
            if total > mx:
                ans = lv
                mx = total
            lv += 1
        return ans


# 1170 - Compare Strings by Frequency of the Smallest Character - MEDIUM
class Solution:
    def numSmallerByFrequency(self, queries: List[str], words: List[str]) -> List[int]:
        f = sorted(w.count(sorted(w)[0]) for w in words)
        return [
            len(words) - bisect.bisect_right(f, q.count(sorted(q)[0])) for q in queries
        ]

    def numSmallerByFrequency(self, queries: List[str], words: List[str]) -> List[int]:
        def f(s: str) -> int:
            cnt = collections.Counter(s)
            return next(cnt[c] for c in string.ascii_lowercase if cnt[c])

        arr = sorted(f(w) for w in words)
        return [len(words) - bisect.bisect_right(arr, f(q)) for q in queries]


# 1171 - Remove Zero Sum Consecutive Nodes from Linked List - MEDIUM
class Solution:
    def removeZeroSumSublists(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        vis = {0: dummy}
        pre = 0
        while head:
            pre += head.val
            vis[pre] = head
            head = head.next
        head = dummy
        pre = 0
        while head:
            pre += head.val
            head.next = vis[pre].next
            head = head.next
        return dummy.next

    def removeZeroSumSublists(self, head: Optional[ListNode]) -> Optional[ListNode]:
        node = dummy = ListNode(0, head)
        vis = {}
        pre = 0
        while node:
            pre += node.val
            vis[pre] = node
            node = node.next
        node = dummy
        pre = 0
        while node:
            pre += node.val
            node.next = vis[pre].next
            node = node.next
        return dummy.next


# 1175 - Prime Arrangements - EASY
class Solution:
    def numPrimeArrangements(self, n: int) -> int:
        def isPrime(n: int) -> int:
            if n == 1:
                return 0
            for i in range(2, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    return 0
            return 1

        def factorial(n: int) -> int:
            res = 1
            for i in range(1, n + 1):
                res *= i
                res %= mod
            return res

        mod = 10**9 + 7
        primes = sum(isPrime(i) for i in range(1, n + 1))
        return factorial(primes) * factorial(n - primes) % mod


# 1178 - Number of Valid Words for Each Puzzle - HARD
class Solution:
    # TLE
    def findNumOfValidWords(self, words: List[str], puzzles: List[str]) -> List[int]:
        puzzleSet = [set(p) for p in puzzles]
        wordSet = [set(w) for w in words]
        # firstLetters = set([p[0] for p in puzzles])
        ans = []
        for i, puzzle in enumerate(puzzles):
            num = 0
            for j in range(len(words)):
                # contain the first letter of puzzle
                if puzzle[0] in wordSet[j]:
                    # every letter is in puzzle
                    if wordSet[j] <= puzzleSet[i]:
                        num += 1
            ans.append(num)

        return ans


# 1184 - Distance Between Bus Stops - EASY
class Solution:
    def distanceBetweenBusStops(
        self, distance: List[int], start: int, destination: int
    ) -> int:
        a = min(destination, start)
        b = max(destination, start)
        return min(sum(distance[a:b]), sum(distance[b:] + distance[:a]))


# 1185 - Day of the Week - EASY
class Solution:
    def dayOfTheWeek(self, day: int, month: int, year: int) -> str:
        return datetime.date(year, month, day).strftime("%A")

    # Zelle formula
    days = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ]

    def dayOfTheWeek(self, d, m, y):
        if m < 3:
            m += 12
            y -= 1
        c, y = y // 100, y % 100
        w = (c // 4 - 2 * c + y + y // 4 + 13 * (m + 1) // 5 + d - 1) % 7
        return self.days[w]


# 1189 - Maximum Number of Balloons - EASY
class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        cnt = collections.Counter(text)
        return min(cnt["b"], cnt["a"], cnt["l"] // 2, cnt["o"] // 2, cnt["n"])
