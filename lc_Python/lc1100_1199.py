import bisect, collections, functools, heapq, itertools, math, queue, operator, string, threading
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
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        len1, len2 = len(text1) + 1, len(text2) + 1
        dp = [[0 for _ in range(len2)] for _ in range(len1)]
        for i in range(1, len1):
            for j in range(1, len2):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[-1][-1]


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


# 1189 - Maximum Number of Balloons - EASY
class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        cnt = collections.Counter(text)
        return min(cnt["b"], cnt["a"], cnt["l"] // 2, cnt["o"] // 2, cnt["n"])
