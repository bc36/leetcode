import bisect, collections, copy, datetime, functools, heapq, itertools, math, operator, re, string
from typing import List, Optional, Tuple
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


# 1000 - Minimum Cost to Merge Stones - HARD
class Solution:
    # 区间 DP, 枚举区间长度, 枚举左端点
    # O(n^3) / O(n^2 * k)
    def mergeStones(self, stones: List[int], k: int) -> int:
        n = len(stones)
        if (n - 1) % (k - 1) != 0:  # (n - k) % (k - 1)
            return -1

        # stone[l] 到 stone[r] 合并成 p 堆的最低成本
        @functools.lru_cache(None)
        def dfs(l: int, r: int, p: int) -> int:
            if r - l + 1 < p:
                return 0
            if p == 1:
                return dfs(l, r, k) + sum(stones[l : r + 1])
            # return min(dfs(l, i, 1) + dfs(i + 1, r, p - 1) for i in range(l, r - p + 2, k - 1))
            return min(dfs(l, i, 1) + dfs(i + 1, r, p - 1) for i in range(l, r, k - 1))

        return dfs(0, n - 1, k)

    def mergeStones(self, stones: List[int], k: int) -> int:
        n = len(stones)
        if (n - 1) % (k - 1):
            return -1
        pre = list(itertools.accumulate(stones, initial=0))

        @functools.lru_cache(None)
        def dfs(l: int, r: int, p: int) -> int:
            if p == 1:
                return 0 if l == r else dfs(l, r, k) + pre[r + 1] - pre[l]
            return min(dfs(l, i, 1) + dfs(i + 1, r, p - 1) for i in range(l, r, k - 1))

        return dfs(0, n - 1, 1)

    def mergeStones(self, stones: List[int], k: int) -> int:
        n = len(stones)
        if (n - 1) % (k - 1):
            return -1
        pre = list(itertools.accumulate(stones, initial=0))

        f = [[[math.inf] * (k + 1) for _ in range(n + 1)] for _ in range(n + 1)]
        for i in range(1, n + 1):
            f[i][i][1] = 0
        for l in range(2, n + 1):
            for i in range(1, n - l + 2):
                j = i + l - 1
                for m in range(1, k + 1):
                    f[i][j][m] = min(
                        f[i][h][1] + f[h + 1][j][m - 1] for h in range(i, j)
                    )
                f[i][j][1] = f[i][j][k] + pre[j] - pre[i - 1]
        return f[1][n][1]


# 1001 - Grid Illumination - HARD
class Solution:
    def gridIllumination(
        self, n: int, lamps: List[List[int]], queries: List[List[int]]
    ) -> List[int]:
        op = set()
        row = collections.defaultdict(int)
        col = collections.defaultdict(int)
        diag = collections.defaultdict(int)
        antidiag = collections.defaultdict(int)

        for i, j in lamps:
            if (i, j) not in op:
                op.add((i, j))
                row[i] += 1
                col[j] += 1
                diag[i - j] += 1  # \
                antidiag[i + j] += 1  # /

        ans = []
        for i, j in queries:
            # whether light on
            if row[i] or col[j] or diag[i - j] or antidiag[i + j]:
                ans.append(1)
            else:
                ans.append(0)
                continue
            # shut down
            for x, y in [
                (i - 1, j - 1),
                (i - 1, j),
                (i - 1, j + 1),
                (i, j - 1),
                (i, j),
                (i, j + 1),
                (i + 1, j - 1),
                (i + 1, j),
                (i + 1, j + 1),
            ]:
                if (x, y) in op:
                    op.remove((x, y))
                    row[x] -= 1
                    col[y] -= 1
                    diag[x - y] -= 1
                    antidiag[x + y] -= 1
        return ans


# 1003 - Check If Word Is Valid After Substitutions - MEDIUM
class Solution:
    def isValid(self, s: str) -> bool:
        while True:
            t = s.replace("abc", "")
            if s == t:
                break
            s = t
        return len(s) == 0

    def isValid(self, s: str) -> bool:
        t = s.replace("abc", "")
        while s != t:
            s = t
            t = s.replace("abc", "")
        return len(s) == 0

    def isValid(self, s: str) -> bool:
        while "abc" in s:
            s = s.replace("abc", "")
            if s == "":
                return True
        return False

    def isValid(self, s: str) -> bool:
        st = []
        for c in s:
            if c == "a" or c == "b":
                st.append(c)
            elif c == "c":
                if len(st) >= 2 and st[-1] == "b" and st[-2] == "a":
                    st.pop()
                    st.pop()
                else:
                    return False
        return len(st) == 0

    def isValid(self, s: str) -> bool:
        st = []
        for c in map(ord, s):
            if c > ord("a") and (len(st) == 0 or c - st.pop() != 1):
                return False
            if c < ord("c"):
                st.append(c)
        return len(st) == 0


# 1005 - Maximize Sum Of Array After K Negations - EASY
class Solution:
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        neg = [x for x in nums if x < 0]
        pos = [x for x in nums if x > 0]
        if len(neg) >= k:
            neg.sort()
            return sum(pos) - sum(neg[:k]) + sum(neg[k:])
        k -= len(neg)
        if not k & 1:
            return sum(pos) - sum(neg)
        tmp = min([abs(x) for x in nums])
        return sum(pos) - sum(neg) - 2 * tmp


# 1009 - Complement of Base 10 Integer - EASY
class Solution:
    def bitwiseComplement(self, n: int) -> int:
        ans = ""
        while n:
            ans += "0" if n & 1 else "1"
            n >>= 1
        return int(ans[::-1], 2) if ans else 1

    def bitwiseComplement(self, n: int) -> int:
        x = 1
        while n > x:
            x = x * 2 + 1
        return x ^ n  # XOR

    def bitwiseComplement(self, n: int) -> int:
        if not n:
            return 1
        mask = n
        mask |= mask >> 1
        mask |= mask >> 2
        mask |= mask >> 4
        mask |= mask >> 8
        mask |= mask >> 16
        return n ^ mask


# 1010 - Pairs of Songs With Total Durations Divisible by 60 - MEDIUM
class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        ans = 0
        d = collections.defaultdict(int)
        for t in time:
            t %= 60
            ans += d[(60 - t) % 60]
            d[t] += 1
        return ans

    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        ans = 0
        d = collections.defaultdict(int)
        for t in time:
            ans += d[-t % 60]
            d[t % 60] += 1
        return ans

    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        ans = 0
        cnt = [0] * 60
        for t in time:
            ans += cnt[-t % 60]
            cnt[t % 60] += 1
        return ans


# 1014 - Best Sightseeing Pair - MEDIUM
class Solution:
    def maxScoreSightseeingPair(self, A: List[int]) -> int:
        pre, ans = A[0], 0
        for i in range(1, len(A)):
            ans = max(ans, pre + A[i] - i)
            pre = max(pre, A[i] + i)
        return ans


# 1015 - Smallest Integer Divisible by K - MEDIUM
class Solution:
    # time consuming, just because Python supports arbitrarily large numbers
    def smallestRepunitDivByK(self, k: int) -> int:
        if not k % 2 or not k % 5:
            return -1
        ans, l = 1, 1
        while True:
            if ans % k == 0:
                return l
            l += 1
            ans = 10 * ans + 1

    # At every iteration, n = kq + r for some quotient q and remainder r.
    # Therefore, 10*n + 1 = 10(kq + r) + 1 = 10kq + 10r + 1.
    # 10kq is divisible by k, so for 10*n + 1 to be divisible by k, it all depends on if 10r + 1 is divisible by k.
    # Therefore, we only have to keep track of r!
    def smallestRepunitDivByK(self, k: int) -> int:
        if not k % 2 or not k % 5:
            return -1
        r = length = 1
        while True:
            r = r % k
            if not r:
                return length
            length += 1
            r = 10 * r + 1

    # k possible remainders from 0 to k-1
    def smallestRepunitDivByK(self, k: int) -> int:
        if k % 2 == 0 or k % 5 == 0:
            return -1
        n = 1
        for i in range(k):
            r = n % k
            if r == 0:
                return i + 1
            n = r * 10 + 1


# 1015 - Smallest Integer Divisible by K - MEDIUM
class Solution:
    # O(k) / O(k)
    def smallestRepunitDivByK(self, k: int) -> int:
        seen = set()
        x = 1 % k
        while x and x not in seen:
            seen.add(x)
            x = (x * 10 + 1) % k
        return -1 if x else len(seen) + 1


# 1016 - Binary String With Substrings Representing 1 To N - MEDIUM
class Solution:
    # O(log(min(m, n)) * mlog(min(m, n))) / O(logn), m = len(s)
    def queryString(self, s: str, n: int) -> bool:
        return all(bin(i)[2:] in s for i in range(1, n + 1))

    # O(mlogn) / O(min(mlogn, n)), m = len(s)
    def queryString(self, s: str, n: int) -> bool:
        seen = set()
        s = list(map(int, s))
        for i, x in enumerate(s):
            if x == 0:
                continue  # start from 1
            j = i + 1
            while x <= n:
                seen.add(x)
                if j == len(s):
                    break
                x = (x << 1) | s[j]  # 子串 s[i : j + 1] 的二进制数
                j += 1
        return len(seen) == n


# 1017 - Convert to Base -2 - MEDIUM
class Solution:
    def baseNeg2(self, n: int) -> str:
        if n == 0:
            return "0"
        ans = ""
        while n:
            k = n % 2
            n = -(n // 2)
            ans = str(k) + ans
        return ans


# 1019 - Next Greater Node In Linked List - MEDIUM
class Solution:
    def nextLargerNodes(self, head: Optional[ListNode]) -> List[int]:
        arr = []
        while head:
            arr.append(head.val)
            head = head.next
        ans = []
        st = []
        while arr:
            while st and st[-1] <= arr[-1]:
                st.pop()
            ans.append(st[-1] if st else 0)
            st.append(arr.pop())
        return ans[::-1]

    def nextLargerNodes(self, head: Optional[ListNode]) -> List[int]:
        arr = []
        while head:
            arr.append(head.val)
            head = head.next
        ans = [0] * len(arr)
        st = []
        for i in range(len(arr)):
            while st and arr[i] > arr[st[-1]]:
                ans[st.pop()] = arr[i]
            st.append(i)
        return ans


# 1020 - Number of Enclaves - MEDIUM
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        def dfs(x, y):
            grid[x][y] = 0
            for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny]:
                    dfs(nx, ny)

        ans, m, n = 0, len(grid), len(grid[0])
        for i in range(m):
            if grid[i][0] == 1:
                dfs(i, 0)
            if grid[i][n - 1] == 1:
                dfs(i, n - 1)
        for j in range(n):
            if grid[0][j] == 1:
                dfs(0, j)
            if grid[m - 1][j] == 1:
                dfs(m - 1, j)
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    ans += 1
        return ans

    def numEnclaves(self, grid: List[List[int]]) -> int:
        def bfs(x, y):
            grid[x][y] = 0
            q = [[x, y]]
            while q:
                x, y = q.pop()
                for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                    if 0 <= nx < m and 0 <= ny < n and grid[nx][ny]:
                        grid[nx][ny] = 0
                        q.append([nx, ny])

        m, n = len(grid), len(grid[0])
        for i in range(m):
            if grid[i][0] == 1:
                bfs(i, 0)
            if grid[i][n - 1] == 1:
                bfs(i, n - 1)
        for j in range(n):
            if grid[0][j] == 1:
                bfs(0, j)
            if grid[m - 1][j] == 1:
                bfs(m - 1, j)

        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j]:
                    ans += 1
        return ans


# 1021 - Remove Outermost Parentheses - EASY
class Solution:
    def removeOuterParentheses(self, s: str) -> str:
        ans = ""
        f = 0
        for c in s:
            if c == ")":
                f -= 1
            if f:
                ans += c
            if c == "(":
                f += 1
        return ans


# 1022 - Sum of Root To Leaf Binary Numbers - EASY
class Solution:
    def sumRootToLeaf(self, root: TreeNode) -> int:
        dq, ans = collections.deque([(root, root.val)]), 0
        while dq:
            for _ in range(len(dq)):
                n, num = dq.popleft()
                if not n.left and not n.right:
                    ans += num
                if n.left:
                    dq.append((n.left, num * 2 + n.left.val))
                if n.right:
                    dq.append((n.right, num * 2 + n.right.val))
        return ans

    def sumRootToLeaf(self, root: TreeNode) -> int:
        def dfs(root: TreeNode, v: int) -> int:
            if root is None:
                return 0
            v = (v << 1) | root.val
            if root.left is None and root.right is None:
                return v
            return dfs(root.left, v) + dfs(root.right, v)

        return dfs(root, 0)

    def sumRootToLeaf(self, root: TreeNode, val=0) -> int:
        if not root:
            return 0
        val = val * 2 + root.val
        if root.left == root.right == None:
            return val
        return self.sumRootToLeaf(root.left, val) + self.sumRootToLeaf(root.right, val)

    def sumRootToLeaf(self, root: TreeNode) -> int:
        def dfs(root: TreeNode, pre: int):
            if not (root.left or root.right):
                self.ans += (pre << 1) + root.val
                return
            if root.left:
                dfs(root.left, (pre << 1) + root.val)
            if root.right:
                dfs(root.right, (pre << 1) + root.val)
            return

        self.ans = 0
        dfs(root, 0)
        return self.ans


# 1023 - Camelcase Matching - MEDIUM
class Solution:
    def camelMatch(self, queries: List[str], pattern: str) -> List[bool]:
        p = "^[a-z]*" + "[a-z]*".join(pattern) + "[a-z]*$"
        return [re.match(p, q) is not None for q in queries]

    def camelMatch(self, queries: List[str], pattern: str) -> List[bool]:
        def check(s: str) -> bool:
            j = 0
            for i in range(len(s)):
                if j < m and s[i] == pattern[j]:
                    j += 1
                elif s[i].isupper():
                    return False
            return j == m

        m = len(pattern)
        return list(check(q) for q in queries)


# 1025 - Divisor Game - EASY
class Solution:
    def divisorGame(self, n: int) -> bool:
        return not n & 1


# 1026 - Maximum Difference Between Node and Ancestor - MEDIUM
class Solution:
    # down to top, calculate the minimum and maximum values then pass them to the root
    def maxAncestorDiff(self, root: TreeNode) -> int:
        self.ans = 0

        def dfs(root: TreeNode) -> Tuple[int, int]:
            if not root:
                return float("inf"), -float("inf")
            lmin, lmax = dfs(root.left)
            rmin, rmax = dfs(root.right)
            rootmin = min(root.val, lmin, rmin)
            rootmax = max(root.val, lmax, rmax)
            self.ans = max(self.ans, rootmax - root.val, root.val - rootmin)
            return rootmin, rootmax

        dfs(root)
        return self.ans

    # top to down, pass the minimum and maximum values to the children
    def maxAncestorDiff(self, root: TreeNode) -> int:
        def dfs(root: TreeNode, low: int, high: int) -> int:
            if root is None:
                return high - low
            low = min(root.val, low)
            high = max(root.val, high)
            return max(dfs(root.left, low, high), dfs(root.right, low, high))

        return dfs(root, root.val, root.val)

    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        def dfs(root: TreeNode, mx: int, mi: int) -> int:
            if not root:
                return 0
            if root.val > mx:
                mx = root.val
            if root.val < mi:
                mi = root.val
            return max(mx - mi, dfs(root.left, mx, mi), dfs(root.right, mx, mi))

        return dfs(root, 0, math.inf)

    # top to down, pass the minimum and maximum values to the children
    def maxAncestorDiff(self, root: TreeNode, mn: int = 100000, mx: int = 0) -> int:
        return (
            max(
                self.maxAncestorDiff(root.left, min(mn, root.val), max(mx, root.val)),
                self.maxAncestorDiff(root.right, min(mn, root.val), max(mx, root.val)),
            )
            if root
            else mx - mn
        )

    def maxAncestorDiff(self, root: TreeNode) -> int:
        q = [(root, root.val, root.val)]  # node, parent, child
        ans = 0
        while q:
            x, fa, ch = q.pop()
            ans = max(ans, fa - ch)
            if x.left:
                q.append((x.left, max(fa, x.left.val), min(ch, x.left.val)))
            if x.right:
                q.append(x.right, max(fa, x.right.val), min(ch, x.right.val))
        return ans


# 1027 - Longest Arithmetic Subsequence - MEDIUM
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        def check(d: int) -> int:
            m = {}
            for v in nums:
                m[v + d] = m.get(v, 0) + 1
            return max(m.values())

        return max(check(i) for i in range(-500, 501))

    def longestArithSeqLength(self, nums: List[int]) -> int:
        f = [{} for _ in range(len(nums))]
        for i in range(len(nums)):
            for j in range(i):
                d = nums[i] - nums[j]
                f[i][d] = f[j].get(d, 1) + 1
        return max(max(d.values(), default=0) for d in f)
        return max(max(d.values()) for d in f[1:])

    def longestArithSeqLength(self, nums: List[int]) -> int:
        vis = set()
        m = {}
        for v in nums:
            for u in vis:
                d = u - v
                m[(v, d)] = m.get((u, d), 1) + 1
            vis.add(v)
        return max(m.values())


# 1029 - Two City Scheduling - MEDIUM
class Solution:
    # O(nlogn) / O(logn)
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        costs.sort(key=lambda x: x[0] - x[1])
        n = len(costs)
        return sum(x for x, _ in costs[: n // 2]) + sum(y for _, y in costs[n // 2 : n])

    # O(nlogn) / O(n)
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        # all go to one city at first, then move to another, find the minimum difference
        first = [x for x, _ in costs]
        diff = [y - x for x, y in costs]  # negative diff means we get a refund
        return sum(first) + sum(sorted(diff)[: len(costs) // 2])


# 1030 - Matrix Cells in Distance Order - EASY
class Solution:
    def allCellsDistOrder(
        self, rows: int, cols: int, rCenter: int, cCenter: int
    ) -> List[List[int]]:
        return sorted(
            [(i, j) for i in range(rows) for j in range(cols)],
            key=lambda x: abs(x[0] - rCenter) + abs(x[1] - cCenter),
        )


# 1032 - Stream of Characters - HARD
class TrieNode:
    def __init__(self) -> None:
        self.ch = collections.defaultdict(TrieNode)
        self.end = False


class Trie:
    def __init__(self, words: List[str]) -> None:
        self.root = TrieNode()
        for w in words:
            r = self.root
            for c in w:
                r = r.ch[c]
            r.end = True
        self.s = [self.root]

    def query(self, letter: str) -> bool:
        f = False
        new = [self.root]
        for node in self.s:
            if letter in node.ch:
                new.append(node.ch[letter])
                if node.ch[letter].end:
                    f = True
        self.s = new
        return f


class StreamChecker:
    # 建树, query 时维护一个 pointer 集合
    def __init__(self, words: List[str]):
        self.trie = Trie(words)

    def query(self, letter: str) -> bool:
        return self.trie.query(letter)


class StreamChecker:
    # 建树, query 时维护一个 pointer 集合
    def __init__(self, words: List[str]):
        trie = dict()
        for w in words:
            r = trie
            for c in w:
                if c not in r:
                    r[c] = dict()
                r = r[c]
            r["end"] = True

        self.trie = trie
        self.s = [self.trie]

    def query(self, letter: str) -> bool:
        f = False
        new = [self.trie]
        for r in self.s:
            if letter in r:
                new.append(r[letter])
                if "end" in r[letter]:
                    f = True
        self.s = new
        return f


# 1034 - Coloring A Border - MEDIUM
class Solution:
    # bfs
    def colorBorder(
        self, grid: List[List[int]], row: int, col: int, color: int
    ) -> List[List[int]]:
        position, borders, originalColor = [(row, col)], [], grid[row][col]
        visited = [[False] * len(grid[0]) for _ in range(len(grid))]
        visited[row][col] = True
        while position:
            x, y = position.pop()
            isBorder = False
            for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if not (
                    0 <= nx < len(grid)
                    and 0 <= ny < len(grid[0])
                    and grid[nx][ny] == originalColor
                ):
                    isBorder = True
                elif not visited[nx][ny]:
                    visited[nx][ny] = True
                    position.append((nx, ny))
            if isBorder:
                borders.append((x, y))
        for i, j in borders:
            grid[i][j] = color
        return grid

    # bfs
    def colorBorder(
        self, grid: List[List[int]], row: int, col: int, color: int
    ) -> List[List[int]]:
        m, n = len(grid), len(grid[0])
        bfs, component, border = [[row, col]], set([(row, col)]), set()
        while bfs:
            r, c = bfs.pop()
            for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                x, y = r + i, c + j
                if 0 <= x < m and 0 <= y < n and grid[x][y] == grid[r][c]:
                    if (x, y) not in component:
                        bfs.append([x, y])
                        component.add((x, y))
                else:
                    border.add((r, c))
        for x, y in border:
            grid[x][y] = color
        return grid

    # dfs
    def colorBorder(
        self, grid: List[List[int]], row: int, col: int, color: int
    ) -> List[List[int]]:
        visited, m, n = set(), len(grid), len(grid[0])

        def dfs(x: int, y: int) -> bool:
            if (x, y) in visited:
                return True
            if not (0 <= x < m and 0 <= y < n and grid[x][y] == grid[row][col]):
                return False
            visited.add((x, y))
            if dfs(x + 1, y) + dfs(x - 1, y) + dfs(x, y + 1) + dfs(x, y - 1) < 4:
                grid[x][y] = color
            return True

        dfs(row, col)
        return grid


# 1036 - Escape a Large Maze - HARD
class Solution:
    # bfs: determine if the start and end points are surrounded
    #      since the amount of data is large
    def isEscapePossible(
        self, blocked: List[List[int]], source: List[int], target: List[int]
    ) -> bool:
        self.blocked_set = set([(r, c) for r, c in blocked])
        bn = len(self.blocked_set)
        if bn <= 1:
            return True
        self.maxblock = bn * (bn - 1) // 2
        return self.bfs(source, target) and self.bfs(target, source)

    def bfs(self, s: List[int], t: List[int]) -> bool:
        row, col = 10**6, 10**6
        visited = set()
        sr, sc = s[0], s[1]
        tr, tc = t[0], t[1]

        dq = collections.deque([(sr, sc)])
        visited.add((sr, sc))
        while dq:
            if len(visited) > self.maxblock:
                return True
            for _ in range(len(dq)):
                r, c = dq.popleft()
                if r == tr and c == tc:
                    return True
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (
                        0 <= nr < row
                        and 0 <= nc < col
                        and ((nr, nc) not in self.blocked_set)
                        and ((nr, nc) not in visited)
                    ):
                        dq.append((nr, nc))
                        visited.add((nr, nc))
        return False

    def isEscapePossible(
        self, blocked: List[List[int]], source: List[int], target: List[int]
    ) -> bool:
        blocked = {tuple(p) for p in blocked}
        # > 400ms
        # blocked_set = set([(r, c) for r, c in blocked])
        # size = len(blocked_set) * (len(blocked_set) - 1) // 2
        # > 800 ms
        size = len(blocked) * (len(blocked) - 1) // 2

        def bfs(source: List[int], target: List[int]) -> bool:
            bfs, seen = [source], {tuple(source)}
            for x0, y0 in bfs:
                for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                    x, y = x0 + i, y0 + j
                    if (
                        0 <= x < 10**6
                        and 0 <= y < 10**6
                        and (x, y) not in seen
                        and (x, y) not in blocked
                    ):
                        if [x, y] == target:
                            return True
                        bfs.append([x, y])
                        seen.add((x, y))
                # > 2000ms
                # if len(bfs) == 20000: return True
                if len(bfs) > size:
                    return True
            return False

        return bfs(source, target) and bfs(target, source)


# 1037 - Valid Boomerang - EASY
class Solution:
    def isBoomerang(self, p: List[List[int]]) -> bool:
        x0, y0 = p[0]
        x1, y1 = p[1]
        x2, y2 = p[2]
        # (y1 - y0) / (x1 - x0) != (y2 - y1) / (x2 - x1)
        return (y1 - y0) * (x2 - x1) != (y2 - y1) * (x1 - x0)


# 1039 - Minimum Score Triangulation of Polygon - MEDIUM
class Solution:
    def minScoreTriangulation(self, values: List[int]) -> int:
        @functools.lru_cache(None)
        def dfs(i: int, j: int) -> int:
            if i + 1 == j:
                return 0
            return min(
                dfs(i, k) + dfs(k, j) + values[i] * values[j] * values[k]
                for k in range(i + 1, j)
            )

        return dfs(0, len(values) - 1)

    def minScoreTriangulation(self, values: List[int]) -> int:
        n = len(values)
        f = [[0] * n for _ in range(n)]
        for r in range(2, n):
            for l in range(r - 2, -1, -1):
                f[l][r] = math.inf
                for m in range(l + 1, r):
                    f[l][r] = min(
                        f[l][r], f[l][m] + f[m][r] + values[l] * values[r] * values[m]
                    )
        return f[0][n - 1]


# 1041 - Robot Bounded In Circle - MEDIUM
class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        x, y, dx, dy = 0, 0, 0, 1
        for i in instructions:
            if i == "R":
                dx, dy = dy, -dx
            if i == "L":
                dx, dy = -dy, dx
            if i == "G":
                x, y = x + dx, y + dy
        return (x, y) == (0, 0) or not (dx == 0 and dy > 0)


# 1042 - Flower Planting With No Adjacent - MEDIUM
class Solution:
    def gardenNoAdj(self, n: int, paths: List[List[int]]) -> List[int]:
        g = [[] for _ in range(n)]
        for x, y in paths:
            g[x - 1].append(y - 1)
            g[y - 1].append(x - 1)
        ans = [0] * n
        for i, adj in enumerate(g):
            tmp = [1, 2, 3, 4]
            for y in adj:
                if ans[y] and ans[y] in tmp:
                    tmp.remove(ans[y])
            ans[i] = tmp[0]
        return ans

    def gardenNoAdj(self, n: int, paths: List[List[int]]) -> List[int]:
        g = [[] for _ in range(n)]
        for x, y in paths:
            g[x - 1].append(y - 1)
            g[y - 1].append(x - 1)
        ans = [0] * n
        for i, adj in enumerate(g):
            ans[i] = ({1, 2, 3, 4} - {ans[j] for j in adj}).pop()
        return ans


# 1046 - Last Stone Weight - EASY
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        q = [-v for v in stones]
        heapq.heapify(q)
        while len(q) > 1:
            a = -heapq.heappop(q)
            b = -heapq.heappop(q)
            if a > b:
                heapq.heappush(q, -a + b)
        return -q[0] if q else 0


# 1047 - Remove All Adjacent Duplicates In String - EASY
class Solution:
    # stack
    def removeDuplicates(self, s: str) -> str:
        stack = [s[0]]
        for i in range(1, len(s)):
            if stack and s[i] == stack[-1]:
                stack.pop()
            else:
                stack.append(s[i])
        return "".join(stack)

    # two pointers
    def removeDuplicates(self, s: str) -> str:
        # pointers: 'ch' and 'end',
        # change 'ls' in-place.
        ls, end = list(s), -1
        for ch in ls:
            if end >= 0 and ls[end] == ch:
                end -= 1
            else:
                end += 1
                ls[end] = ch
        return "".join(ls[: end + 1])


# 1048 - Longest String Chain - MEDIUM
class Solution:
    # O(n * m * m + n * m * logn + n * m * m + m + n) / O(n * m), m = len(words[i])
    def longestStrChain(self, words: List[str]) -> int:
        f = collections.defaultdict(list)  # n 个单词, 每个单词 m 个 candidate -> f 最大 n * m
        for w in words:  # O(n)
            for i in range(len(w)):  # O(m)
                f[w[:i] + "*" + w[i + 1 :]].append(w)  # O(m)

        def dfs(w: str) -> int:  # O(m * m + m + n)
            r = 1
            for i in range(len(w) + 1):  # O(m)
                new = w[:i] + "*" + w[i:]  # O(m)
                for nxt in f[new]:  # 根据 vis 大小决定, 最多循环 m + n 次
                    if nxt not in vis:
                        vis.add(nxt)
                        x = dfs(nxt) + 1
                        r = x if x > r else r
            return r

        vis = set()  # n 个单词, 每个单词 m 个 candidate -> vis 最大 n * m
        ans = 1
        # sort -> O(n * m * logn)
        for w in sorted(words, key=len):  # O(n)
            if w not in vis:
                vis.add(w)
                ans = max(ans, dfs(w))  # O(m * m)
        return ans

    # # O(n * m * logn + n * m + n) / O(n), m = len(words[i])
    def longestStrChain(self, words: List[str]) -> int:
        f = {}  # dp
        for w in sorted(words, key=len):
            f[w] = 1
            for i in range(len(w)):
                tmp = w[:i] + w[i + 1 :]
                if tmp in f:
                    f[w] = max(f[w], f[tmp] + 1)
        return max(f.values())

    def longestStrChain(self, words: List[str]) -> int:
        f = {}
        for w in sorted(words, key=len):
            f[w] = max(f.get(w[:i] + w[i + 1 :], 0) + 1 for i in range(len(w)))
        return max(f.values())


# 1051 - Height Checker - EASY
class Solution:
    def heightChecker(self, heights: List[int]) -> int:
        exp = sorted(heights)
        ans = 0
        for i in range(len(heights)):
            if heights[i] != exp[i]:
                ans += 1
        return ans


# 1053 - Previous Permutation With One Swap - MEDIUM
class Solution:
    # O(n) / O(1)
    def prevPermOpt1(self, arr: List[int]) -> List[int]:
        for i in range(len(arr) - 1, 0, -1):
            if arr[i - 1] > arr[i]:
                t = i
                for j in range(i, len(arr)):
                    if arr[i - 1] > arr[j] and arr[j] > arr[t]:
                        t = j
                arr[i - 1], arr[t] = arr[t], arr[i - 1]
                break
        return arr


# 1078 - Occurrences After Bigram - EASY
class Solution:
    def findOcurrences(self, text: str, first: str, second: str) -> List[str]:
        text, ans = text.split(), []
        for i in range(2, len(text)):
            if text[i - 2] == first and text[i - 1] == second:
                ans.append(text[i])
        return ans


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


# 1089 - Duplicate Zeros - EASY
class Solution:
    def duplicateZeros(self, arr: List[int]) -> None:
        """
        Do not return anything, modify arr in-place instead.
        """
        n = len(arr)
        i = 0
        while i < n:
            if arr[i] == 0:
                arr.insert(i, 0)
                arr.pop()
                i += 2
            else:
                i += 1
        return

    def duplicateZeros(self, arr: List[int]) -> None:
        l = len(arr)
        dq = collections.deque(arr)
        ans = []
        while dq and l > 0:
            ans.append(dq.popleft())
            if ans[-1] == 0 and l > 0:
                ans.append(0)
                l -= 1
            l -= 1
        for i in range(len(arr)):
            arr[i] = ans[i]
        return

    def duplicateZeros(self, arr: List[int]) -> None:
        arr[:] = [x for v in arr for x in ([v] if v else [0, 0])][: len(arr)]


# 1091 - Shortest Path in Binary Matrix - MEDIUM
class Solution:
    # TLE, not suitable for 'dfs', be careful with 'visited2'(cycle)
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if grid[-1][-1] == 1 or grid[0][0] == 1:
            return -1

        def dfs(x, y, step, visited):
            n = len(grid)
            if x == y == n - 1:
                self.ans = min(self.ans, step)
                return
            for i, j in (
                (x - 1, y - 1),
                (x - 1, y),
                (x - 1, y + 1),
                (x, y - 1),
                (x, y + 1),
                (x + 1, y - 1),
                (x + 1, y),
                (x + 1, y + 1),
            ):
                if (
                    0 <= i < n
                    and 0 <= j < n
                    and grid[i][j] == 0
                    and (i, j) not in visited
                ):
                    visited.add((i, j))
                    visited2 = copy.deepcopy(visited)
                    dfs(i, j, step + 1, visited2)
            return

        self.ans = math.inf
        dfs(0, 0, 1, set())
        return self.ans if self.ans != math.inf else -1

    # bfs, O(n^2) + O(n)
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if grid[-1][-1] == 1 or grid[0][0] == 1:
            return -1
        dq, n = collections.deque([(0, 0, 1)]), len(grid)
        # no need to use 'visited', space complexity from O(n^2) to O(n)
        grid[0][0] = 1
        while dq:
            x, y, step = dq.popleft()
            if x == y == n - 1:
                return step
            for i, j in (
                (x - 1, y - 1),
                (x - 1, y),
                (x - 1, y + 1),
                (x, y - 1),
                (x, y + 1),
                (x + 1, y - 1),
                (x + 1, y),
                (x + 1, y + 1),
            ):
                if 0 <= i < n and 0 <= j < n and grid[i][j] == 0:
                    grid[i][j] = 1
                    dq.append((i, j, step + 1))
        return -1


# 1092 - Shortest Common Supersequence  - HARD
class Solution:
    # 最长公共子序列
    # f[i][j] 表示字符串 str1 的前 i 个字符和字符串 str2 的前 j 个字符的最长公共子序列的长度
    # f[i][j] = 0                                   if i == 0 or j == 0
    #         = f[i - 1][j - 1]                     if str1[i] == str2[j]
    #         = min(f[i - 1][j], f[i][j - 1]) + 1   if str1[i] != str2[j]
    # O(nm) / O(nm)
    def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
        n, m = len(str1), len(str2)
        f = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if str1[i - 1] == str2[j - 1]:
                    f[i][j] = f[i - 1][j - 1] + 1
                else:
                    f[i][j] = max(f[i - 1][j], f[i][j - 1])
        ans = []
        i, j = n, m
        while i and j:
            # 如果 str1[i] != str2[j], 则将 f[i][j] 与 f[i - 1][j] 和 f[i][j - 1] 中的最大值进行比较
            if f[i][j] == f[i - 1][j]:
                i -= 1
                ans.append(str1[i])
            elif f[i][j] == f[i][j - 1]:
                j -= 1
                ans.append(str2[j])
            else:
                i, j = i - 1, j - 1
                ans.append(str1[i])
        return "".join(itertools.chain.from_iterable((str1[:i], str2[:j], ans[::-1])))

    # f[i + 1][j + 1] 表示 s 的前 i 个字母和 t 的前 j 个字母的最短公共超序列的长度
    # f[i][j] = f[i - 1][j - 1] + 1                 if s[i] == t[j]
    #         = min(f[i - 1][j], f[i][j - 1]) + 1   if s[i] != t[j]
    # 初始值 f[i][0] = i, f[0][j] = j
    # O(nm) / O(nm)
    def shortestCommonSupersequence(self, s: str, t: str) -> str:
        n, m = len(s), len(t)
        f = [[0] * (m + 1) for _ in range(n + 1)]
        f[0] = list(range(m + 1))
        for i in range(1, n + 1):
            f[i][0] = i
        for i, x in enumerate(s):
            for j, y in enumerate(t):
                if x == y:
                    f[i + 1][j + 1] = f[i][j] + 1
                else:
                    f[i + 1][j + 1] = min(f[i][j + 1], f[i + 1][j]) + 1
        ans = []
        i, j = n - 1, m - 1
        while i >= 0 and j >= 0:
            if s[i] == t[j]:
                ans.append(s[i])
                i -= 1
                j -= 1
            elif f[i + 1][j + 1] == f[i][j + 1] + 1:
                ans.append(s[i])
                i -= 1
            else:
                ans.append(t[j])
                j -= 1
        return s[: i + 1] + t[: j + 1] + "".join(ans[::-1])

    def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
        if len(str1) < len(str2):
            str1, str2 = str2, str1
        l2 = len(str2)
        # 先找最长公共串, 再把没有的加上
        f = [""] * (l2 + 1)
        g = [""] * (l2 + 1)
        for c1 in str1:
            for i, c2 in enumerate(str2):
                if c2 == c1:
                    f[i + 1] = g[i] + c1
                else:
                    f[i + 1] = f[i] if len(f[i]) > len(g[i + 1]) else g[i + 1]
            g, f = f, g
        lcs = g[-1]
        if not lcs:  # == 0
            return str1 + str2
        ans = ""
        i = j = k = 0
        while k < len(lcs):
            while str1[i] != lcs[k]:
                ans += str1[i]
                i += 1
            while str2[j] != lcs[k]:
                ans += str2[j]
                j += 1
            ans += lcs[k]
            k += 1
            i += 1
            j += 1
        return ans + str1[i:] + str2[j:]


# 1094 - Car Pooling - MEDIUM
class Solution:
    def carPooling(self, trips, capacity):
        for _, v in sorted(x for n, i, j in trips for x in [[i, n], [j, -n]]):
            capacity -= v
            if capacity < 0:
                return False
        return True

    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        heap = []
        for n, i, j in trips:
            heapq.heappush(heap, (i, n))
            heapq.heappush(heap, (j, -n))
        while heap:
            capacity -= heapq.heappop(heap)[1]
            if capacity < 0:
                return False
        return True
