from typing import List
import collections, itertools, math


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 1200 - Minimum Absolute Difference - EASY
class Solution:
    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        diff = float("inf")
        arr.sort()
        for i in range(len(arr) - 1):
            diff = min(arr[i + 1] - arr[i], diff)
        ans, left, right = [], 0, 1
        while right < len(arr):
            if arr[right] - arr[left] > diff:
                left += 1
            elif arr[right] - arr[left] < diff:
                right += 1
            else:
                ans.append([arr[left], arr[right]])
                left += 1
                right += 1
        return ans

    def minimumAbsDifference(self, a: List[int]) -> List[List[int]]:
        a.sort()
        diff = min(a[i] - a[i - 1] for i in range(1, len(a)))
        return [[a[i - 1], a[i]] for i in range(1, len(a)) if a[i] - a[i - 1] == diff]

    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        arr.append(math.inf)
        arr.sort()
        ans = []
        d = math.inf
        for i in range(1, len(arr)):
            if arr[i] - arr[i - 1] < d:
                ans = [[arr[i - 1], arr[i]]]
                d = arr[i] - arr[i - 1]
            elif arr[i] - arr[i - 1] == d:
                ans.append([arr[i - 1], arr[i]])
        return ans


# 1202 - Smallest String With Swaps - MEDIUM
class Solution:
    # O(n + m + nlogn) / O(n), n = len(s), m = len(pairs)
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        g = [[] for _ in range(n)]
        for x, y in pairs:
            g[x].append(y)
            g[y].append(x)
        seen = [0] * n
        ans = list(s)
        for i in range(n):
            if not seen[i]:
                conn = []
                dq = collections.deque([i])
                seen[i] = 1
                conn.append(i)
                while dq:
                    cur = dq.popleft()
                    for nxt in g[cur]:
                        if not seen[nxt]:
                            seen[nxt] = 1
                            conn.append(nxt)
                            dq.append(nxt)
                idx = sorted(conn)
                val = sorted(ans[x] for x in conn)
                for j, ch in zip(idx, val):
                    ans[j] = ch
        return "".join(ans)

    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        def dfs(i):
            seen[i] = True
            conn.append(i)
            for j in g[i]:
                if not seen[j]:
                    dfs(j)
            return

        n = len(s)
        g = [[] for _ in range(n)]
        for i, j in pairs:
            g[i].append(j)
            g[j].append(i)
        seen = [False for _ in range(n)]
        ans = list(s)
        for i in range(n):
            if not seen[i]:
                conn = []
                dfs(i)
                conn.sort()
                chars = [ans[k] for k in conn]
                chars.sort()
                for j in range(len(conn)):
                    ans[conn[j]] = chars[j]
        return "".join(ans)

    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        n = len(s)
        g = [[] for _ in range(n)]
        for a, b in pairs:
            g[a].append(b)
            g[b].append(a)
        s = list(s)
        seen = set()
        for i in range(n):
            if i not in seen:
                dq = collections.deque([i])
                conn = []
                while dq:
                    j = dq.popleft()
                    for k in g[j]:
                        if k not in seen:
                            seen.add(k)
                            dq.append(k)
                            conn.append(k)
                char = [s[j] for j in conn]
                conn.sort()
                char.sort()
                for j in range(len(conn)):
                    s[conn[j]] = char[j]
        return "".join(s)

    # union find, disjoint Set
    # O((E + V) * αV + VlogV) / O(V)
    # where α is The Inverse Ackermann Function
    # union: E * αV, find: V * αV, sort: VlogV
    # V represents the number of vertices (the length of the given string)
    # E represents the number of edges (the number of pairs)
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class UF:
            def __init__(self, n):
                self.p = list(range(n))

            def union(self, x, y):
                self.p[self.find(x)] = self.find(y)
                return

            def find(self, x):
                if x != self.p[x]:
                    self.p[x] = self.find(self.p[x])
                return self.p[x]

        uf = UF(len(s))
        ans = []
        d = collections.defaultdict(list)
        for a, b in pairs:
            uf.union(a, b)
        for i in range(len(s)):
            d[uf.find(i)].append(s[i])
        for group in d:
            d[group].sort(reverse=True)
        for i in range(len(s)):
            ans.append(d[uf.find(i)].pop())
        return "".join(ans)


class UF:
    def __init__(self, n):
        self.r = [i for i in range(n)]
        self.rank = [1] * n

    def find(self, x):
        # if self.r[x] != x:
        #     self.r[x] = self.find(self.r[x])
        # return self.r[x]
        if self.r[x] == x:
            return x
        return self.find(self.r[x])

    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            if self.rank[rx] > self.rank[ry]:
                self.r[ry] = rx
                self.rank[rx] += 1
            else:
                self.r[rx] = ry
                self.rank[ry] += 1
        return


class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        uf = UF(len(s))
        for a, b in pairs:
            uf.union(a, b)
        g = collections.defaultdict(list)
        for i in range(len(s)):
            g[uf.find(i)].append(s[i])
        for k in g:
            g[k].sort(reverse=True)
        ans = []
        for i in range(len(s)):
            ans.append(g[uf.find(i)].pop())
        return "".join(ans)

    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        uf = UF(len(s))
        for a, b in pairs:
            uf.union(a, b)
        p = {i: uf.find(i) for i in range(len(s))}
        conn = collections.defaultdict(list)
        for i in range(len(s)):
            conn[p[i]].append(i)
        s = list(s)
        for _, v in conn.items():
            char = [s[i] for i in v]
            v.sort()
            char.sort()
            for i, ch in zip(v, char):
                s[i] = ch
        return "".join(s)


# 1217 - Minimum Cost to Move Chips to The Same Position - EASY
class Solution:
    def minCostToMoveChips(self, position: List[int]) -> int:
        odd, even = 0, 0
        for chip in position:
            if chip & 1:
                even += 1
            else:
                odd += 1
        return min(odd, even)


class Solution:
    def minCostToMoveChips(self, position: List[int]) -> int:
        cost = [0, 0]
        for chip in position:
            cost[chip & 1] += 1
        return min(cost)


# 1218 - Longest Arithmetic Subsequence of Given Difference - MEDIUM
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        dp = collections.defaultdict(int)
        for i in arr:
            dp[i] = dp[i - difference] + 1
        return max(dp.values())


# 1219 - Path with Maximum Gold - MEDIUM
class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:
        def backtrack(i, j, total):
            self.ans = max(self.ans, total)
            t = grid[i][j]
            grid[i][j] = 0
            for dx, dy in dire:
                if 0 <= i + dx < m and 0 <= j + dy < n and grid[i + dx][j + dy] != 0:
                    backtrack(i + dx, j + dy, total + grid[i + dx][j + dy])
            grid[i][j] = t
            return

        # not every need to be searched
        def find_corner(i, j):
            count = 0
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = dx + i, dy + j
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny]:
                    count += 1
            return count <= 2

        self.ans = 0
        dire = ((1, 0), (-1, 0), (0, 1), (0, -1))
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] != 0 and find_corner(j, i):
                    backtrack(i, j, grid[i][j])
        return self.ans

    def getMaximumGold(self, grid: List[List[int]]) -> int:
        def backtrack(x, y, count, seen):
            nonlocal ans
            if count > ans:
                ans = count
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and seen[nx][ny] and grid[nx][ny]:
                    seen[nx][ny] = 0
                    backtrack(nx, ny, count + grid[nx][ny], seen)
                    seen[nx][ny] = 1

        def find_corner(i, j):
            count = 0
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = dx + i, dy + j
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny]:
                    count += 1
            return count <= 2

        m, n = len(grid), len(grid[0])
        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] and find_corner(i, j):
                    seen = [[1] * n for _ in range(m)]
                    seen[i][j] = 0
                    backtrack(i, j, grid[i][j], seen)
        return ans


# 1220 - Count Vowels Permutation - HARD
class Solution:
    def countVowelPermutation(self, n: int) -> int:
        # only: ae, ea, ei, ia, ie, io, iu, oi, ou, ua
        a, e, i, o, u = 1, 1, 1, 1, 1
        for _ in range(n - 1):
            a, e, i, o, u = e + u + i, a + i, o + e, i, i + o
        return (a + e + i + o + u) % (10**9 + 7)


# 1232 - Check If It Is a Straight Line - EASY
class Solution:
    # use multiplication '*' instead of division '/'
    def checkStraightLine(self, c: List[List[int]]) -> bool:
        x0 = c[0][0]
        y0 = c[0][1]
        x1 = c[1][0]
        y1 = c[1][1]
        for i in range(2, len(c)):
            x = c[i][0]
            y = c[i][1]
            if (y - y0) * (x1 - x0) != (y1 - y0) * (x - x0):
                return False
        return True

    def checkStraightLine(self, c: List[List[int]]) -> bool:
        dx = c[1][0] - c[0][0]
        dy = c[1][1] - c[0][1]
        for i in range(2, len(c)):
            if (c[i][1] - c[i - 1][1]) * dx != (c[i][0] - c[i - 1][0]) * dy:
                return False
        return True

    def checkStraightLine(self, c: List[List[int]]) -> bool:
        return all(
            (c[1][1] - c[0][1]) * (c[i][0] - c[0][0])
            == (c[i][1] - c[0][1]) * (c[1][0] - c[0][0])
            for i in range(2, len(c))
        )


# 1249 - Minimum Remove to Make Valid Parentheses - MEDIUM
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        s = list(s)
        left = 0
        for i in range(len(s)):
            if s[i] == "(":
                left += 1
            elif s[i] == ")":
                if left <= 0:
                    s[i] = ""  # remove extra ")"
                else:
                    left -= 1
        # remove extra "("
        i = len(s) - 1
        while left > 0:
            if s[i] == "(":
                s[i] = ""
                left -= 1
            i -= 1
        return "".join(s)

    def minRemoveToMakeValid(self, s: str) -> str:
        left = []
        s = list(s)
        for i in range(len(s)):
            if s[i] == "(":
                left.append(i)
            elif s[i] == ")":
                if not left:
                    s[i] = ""
                else:
                    left.pop()
        for i in left:
            s[i] = ""
        return "".join(s)

    def minRemoveToMakeValid(self, s: str) -> str:
        arr = list(s)
        stack = []
        for i, ch in enumerate(s):
            if ch == "(":
                stack.append(i)
            elif ch == ")":
                if stack:
                    stack.pop()  # pop the rightmost '('
                else:
                    arr[i] = ""  # remove extra ')', the leftmsot ')'
        while stack:
            arr[stack.pop()] = ""  # remove extra '('
        return "".join(arr)

    def minRemoveToMakeValid(self, s: str) -> str:
        arr = []
        l = 0
        r = s.count(")")
        for ch in s:
            if ch == "(":
                if r > 0:
                    arr.append(ch)
                    l += 1
                    r -= 1
            elif ch == ")":
                if l > 0:
                    arr.append(ch)
                    l -= 1
                else:
                    r -= 1
            else:
                arr.append(ch)
        return "".join(arr)


# 1281 - Subtract the Product and Sum of Digits of an Integer - EASY
class Solution:
    def subtractProductAndSum(self, n: int) -> int:
        s = 0
        p = 1
        while n:
            s += n % 10
            p *= n % 10
            n //= 10
        return p - s


# 1286 - Iterator for Combination - MEDIUM
class CombinationIterator:
    def __init__(self, characters: str, combinationLength: int):
        self.dq = collections.deque(
            itertools.combinations(characters, combinationLength)
        )

    def next(self) -> str:
        return "".join(self.dq.popleft())

    def hasNext(self) -> bool:
        return len(self.dq) > 0


# 1290 - Convert Binary Number in a Linked List to Integer - EASY
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        s = ""
        while head:
            s += str(head.val)
            head = head.next
        return int(s, 2)

    def getDecimalValue(self, head: ListNode) -> int:
        ans = head.val
        while head := head.next:
            ans = (ans << 1) + head.val
        return ans


# 1291 - Sequential Digits - MEDIUM
class Solution:
    def sequentialDigits(self, low: int, high: int) -> List[int]:
        def dfs(now: int):
            if now > high or now % 10 == 9:
                return
            new = now * 10 + now % 10 + 1
            if low <= new <= high:
                ans.append(new)
            dfs(new)
            return

        ans = []
        for i in range(1, 10):
            dfs(i)
        return sorted(ans)

    def sequentialDigits(self, low: int, high: int) -> List[int]:
        ans = []
        dq = collections.deque(range(1, 10))
        while dq:
            cur = dq.popleft()
            if low <= cur <= high:
                ans.append(cur)
            mod = cur % 10
            if mod < 9:
                dq.append(cur * 10 + mod + 1)
        return ans

    def sequentialDigits(self, low: int, high: int) -> List[int]:
        ans = []
        for i in range(1, 9):
            num = nxt = i
            while num <= high and nxt < 10:
                if num >= low:
                    ans.append(num)
                nxt += 1
                num = num * 10 + nxt
        return sorted(ans)
