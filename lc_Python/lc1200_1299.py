import bisect, collections, functools, heapq, itertools, math, string, operator
from typing import List, Optional, Tuple
import sortedcontainers


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


# 1206 - Design Skiplist - HARD
class Skiplist:
    def __init__(self):
        self.d = dict()

    def search(self, target: int) -> bool:
        return True if target in self.d else False

    def add(self, num: int) -> None:
        self.d[num] = self.d.get(num, 0) + 1
        return

    def erase(self, num: int) -> bool:
        if num not in self.d:
            return False
        self.d[num] -= 1
        if self.d[num] == 0:
            del self.d[num]
        return True


class Skiplist:
    def __init__(self):
        self.d = dict()

    def search(self, target: int) -> bool:
        return self.d.get(target, 0) != 0

    def add(self, num: int) -> None:
        self.d[num] = self.d.get(num, 0) + 1
        return

    def erase(self, num: int) -> bool:
        if self.d.get(num, 0) == 0:
            return False
        self.d[num] -= 1
        return True


# 1207 - Unique Number of Occurrences - EASY
class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        cnt = collections.Counter(arr)
        return len(set(cnt)) == len(set(cnt.values()))


# 1209 - Remove All Adjacent Duplicates in String II - MEDIUM
class Solution:
    def removeDuplicates(self, s: str, k: int) -> str:
        st = [("", 1)]
        for c in s:
            if st[-1][0] == c:
                st.append((c, st[-1][1] + 1))
            else:
                st.append((c, 1))
            if st[-1][1] == k:
                for _ in range(k):
                    st.pop()
        return "".join(x for x, _ in st)

    def removeDuplicates(self, s: str, k: int) -> str:
        st = [["", 1]]
        for c in s:
            if st[-1][0] == c:
                st[-1][1] += 1
                if st[-1][1] == k:
                    st.pop()
            else:
                st.append([c, 1])
        return "".join(c * t for c, t in st)


# 1210 - Minimum Moves to Reach Target with Rotations - HARD
class Solution:
    # O(n^2) / O(n^2)
    def minimumMoves(self, grid: List[List[int]]) -> int:
        vis = {(0, 0, 0)}
        q = [(0, 0, 0)]  # 0: horizontal / 1: vertical
        n = len(grid)
        step = 0
        while q:
            new = []
            for x, y, f in q:
                if f:
                    if x + 2 < n and not grid[x + 2][y] and (x + 1, y, f) not in vis:
                        new.append((x + 1, y, f))
                        vis.add((x + 1, y, f))
                    if (
                        x + 1 < n
                        and y + 1 < n
                        and not grid[x][y + 1]
                        and not grid[x + 1][y + 1]
                        and (x, y + 1, f) not in vis
                    ):
                        new.append((x, y + 1, f))
                        vis.add((x, y + 1, f))
                    if (
                        x + 1 < n
                        and y + 1 < n
                        and not grid[x][y + 1]
                        and not grid[x + 1][y + 1]
                        and (x, y, 1 - f) not in vis
                    ):
                        new.append((x, y, 1 - f))
                        vis.add((x, y, 1 - f))
                else:
                    if x == n - 1 and y == n - 2:
                        return step
                    if y + 2 < n and not grid[x][y + 2] and (x, y + 1, f) not in vis:
                        new.append((x, y + 1, f))
                        vis.add((x, y + 1, f))
                    if (
                        x + 1 < n
                        and y + 1 < n
                        and not grid[x + 1][y]
                        and not grid[x + 1][y + 1]
                        and (x + 1, y, f) not in vis
                    ):
                        new.append((x + 1, y, f))
                        vis.add((x + 1, y, f))
                    if (
                        x + 1 < n
                        and y + 1 < n
                        and not grid[x + 1][y]
                        and not grid[x + 1][y + 1]
                        and (x, y, 1 - f) not in vis
                    ):
                        new.append((x, y, 1 - f))
                        vis.add((x, y, 1 - f))
            q = new
            step += 1
        return -1

    def minimumMoves(self, g: List[List[int]]) -> int:
        step = 0
        n = len(g)
        vis = {(0, 0, 0)}
        q = [(0, 0, 0)]  # 0: horizontal / 1: vertical
        while q:
            new = []
            for x, y, f in q:
                for nx, ny, nf in (x + 1, y, f), (x, y + 1, f), (x, y, f ^ 1):
                    a = nx + nf  # head's position
                    b = ny + (nf ^ 1)
                    if (
                        a < n
                        and b < n
                        and not g[nx][ny]
                        and not g[a][b]
                        and (nx, ny, nf) not in vis
                        and (nf == f or g[nx + 1][ny + 1] == 0)
                    ):
                        if nx == n - 1 and ny == n - 2:  # 此时蛇头一定在 (n-1,n-1)
                            return step + 1
                        vis.add((nx, ny, nf))
                        new.append((nx, ny, nf))
            q = new
            step += 1
        return -1


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

    def minCostToMoveChips(self, position: List[int]) -> int:
        cost = [0, 0]
        for chip in position:
            cost[chip & 1] += 1
        return min(cost)

    def minCostToMoveChips(self, position: List[int]) -> int:
        odd = sum(1 for v in position if v & 1)
        even = len(position) - odd
        if odd > even:
            return even
        return odd


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
        a = e = i = o = u = 1
        mod = 10**9 + 7
        for _ in range(n - 1):
            a, e, i, o, u = (
                (e + u + i) % mod,
                (a + i) % mod,
                (o + e) % mod,
                i % mod,
                (i + o) % mod,
            )
        return (a + e + i + o + u) % mod


# 1223 - Dice Roll Simulation - HARD
class Solution:
    def dieSimulator(self, n: int, rollMax: List[int]) -> int:
        mod = 1000000007
        f = [[0] * 6 for _ in range(n)]
        f[0] = [1] * 6
        for i in range(1, n):
            for j, cap in enumerate(rollMax):
                cur = sum(f[i - 1])
                if i > cap:
                    cur -= sum(f[i - cap - 1]) - f[i - cap - 1][j]
                elif i == cap:
                    cur -= 1
                f[i][j] = cur % mod
        return sum(f[n - 1]) % mod

    def dieSimulator(self, n: int, rollMax: List[int]) -> int:
        mod = 1000000007
        f = [[0] * 6 for _ in range(n + 1)]
        summ = [1] + [0] * n
        for i in range(1, n + 1):
            for j, cap in enumerate(rollMax):
                p = max(i - cap - 1, 0)
                sub = (summ[p] - f[p][j]) % mod
                f[i][j] = (summ[i - 1] - sub) % mod
                if i <= cap:
                    f[i][j] = (f[i][j] + 1) % mod
                summ[i] = (summ[i] + f[i][j]) % mod
        return summ[n]


# 1224 - Maximum Equal Frequency - HARD
class Solution:
    def maxEqualFreq(self, nums: List[int]) -> int:
        d = collections.defaultdict(int)
        ans = fmx = t = tmx = 0
        for i, n in enumerate(nums):
            if d[n] == 0:
                t += 1
            d[n] += 1
            if d[n] > fmx:
                fmx = d[n]
                tmx = 1
            elif d[n] == fmx:
                tmx += 1
            if fmx == 1 or fmx * tmx == i or tmx == 1 and (fmx - 1) * t == i:
                ans = i + 1
        return ans


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


# 1233 - Remove Sub-Folders from the Filesystem - MEDIUM
class Solution:
    # O(nmlogn) / O(nl), n = len(folder), m = element_average_length(folder)
    def removeSubfolders(self, folder: List[str]) -> List[str]:
        def build() -> dict:
            folder.sort(reverse=True)  # make '/a/b/c' before to '/a/b'
            trie = {}
            for x in folder:
                r = trie
                for y in x.split("/")[1:]:
                    if y not in r:
                        r[y] = {}
                    r = r[y]
                r.clear()

                # ERROR, It won't clear the dict() it points to, instead it points to a new dict().
                # r = {}

            return trie

        def calc(r: dict, prefix: str) -> None:
            for x in r:
                pwd = prefix + "/" + x
                if len(r[x]) == 0:
                    ans.append(pwd)
                else:
                    calc(r[x], pwd)
            return

        ans = []
        calc(build(), "")
        return ans

    # O(nm) / O(nm)
    def removeSubfolders(self, folder: List[str]) -> List[str]:
        def build() -> dict:
            trie = {}
            for x in folder:
                r = trie
                for y in x.split("/")[1:]:
                    if y not in r:
                        r[y] = {}
                    r = r[y]
                r["#"] = 1
            return trie

        def calc(r: dict, prefix: str) -> None:
            for x in r:
                pwd = prefix + "/" + x
                if "#" in r[x]:
                    ans.append(pwd)
                else:
                    calc(r[x], pwd)
            return

        ans = []
        calc(build(), "")
        return ans

    def removeSubfolders(self, folder: List[str]) -> List[str]:
        def build() -> dict:
            trie = {}
            for i, x in enumerate(folder):
                r = trie
                for y in x.split("/")[1:]:
                    if y not in r:
                        r[y] = {}
                    r = r[y]
                r["#"] = i
            return trie

        def dfs(r: dict) -> None:
            for x in r:
                if "#" in r:
                    ans.append(folder[r["#"]])
                    return
                else:
                    dfs(r[x])
            return

        ans = []
        dfs(build())
        return ans


# 1234 - Replace the Substring for Balanced String - MEDIUM
class Solution:
    # 如果子串之外的任意字符的出现次数都超过 t, 无论怎么换, 都无法使这些字符的出现次数等于 t
    # 如果子串之外的任意字符的出现次数都不超过 t, 才可以通过替换子串, 使得每个字符出现次数为 t
    # O(n) / O(1)
    def balancedString(self, s: str) -> int:
        cnt = collections.Counter(s)
        t = len(s) // 4
        if all(v == t for v in cnt.values()):
            return 0
        l = 0
        ans = 1e5
        for r, c in enumerate(s):
            cnt[c] -= 1
            while all(v <= t for v in cnt.values()):
                ans = min(ans, r - l + 1)
                cnt[s[l]] += 1
                l += 1
        return ans


# 1237 - Find Positive Integer Solution for a Given Equation - MEDIUM
# print(inspect.getsource(customfunction.__class__))
class CustomFunction:
    def __init__(self, x):
        self.num = x

    def f(self, x, y):
        if self.num == 1:
            return x + y
        elif self.num == 2:
            return x * y
        elif self.num == 3:
            return x * x + y
        elif self.num == 4:
            return x + y * y
        elif self.num == 5:
            return x * x + y * y
        elif self.num == 6:
            return (x + y) * (x + y)
        elif self.num == 7:
            return x * x * x + y * y * y
        elif self.num == 8:
            return x * x * y
        elif self.num == 9:
            return x * y * y
        else:
            return 0


class Solution:
    def findSolution(self, customfunction: "CustomFunction", z: int) -> List[List[int]]:
        ans = []
        x = 1
        y = 1000
        while x <= 1000 and y:
            r = customfunction.f(x, y)
            if r < z:
                x += 1
            elif r > z:
                y -= 1
            else:
                ans.append([x, y])
                x += 1
                y -= 1
        return ans

    def findSolution(self, customfunction: "CustomFunction", z: int) -> List[List[int]]:
        ans = []
        y = 1000
        for x in range(1, 1001):
            while y and customfunction.f(x, y) > z:
                y -= 1
            if y == 0:
                break
            if customfunction.f(x, y) == z:
                ans.append((x, y))
        return ans


# 1238 - Circular Permutation in Binary Representation - MEDIUM
class Solution:
    def circularPermutation(self, n: int, start: int) -> List[int]:
        grayCode = lambda x: x ^ (x >> 1)
        p = [grayCode(i) for i in range(1 << n)]
        i = p.index(start)
        return p[i:] + p[:i]

    def circularPermutation(self, n: int, start: int) -> List[int]:
        return (
            [start, start ^ 1]
            if n == 1
            else self.circularPermutation(n - 1, start)
            + self.circularPermutation(n - 1, start ^ 1 << (n - 1))[::-1]
        )

    def circularPermutation(self, n: int, start: int) -> List[int]:
        return [i ^ (i >> 1) ^ start for i in range(1 << n)]


# 1247 - Minimum Swaps to Make Strings Equal - MEDIUM
class Solution:
    # skip if s1[i] == s2[i]
    # if xx, yy -> 1
    # if xy, yx -> 2
    # consider the number of (s1[i] == 'x' and s2[i] == 'y'), (s1[i] == 'y' and s2[i] == 'x')
    # O(n) / O(1)
    def minimumSwap(self, s1: str, s2: str) -> int:
        # if (s1.count("x") + s2.count("x")) % 2:
        #     return -1

        xy = yx = 0
        for a, b in zip(s1, s2):
            if a == "x" and b == "y":
                xy += 1
            if a == "y" and b == "x":
                yx += 1

            # xy += a == "x" and b == "y"
            # yx += a == "y" and b == "x"

            # xy += a < b
            # yx += a > b

        if (xy + yx) % 2:
            return -1

        # xy % 2 equals to yx % 2
        # may be one pair and at most one pair of 'xy', 'yx'

        return xy // 2 + yx // 2 + (xy % 2) * 2

    def minimumSwap(self, s1: str, s2: str) -> int:
        cnt = collections.Counter(a for a, b in zip(s1, s2) if a != b)
        d = cnt["x"] + cnt["y"]
        return -1 if d % 2 else d // 2 + cnt["x"] % 2


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


# 1250 - Check If It Is a Good Array - HARD
class Solution:
    # 裴蜀定理
    def isGoodArray(self, nums: List[int]) -> bool:
        return functools.reduce(math.gcd, nums) == 1

    # >= 3.9, 支持使用 *args 解包多个数求最大公约数的情况
    def isGoodArray(self, nums: List[int]) -> bool:
        return math.gcd(*nums) == 1


# 1252 - Cells with Odd Values in a Matrix - EASY
class Solution:
    def oddCells(self, m: int, n: int, indices: List[List[int]]) -> int:
        f = [[0] * n for _ in range(m)]
        for r, c in indices:
            for i in range(n):
                f[r][i] += 1
            for i in range(m):
                f[i][c] += 1
        return sum(v & 1 for r in f for v in r)

    def oddCells(self, m: int, n: int, indices: List[List[int]]) -> int:
        rows = [0] * m
        cols = [0] * n
        for r, c in indices:
            rows[r] += 1
            cols[c] += 1
        return sum((r + c) % 2 for r in rows for c in cols)


# 1255 - Maximum Score Words Formed by Letters - HARD
class Solution:
    def maxScoreWords(
        self, words: List[str], letters: List[str], score: List[int]
    ) -> int:
        cnt = collections.Counter(letters)
        n = len(words)
        sc = [0] * n
        wcnt = [None] * n
        for i, w in enumerate(words):
            wcnt[i] = collections.Counter(w)
            for c in w:
                sc[i] += score[ord(c) - 97]

        def dfs(i: int, cnt: collections.Counter) -> int:
            if i == n:
                return 0
            if not wcnt[i] - cnt:
                # if all(wcnt[i][k] <= cnt[k] for k in wcnt[i]):
                return max(dfs(i + 1, cnt), dfs(i + 1, cnt - wcnt[i]) + sc[i])
            return dfs(i + 1, cnt)

        return dfs(0, cnt)

    def maxScoreWords(
        self, words: List[str], letters: List[str], score: List[int]
    ) -> int:
        n = len(words)
        words = list(collections.Counter(w) for w in words)
        letters = collections.Counter(letters)

        @functools.lru_cache(None)
        def dfs(state: int) -> int:
            d = {}
            for i in range(n):
                if state >> i & 1 == 1:
                    for k, v in words[i].items():
                        d[k] = d.get(k, 0) + v
                        if d[k] > letters[k]:
                            return 0
            cur = sum(score[ord(k) - 97] * d[k] for k in d)
            for i in range(n):
                if state >> i & 1 == 0:
                    cur = max(cur, dfs(state | 1 << i))
            return cur

        return dfs(0)

    def maxScoreWords(
        self, words: List[str], letters: List[str], score: List[int]
    ) -> int:
        letters = collections.Counter(letters)
        n = len(words)
        ans = 0
        # words = list(collections.Counter(w) for w in words)
        for state in range(1 << n):
            # cnt = collections.Counter()
            # for i in range(n):
            #     if state >> i & 1:
            #         cnt += words[i]

            cnt = collections.Counter(
                "".join(words[i] for i in range(n) if state >> i & 1)
            )

            if all(v <= letters[c] for c, v in cnt.items()):
                ans = max(ans, sum(v * score[ord(k) - 97] for k, v in cnt.items()))
        return ans


# 1260 - Shift 2D Grid - EASY
class Solution:
    def shiftGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
        arr = [v for r in grid for v in r]
        k %= len(arr)
        arr = arr[-k:] + arr[:-k]
        return [arr[i : i + len(grid[0])] for i in range(0, len(arr), len(grid[0]))]


# 1266 - Minimum Time Visiting All Points - EASY
class Solution:
    def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
        ans = 0
        px, py = points[0]
        for x, y in points[1:]:
            ans += max(abs(x - px), abs(y - py))
            px, py = x, y
        return ans


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


# 1282 - Group the People Given the Group Size They Belong To - MEDIUM
class Solution:
    def groupThePeople(self, groupSizes: List[int]) -> List[List[int]]:
        g = collections.defaultdict(list)
        for i, v in enumerate(groupSizes):
            g[v].append(i)
        ans = []
        for k, v in g.items():
            ans.extend((v[i : i + k] for i in range(0, len(v), k)))
            # for i in range(0, len(v), k):
            #     ans.append(v[i : i + k])
        return ans


# 1283 - Find the Smallest Divisor Given a Threshold - MEDIUM
class Solution:
    def smallestDivisor(self, nums: List[int], threshold: int) -> int:
        l = 1
        r = max(nums)
        while l < r:
            m = l + r >> 1
            if sum((v + m - 1) // m for v in nums) <= threshold:
                r = m
            else:
                l = m + 1
        return l

    def smallestDivisor(self, nums: List[int], threshold: int) -> int:
        return 1 + bisect.bisect_left(
            range(1, max(nums) + 1),
            x=True,
            key=lambda m: sum((v + m - 1) // m for v in nums) <= threshold,
        )


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


# 1287 - Element Appearing More Than 25% In Sorted Array - EASY
class Solution:
    def findSpecialInteger(self, arr: List[int]) -> int:
        t = 0
        most = arr[0]
        for v in arr:
            if v == most:
                t += 1
            else:
                t = 1
            most = v
            if t * 4 > len(arr):
                return most
        return -1

    def findSpecialInteger(self, arr: List[int]) -> int:
        cnt = len(arr) // 4
        for i in range(len(arr) - cnt):
            if arr[i] == arr[i + cnt]:
                return arr[i]
        return -1


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


# 1295 - Find Numbers with Even Number of Digits - EASY
class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        return sum(len(str(v)) % 2 == 0 for v in nums)


# 1299 - Replace Elements with Greatest Element on Right Side - EASY
class Solution:
    def replaceElements(self, arr: List[int]) -> List[int]:
        ans = [0] * (len(arr) - 1) + [-1]
        for i in range(len(arr) - 2, -1, -1):
            ans[i] = max(ans[i + 1], arr[i + 1])
        return ans
