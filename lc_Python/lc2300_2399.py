import bisect, collections, functools, heapq, itertools, math, operator, string
from typing import List, Optional, Tuple
import sortedcontainers


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 2300 - Successful Pairs of Spells and Potions - MEDIUM
class Solution:
    def successfulPairs(self, sp: List[int], p: List[int], success: int) -> List[int]:
        p = sorted(p)
        ans = []
        n = len(p)
        for s in sp:
            l = 0
            r = n
            while l < r:
                m = (l + r) // 2
                if s * p[m] < success:
                    l = m + 1
                else:
                    r = m
            ans.append(n - l)
        return ans

    def successfulPairs(self, sp: List[int], p: List[int], success: int) -> List[int]:
        p = sorted(p)
        ans = []
        for s in sp:
            t = (success + s - 1) // s
            i = bisect.bisect_left(p, t)
            ans.append(len(p) - i)
        return ans

    def successfulPairs(self, sp: List[int], p: List[int], success: int) -> List[int]:
        p = sorted(p)
        ans = []
        for s in sp:
            i = bisect.bisect_right(p, success / s - 1e-7)
            ans.append(len(p) - i)
        return ans


# 2301 - Match Substring After Replacement - HARD
class Solution:
    # O(S * T) / O(M)
    def matchReplacement(self, s: str, sub: str, mappings: List[List[str]]) -> bool:
        m = set(map(lambda x: x[0] + x[1], mappings))
        for i in range(len(s) - len(sub) + 1):
            ok = True
            for j in range(len(sub)):
                if s[i + j] != sub[j] and sub[j] + s[i + j] not in m:
                    ok = False
                    break
            if ok:
                return True
        return False

    def matchReplacement(self, s: str, sub: str, mappings: List[List[str]]) -> bool:
        m = {c: set([c]) for c in set(s)}
        for old, new in mappings:
            if new in m:
                m[new].add(old)
        for i in range(len(s) - len(sub) + 1):
            ok = True
            for j in range(len(sub)):
                if not sub[j] in m[s[i + j]]:
                    ok = False
                    break
            if ok:
                return True
        return False

    def matchReplacement(self, s: str, sub: str, mappings: List[List[str]]) -> bool:
        m = collections.defaultdict(set)
        for old, new in mappings:
            m[old].add(new)
        for i in range(len(s) - len(sub) + 1):
            ok = True
            for j in range(len(sub)):
                if sub[j] != s[i + j] and s[i + j] not in m[sub[j]]:
                    ok = False
                    break
            if ok:
                return True
        return False


# 2302 - Count Subarrays With Score Less Than K - HARD
class Solution:
    # O(n) / O(1), sliding window, two pointer
    # HOW MANY SUBARRAY in A[i:j]?
    def countSubarrays(self, nums: List[int], k: int) -> int:
        ans = cur = r = 0
        n = len(nums)
        for l in range(n):
            while r < n and (cur + nums[r]) * (r - l + 1) < k:
                cur += nums[r]
                r += 1
            ans += r - l
            cur -= nums[l]
        return ans

    def countSubarrays(self, nums: List[int], k: int) -> int:
        ans = summ = l = 0
        for r, v in enumerate(nums):
            summ += v
            while summ * (r - l + 1) >= k:
                summ -= nums[l]
                l += 1
            ans += r - l + 1
        return ans

    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + nums[i]
        l = ans = 0
        for r in range(n):
            while (pre[r + 1] - pre[l]) * (r - l + 1) >= k:
                l += 1
            ans += r - l + 1
        return ans

    def countSubarrays(self, nums: List[int], k: int) -> int:
        ans = pre = length = 0
        for i, num in enumerate(nums):
            length += 1
            pre += num
            while length * pre >= k:
                pre -= nums[i - length + 1]
                length -= 1
            ans += length
        return ans

    # O(n ^ 2) / O(n), pre sum, bisect
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        pre = [0] * (n + 1)
        for i in range(n):
            pre[i + 1] = pre[i] + nums[i]
        for i in range(n):
            l = i
            r = n
            while l < r:
                m = (l + r) >> 1
                if (pre[m + 1] - pre[i]) * (m + 1 - i) < k:
                    l = m + 1
                else:
                    r = m
            ans += l - i
        return ans


# 2303 - Calculate Amount Paid in Taxes - EASY
class Solution:
    def calculateTax(self, brackets: List[List[int]], income: int) -> float:
        ans = pre = 0
        for u, p in brackets:
            if income >= u:
                ans += (u - pre) * p / 100
            else:
                ans += (income - pre) * p / 100
                break
            pre = u
        return ans


# 2304 - Minimum Path Cost in a Grid - MEDIUM
class Solution:
    # O(m * n * n) / O(n), dp
    def minPathCost(self, grid: List[List[int]], moveCost: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        arr = [g for _, g in enumerate(grid[0])]
        mp = {i: v for i, v in enumerate(moveCost)}
        for i in range(m - 1):  #  O(m)
            nxt = [math.inf for _ in range(n)]
            for j in range(n):  #  O(n)
                for jj, cost in enumerate(mp[grid[i][j]]):  #  O(n)
                    if arr[j] + cost < nxt[jj]:
                        nxt[jj] = arr[j] + cost
            for j in range(n):
                nxt[j] += grid[i + 1][j]
            arr = nxt
        return min(arr)

    # O(m * n * n) / O(n * n)
    def minPathCost(self, grid: List[List[int]], moveCost: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(n):
            dp[0][i] = grid[0][i]
        for r in range(1, m):
            for c in range(n):
                dp[r][c] = grid[r][c] + min(
                    dp[r - 1][i] + moveCost[grid[r - 1][i]][c] for i in range(n)
                )
        return min(dp[m - 1])

    def minPathCost(self, grid: List[List[int]], moveCost: List[List[int]]) -> int:
        m = len(grid)
        f = grid[0]
        for i in range(1, m):
            f = [
                g + min(f[j] + moveCost[v][jj] for j, v in enumerate(grid[i - 1]))
                for jj, g in enumerate(grid[i])
            ]
        return min(f)

    def minPathCost(self, grid: List[List[int]], moveCost: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])

        @functools.lru_cache(None)
        def fn(i: int, j: int) -> int:
            if i == m - 1:
                return grid[i][j]
            ans = math.inf
            for jj in range(n):
                ans = min(ans, grid[i][j] + fn(i + 1, jj) + moveCost[grid[i][j]][jj])
            return ans

        return min(fn(0, j) for j in range(n))


# 2305 - Fair Distribution of Cookies - MEDIUM
class Solution:
    # O(k ^ n) / , backtrack + prune
    def distributeCookies(self, cookies: List[int], k: int) -> int:
        def dfs(x: int):
            if x == len(cookies):
                self.ans = min(self.ans, max(arr))
                return
            for i in range(k):
                arr[i] += cookies[x]
                if arr[i] < self.ans:  # if not, TLE
                    dfs(x + 1)
                arr[i] -= cookies[x]
            return

        arr = [0] * k
        self.ans = 1e9
        dfs(0)
        return self.ans

    def distributeCookies(self, cookies: List[int], k: int) -> int:
        def dfs(i: int):
            nonlocal ans
            if i == len(cookies):
                ans = min(ans, max(arr))
                return
            if ans <= max(arr):  # not helpful to optimize 'ans'
                return
            for j in range(k):
                arr[j] += cookies[i]
                dfs(i + 1)
                arr[j] -= cookies[i]
            return

        arr = [0] * k
        ans = 1e9
        dfs(0)
        return ans

    # no pruning
    def distributeCookies(self, cookies: List[int], k: int) -> int:
        def dfs(i):
            if i == len(cookies):
                v = max(arr)
                # ans[0] = min(ans[0], v) -> TLE, min(), max() are bit slower than 'if'
                if ans[0] == -1 or ans[0] > v:
                    ans[0] = v
                return
            for j in range(k):
                arr[j] += cookies[i]
                dfs(i + 1)
                arr[j] -= cookies[i]

        arr = [0] * k
        ans = [-1]
        dfs(0)
        return ans[0]

    # O(k * 3 ^ n) / O(2 ^ n), state compression dp
    # f[i][j]: The minimum value of inequity of the in total 'i' subarray
    #          when the in total 'i' subarray are assigned the set of cookies 'j'.
    def distributeCookies(self, cookies: List[int], k: int) -> int:
        n = len(cookies)
        summ = [0] * (1 << n)
        for i in range(1, 1 << n):
            for j in range(n):
                if i >> j & 1:
                    summ[i] += cookies[j]

        f = [[0] * (1 << n) for _ in range(k)]
        f[0] = summ
        for i in range(1, k):
            for j in range(1 << n):
                s = j
                f[i][j] = 1e9
                while s:
                    # 'i' always comes from 'i-1' -> Scrolling array, save space
                    # 'j' always comes from a number smaller than 'j' -> 01 knapsack -> reverse enumeration
                    f[i][j] = min(f[i][j], max(f[i - 1][j ^ s], summ[s]))
                    s = (s - 1) & j
        return f[k - 1][(1 << n) - 1]

    # optimize the first dimension of 'f'
    def distributeCookies(self, cookies: List[int], k: int) -> int:
        n = len(cookies)
        summ = [0] * (1 << n)
        for i, v in enumerate(cookies):
            bit = 1 << i
            for j in range(bit):
                summ[bit | j] = summ[j] + v

        f = summ.copy()
        for _ in range(1, k):
            for j in range((1 << n) - 1, 0, -1):
                s = j
                while s:
                    f[j] = min(f[j], max(f[j ^ s], summ[s]))
                    s = (s - 1) & j
        return f[-1]

    def distributeCookies(self, cookies: List[int], k: int) -> int:
        @functools.lru_cache(None)
        def fn(mask: int, k: int) -> int:
            if mask == 0:
                return 0
            if k == 0:
                return 1e9
            ans = 1e9
            orig = mask
            while mask:
                mask = orig & (mask - 1)  # choose another subset
                s = sum(cookies[i] for i in range(n) if (orig ^ mask) & 1 << i)
                ans = min(ans, max(s, fn(mask, k - 1)))
            return ans

        n = len(cookies)
        return fn((1 << n) - 1, k)


# 2306 - Naming a Company - HARD
class Solution:
    def distinctNames(self, ideas: List[str]) -> int:
        group = collections.defaultdict(int)
        for s in ideas:
            group[s[1:]] |= 1 << (ord(s[0]) - ord("a"))
        ans = 0
        cnt = [[0] * 26 for _ in range(26)]
        for mask in group.values():
            for i in range(26):
                if mask >> i & 1 == 0:
                    for j in range(26):
                        if mask >> j & 1:
                            cnt[i][j] += 1
                else:
                    for j in range(26):
                        if mask >> j & 1 == 0:
                            ans += cnt[i][j]
        return ans * 2

    def distinctNames(self, ideas: List[str]) -> int:
        group = collections.defaultdict(set)
        for s in ideas:
            group[s[0]].add(s[1:])
        ans = 0
        for a, b in itertools.combinations(group.values(), 2):
            m = len(a & b)
            ans += (len(a) - m) * (len(b) - m)
        return ans * 2

    def distinctNames(self, ideas: List[str]) -> int:
        group = collections.defaultdict(set)
        for s in ideas:
            group[s[0]].add(s[1:])
        ans = 0
        for a, seta in group.items():
            for b, setb in group.items():
                if a >= b:
                    continue
                same = len(seta & setb)
                ans += (len(seta) - same) * (len(setb) - same)
        return ans * 2

    def distinctNames(self, ideas: List[str]) -> int:
        ss = [set() for _ in range(26)]
        for x in ideas:
            ss[ord(x[0]) - ord("a")].add(x[1:])
        ans = 0
        for i in range(26):
            for j in range(i + 1, 26):
                # setA - setB = setA.difference(setB)
                ans += len(ss[i] - ss[j]) * len(ss[j] - ss[i])
        return ans * 2

    def distinctNames(self, ideas: List[str]) -> int:
        ideas = set(ideas)
        d = collections.Counter()
        for s in ideas:
            for c in string.ascii_lowercase:
                if c + s[1:] not in ideas:
                    d[c, s[0]] += 1
        ans = 0
        for s in ideas:
            cnt = 0
            for c in string.ascii_lowercase:
                if c + s[1:] not in ideas:
                    cnt += d[s[0], c]
            ans += cnt
        return ans


# 2309. Greatest English Letter in Upper and Lower Case - EASY
class Solution:
    # O(n) / O(n)
    def greatestLetter(self, s: str) -> str:
        s = set(s)
        for i in range(26, -1, -1):
            if chr(ord("a") + i) in s and chr(ord("A") + i) in s:
                return chr(ord("A") + i)
        return ""

    # O(n) / O(1)
    def greatestLetter(self, s: str) -> str:
        mask: int = 0
        for c in s:
            mask |= 1 << (ord(c) - ord("A"))
        mask &= mask >> (ord("a") - ord("A"))
        if mask == 0:
            return ""
        l = mask.bit_length()
        return chr(ord("A") + l - 1)


# 2310 - Sum of Numbers With Units Digit K - MEDIUM
class Solution:
    # O(1) / O(1)
    def minimumNumbers(self, num: int, k: int) -> int:
        if num == 0:
            return 0
        for x in range(1, 11):
            if x * k % 10 == num % 10 and x * k <= num:
                return x
        return -1

    def minimumNumbers(self, num: int, k: int) -> int:
        if num == 0:
            return 0
        for t in range(1, 11):
            if t * k % 10 == num % 10:
                if t * k <= num:
                    return t
                else:
                    return -1
        return -1


# 2311 - Longest Binary Subsequence Less Than or Equal to K - MEDIUM
class Solution:
    # O(n) / O(1)
    def longestSubsequence(self, s: str, k: int) -> int:
        z = 0
        for i, c in enumerate(s):
            if int(s[i:], 2) <= k:
                return z + len(s) - i
            if c == "0":
                z += 1
        return -1

    def longestSubsequence(self, s: str, k: int) -> int:
        bits = ""
        for i in range(len(s) - 1, -1, -1):
            bits = s[i] + bits
            if int(bits, 2) > k:
                return s[:i].count("0") + len(bits) - 1
        return len(s)

    def longestSubsequence(self, s: str, k: int) -> int:
        n = len(s)
        m = k.bit_length()
        if n < m:
            return n
        can = m - 1
        if int(s[n - m :], 2) <= k:
            can = m
        return can + s.count("0", 0, n - m)

    def longestSubsequence(self, s: str, k: int) -> int:
        ans = s.count("0")
        x = 0
        p = 1
        for c in s[::-1]:
            x += int(c) * p
            p *= 2
            if x > k:
                break
            if c == "1":
                ans += 1
        return ans

    # O(n ^ 2) / O(n), dp[i] means the minimum value of subsequence with length i
    def longestSubsequence(self, s: str, k: int) -> int:
        dp = [0]
        for v in map(int, s):
            if dp[-1] * 2 + v <= k:
                dp.append(dp[-1] * 2 + v)
            for i in range(len(dp) - 1, 0, -1):
                dp[i] = min(dp[i], dp[i - 1] * 2 + v)
        return len(dp) - 1


# 2312 - Selling Pieces of Wood - HARD
class Solution:
    # O(m * n * (m + n)) / O(m * n)
    def sellingWood(self, m: int, n: int, prices: List[List[int]]) -> int:
        d = collections.defaultdict(dict)
        for h, w, p in prices:
            d[h][w] = p

        @functools.lru_cache(None)
        def fn(h: int, w: int) -> int:
            p = d[h].get(w, 0)
            for i in range(1, h // 2 + 1):
                p = max(p, fn(i, w) + fn(h - i, w))
            for i in range(1, w // 2 + 1):
                p = max(p, fn(h, i) + fn(h, w - i))
            return p

        return fn(m, n)

    # f[i][j], the maximum amount that can be earned by cutting a block of size (i, j)
    def sellingWood(self, m: int, n: int, prices: List[List[int]]) -> int:
        p = [[0] * (n + 1) for _ in range(m + 1)]
        for h, w, pr in prices:
            p[h][w] = pr
        f = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                f[i][j] = p[i][j]
                for k in range(1, i // 2 + 1):
                    f[i][j] = max(f[i][j], f[i - k][j] + f[k][j])
                for k in range(1, j // 2 + 1):
                    f[i][j] = max(f[i][j], f[i][j - k] + f[i][k])
        return f[m][n]


# 2315 - Count Asterisks - EASY
class Solution:
    def countAsterisks(self, s: str) -> int:
        f = 2
        ans = 0
        for c in s:
            if c == "|":
                f = 3 - f
            elif c == "*" and f == 2:
                ans += 1
        return ans

    def countAsterisks(self, s: str) -> int:
        f = 0
        ans = 0
        for c in s:
            if c == "|":
                f += 1
            elif c == "*" and not f & 1:
                ans += 1
        return ans

    def countAsterisks(self, s: str) -> int:
        return sum(t.count("*") for t in s.split("|")[::2])


# 2316 - Count Unreachable Pairs of Nodes in an Undirected Graph - MEDIUM
class Solution:
    # BFS
    def countPairs(self, n: int, edges: List[List[int]]) -> int:
        ans = n * (n - 1) // 2
        vis = set()
        g = collections.defaultdict(set)
        for a, b in edges:
            g[a].add(b)
            g[b].add(a)

        def bfs(start: int) -> int:
            cnt = 1
            dq = collections.deque([start])
            vis.add(start)
            while dq:
                n = dq.popleft()
                for x in g[n]:
                    if x not in vis:
                        vis.add(x)
                        cnt += 1
                        dq.append(x)
            return cnt

        for i in range(n):
            if i not in vis:
                group = bfs(i)
                ans -= group * (group - 1) // 2
        return ans

    # DFS
    def countPairs(self, n: int, edges: List[List[int]]) -> int:
        vis = [False] * n
        g = [[] for _ in range(n)]
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)

        def dfs(i: int):
            nonlocal group
            vis[i] = True
            group += 1
            for j in g[i]:
                if not vis[j]:
                    dfs(j)
            return

        ans = pre = group = 0
        for i in range(n):
            if not vis[i]:
                group = 0
                dfs(i)
                ans += group * pre
                pre += group
        return ans

    # DSU, disjoint set union
    def countPairs(self, n: int, edges: List[List[int]]) -> int:
        dsu = [i for i in range(n)]
        size = [1] * n

        def find(x: int) -> int:
            if dsu[x] != x:
                dsu[x] = find(dsu[x])
            return dsu[x]

        for x, y in edges:
            a = find(dsu[x])
            b = find(dsu[y])
            if a != b:
                size[b] += size[a]
                dsu[a] = b
        ans = n * (n - 1) // 2
        for i in range(n):
            if dsu[i] == i:
                ans -= size[i] * (size[i] - 1) // 2
        return ans


# 2317 - Maximum XOR After Operations - MEDIUM
class Solution:
    def maximumXOR(self, nums: List[int]) -> int:
        ans = 0
        for n in nums:
            ans = ans | n
        return ans

    def maximumXOR(self, nums: List[int]) -> int:
        return functools.reduce(operator.or_, nums)


# 2318 - Number of Distinct Roll Sequences - HARD
@functools.lru_cache(None)
def dfs(n: int, l1: int, l2: int) -> int:
    """
    leetcode will new many Solution() objects, if write dfs() in the obj, the new func won't use the previous lru_cache()
    """

    if n == 0:
        return 1
    ans = 0
    for cur in range(1, 7):
        if cur != l1 and cur != l2 and math.gcd(cur, l1) == 1:
            ans += dfs(n - 1, cur, l1)
    return ans % (10**9 + 7)


class Solution:
    # O(n * 6 ^ 3) / O(n * 6 ^ 2)
    def distinctSequences(self, n: int) -> int:
        return dfs(n, 7, 7)  # greatest common divisor of 7 and each number is 1


class Solution:
    def distinctSequences(self, n: int) -> int:
        def gcd(a: int, b: int):
            return gcd(b, a % b) if b else a

        if n == 1:
            return 6
        f = [[[0] * 6 for _ in range(6)] for _ in range(n + 1)]
        mod = 10**9 + 7
        for i in range(6):
            for j in range(6):
                if i != j and gcd(i + 1, j + 1) == 1:
                    f[2][i][j] = 1
        for i in range(3, n + 1):
            for j in range(6):
                for k in range(6):
                    if j != k and gcd(j + 1, k + 1) == 1:
                        for kk in range(6):
                            if kk != j:
                                f[i][j][k] = (f[i][j][k] + f[i - 1][k][kk]) % mod
        ans = 0
        for i in range(6):
            for j in range(6):
                ans = (ans + f[n][i][j]) % mod
        return ans


# 2319 - Check if Matrix Is X-Matrix - EASY
class Solution:
    def checkXMatrix(self, grid: List[List[int]]) -> bool:
        n = len(grid)
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if i == j or i + j == n - 1:
                    if v == 0:
                        return False
                elif v != 0:
                    return False
        return True


# 2320 - Count Number of Ways to Place Houses - MEDIUM
class Solution:
    def countHousePlacements(self, n: int) -> int:
        ans = 0
        f = [[0] * 4 for _ in range(n)]
        f[0][0] = 1  # no left or right
        f[0][1] = 1  # left
        f[0][2] = 1  # right
        f[0][3] = 1  # left and right
        mod = 10**9 + 7
        for i in range(1, n):
            f[i][0] = (f[i - 1][1] + f[i - 1][2] + f[i - 1][3] + f[i - 1][0]) % mod
            f[i][1] = (f[i - 1][0] + f[i - 1][2]) % mod
            f[i][2] = (f[i - 1][0] + f[i - 1][1]) % mod
            f[i][3] = (f[i - 1][0]) % mod
        ans = sum(f[-1])
        return ans % mod

    def countHousePlacements(self, n: int) -> int:
        mod = 10**9 + 7
        f = [1, 2]
        for _ in range(n):
            f.append((f[-1] + f[-2]) % mod)
        return f[n] ** 2 % mod


# 2321 - Maximum Score Of Spliced Array - HARD
class Solution:
    def maximumsSplicedArray(self, nums1: List[int], nums2: List[int]) -> int:
        def maxSubArray(diff: List[int]):
            cur = 0
            r = 0
            for d in diff:
                cur += d
                if cur < 0:
                    cur = 0
                r = max(r, cur)
            return r

        def maxSubArray2(diff: List[int]) -> int:
            pre = 0
            r = diff[0]
            for d in diff:
                pre = max(pre + d, d)
                r = max(r, pre)
            return r

        diff1 = []
        diff2 = []
        for n1, n2 in zip(nums1, nums2):
            diff1.append(n1 - n2)
            diff2.append(n2 - n1)
        ans = max(sum(nums1) + maxSubArray(diff2), sum(nums2) + maxSubArray(diff1))
        return ans


# 2322 - Minimum Score After Removals on a Tree - HARD
class Solution:
    # O(n^2) / O(n)

    # 固定树根, 枚举删除的边, 两个边的关系有两种情况
    # 1. 互为祖先 / 2. 并列不相交
    def minimumScore(self, nums: List[int], edges: List[List[int]]) -> int:
        n = len(nums)
        g = [[] for _ in range(n)]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)

        xor = [0] * n
        inn = [0] * n
        out = [0] * n
        t = 0  # timestamp

        def dfs(x: int, fa: int) -> None:
            nonlocal t
            t += 1
            inn[x] = t
            xor[x] = nums[x]
            for y in g[x]:
                if y != fa:
                    dfs(y, x)
                    xor[x] ^= xor[y]
            out[x] = t

        dfs(0, -1)

        ans = 10**9
        for i in range(2, n):
            for j in range(1, i):
                if inn[i] < inn[j] <= out[i]:  # i is an ancestor of j
                    a = xor[j]
                    b = xor[i] ^ xor[j]
                    c = xor[0] ^ xor[i]
                elif inn[j] < inn[i] <= out[j]:  # j is an ancestor of i
                    a = xor[i]
                    b = xor[i] ^ xor[j]
                    c = xor[0] ^ xor[j]
                else:  # i and j do not overlap
                    a = xor[i]
                    b = xor[j]
                    c = xor[0] ^ xor[i] ^ xor[j]

                ans = min(ans, max(a, b, c) - min(a, b, c))
                # speed up from min/max

                # mn = mx = a
                # if b < mn:
                #     mn = b
                # elif b > mx:
                #     mx = b
                # if c < mn:
                #     mn = c
                # elif c > mx:
                #     mx = c
                # if mx - mn < ans:
                #     ans = mx - mn

                # if a > b:
                #     a, b = b, a
                # if a > c:
                #     a, c = c, a
                # if b > c:
                #     b, c = c, b
                # ans = min(ans, c - a)

                if ans == 0:
                    return 0
        return ans

    # 枚举树根, 删除树根与某子节点的连边, 树 -> A + B, 不包含树根的为连通块A, 包含的为B
    # B中枚举删除第二条边, B -> C + D, 包含树根的为连通块C, 不包含的为D, D为整棵树的一棵子树
    # 求块A, C, D的异或和, T为整棵树异或和
    # A = T ^ B, C = D ^ B
    def minimumScore(self, nums: List[int], edges: List[List[int]]) -> int:
        ans = 10**9
        g = collections.defaultdict(list)
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)
        t = 0
        for v in nums:
            t ^= v

        def dfs(x: int, fa: int, ban: int) -> int:
            """以 x 为根, 不含 ban 的子树的异或和"""
            r = nums[x]
            for y in g[x]:
                if y != fa and y != ban:
                    r ^= dfs(y, x, ban)
            return r

        def dfs2(x: int, fa: int, ban: int) -> int:
            """以 x 为根，不含 ban 的子树中枚举第二条被删除的边"""
            nonlocal ans
            r = nums[x]
            # 枚举连通块 D 的根 y
            for y in g[x]:
                if y != fa and y != ban:
                    # 连通块 D 的异或和
                    a = dfs2(y, x, ban)
                    # 连通块 C 的异或和
                    b = xorB ^ a
                    # 连通块 A 的异或和
                    c = t ^ xorB
                    ans = min(ans, max(a, b, c) - min(a, b, c))
                    r ^= a
            return r

        for i in range(len(nums)):
            for j in g[i]:
                # 计算B, 即以 i 为根, 不含 j 的子树的异或和
                xorB = dfs(i, -1, j)
                # 删除第二条边
                dfs2(i, -1, j)
        return ans

    # 枚举根节点, 对每个根节点, 求一次每个子树的异或和
    # 从根节点删除一个边, 枚举子树中删除一条边
    def minimumScore(self, nums: List[int], edges: List[List[int]]) -> int:
        ans = 10**9
        g = collections.defaultdict(list)
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)
        t = 0
        for v in nums:
            t ^= v
        xor = [0] * len(nums)

        def dfs(x: int, fa: int) -> None:
            xor[x] = nums[x]
            for y in g[x]:
                if y != fa:
                    dfs(y, x)
                    xor[x] ^= xor[y]
            return

        def calc(x: int, fa: int, summ: int) -> None:
            nonlocal ans
            for y in g[x]:
                if y != fa:
                    a = t ^ summ
                    b = summ ^ xor[y]
                    c = xor[y]
                    ans = min(ans, max(a, b, c) - min(a, b, c))
                    calc(y, x, summ)
            return

        for i in range(len(nums)):
            dfs(i, -1)
            for j in g[i]:
                calc(j, i, xor[j])
        return ans


# 2325 - Decode the Message - EASY
class Solution:
    def decodeMessage(self, key: str, message: str) -> str:
        d = dict()
        i = 0
        for c in key:
            if c != " " and c not in d:
                d[c] = chr(ord("a") + i)
                i += 1
            if i == 26:
                break
        ans = ""
        for c in message:
            if c == " ":
                ans += c
            else:
                ans += d[c]
        return ans

    def decodeMessage(self, key: str, message: str) -> str:
        # d = {" ": " "}
        d = dict()
        i = 0
        for c in key:
            if c != " " and c not in d:
                d[c] = string.ascii_lowercase[i]
                i += 1
            if i == 26:
                break
        return "".join(d[c] if c != " " else " " for c in message)


# 2326 - Spiral Matrix IV - MEDIUM
class Solution:
    def spiralMatrix(self, m: int, n: int, head: Optional[ListNode]) -> List[List[int]]:
        g = [[-1] * n for _ in range(m)]
        d = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        r = c = dd = 0
        while head:
            g[r][c] = head.val
            dr, dc = d[dd]
            if (
                r + dr < 0
                or r + dr >= m
                or c + dc < 0
                or c + dc >= n
                or g[r + dr][c + dc] != -1
            ):
                dd = (dd + 1) % 4
                dr, dc = d[dd]
            r += dr
            c += dc
            head = head.next
        return g


# 2327 - Number of People Aware of a Secret - MEDIUM
class Solution:
    # O(n ^ 2) / O(n)
    def peopleAwareOfSecret(self, n: int, delay: int, forget: int) -> int:
        # number of 'new' peple who know secret on day i
        f = [0] * (n + 1)
        f[1] = 1
        mod = 10**9 + 7
        for i in range(2, n + 1):
            # for j in range(max(i - forget + 1, 0), i - delay + 1):
            for j in range(i - forget + 1, i - delay + 1):
                if 0 <= j:
                    f[i] = (f[i] + f[j]) % mod
        return sum(f[-forget:]) % mod

    def peopleAwareOfSecret(self, n: int, delay: int, forget: int) -> int:
        new = [0] * (n + 1)
        new[1] = 1
        fgt = [0] * (n + 1)
        mod = 10**9 + 7
        ans = 0
        for i in range(1, n + 1):
            for j in range(i + delay, i + forget):
                if j < n + 1:
                    new[j] = (new[i] + new[j]) % mod
            if i + forget < n + 1:
                fgt[i + forget] = (fgt[i + forget] + new[i]) % mod
            ans = (ans + new[i] - fgt[i]) % mod
        return ans

    def peopleAwareOfSecret(self, n: int, delay: int, forget: int) -> int:
        # 0: cannot share secret / 1: can, on day i
        dp = [[0, 0] for _ in range(n + 1)]
        dp[1][1] = 1
        mod = 10**9 + 7
        for i in range(1, n + 1):
            for j in range(1, min(i + delay, n + 1)):
                dp[j][0] = (dp[j][0] + dp[i][1]) % mod
            for j in range(i + delay, min(i + forget, n + 1)):
                dp[j][1] = (dp[j][1] + dp[i][1]) % mod
        return (dp[n][0] + dp[n][1]) % mod

    # optimize dp[n][0]
    def peopleAwareOfSecret(self, n: int, delay: int, forget: int) -> int:
        # 0: cannot share secret / 1: can, on day i
        dp = [0] * (n + 1)
        dp[1] = 1
        cnt0 = 0  # dp[n][0]
        mod = 10**9 + 7
        for i in range(1, n + 1):
            if i + delay >= n + 1:
                cnt0 = (cnt0 + dp[i]) % mod
            for j in range(i + delay, min(i + forget, n + 1)):
                dp[j] = (dp[j] + dp[i]) % mod
        return (cnt0 + dp[n]) % mod

    def peopleAwareOfSecret(self, n: int, delay: int, forget: int) -> int:
        # people who kept secret for j days on day i
        f = [[0] * (n + 1) for _ in range(n + 1)]
        mod = 10**9 + 7
        for i in range(1, n + 1):
            f[1][i] = 1
        for i in range(2, n + 1):
            for j in range(1, forget + 1):
                if j == 1:
                    f[i][j] = (f[i - 1][forget - 1] - f[i - 1][delay - 1]) % mod
                else:
                    f[i][j] = (f[i - 1][j - 1] - f[i - 1][j - 2]) % mod
                f[i][j] = (f[i][j] + f[i][j - 1]) % mod
        return (f[n][forget] + mod) % mod

    # O(n) / O(n), optimize with prefix sum
    def peopleAwareOfSecret(self, n: int, delay: int, forget: int) -> int:
        # number of 'new' peple who know secret on day i
        dp = 1
        p = [0] * (n + 2)  # prefix sum
        p[2] = 1
        mod = 10**9 + 7
        for i in range(2, n + 1):
            dp = p[max(i - delay + 1, 0)] - p[max(i - forget + 1, 0)]
            p[i + 1] = (p[i] + dp) % mod
        return (p[i + 1] - p[max(i - forget + 1, 0)] + mod) % mod


# 2328 - Number of Increasing Paths in a Grid - HARD
class Solution:
    # O(mn) / O(mn)
    def countPaths(self, grid: List[List[int]]) -> int:
        @functools.lru_cache(None)
        def dfs(x: int, y: int) -> int:
            cur = 1
            for nx, ny in [x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]:
                # It doesn't matter if (x, y) is the start or the end
                # Both can be accepted, just thinking from a different perspective
                # grid[x][y] > grid[nx][ny] or grid[x][y] < grid[nx][ny]
                if 0 <= nx < m and 0 <= ny < n and grid[x][y] < grid[nx][ny]:
                    cur += dfs(nx, ny)
            return cur % mod

        m = len(grid)
        n = len(grid[0])
        mod = 10**9 + 7
        return sum(dfs(i, j) for i in range(m) for j in range(n)) % mod

    def countPaths(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        mod = 10**9 + 7
        dp = [[1 for _ in range(n)] for j in range(m)]
        pair = []
        # state update require all elements to be sorted
        for i in range(m):
            for j in range(n):
                pair.append([grid[i][j], i, j])
        pair.sort()
        for i in range(m * n):
            v, x, y = pair[i]
            for nx, ny in [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]:
                # why below is not work?
                # if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] > v:

                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] < v:
                    dp[x][y] += dp[nx][ny]
            dp[x][y] %= mod
        return sum(sum(r) for r in dp) % mod

    def countPaths(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        mod = 10**9 + 7
        dp = [[1 for _ in range(n)] for j in range(m)]
        pair = []
        for i in range(m):
            for j in range(n):
                pair.append([grid[i][j], i, j])
        pair.sort()
        while pair:
            v, x, y = pair.pop()
            for nx, ny in [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]:
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] > v:
                    dp[x][y] += dp[nx][ny]
            dp[x][y] %= mod
        return sum(sum(r) for r in dp) % mod


# 2331 - Evaluate Boolean Binary Tree - EASY
class Solution:
    def evaluateTree(self, root: Optional[TreeNode]) -> bool:
        if root.val == 2:
            return self.evaluateTree(root.left) or self.evaluateTree(root.right)
        if root.val == 3:
            return self.evaluateTree(root.left) and self.evaluateTree(root.right)
        return root.val == 1


# 2332 - The Latest Time to Catch a Bus - MEDIUM
class Solution:
    # O(nlogn + mlogm) / O(1)
    def latestTimeCatchTheBus(
        self, buses: List[int], passengers: List[int], capacity: int
    ) -> int:
        buses.sort()
        passengers.sort()
        j = 0
        for v in buses:
            c = capacity
            while j < len(passengers) and c and passengers[j] <= v:
                j += 1
                c -= 1
        j -= 1
        ans = buses[-1] if c else passengers[j]
        while j > -1 and passengers[j] == ans:
            j -= 1
            ans -= 1
        return ans

    def latestTimeCatchTheBus(
        self, buses: List[int], p: List[int], capacity: int
    ) -> int:
        buses.sort()
        p.sort()
        s = set(p)
        ans = cnt = j = 0
        for i in range(len(buses)):
            while cnt < capacity and j < len(p) and p[j] <= buses[i]:
                if p[j] == p[0] or p[j - 1] != p[j] - 1:
                    ans = p[j] - 1
                j += 1
                cnt += 1
            if cnt < capacity and buses[i] not in s:
                ans = buses[i]
            cnt = 0
        return ans

    def latestTimeCatchTheBus(
        self, buses: List[int], passengers: List[int], capacity: int
    ) -> int:
        s = set(passengers)
        p = collections.deque(sorted(passengers))
        ans = 1
        for t in sorted(buses):
            c = capacity
            while c and p and p[0] <= t:
                ans = p.popleft()
                c -= 1
            if c:
                ans = t
        while ans in s:
            ans -= 1
        return ans


# 2333 - Minimum Sum of Squared Difference - MEDIUM
class Solution:
    def minSumSquareDiff(
        self, nums1: List[int], nums2: List[int], k1: int, k2: int
    ) -> int:
        d = collections.defaultdict(int)
        for a, b in zip(nums1, nums2):
            d[abs(a - b)] += 1
        k = k1 + k2
        i = max(d.keys())
        while i > 0 and k > 0:
            change = min(k, d[i])
            d[i - 1] += change
            k -= change
            d[i] -= change
            i -= 1
        return sum(k * k * v for k, v in d.items())

    def minSumSquareDiff(
        self, nums1: List[int], nums2: List[int], k1: int, k2: int
    ) -> int:
        ans = 0
        k = k1 + k2
        diff = []
        for a, b in zip(nums1, nums2):
            d = abs(a - b)
            diff.append(d)
            ans += d * d
        if sum(diff) <= k:
            return 0
        diff.append(0)
        diff.sort(reverse=True)
        for i, v in enumerate(diff):
            ans -= v * v
            j = i + 1
            c = j * (v - diff[j])
            if c < k:
                k -= c
                continue
            v -= k // j
            return ans + k % j * (v - 1) * (v - 1) + (j - k % j) * v * v

    def minSumSquareDiff(
        self, nums1: List[int], nums2: List[int], k1: int, k2: int
    ) -> int:
        diff = sorted(abs(a - b) for a, b in zip(nums1, nums2) if a != b)
        k = k1 + k2
        sub = sum(diff)
        if sub <= k:
            return 0
        for i, v in enumerate(diff):
            # whether all remaining elements of diff can be reduced to v
            if sub - (len(diff) - i) * v <= k:
                k -= sub - (len(diff) - i) * v
                # all remaining elements be reduced to v-m, k elements to v-m-1
                m = k // (len(diff) - i)
                k %= len(diff) - i
                break
            sub -= v
        diff = diff[:i] + [v - m - 1] * k + [v - m] * (len(diff) - i - k)
        return sum([d * d for d in diff])

    def minSumSquareDiff(
        self, nums1: List[int], nums2: List[int], k1: int, k2: int
    ) -> int:
        n = len(nums1)
        k = k1 + k2
        d = [abs(a - b) for a, b in zip(nums1, nums2)]
        l = 0
        r = 10**5
        while l < r:
            m = l + r >> 1
            summ = 0
            for i in range(n):
                if d[i] > m:
                    summ += d[i] - m
            if summ <= k:
                r = m
            else:
                l = m + 1
        summ = 0
        for i in range(n):
            if d[i] > r:
                summ += d[i] - r
        k -= summ
        ans = 0
        for i in range(n):
            if d[i] >= r:
                if r and k:
                    ans += (r - 1) * (r - 1)
                    k -= 1
                else:
                    ans += r * r
            else:
                ans += d[i] * d[i]
        return ans

    def minSumSquareDiff(
        self, nums1: List[int], nums2: List[int], k1: int, k2: int
    ) -> int:
        hp = [-abs(a - b) for a, b in zip(nums1, nums2)]
        s = -sum(hp)
        k = k1 + k2
        if s <= k:
            return 0
        heapq.heapify(hp)
        while k > 0:
            d = -heapq.heappop(hp)
            can = max(k // len(nums1), 1)
            d -= can
            heapq.heappush(hp, -d)
            k -= can
        return sum(d * d for d in hp)


# 2334 - Subarray With Elements Greater Than Varying Threshold - HARD
class Solution:
    def validSubarraySize(self, nums: List[int], threshold: int) -> int:
        def find(x: int) -> int:
            if p[x] != x:
                p[x] = find(p[x])
            return p[x]

        n = len(nums)
        p = list(range(n + 1))
        sz = [1] * (n + 1)
        q = list(range(n))
        q = [y for _, y in sorted(zip(nums, q), reverse=True)]
        i = 0
        for k in range(1, n + 1):
            while i < n and nums[q[i]] > threshold // k:
                a = q[i]
                b = find(q[i] + 1)
                sz[b] += sz[a]
                if sz[b] - 1 >= k:
                    return k
                p[a] = b
                i += 1
        return -1

    def validSubarraySize(self, nums: List[int], threshold: int) -> int:
        def find(x: int) -> int:
            if p[x] != x:
                p[x] = find(p[x])
            return p[x]

        n = len(nums)
        p = list(range(n + 1))
        sz = [1] * (n + 1)
        for num, i in sorted(zip(nums, range(n)), reverse=True):
            j = find(i + 1)
            p[i] = j
            sz[j] += sz[i]
            if num > threshold // (sz[j] - 1):
                return sz[j] - 1
        return -1

    def validSubarraySize(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        left = [-1] * n
        st = []
        for i in range(len(nums)):
            while st and nums[st[-1]] >= nums[i]:
                st.pop()
            if st:
                left[i] = st[-1]
            st.append(i)
        right = [n] * n
        st = []
        for i in range(n - 1, -1, -1):
            while st and nums[st[-1]] >= nums[i]:
                st.pop()
            if st:
                right[i] = st[-1]
            st.append(i)
        for num, l, r in zip(nums, left, right):
            k = r - l - 1
            if num > threshold // k:
                return k
        return -1


# 2335 - Minimum Amount of Time to Fill Cups - EASY
class Solution:
    def fillCups(self, amount: List[int]) -> int:
        ans = 0
        hp = [-v for v in amount if v != 0]
        heapq.heapify(hp)
        while hp:
            if len(hp) >= 2:
                f = heapq.heappop(hp)
                s = heapq.heappop(hp)
                f += 1
                s += 1
                if f:
                    heapq.heappush(hp, f)
                if s:
                    heapq.heappush(hp, s)
                ans += 1
            elif len(hp) == 1:
                return ans - hp[0]
        return ans

    def fillCups(self, a: List[int]) -> int:
        ans = 0
        a.sort()
        while a[1] > 0:
            ans += 1
            a[1] -= 1
            a[2] -= 1
            a.sort()
        return ans + a[2]

    def fillCups(self, a: List[int]) -> int:
        a.sort()
        if a[0] + a[1] <= a[2]:
            return a[2]
        t = a[0] + a[1] - a[2]
        return (t + 1) // 2 + a[2]


# 2336 - Smallest Number in Infinite Set - MEDIUM
class SmallestInfiniteSet:
    def __init__(self):
        self.s = set()

    def popSmallest(self) -> int:
        for i in range(1, 1001):
            if i not in self.s:
                self.s.add(i)
                return i

    def addBack(self, num: int) -> None:
        if num in self.s:
            self.s.remove(num)
        return


class SmallestInfiniteSet:
    def __init__(self):
        self.p = 1
        self.l = []
        self.s = set()

    def popSmallest(self) -> int:
        if self.l:
            a = heapq.heappop(self.l)
            self.s.remove(a)
            return a
        else:
            a = self.p
            self.p += 1
            return a

    def addBack(self, num: int) -> None:
        if num >= self.p:
            pass
        elif num not in self.s:
            self.s.add(num)
            heapq.heappush(self.l, num)


# 2337 - Move Pieces to Obtain a String - MEDIUM
class Solution:
    def canChange(self, start: str, target: str) -> bool:
        if start.replace("_", "") != target.replace("_", ""):
            return False
        i = j = 0
        n = len(start)
        while i < n and j < n:
            if start[i] == "_":
                i += 1
                continue
            if target[j] == "_":
                j += 1
                continue
            if start[i] == "L" and i < j:
                return False
            if start[i] == "R" and i > j:
                return False
            i += 1
            j += 1
        return True

    def canChange(self, start: str, target: str) -> bool:
        if start.replace("_", "") != target.replace("_", ""):
            return False
        i = j = 0
        n = len(start)
        while i < n and j < n:
            while i < n and start[i] == "_":
                i += 1
            while j < n and target[j] == "_":
                j += 1
            if i < n and j < n:
                if start[i] == "L" and i < j:
                    return False
                if start[i] == "R" and i > j:
                    return False
            i += 1
            j += 1
        return True

    def canChange(self, start: str, target: str) -> bool:
        if start.replace("_", "") != target.replace("_", ""):
            return False
        j = 0
        for i, c in enumerate(start):
            if c == "_":
                continue
            while target[j] == "_":
                j += 1
            if i != j and (c == "L") == (i < j):
                return False
            j += 1
        return True

    def canChange(self, start: str, target: str) -> bool:
        ls = []
        rs = []
        lt = []
        rt = []
        l = r = ""
        for i, v in enumerate(start):
            if v == "L":
                ls.append(i)
                l += "L"
            if v == "R":
                rs.append(i)
                l += "R"
        for i, v in enumerate(target):
            if v == "L":
                lt.append(i)
                r += "L"
            if v == "R":
                rt.append(i)
                r += "R"
        if l != r:
            return False
        for i in range(len(ls)):
            if ls[i] < lt[i]:
                return False
        for i in range(len(rs)):
            if rs[i] > rt[i]:
                return False
        return True


# 2338 - Count the Number of Ideal Arrays - HARD


# 2341 - Maximum Number of Pairs in Array - EASY
class Solution:
    def numberOfPairs(self, nums: List[int]) -> List[int]:
        cnt = collections.Counter(nums)
        ans = [0, 0]
        for v in cnt.values():
            ans[0] += v >> 1
            ans[1] += v & 1
        return ans


# 2342 - Max Sum of a Pair With Equal Sum of Digits - MEDIUM
class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        d = collections.defaultdict(list)
        for n in nums:
            x = n
            t = 0
            while x:
                t += x % 10
                x //= 10
            d[t].append(n)
        ans = -1
        for v in d.values():
            if len(v) >= 2:
                x = sorted(v, reverse=True)
                ans = max(ans, x[0] + x[1])
        return ans

    def maximumSum(self, nums: List[int]) -> int:
        ans = -1
        d = collections.defaultdict(int)
        for n in nums:
            # t = sum(int(x) for x in str(n)) # slow
            t = 0
            x = n
            while x:
                t += x % 10
                x //= 10
            if t in d:
                ans = max(ans, d[t] + n)
            d[t] = max(d[t], n)
        return ans

    def maximumSum(self, nums: List[int]) -> int:
        g = collections.defaultdict(list)
        for n in nums:
            t = 0
            x = n
            while x > 0:
                t += x % 10
                x //= 10
            if len(g[t]) < 2:
                heapq.heappush(g[t], n)
            else:
                heapq.heappushpop(g[t], n)
        return max((g[0] + g[1] for g in g.values() if len(g) == 2), default=-1)


# 2343 - Query Kth Smallest Trimmed Number - MEDIUM
class Solution:
    def smallestTrimmedNumbers(
        self, nums: List[str], queries: List[List[int]]
    ) -> List[int]:
        ans = []
        for k, t in queries:
            arr = []
            for i, n in enumerate(nums):
                arr.append((n[-t:], i))
            arr = sorted(arr)
            ans.append(arr[k - 1][1])
        return ans


# 2344 - Minimum Deletions to Make Array Divisible - HARD
class Solution:
    def minOperations(self, nums: List[int], numsDivide: List[int]) -> int:
        ans = 0
        cnt = collections.Counter(nums)
        numsDivide = set(numsDivide)
        for x, v in sorted(cnt.items()):
            f = True
            for y in numsDivide:
                if y % x != 0:
                    f = False
                    break
            if f:
                return ans
            ans += v
        return -1

    def minOperations(self, nums: List[int], numsDivide: List[int]) -> int:
        ans = 0
        cnt = collections.Counter(nums)
        numsDivide = set(numsDivide)
        for x, v in sorted(cnt.items()):
            # if all(y % x == 0 for y in numsDivide):
            #     return ans
            if not any(y % x for y in numsDivide):
                return ans
            ans += v
        return -1

    def minOperations(self, nums: List[int], numsDivide: List[int]) -> int:
        x = numsDivide[0]
        for i in range(1, len(numsDivide)):
            x = math.gcd(x, numsDivide[i])
        cnt = collections.Counter(nums)
        ans = 0
        for k, v in sorted(cnt.items()):
            if x % k == 0:
                return ans
            ans += v
        return -1

    def minOperations(self, nums: List[int], numsDivide: List[int]) -> int:
        g = 0
        for n in numsDivide:
            g = math.gcd(g, n)
        nums.sort()
        for i in range(len(nums)):
            if g % nums[i] == 0:
                return i
        return -1


# 2347 - Best Poker Hand - EASY
class Solution:
    def bestHand(self, ranks: List[int], suits: List[str]) -> str:
        if len(set(suits)) == 1:
            return "Flush"
        cnt = collections.Counter(ranks)
        if any(v >= 3 for v in cnt.values()):
            return "Three of a Kind"
        if any(v >= 2 for v in cnt.values()):
            return "Pair"
        if len(cnt.keys()) == 5:
            return "High Card"
        return ""

    def bestHand(self, ranks: List[int], suits: List[str]) -> str:
        if len(set(suits)) == 1:
            return "Flush"
        for i in ranks:
            if ranks.count(i) >= 3:
                return "Three of a Kind"
        for i in ranks:
            if ranks.count(i) == 2:
                return "Pair"
        return "High Card"


# 2348 - Number of Zero-Filled Subarrays - MEDIUM
class Solution:
    def zeroFilledSubarray(self, nums: List[int]) -> int:
        ans = 0
        nums.append(1)
        cur = 0
        for v in nums:
            if v == 0:
                cur += 1
            else:
                ans += cur * (cur + 1) // 2
                cur = 0
        return ans

    def zeroFilledSubarray(self, nums: List[int]) -> int:
        ans = 0
        cur = 0
        for v in nums:
            if v == 0:
                cur += 1
                ans += cur
            else:
                cur = 0
        return ans


# 2349 - Design a Number Container System - MEDIUM
class NumberContainers:
    def __init__(self):
        self.cur = collections.defaultdict(int)
        self.n = collections.defaultdict(list)

    def change(self, index: int, number: int) -> None:
        self.cur[index] = number
        heapq.heappush(self.n[number], index)
        return

    def find(self, number: int) -> int:
        while self.n[number]:
            x = self.n[number][0]
            if self.cur[x] == number:
                return x
            heapq.heappop(self.n[number])
        return -1


class NumberContainers:
    def __init__(self):
        self.cur = collections.defaultdict(int)
        self.arr = collections.defaultdict(sortedcontainers.SortedList)

    def change(self, index: int, number: int) -> None:
        if index in self.cur:
            self.arr[self.cur[index]].remove(index)
        self.cur[index] = number
        self.arr[number].add(index)

    def find(self, number: int) -> int:
        return self.arr[number][0] if self.arr[number] else -1


# 2350 - Shortest Impossible Sequence of Rolls - HARD
class Solution:
    # O(n) / O(k)
    def shortestSequence(self, rolls: List[int], k: int) -> int:
        ans = 1
        s = set()
        for v in rolls:
            s.add(v)
            if len(s) == k:
                ans += 1
                s.clear()
        return ans

    def shortestSequence(self, rolls: List[int], k: int) -> int:
        ans = 1
        vis = [0] * (k + 1)
        need = k
        for v in rolls:
            if vis[v] < ans:
                vis[v] = ans
                need -= 1
                if need == 0:
                    need = k
                    ans += 1
        return ans


# 2351 - First Letter to Appear Twice - EASY
class Solution:
    def repeatedCharacter(self, s: str) -> str:
        st = set()
        for c in s:
            if c in st:
                return c
            st.add(c)
        return ""

    def repeatedCharacter(self, s: str) -> str:
        b = 0
        for c in s:
            # if 1 << ord(c) - ord("a") & b > 0:
            if b >> ord(c) - ord("a") & 1 > 0:
                return c
            b |= 1 << ord(c) - ord("a")
        return ""


# 2352 - Equal Row and Column Pairs - MEDIUM
class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        d = collections.defaultdict(int)
        for r in grid:
            d[tuple(r)] += 1
        ans = 0
        n = len(grid)
        for j in range(n):
            t = tuple([grid[i][j] for i in range(n)])
            ans += d[t]
        return ans

    def equalPairs(self, grid: List[List[int]]) -> int:
        cnt = collections.Counter(tuple(r) for r in grid)
        return sum(cnt[c] for c in zip(*grid))


# 2353 - Design a Food Rating System - MEDIUM
class FoodRatings:
    def __init__(self, foods: List[str], cuisines: List[str], ratings: List[int]):
        self.frc = collections.defaultdict(list)
        self.crf = collections.defaultdict(list)
        for f, c, r in zip(foods, cuisines, ratings):
            self.frc[f] = [r, c]
            heapq.heappush(self.crf[c], (-r, f))

    def changeRating(self, food: str, newRating: int) -> None:
        _, c = self.frc[food]
        heapq.heappush(self.crf[c], (-newRating, food))
        self.frc[food][0] = newRating

    def highestRated(self, cuisine: str) -> str:
        hp = self.crf[cuisine]
        while -hp[0][0] != self.frc[hp[0][1]][0]:
            heapq.heappop(hp)
        return hp[0][1]


class FoodRatings:
    def __init__(self, foods: List[str], cuisines: List[str], ratings: List[int]):
        self.frc = collections.defaultdict(list)
        self.crf = collections.defaultdict(sortedcontainers.SortedSet)
        for f, c, r in zip(foods, cuisines, ratings):
            self.frc[f] = [r, c]
            self.crf[c].add((-r, f))

    def changeRating(self, food: str, newRating: int) -> None:
        r, c = self.frc[food]
        s = self.crf[c]
        s.remove((-r, food))
        s.add((-newRating, food))
        self.frc[food][0] = newRating

    def highestRated(self, cuisine: str) -> str:
        return self.crf[cuisine][0][1]


class FoodRatings:
    def __init__(self, foods: List[str], cuisines: List[str], ratings: List[int]):
        self.f2r = collections.defaultdict(int)
        self.f2c = collections.defaultdict(str)
        self.crf = collections.defaultdict(sortedcontainers.SortedList)
        for f, c, r in zip(foods, cuisines, ratings):
            self.f2r[f] = r
            self.f2c[f] = c
            self.crf[c].add((-r, f))

    def changeRating(self, food: str, newRating: int) -> None:
        pre = self.f2r[food]
        c = self.f2c[food]
        self.crf[c].discard((-pre, food))
        self.f2r[food] = newRating
        self.f2c[food] = c
        self.crf[c].add((-newRating, food))

    def highestRated(self, cuisine: str) -> str:
        return self.crf[cuisine][0][1]


# 2354 - Number of Excellent Pairs - HARD
class Solution:
    """
    py3.10 builtin
    int.bit_count(x)
    x.bit_count()

    def bit_count(self) -> str:
        return bin(self).count("1")
    """

    def countExcellentPairs(self, nums: List[int], k: int) -> int:
        def fn(x):
            r = 0
            while x:
                if x & 1:
                    r += 1
                x //= 2
            return r

        cnt = collections.Counter(x.bit_count() for x in set(nums))  # 100ms
        # cnt = collections.Counter(bin(x).count('1') for x in set(nums)) # 200ms
        # cnt = collections.Counter(fn(x) for x in set(nums)) # 1000ms
        ans = 0
        for k1, v1 in cnt.items():
            for k2, v2 in cnt.items():
                if k1 + k2 >= k:
                    ans += v1 * v2
        return ans

    def countExcellentPairs(self, nums: List[int], k: int) -> int:
        arr = [int.bit_count(x) for x in set(nums)]
        arr.sort()
        ans = 0
        for v in arr:
            p = bisect.bisect_left(arr, k - v)
            ans += len(arr) - p
        return ans


# 2357 - Make Array Zero by Subtracting Equal Amounts - EASY
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        return len(set(n for n in nums if n > 0))

    def minimumOperations(self, nums: List[int]) -> int:
        return len(set(nums) - {0})

    def minimumOperations(self, nums: List[int]) -> int:
        s = set(nums)
        if 0 in s:
            s.remove(0)
        return len(s)


# 2358 - Maximum Number of Groups Entering a Competition - MEDIUM
class Solution:
    def maximumGroups(self, grades: List[int]) -> int:
        # grades.sort()
        ans = 1
        cur = 0
        pre = 1
        for _ in range(1, len(grades)):
            cur += 1
            if cur > pre:
                ans += 1
                pre = cur
                cur = 0
        return ans

    def maximumGroups(self, grades: List[int]) -> int:
        n = len(grades)
        k = 0
        while n >= k + 1:
            k += 1
            n -= k
        return k

    def maximumGroups(self, grades: List[int]) -> int:
        for i in range(1, len(grades)):
            if i * (i + 1) // 2 > len(grades):
                return i - 1
        return 1

    def maximumGroups(self, grades: List[int]) -> int:
        # 1 + 2 + ... + x <= n
        # x(x + 1) / 2 <= n
        # x**2 + x - 2n <= n
        return int((-1 + math.sqrt(1 + 8 * len(grades))) / 2)

    def maximumGroups(self, grades: List[int]) -> int:
        # 如果怕掉精度, 1.99999 -> 1
        n = 2 * len(grades)
        x = int(math.sqrt(n))
        return x - 1 if x * (x + 1) > n else x


# 2359 - Find Closest Node to Given Two Nodes - MEDIUM
class Solution:
    # only one cycle in each block
    def closestMeetingNode(self, edges: List[int], n1: int, n2: int) -> int:
        p1 = [n1]
        vis = set([n1])
        while edges[n1] != -1 and edges[n1] not in vis:
            p1.append(edges[n1])
            vis.add(edges[n1])
            n1 = edges[n1]
        p2 = [n2]
        vis = set([n2])
        while edges[n2] != -1 and edges[n2] not in vis:
            p2.append(edges[n2])
            vis.add(edges[n2])
            n2 = edges[n2]
        s1 = set(p1)
        s2 = set(p2)
        dis1 = {v: i for i, v in enumerate(p1)}
        dis2 = {v: i for i, v in enumerate(p2)}
        ans = -1
        d = math.inf
        for n in s1.intersection(s2):
            d1 = dis1[n]
            d2 = dis2[n]
            if max(d1, d2) < d:
                d = max(d1, d2)
                ans = n
            elif max(d1, d2) == d:
                ans = min(ans, n)
        return ans

    def closestMeetingNode(self, edges: List[int], node1: int, node2: int) -> int:
        def path(x: int) -> List[int]:
            dis = [math.inf] * n  # calculate distance and meanwhile mark it as visited
            d = 0
            while x >= 0 and dis[x] == math.inf:
                dis[x] = d
                d += 1
                x = edges[x]
            return dis

        n = len(edges)
        mi = math.inf
        ans = -1
        # traverse from smaller to larger index, do not require mapping as above solution
        for i, d1, d2 in zip(range(n), path(node1), path(node2)):
            d = max(d1, d2)
            if d < mi:
                mi = d
                ans = i
        return ans


# 2360 - Longest Cycle in a Graph - HARD
class Solution:
    def longestCycle(self, edges: List[int]) -> int:
        def dfs(x: int) -> bool:
            if vis[x]:
                nonlocal end
                end = x
                return True
            path.append(x)
            vis[x] = True
            if edges[x] == -1:
                return False
            return dfs(edges[x])

        vis = [False] * len(edges)
        ans = -1
        for i in range(len(edges)):
            if not vis[i]:
                end = -1
                path = []
                if dfs(i):
                    for i, p in enumerate(path):  # dfs with depth to optimize this step
                        if p == end:
                            ans = max(ans, len(path) - i)
        return ans

    def longestCycle(self, edges: List[int]) -> int:
        def dfs(x: int, depth: int) -> None:
            nonlocal ans
            vis[x] = True
            in_stk[x] = depth
            y = edges[x]  # y is next vertex / node
            if y != -1:
                if not vis[y]:
                    dfs(y, depth + 1)
                elif in_stk[y]:
                    ans = max(ans, depth + 1 - in_stk[y])
            in_stk[x] = 0
            return

        vis = [False] * len(edges)
        in_stk = [0] * len(edges)  # in stack, idea is like marking time, rank or depth
        ans = -1
        for i in range(len(edges)):
            if not vis[i]:
                dfs(i, 1)
        return ans

    def longestCycle(self, edges: List[int]) -> int:
        vis = [False] * len(edges)
        ans = -1
        for i in range(len(edges)):
            if not vis[i]:
                end = -1
                path = []
                while i != -1:
                    if vis[i]:
                        end = i
                        break
                    path.append(i)
                    vis[i] = True
                    i = edges[i]
                if end != -1:
                    for i, p in enumerate(path):  # use time array to optimize this step
                        if p == end:
                            ans = max(ans, len(path) - i)
        return ans

    def longestCycle(self, edges: List[int]) -> int:
        time = [0] * len(edges)
        now = 1
        ans = -1
        for x in range(len(edges)):
            if time[x]:  # visited
                continue
            start = now
            while x != -1:
                if time[x]:
                    if time[x] >= start:  # find a cycle
                        ans = max(ans, now - time[x])
                    break
                time[x] = now
                now += 1
                x = edges[x]
        return ans


# 2363 - Merge Similar Items - EASY
class Solution:
    def mergeSimilarItems(
        self, items1: List[List[int]], items2: List[List[int]]
    ) -> List[List[int]]:
        d = collections.defaultdict(int)
        for v, w in items1:
            d[v] += w
        for v, w in items2:
            d[v] += w
        return sorted(d.items())

    def mergeSimilarItems(
        self, items1: List[List[int]], items2: List[List[int]]
    ) -> List[List[int]]:
        cnt = collections.Counter()
        cnt += dict(items1)
        cnt += dict(items2)
        return sorted(cnt.items())


# 2364 - Count Number of Bad Pairs - MEDIUM
class Solution:
    def countBadPairs(self, nums: List[int]) -> int:
        d = collections.defaultdict(int)
        ans = 0
        summ = 0
        for i, v in enumerate(nums):
            ans += summ - d[v - i]
            summ += 1
            d[v - i] += 1
        return ans

    def countBadPairs(self, nums: List[int]) -> int:
        d = collections.defaultdict(int)
        for i, v in enumerate(nums):
            d[v - i] += 1
        ans = 0
        for v in d.values():
            ans += v * (v - 1) // 2
        return len(nums) * (len(nums) - 1) // 2 - ans

    def countBadPairs(self, nums: List[int]) -> int:
        cnt = collections.Counter(v - i for i, v in enumerate(nums))
        return len(nums) * (len(nums) - 1) // 2 - sum(
            v * (v - 1) // 2 for v in cnt.values()
        )


# 2365 - Task Scheduler II - MEDIUM
class Solution:
    # solution 1: maintain the current time
    def taskSchedulerII(self, tasks: List[int], space: int) -> int:
        d = {}
        cur = 0  # current time
        for t in tasks:
            cur += 1
            if t in d and cur <= d[t] + space:
                cur = d[t] + space + 1
            # or
            # if t in d:
            #     cur = max(cur, d[t] + space + 1)
            d[t] = cur
        return cur

    def taskSchedulerII(self, tasks: List[int], space: int) -> int:
        d = collections.defaultdict(lambda: -1e9)
        cur = 0
        for t in tasks:
            cur += 1
            if cur <= d[t] + space:
                cur = d[t] + space + 1
            d[t] = cur
        return cur

    # solution 2: the time to finish the last task
    def taskSchedulerII(self, tasks: List[int], space: int) -> int:
        d = {}
        r = 0
        for t in tasks:
            if t in d:
                d[t] = max(r + 1, d[t] + space + 1)
            else:
                d[t] = r + 1
            r = d[t]
        return r


# 2366 - Minimum Replacements to Sort the Array - HARD
class Solution:
    def minimumReplacement(self, nums: List[int]) -> int:
        ans = 0
        nxt = nums[-1]
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] <= nxt:
                nxt = nums[i]
            else:
                k = nums[i] // nxt
                while k * nxt < nums[i]:
                    k += 1
                ans += k - 1
                nxt = nums[i] // k
        return ans

    def minimumReplacement(self, nums: List[int]) -> int:
        ans = 0
        nxt = nums[-1]
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] <= nxt:
                nxt = nums[i]
            else:
                k = (nums[i] + nxt - 1) // nxt  # math.ceil(nums[i] / nxt)
                ans += k - 1
                nxt = nums[i] // k
        return ans

    def minimumReplacement(self, nums: List[int]) -> int:
        ans = 0
        nxt = nums[-1]
        for i in range(len(nums) - 2, -1, -1):
            k = (nums[i] - 1) // nxt
            ans += k
            nxt = nums[i] // (k + 1)
        return ans

    def minimumReplacement(self, nums: List[int]) -> int:
        ans = 0
        nxt = nums[-1]
        for n in reversed(nums):
            k = (n - 1) // nxt
            ans += k
            nxt = n // (k + 1)
        return ans


# 2367 - Number of Arithmetic Triplets - EASY
class Solution:
    def arithmeticTriplets(self, nums: List[int], diff: int) -> int:
        s = set(nums)
        return sum(x - diff in s and x + diff in s for x in nums)

    def arithmeticTriplets(self, nums: List[int], diff: int) -> int:
        s = set()
        ans = 0
        for x in nums:
            if x - diff in s and x - diff * 2 in s:
                ans += 1
            s.add(x)
        return ans


# 2368 - Reachable Nodes With Restrictions - MEDIUM
class Solution:
    def reachableNodes(
        self, n: int, edges: List[List[int]], restricted: List[int]
    ) -> int:
        g = collections.defaultdict(list)
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        r = set(restricted)
        dq = collections.deque([0])
        ans = 0
        vis = [False] * n
        vis[0] = True
        while dq:
            ans += len(dq)
            for _ in range(len(dq)):
                n = dq.popleft()
                for x in g[n]:
                    if not vis[x] and x not in r:
                        dq.append(x)
                        vis[x] = True
        return ans

    def reachableNodes(
        self, n: int, edges: List[List[int]], restricted: List[int]
    ) -> int:
        g = collections.defaultdict(list)
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        ans = 0
        r = set(restricted)

        def dfs(x: int, fa: int):
            nonlocal ans
            ans += 1
            for y in g[x]:
                # we need father? -> cuz it is a tree
                if y != fa and y not in r:
                    dfs(y, x)

        dfs(0, -1)
        return ans

    def reachableNodes(
        self, n: int, edges: List[List[int]], restricted: List[int]
    ) -> int:
        g = [[] for _ in range(n)]
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        ans = 0
        r = set(restricted)
        vis = [False] * n
        vis[0] = True

        def dfs(x: int):
            nonlocal ans
            ans += 1
            for y in g[x]:
                if not vis[y] and y not in r:
                    vis[y] = True
                    dfs(y)

        dfs(0)
        return ans


# 2369 - Check if There is a Valid Partition For The Array - MEDIUM
class Solution:
    # dp
    def validPartition(self, nums: List[int]) -> bool:
        @functools.lru_cache(None)
        def dfs(i: int) -> bool:
            if i == 2:
                return (
                    nums[0] == nums[1] == nums[2]
                    or nums[0] + 2 == nums[1] + 1 == nums[2]
                )
            if i == 1:
                return nums[0] == nums[1]
            if i <= 0:
                return False
            if nums[i - 1] == nums[i] and dfs(i - 2):
                return True
            if nums[i - 2] == nums[i - 1] == nums[i] and dfs(i - 3):
                return True
            if nums[i - 2] + 2 == nums[i - 1] + 1 == nums[i] and dfs(i - 3):
                return True
            return False

        return dfs(len(nums) - 1)

    def validPartition(self, nums: List[int]) -> bool:
        n = len(nums)

        @functools.lru_cache(None)
        def dfs(i: int) -> bool:
            if i == n:
                return True
            if i + 1 < n and nums[i] == nums[i + 1] and dfs(i + 2):
                return True
            if i + 2 < n and nums[i] == nums[i + 1] == nums[i + 2] and dfs(i + 3):
                return True
            if (
                i + 2 < n
                and nums[i] + 1 == nums[i + 1] == nums[i + 2] - 1
                and dfs(i + 3)
            ):
                return True
            return False

        return dfs(0)

    def validPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        f = [True] + [False] * n
        for i in range(1, n):
            if f[i - 1] and nums[i] == nums[i - 1]:
                f[i + 1] = True
            if i > 1:
                if f[i - 2] and nums[i] == nums[i - 1] == nums[i - 2]:
                    f[i + 1] = True
                if f[i - 2] and nums[i] == nums[i - 1] + 1 == nums[i - 2] + 2:
                    f[i + 1] = True
        return f[n]

    def validPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        f = [True] + [False] * n
        for i in range(n):
            if i >= 1 and nums[i] == nums[i - 1]:
                f[i + 1] |= f[i - 1]
            if i >= 2 and nums[i] == nums[i - 1] and nums[i] == nums[i - 2]:
                f[i + 1] |= f[i - 2]
            if i >= 2 and nums[i] == nums[i - 1] + 1 and nums[i] == nums[i - 2] + 2:
                f[i + 1] |= f[i - 2]
        return f[n]

    def validPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        f = [False] * (n + 1)
        f[0] = True
        for i in range(n):
            if not f[i]:
                continue
            if i + 2 <= n and nums[i] == nums[i + 1]:
                f[i + 2] = True
            if i + 3 <= n:
                if (
                    nums[i] == nums[i + 1] == nums[i + 2]
                    or nums[i] + 2 == nums[i + 1] + 1 == nums[i + 2]
                ):
                    f[i + 3] = True
        return f[n]

    # save space
    def validPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        f = [False, False, False, True]
        for i in range(n):
            f[i % 4] = False
            if i - 1 >= 0 and nums[i] == nums[i - 1]:
                f[i % 4] |= f[(i - 2) % 4]
            if i - 2 >= 0 and nums[i] == nums[i - 1] == nums[i - 2]:
                f[i % 4] |= f[(i - 3) % 4]
            if i - 2 >= 0 and nums[i] == nums[i - 1] + 1 == nums[i - 2] + 2:
                f[i % 4] |= f[(i - 3) % 4]
        return f[(n - 1) % 4]


# 2370 - Longest Ideal Subsequence - MEDIUM
class Solution:
    def longestIdealString(self, s: str, k: int) -> int:
        f = [0] * 26
        for c in s:
            i = ord(c) - ord("a")
            f[i] = 1 + max(f[max(0, i - k) : i + k + 1])  # [ , )
            # f[i] = 1 + max(f[max(0, i - k) : min(25, i + k) + 1])
        return max(f)

    def longestIdealString(self, s, k):
        f = [0] * 123  # ord('z') = 122
        for c in s:
            i = ord(c)
            f[i] = max(f[i - k : i + k + 1]) + 1
        return max(f)

    def longestIdealString(self, s: str, k: int) -> int:
        f = collections.defaultdict(int)
        for c in s:
            i = ord(c)
            f[i] = 1 + max(f[i + j] for j in range(-k, k + 1))
        return max(f.values())


# 2373 - Largest Local Values in a Matrix - EASY
class Solution:
    def largestLocal(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        for i in range(n - 2):
            for j in range(n - 2):
                grid[i][j] = max(max(r[j : j + 3]) for r in grid[i : i + 3])
            grid[i].pop()
            grid[i].pop()
        grid.pop()
        grid.pop()
        return grid


# 2374 - Node With Highest Edge Score - MEDIUM
class Solution:
    def edgeScore(self, edges: List[int]) -> int:
        s = [0] * len(edges)
        for i, v in enumerate(edges):
            s[v] += i
        ans = mx = 0
        for i, sc in enumerate(s):
            if sc > mx:
                mx = sc
                ans = i
        return ans


# 2375 - Construct Smallest Number From DI String - MEDIUM
class Solution:
    # O(n!) / O(n)
    def smallestNumber(self, pattern: str) -> str:
        for lst in itertools.permutations(range(1, 2 + len(pattern)), len(pattern) + 1):
            for i in range(len(pattern)):
                if lst[i] < lst[i + 1] and pattern[i] == "D":
                    break
                if lst[i] > lst[i + 1] and pattern[i] == "I":
                    break
            else:
                return "".join([str(x) for x in lst])

    # O(n!) / O(n), backtracking
    def smallestNumber(self, pattern: str) -> str:
        def dfs(x: int, pre: int, path: List[int]) -> bool:
            if x == len(pattern):
                return True
            if pattern[x] == "I":
                for i in range(pre + 1, 10):
                    if used[i]:
                        continue
                    used[i] = True
                    path.append(i)
                    if dfs(x + 1, i, path):
                        return True
                    path.pop()
                    used[i] = False
            else:
                for i in range(pre - 1, 0, -1):
                    if used[i]:
                        continue
                    used[i] = True
                    path.append(i)
                    if dfs(x + 1, i, path):
                        return True
                    path.pop()
                    used[i] = False
            return False

        for i in range(1, 10):
            used = [False] * 10
            arr = [i]
            used[i] = True
            if dfs(0, i, arr):
                return "".join(map(str, arr))
        return ""

    def smallestNumber(self, pattern: str) -> str:
        def dfs(x: int) -> None:
            nonlocal find, ans
            if find:
                return
            if x > len(pattern):
                find = True
                ans = "".join(str(p[i]) for i in range(x))
                return
            for i in range(1, 10):
                if used[i]:
                    continue
                if x != 0:
                    if pattern[x - 1] == "I" and p[x - 1] >= i:
                        continue
                    if pattern[x - 1] == "D" and p[x - 1] <= i:
                        continue
                used[i] = True
                p[x] = i
                dfs(x + 1)
                used[i] = False
                p[x] = 0
            return

        find = False
        ans = ""
        used = [False] * 10
        p = [0] * 10
        dfs(0)
        return ans

    # O(n) / O(n)
    def smallestNumber(self, pattern: str) -> str:
        ans = ""
        st = []
        for i, p in enumerate(pattern + "I", 1):
            if p == "D":
                st.append(str(i))
            else:
                ans += str(i)
                while st:
                    ans += st.pop()
        return ans

    def smallestNumber(self, pattern: str) -> str:
        st = []
        ans = []
        for i, p in enumerate(pattern + "I", 1):
            st.append(str(i))
            if p == "I":
                ans += st[::-1]
                st = []
        return "".join(ans)

    def smallestNumber(self, pattern: str) -> str:
        ans = []
        for i, p in enumerate(pattern + "I", 1):
            if p == "I":
                ans += range(i, len(ans), -1)
        return "".join(map(str, ans))

    def smallestNumber(self, s):
        res = []
        for i, c in enumerate(s + "I", 1):
            if c == "I":
                res += range(i, len(res), -1)
        return "".join(map(str, res))


# 2376 - Count Special Integers - HARD
class Solution:
    # digit dp
    def countSpecialNumbers(self, n: int) -> int:
        s = str(n)

        @functools.lru_cache(None)
        def fn(i: int, mask: int, is_limit: bool, is_num: bool) -> int:
            if i == len(s):
                return int(is_num)
            ans = 0
            if not is_num:
                ans = fn(i + 1, mask, False, False)
            up = int(s[i]) if is_limit else 9
            for d in range(0 if is_num else 1, up + 1):
                if mask >> d & 1 == 0:
                    ans += fn(i + 1, mask | (1 << d), is_limit and d == up, True)
            return ans

        return fn(0, 0, True, False)

    def countSpecialNumbers(self, n: int) -> int:
        # all numbers
        nums = []
        while n:
            nums.append(n % 10)
            n //= 10
        ans = 0
        # all numbers that length < m, m is the length of original number
        for i in range(1, len(nums)):
            t = k = 9
            for j in range(i - 1):  # calculate permutation
                t *= k
                k -= 1
            ans += t
        nums.reverse()
        vis = [0] * 10
        for i in range(len(nums)):
            for j in range(int(i == 0), nums[i]):  # no leading zore
                if vis[j]:
                    continue
                t = 1
                u = 9 - i
                for k in range(len(nums) - i - 1):  # calculate permutation
                    t *= u
                    u -= 1
                ans += t
            if vis[nums[i]]:
                break
            vis[nums[i]] = True

        # whether original number is qualified
        s = set(nums)
        if len(s) == len(nums):
            ans += 1
        return ans

    # shorter version of above
    def countSpecialNumbers(self, n: int) -> int:
        nums = list(map(int, str(n + 1)))
        ans = sum(9 * math.perm(9, i) for i in range(len(nums) - 1))
        s = set()
        for i, x in enumerate(nums):
            for y in range(i == 0, x):
                if y not in s:
                    ans += math.perm(9 - i, len(nums) - i - 1)
            if x in s:
                break
            s.add(x)
        return ans


# 2379 - Minimum Recolors to Get K Consecutive Black Blocks - EASY
class Solution:
    def minimumRecolors(self, blocks: str, k: int) -> int:
        ans = 100
        for i in range(len(blocks) - k + 1):
            ans = min(ans, blocks[i : i + k].count("W"))
        return ans

    def minimumRecolors(self, blocks: str, k: int) -> int:
        cnt = 0
        for i in range(k):
            if blocks[i] == "W":
                cnt += 1
        ans = cnt
        for i in range(k, len(blocks)):
            if blocks[i] == "W":
                cnt += 1
            if blocks[i - k] == "W":
                cnt -= 1
            ans = min(ans, cnt)
        return ans


# 2380 - Time Needed to Rearrange a Binary String - MEDIUM
class Solution:
    def secondsToRemoveOccurrences(self, s: str) -> int:
        ans = 0
        while "01" in s:
            ans += 1
            s = s.replace("01", "10", -1)
        return ans

    def secondsToRemoveOccurrences(self, s: str) -> int:
        ans = pre0 = 0
        for c in s:
            if c == "0":
                pre0 += 1
            if c == "1" and pre0:
                ans = max(ans + 1, pre0)
        return ans


# 2381 - Shifting Letters II - MEDIUM
class Solution:
    # O(n) / O(n)
    def shiftingLetters(self, s: str, shifts: List[List[int]]) -> str:
        n = len(s)
        d = [0] * (n + 1)
        for l, r, t in shifts:
            if t:
                d[l] += 1
                d[r + 1] -= 1
            else:
                d[l] -= 1
                d[r + 1] += 1
        # prefix_sum[i+1] = prefix_sum[i] + diff[i]
        cur = 0
        a = []
        for i in range(n):
            cur += d[i]
            z = (ord(s[i]) - 97 + cur) % 26
            a.append(chr(z + 97))
        return "".join(a)


# 2382 - Maximum Segment Sum After Removals - HARD
class Solution:
    # O(nlogn) / O(n)
    def maximumSegmentSum(self, nums: List[int], removeQueries: List[int]) -> List[int]:
        class UnionFind:
            def __init__(self, n: int) -> None:
                self.p = [i for i in range(n)]
                self.sc = [0] * n

            def find(self, x: int) -> int:
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
                self.sc[py] += self.sc[px]
                return

        n = len(nums)
        uf = UnionFind(n)
        ans = [0]
        mx = 0
        for i in range(n - 1, 0, -1):
            p = removeQueries[i]
            v = nums[p]
            uf.sc[p] = v
            if p - 1 >= 0 and uf.sc[p - 1] != 0:
                uf.union(p - 1, p)
            if p + 1 < n and uf.sc[p + 1] != 0:
                uf.union(p, p + 1)
            f = uf.find(p)
            mx = uf.sc[f] if uf.sc[f] > mx else mx
            ans.append(mx)
        return ans[::-1]

    # O(nlogn) / O(n)
    def maximumSegmentSum(self, nums: List[int], removeQueries: List[int]) -> List[int]:
        n = len(nums)
        p = list(range(n + 1))
        summ = [0] * (n + 1)

        def find(x: int) -> int:
            if p[x] != x:
                p[x] = find(p[x])
            return p[x]

        ans = [0] * n
        for i in range(n - 1, 0, -1):
            x = removeQueries[i]
            fa = find(x + 1)
            p[x] = fa
            summ[fa] += summ[x] + nums[x]
            ans[i - 1] = max(ans[i], summ[fa])
        return ans

    # O(n) / O(n)
    def maximumSegmentSum(self, nums: List[int], removeQueries: List[int]) -> List[int]:
        n = len(nums)
        # left[i] 为位置 i 对应连续子段的左端点，right[i] 为位置 i 对应连续子段的右端点
        left = list(range(n))
        right = list(range(n))
        vis = [False] * n
        p = [0] + list(itertools.accumulate(nums))  # prefix sum
        ans = []
        mx = 0
        for i in removeQueries[::-1]:
            ans.append(mx)
            l = r = i
            if i - 1 >= 0 and vis[i - 1]:
                l = left[i - 1]
            if i + 1 < n and vis[i + 1]:
                r = right[i + 1]
            vis[i] = True
            left[r] = l
            right[l] = r
            mx = max(mx, p[r + 1] - p[l])
        return ans[::-1]

    # O(n) / O(n)
    def maximumSegmentSum(self, nums: List[int], removeQueries: List[int]) -> List[int]:
        ans = []
        d = {}  # segments
        mx = 0
        for i in removeQueries[::-1]:
            ans.append(mx)
            v = nums[i]
            if i - 1 in d and i + 1 in d:
                x, l, _ = d.pop(i - 1)
                y, _, r = d.pop(i + 1)
                d[l] = [x + y + v, l, r]
                d[r] = [x + y + v, l, r]
                v += x + y
            elif i - 1 in d:
                x, l, r = d.pop(i - 1)
                d[l] = [v + x, l, i]
                d[i] = [v + x, l, i]
                v += x
            elif i + 1 in d:
                x, l, r = d.pop(i + 1)
                d[i] = [v + x, i, r]
                d[r] = [v + x, i, r]
                v += x
            else:
                d[i] = [v, i, i]
            mx = max(mx, v)
        return ans[::-1]

    # O(n) / O(n)
    def maximumSegmentSum(self, nums: List[int], removeQueries: List[int]) -> List[int]:
        ans = []
        d = {}
        mx = 0
        for i in removeQueries[::-1]:
            ans.append(mx)
            d[i] = (nums[i], 1)
            rv, rl = d.get(i + 1, (0, 0))
            lv, ll = d.get(i - 1, (0, 0))
            t = nums[i] + rv + lv
            d[i + rl] = (t, ll + rl + 1)
            d[i - ll] = (t, ll + rl + 1)
            mx = max(mx, t)
        return ans[::-1]

    def maximumSegmentSum(self, nums, removeQueries):
        from sortedcontainers import SortedList

        n = len(nums)
        p = [0]  # prefix sum
        ans = []
        segments = SortedList()
        summ = SortedList([0])
        for v in nums:
            p.append(p[-1] + v)

        def add(l: int, r: int) -> None:
            """adding a segment with (left, right) borders and its sum"""
            if l > r:
                return
            segments.add((l, r))
            summ.add(p[r + 1] - p[l])
            return

        def remove(l: int, r: int) -> None:
            """removing a segment"""
            segments.remove((l, r))
            summ.remove(p[r + 1] - p[l])
            return

        add(0, n - 1)
        for i in removeQueries:
            idx = segments.bisect_left((i + 1, -1)) - 1
            l, r = segments[idx]
            remove(l, r)
            add(l, i - 1)
            add(i + 1, r)
            ans.append(summ[-1])
        return ans


# 2383 - Minimum Hours of Training to Win a Competition - EASY
class Solution:
    def minNumberOfHours(
        self, inEn: int, inEx: int, energy: List[int], experience: List[int]
    ) -> int:
        ans = 0
        for n, x in zip(energy, experience):
            if inEn <= n:
                ans += n - inEn + 1
                inEn = n + 1
            inEn -= n
            if inEx <= x:
                ans += x - inEx + 1
                inEx = x + 1
            inEx += x
        return ans


# 2384 - Largest Palindromic Number - MEDIUM
class Solution:
    def largestPalindromic(self, num: str) -> str:
        cnt = collections.Counter(num)
        ans = ""
        for i in range(9, -1, -1):
            if i == 0 and ans == "":
                break
            c = str(i)
            ans += c * (cnt[c] // 2)
            cnt[c] -= cnt[c] // 2 * 2
        half = ans[::-1]
        for i in range(9, -1, -1):
            c = str(i)
            if cnt[c] > 0:
                ans += c
                break
        return ans + half


# 2385 - Amount of Time for Binary Tree to Be Infected - MEDIUM
class Solution:
    def amountOfTime(self, root: Optional[TreeNode], start: int) -> int:
        def dfs(r: TreeNode) -> None:
            if r.left:
                g[r.val].append(r.left.val)
                g[r.left.val].append(r.val)
                dfs(r.left)
            if r.right:
                g[r.val].append(r.right.val)
                g[r.right.val].append(r.val)
                dfs(r.right)
            return

        g = collections.defaultdict(list)
        dfs(root)

        # def dfs2(r: TreeNode) -> int:
        #     if r in vis:
        #         return 0
        #     vis.add(r)
        #     depth = 1
        #     for nei in g[r]:
        #         depth = max(depth, 1 + dfs2(nei))
        #     return depth

        # vis = set()
        # return dfs2(start) - 1

        # unrooted tree(无根树), node.left / node.right / fa[node] three directions
        def dfs3(son: TreeNode, fa: TreeNode) -> int:
            depth = 0
            for n in g[son]:
                if n != fa:
                    depth = max(depth, dfs3(n, son) + 1)
            return depth

        return dfs3(start, -1)

        # def bfs() -> int:
        #     q = [start]
        #     step = -1
        #     while q:
        #         new = []
        #         for _ in range(len(q)):
        #             n = q.pop()
        #             for nei in g[n]:
        #                 if nei not in vis:
        #                     new.append(nei)
        #                     vis.add(nei)
        #         q = new
        #         step += 1
        #     return step

        # vis = {start}
        # return bfs()


# 2386 - Find the K-Sum of an Array - HARD
class Solution:
    # 求出非负最大和, 如何处理负数, 简化: 减去正数, 加上负数 操作 -> 取绝对值
    # 类似 dijkstra
    # 思考: 不重不漏生成所有子序列
    # O(nlogn + klogk) / O(n + k)
    def kSum(self, nums: List[int], k: int) -> int:
        total = 0
        for i, v in enumerate(nums):
            if v >= 0:
                total += v
            else:
                nums[i] = -v
        nums.sort()
        h = [(-total, 0)]
        for _ in range(k - 1):
            s, i = heapq.heappop(h)
            if i < len(nums):
                heapq.heappush(h, (s + nums[i], i + 1))
                if i:
                    heapq.heappush(h, (s + nums[i] - nums[i - 1], i + 1))
        return -h[0][0]

    def kSum(self, nums: List[int], k: int) -> int:
        total = sum(max(0, v) for v in nums)
        nums = sorted(abs(v) for v in nums)
        s = 0
        h = [(nums[0], 0)]
        for _ in range(k - 1):
            s, i = heapq.heappop(h)
            if i + 1 < len(nums):
                heapq.heappush(h, (s + nums[i + 1] - nums[i], i + 1))
                heapq.heappush(h, (s + nums[i + 1], i + 1))
        return total - s

    def kSum(self, nums: List[int], k: int) -> int:
        total = 0
        for i, v in enumerate(nums):
            nums[i] = abs(v)
            total += v if v > 0 else 0
        nums.sort()
        h = [(nums[0], 0)]
        s = 0
        while k > 1:
            s, i = heapq.heappop(h)
            if i + 1 < len(nums):
                heapq.heappush(h, (s + nums[i + 1] - nums[i], i + 1))
                heapq.heappush(h, (s + nums[i + 1], i + 1))
            k -= 1
        return total - s

    def kSum(self, nums: List[int], k: int) -> int:
        total = sum(max(0, v) for v in nums)
        nums = sorted(abs(v) for v in nums)
        h = [(-total + nums[0], 0)]
        while k > 1:
            s, i = heapq.heappop(h)
            total = -s
            if i + 1 < len(nums):
                heapq.heappush(h, (s + nums[i + 1] - nums[i], i + 1))
                heapq.heappush(h, (s + nums[i + 1], i + 1))
            k -= 1
        return total

    # O(nlogn + k * (2k * log2k))
    def kSum(self, nums: List[int], k: int) -> int:
        total = sum(x for x in nums if x > 0)
        nums = [abs(x) for x in nums]
        p = [0]
        nums.sort()
        for x in nums[:k]:
            q = [x + y for y in p]
            p = p + q
            p.sort()
            p = p[:k]
        return total - p[k - 1]


# 2389 - Longest Subsequence With Limited Sum - EASY
class Solution:
    # O(nlogn + nm) / O(n)
    def answerQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        nums.sort()
        ans = [0] * len(queries)
        for i, v in enumerate(queries):
            x = 0
            for j in nums:
                if v >= j:
                    v -= j
                    x += 1
                else:
                    break
            ans[i] = x
        return ans

    # O((n+m)logn) / O(n)
    def answerQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        nums.sort()
        p = [0] * (len(nums) + 1)
        for i, v in enumerate(nums):
            p[i + 1] = p[i] + v
        return [bisect.bisect_right(p, q) - 1 for q in queries]

    # O((n+m)logn) / O(1)
    def answerQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        nums.sort()
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        return [bisect.bisect_right(nums, q) for q in queries]


# 2390 - Removing Stars From a String - MEDIUM
class Solution:
    def removeStars(self, s: str) -> str:
        l = list(s)
        dq = collections.deque()
        for i in range(len(s) - 1, -1, -1):
            if s[i] == "*":
                dq.append(i)
            else:
                if dq:
                    p = dq.popleft()
                    l[p] = ""
                    l[i] = ""
        return "".join(l)

    def removeStars(self, s: str) -> str:
        ans = []
        for c in s:
            if c == "*":
                ans.pop()
            else:
                ans.append(c)
        return "".join(ans)


# 2391 - Minimum Amount of Time to Collect Garbage - MEDIUM
class Solution:
    def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
        ans = 0
        m = p = g = 0  # pre location
        for i, v in enumerate(garbage):
            cnt = collections.Counter(v)
            ans += sum(cnt.values())
            if "M" in cnt:
                ans += sum(travel[m:i])
                m = i
            if "G" in cnt:
                ans += sum(travel[g:i])
                g = i
            if "P" in cnt:
                ans += sum(travel[p:i])
                p = i
        return ans

    def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
        ans = 0
        m = p = g = 0
        for i in reversed(range(1, len(garbage))):
            pp = garbage[i].count("P")
            gg = garbage[i].count("G")
            mm = garbage[i].count("M")
            p += pp
            g += gg
            m += mm
            if p != 0:
                p += travel[i - 1]
            if g != 0:
                g += travel[i - 1]
            if m != 0:
                m += travel[i - 1]
        p += garbage[0].count("P")
        g += garbage[0].count("G")
        m += garbage[0].count("M")
        return ans + m + g + p

    def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
        ans = 0
        d = {}  # rightmost position
        for i, g in enumerate(garbage):
            ans += len(g)
            for c in g:
                d[c] = i
        return ans + sum(sum(travel[:v]) for v in d.values())

    def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
        p = [0] * (len(travel) + 1)
        for i, t in enumerate(travel):
            p[i + 1] = p[i] + t
        ans = sum(map(len, garbage))
        for c in "GMP":
            idx = 0
            for i, g in enumerate(garbage):
                idx = i if c in g else idx
            ans += p[idx]
        return ans


# 2392 - Build a Matrix With Conditions - HARD
class Solution:
    # 拓扑排序找环两种方法:
    # 1. 检查入度是否都为 0
    # 2. 检查排序后长度是否为 n, (方便一点)
    def buildMatrix(
        self, k: int, rowConditions: List[List[int]], colConditions: List[List[int]]
    ) -> List[List[int]]:
        # 先排序, 而后将所有 孤立点 加入序列尾部
        def topo_sort(edges: List[List[int]]) -> List[int]:
            e = collections.defaultdict(list)
            ind = [0] * k
            vis = set()
            for x, y in edges:
                x -= 1
                y -= 1
                vis.add(x)
                vis.add(y)
                e[x].append(y)
                ind[y] += 1
            q = [i for i, d in enumerate(ind) if i in vis and d == 0]
            arr = q.copy()
            while q:
                new = []
                for x in q:
                    for y in e[x]:
                        ind[y] -= 1
                        if ind[y] == 0:
                            new.append(y)
                            arr.append(y)
                            # vis.add(y)  # 可以不要, 因为若可排序排序, arr 会和 vis 有相同元素
                q = new

            # 注释掉 any, 改为如下判断, 由此可以证明: 若可排序, arr 会和 vis 有相同元素
            # if len(arr) != len(vis):
            #     return []

            if any(d > 0 for d in ind):
                return []
            for i in range(k):
                if i not in vis:
                    arr.append(i)
            return arr

        row = topo_sort(rowConditions)
        if not row:
            return []
        col = topo_sort(colConditions)
        if not col:
            return []

        r = {v: i for i, v in enumerate(row)}
        c = {v: i for i, v in enumerate(col)}
        g = [[0] * k for _ in range(k)]
        for i in range(k):
            g[r[i]][c[i]] = i + 1
        return g

    def buildMatrix(
        self, k: int, rowConditions: List[List[int]], colConditions: List[List[int]]
    ) -> List[List[int]]:
        # 直接排序, 孤立点 穿插在序列中
        # 孤立点会进入队列一次然后 append 到 arr 中
        def topo_sort(edges: List[List[int]]) -> List[int]:
            e = [[] for _ in range(k)]
            ind = [0] * k
            for x, y in edges:
                x -= 1
                y -= 1
                e[x].append(y)
                ind[y] += 1
            arr = []
            dq = collections.deque(i for i, d in enumerate(ind) if d == 0)
            while dq:
                x = dq.popleft()
                arr.append(x)
                for y in e[x]:
                    ind[y] -= 1
                    if ind[y] == 0:
                        dq.append(y)
            return arr if len(arr) == k else []

        row = topo_sort(rowConditions)
        if not row:
            return []
        col = topo_sort(colConditions)
        if not col:
            return []

        r = {v: i for i, v in enumerate(row)}
        c = {v: i for i, v in enumerate(col)}
        g = [[0] * k for _ in range(k)]
        for i in range(k):
            g[r[i]][c[i]] = i + 1
        return g

    def buildMatrix(
        self, k: int, rowConditions: List[List[int]], colConditions: List[List[int]]
    ) -> List[List[int]]:
        # dfs排序:
        # 1. 如下, 找一个最优结果, 然后验证
        # 2. vis 打标记, dfs 中已 vis 则退出
        def topo_sort(edges: List[List[int]]) -> List[int]:
            e = [[] for _ in range(k + 1)]
            for x, y in edges:
                e[x].append(y)
            vis = [False] * (k + 1)
            arr = []

            def dfs(x: int) -> None:
                vis[x] = True
                for y in e[x]:
                    if not vis[y]:
                        dfs(y)
                arr.append(x)
                return

            for i in range(1, k + 1):
                if not vis[i]:
                    dfs(i)
            arr.reverse()
            d = {x: i for i, x in enumerate(arr)}
            if all(d[x] < d[y] for x, y in edges):
                return arr
            return []

        row = topo_sort(rowConditions)
        if not row:
            return []
        col = topo_sort(colConditions)
        if not col:
            return []
        g = [[0] * k for _ in range(k)]
        r = {v: i for i, v in enumerate(row)}
        c = {v: i for i, v in enumerate(col)}
        for i in range(1, k + 1):
            g[r[i]][c[i]] = i
        return g

    def buildMatrix(
        self, k: int, rowConditions: List[List[int]], colConditions: List[List[int]]
    ) -> List[List[int]]:
        def topo_sort(edges: List[List[int]]) -> List[int]:
            e = [[] for _ in range(k)]
            for x, y in edges:
                e[x - 1].append(y - 1)
            arr = []
            vis = [0] * k
            q = list(range(k))
            while q:
                n = q.pop()
                if n < 0:
                    arr.append(~n)
                elif not vis[n]:
                    vis[n] = 1
                    q.append(~n)
                    q += e[n]
            # cycle check
            for x in arr:
                if any(vis[y] for y in e[x]):
                    return []
                vis[x] = 0
            return arr[::-1]

        row = topo_sort(rowConditions)
        if not row:
            return []
        col = topo_sort(colConditions)
        if not col:
            return []

        r = {v: i for i, v in enumerate(row)}
        c = {v: i for i, v in enumerate(col)}
        g = [[0] * k for _ in range(k)]
        for i in range(k):
            g[r[i]][c[i]] = i + 1
        return g


# 2395 - Find Subarrays With Equal Sum - EASY
class Solution:
    def findSubarrays(self, nums: List[int]) -> bool:
        s = set()
        for i in range(len(nums) - 1):
            summ = nums[i] + nums[i + 1]
            if summ in s:
                return True
            s.add(summ)
        return False


# 2396 - Strictly Palindromic Number - MEDIUM
class Solution:
    # 对于所有 n > 4, 它们的 n - 1 进制表示是 11,  n - 2 进制表示都是 12 (已经不是回文)
    def isStrictlyPalindromic(self, n: int) -> bool:
        return False


# 2397 - Maximum Rows Covered by Columns - MEDIUM
class Solution:
    # O(2 ** n * (mn)) / O(1)
    def maximumRows(self, mat: List[List[int]], cols: int) -> int:
        def to_num(row):
            a = 0
            for v in row:
                a = (a << 1) + v
            return a

        ans = 0
        nums = [to_num(r) for r in mat]
        for comb in itertools.combinations(range(len(mat[0])), cols):
            mask = sum(1 << c for c in comb)
            ans = max(ans, sum((mask & v) == v for v in nums))
        return ans

    def maximumRows(self, mat: List[List[int]], cols: int) -> int:
        m = len(mat)
        n = len(mat[0])
        ans = 0
        b = [0] * m
        for i, r in enumerate(mat):
            for j, v in enumerate(r):
                b[i] |= v << j

        def calc(sub: int) -> int:
            cnt = 0
            for v in b:
                if sub & v == v:
                    cnt += 1
            return cnt

        for sub in range(1 << n):
            if bin(sub).count("1") != cols:
                continue
            ans = max(ans, calc(sub))
        return ans

    def maximumRows(self, mat: List[List[int]], cols: int) -> int:
        m = len(mat)
        n = len(mat[0])
        ans = 0
        for mask in range(1 << n):
            if bin(mask).count("1") != cols:
                continue
            cnt = 0
            for i in range(m):
                ok = 1
                for j in range(n):
                    if mat[i][j] == 1 and not ((mask >> j) & 1):
                        ok = 0
                        break
                cnt += ok
            ans = max(ans, cnt)
        return ans

    def maximumRows(self, mat: List[List[int]], cols: int) -> int:
        ans = 0
        mask = [sum(v << j for j, v in enumerate(r)) for r in mat]
        for s in range(1 << len(mat[0])):
            if bin(s).count("1") == cols:  # s.bit_count()
                ans = max(ans, sum(r & s == r for r in mask))
        return ans

    # O(m * comb(n, cols)) / O(m), 优化掉上述的无效枚举, 直接枚举所有包含 cols 个 1 的集合
    def maximumRows(self, mat: List[List[int]], cols: int) -> int:
        ans = 0
        mask = [sum(v << j for j, v in enumerate(r)) for r in mat]
        s = (1 << cols) - 1
        while s < 1 << len(mat[0]):
            # r & s = r 表示 r 是 s 的子集，所有 1 都被覆盖
            ans = max(ans, sum(r & s == r for r in mask))
            lb = s & -s  # low bit
            x = s + lb
            s = (s ^ x) // lb >> 2 | x
        return ans


# 2398 - Maximum Number of Robots Within Budget - HARD
class Solution:
    # O(nlogn) / O(n)
    def maximumRobots(
        self, chargeTimes: List[int], runningCosts: List[int], budget: int
    ) -> int:
        n = len(chargeTimes)
        pre = [0]
        for i in range(n):
            pre.append(pre[-1] + runningCosts[i])
        ans = l = summ = 0
        mx = []
        for r in range(n):
            heapq.heappush(mx, (-chargeTimes[r], r))
            summ += runningCosts[r] * (r - l) + (pre[r + 1] - pre[l])
            while mx and summ - mx[0][0] > budget:
                summ -= runningCosts[l] * (r - l) + (pre[r + 1] - pre[l])
                l += 1
                if mx and mx[0][1] < l:
                    heapq.heappop(mx)
            ans = max(ans, r - l + 1)
        return ans

    def maximumRobots(
        self, chargeTimes: List[int], runningCosts: List[int], budget: int
    ) -> int:
        def get_mx(pq: List[int], i: int) -> int:
            while pq and pq[0][1] <= i:
                heapq.heappop(pq)
            return -pq[0][0] if pq else 0

        ans = summ = 0
        l = -1
        pq = []
        for r in range(len(runningCosts)):
            summ += runningCosts[r]
            heapq.heappush(pq, (-chargeTimes[r], r))
            while summ * (r - l) + get_mx(pq, l) > budget:
                l += 1
                summ -= runningCosts[l]
            ans = max(ans, r - l)
        return ans

    # O(n) / O(n)
    def maximumRobots(
        self, chargeTimes: List[int], runningCosts: List[int], budget: int
    ) -> int:
        n = len(chargeTimes)
        cur = l = 0
        dq = collections.deque()
        for r in range(n):
            cur += runningCosts[r]
            while dq and chargeTimes[dq[-1]] < chargeTimes[r]:
                dq.pop()
            dq.append(r)
            if chargeTimes[dq[0]] + (r - l + 1) * cur > budget:
                if dq[0] == l:
                    dq.popleft()
                cur -= runningCosts[l]
                l += 1
        return n - l

    # 单调队列 / 单调栈原理: 及时弹出无用数据
    def maximumRobots(
        self, chargeTimes: List[int], runningCosts: List[int], budget: int
    ) -> int:
        ans = s = l = 0
        dq = collections.deque()
        for r, (t, c) in enumerate(zip(chargeTimes, runningCosts)):
            while dq and t >= chargeTimes[dq[-1]]:
                dq.pop()
            dq.append(r)
            s += c
            while dq and chargeTimes[dq[0]] + (r - l + 1) * s > budget:
                if dq[0] == l:
                    dq.popleft()
                s -= runningCosts[l]
                l += 1
            ans = max(ans, r - l + 1)
        return ans

    # 如果题目改为子序列该怎么做?
    def maximumRobotsSubseq(
        self, chargeTimes: List[int], runningCosts: List[int], budget: int
    ) -> int:
        ans = summ = 0
        hp = []
        # 按照限制排序: max(chargeTimes), 左边都是可选的
        for t, c in sorted(zip(chargeTimes, runningCosts)):
            heapq.heappush(hp, -c)
            summ += c
            while hp and t + len(hp) * summ > budget:
                # 弹出一个最大花费, 即使弹出的是当前的 c 也没关系，这不会得到更大的 ans
                summ += heapq.heappop(hp)
            ans = max(ans, len(hp))
        return ans


# 2399 - Check Distances Between Same Letters - EASY
class Solution:
    def checkDistances(self, s: str, distance: List[int]) -> bool:
        pre = dict()
        for i in range(len(s)):
            if s[i] in pre:
                if i - pre[s[i]] - 1 != distance[ord(s[i]) - ord("a")]:
                    return False
            else:
                pre[s[i]] = i
        return True

    def checkDistances(self, s: str, distance: List[int]) -> bool:
        vis = [0] * 26
        for i, ch in enumerate(s):
            c = ord(ch) - ord("a")
            if vis[c] and i - vis[c] != distance[c]:
                return False
            vis[c] = i + 1
        return True
