import bisect, collections, functools, math, itertools, heapq, string, operator
from typing import List, Optional

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

        def find(x: int):
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
        for i in range(n):
            for j in range(n):
                if i == j or i + j == n - 1:
                    if grid[i][j] == 0:
                        return False
                else:
                    if grid[i][j] != 0:
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
