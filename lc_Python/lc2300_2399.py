import bisect, collections, functools, math, itertools, heapq
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
