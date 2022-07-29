import bisect, collections, functools, math, itertools, heapq
from typing import List, Optional, Tuple
import sortedcontainers


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 2200 - Find All K-Distant Indices in an Array - EASY
class Solution:
    def findKDistantIndices(self, nums: List[int], key: int, k: int) -> List[int]:
        ans = []
        n = len(nums)
        for i in range(n):
            for j in range(i - k, i + k + 1):
                if 0 <= j < n and nums[j] == key:
                    ans.append(i)
                    break
        return ans


# 2201 - Count Artifacts That Can Be Extracted - MEDIUM
class Solution:
    def digArtifacts(
        self, n: int, artifacts: List[List[int]], dig: List[List[int]]
    ) -> int:
        s = set((i, j) for i, j in dig)
        ans = 0
        for r1, c1, r2, c2 in artifacts:
            have = True
            f = False
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    if (r, c) not in s:
                        have = False
                        f = True
                        break
                if f:
                    break
            if have:
                ans += 1
        return ans


# 2202 - Maximize the Topmost Element After K Moves - MEDIUM
class Solution:
    def maximumTop(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 1 or k == 0:
            if k & 1:
                return -1
            else:
                return nums[0]

        f = max(nums[: k - 1]) if k > 1 else 0
        s = nums[k] if k < n else 0
        return max(f, s)


# 2203 - Minimum Weighted Subgraph With the Required Paths - HARD
class Solution:
    def minimumWeight(
        self, n: int, edges: List[List[int]], src1: int, src2: int, dest: int
    ) -> int:
        def dijkstra(g: List[List[tuple]], start: int) -> List[int]:
            dis = [math.inf] * n
            dis[start] = 0
            pq = [(0, start)]
            while pq:
                d, x = heapq.heappop(pq)
                if dis[x] < d:
                    continue
                for y, wt in g[x]:
                    new_d = dis[x] + wt
                    if new_d < dis[y]:
                        dis[y] = new_d
                        heapq.heappush(pq, (new_d, y))
            return dis

        g = [[] for _ in range(n)]
        rg = [[] for _ in range(n)]
        for x, y, wt in edges:
            g[x].append((y, wt))
            rg[y].append((x, wt))

        d1 = dijkstra(g, src1)
        d2 = dijkstra(g, src2)
        d3 = dijkstra(rg, dest)

        ans = min(sum(d) for d in zip(d1, d2, d3))
        return ans if ans < math.inf else -1

    def minimumWeight(
        self, n: int, edges: List[List[int]], src1: int, src2: int, dest: int
    ) -> int:
        g = collections.defaultdict(list)
        reverse_g = collections.defaultdict(list)
        for i, j, w in edges:
            g[i].append((j, w))
            reverse_g[j].append((i, w))

        def dijkstra(src: int, G: collections.defaultdict):
            dis = [math.inf] * n
            pq = [(0, src)]
            while pq:
                w, node = heapq.heappop(pq)
                if (
                    dis[node] <= w
                ):  # see the different symbols between here and solution above
                    continue
                dis[node] = w
                for nxt, wt in G[node]:
                    if w + wt < dis[nxt]:  # and here
                        heapq.heappush(pq, (w + wt, nxt))
            return dis

        l1 = dijkstra(src1, g)
        l2 = dijkstra(src2, g)
        l3 = dijkstra(dest, reverse_g)
        ans = math.inf
        for i in range(n):
            ans = min(ans, l1[i] + l2[i] + l3[i])
        return ans if ans != math.inf else -1

    def minimumWeight(
        self, n: int, edges: List[List[int]], src1: int, src2: int, dest: int
    ) -> int:
        G1 = collections.defaultdict(list)
        G2 = collections.defaultdict(list)
        for a, b, w in edges:
            G1[a].append((b, w))
            G2[b].append((a, w))

        def dijkstra(graph: collections.defaultdict, src: int):
            pq = [(0, src)]
            t = {}
            while pq:
                time, node = heapq.heappop(pq)
                if node not in t:
                    t[node] = time
                    for v, w in graph[node]:
                        heapq.heappush(pq, (time + w, v))
            return [t.get(i, float("inf")) for i in range(n)]

        arr1 = dijkstra(G1, src1)
        arr2 = dijkstra(G1, src2)
        arr3 = dijkstra(G2, dest)

        ans = float("inf")
        for i in range(n):
            ans = min(ans, arr1[i] + arr2[i] + arr3[i])
        return ans if ans != float("inf") else -1


# 2206 - Divide Array Into Equal Pairs - EASY
class Solution:
    def divideArray(self, nums: List[int]) -> bool:
        # cnt = collections.Counter(nums)
        # for n in cnt:
        #     if cnt[n] & 1:
        #         return False
        # return True
        return all(v % 2 == 0 for _, v in collections.Counter(nums).items())
        return not any(v & 1 for _, v in collections.Counter(nums).items())


# 2207 - Maximize Number of Subsequences in a String - MEDIUM
class Solution:
    def maximumSubsequenceCount(self, t: str, p: str) -> int:
        a = ans = 0
        b = t.count(p[1])
        if p[1] == p[0]:
            ans = b * (b + 1) // 2
        else:
            for ch in t:
                if ch == p[0]:
                    ans += b
                    a += 1
                elif ch == p[1]:
                    # ans += a
                    b -= 1
            # ans //= 2
            ans += max(a, t.count(p[1]))
        return ans

    # O(n) / O(1)
    def maximumSubsequenceCount(self, t: str, p: str) -> int:
        ans = c1 = c2 = 0
        for ch in t:
            if ch == p[1]:
                ans += c1
                c2 += 1
            if ch == p[0]:  # two 'if' to handle the case where p[0] == p[1]
                c1 += 1
        return ans + max(c1, c2)


# 2208 - Minimum Operations to Halve Array Sum -  MEDIUM
class Solution:
    # O(n * logn + m * logn) / O(n), where m is the number of operations
    def halveArray(self, nums: List[int]) -> int:
        t = sum(nums)
        half = t / 2
        hp = [-n for n in nums]
        heapq.heapify(hp)  # nlogn
        ans = 0
        while t > half:
            n = -heapq.heappop(hp) / 2
            t -= n
            heapq.heappush(hp, -n)
            ans += 1
        return ans


# 2209 - Minimum White Tiles After Covering With Carpets - HARD
class Solution:
    # dp[i][j], means that consider the previous floor with length 'j',
    # 'i' carpets are used, the minimum remaining white tiles.
    # i == 0, not use carpet, dp[0][j] is the number of white tiles before index j
    # i != 0, dp[i][j] = min(dp[i][j-1] + isWhite[j], dp[i-1][j-carpetLen])
    # dp[i][j-1] + isWhite[j]: not use carpet in 'floor[j]'
    # dp[i-1][j-carpetLen]: use carpet in 'floor[j]'
    # O(nm) / O(nm), n = len(floor), m = numCarpets
    def minimumWhiteTiles(self, floor: str, ncp: int, l: int) -> int:
        n = len(floor)
        dp = [[0] * n for _ in range(ncp + 1)]
        dp[0][0] = 1 if floor[0] == "1" else 0
        isWhite = [0] * n
        for i in range(1, n):
            dp[0][i] = dp[0][i - 1]
            if floor[i] == "1":
                dp[0][i] += 1
                isWhite[i] = 1
        for i in range(1, ncp + 1):
            for j in range(n):
                # less than 'carpetLen' bricks will end up with 0 white bricks left after using the carpet
                if j < l:
                    dp[i][j] = 0
                else:
                    dp[i][j] = min(dp[i][j - 1] + isWhite[j], dp[i - 1][j - l])
        return dp[ncp][n - 1]

    def minimumWhiteTiles(self, floor: str, numCarpets: int, carpetLen: int) -> int:
        # define dp[i][numCarpet]
        # choose use or not use
        # if use: dp[i][use] = dp[i+carpetLen][use-1]
        @functools.lru_cache(None)
        def dfs(i, num):
            if i >= len(floor):
                return 0
            res = float("inf")
            # use
            if num:
                res = dfs(i + carpetLen, num - 1)
            # not use
            res = min(res, (floor[i] == "1") + dfs(i + 1, num))
            return res

        return dfs(0, numCarpets)

    def minimumWhiteTiles(self, floor: str, numCarpets: int, carpetLen: int) -> int:
        pre = [0]
        n = len(floor)
        for i in range(n):
            if floor[i] == "1":
                pre.append(pre[-1] + 1)
            else:
                pre.append(pre[-1])

        @functools.lru_cache(None)
        def dp(i, j):
            if i < 0:
                return 0
            if j == 0:
                return pre[i + 1]
            return min(
                dp(i - 1, j) + (1 if floor[i] == "1" else 0), dp(i - carpetLen, j - 1)
            )

        return dp(n - 1, numCarpets)


# 2210 - Count Hills and Valleys in an Array - EASY
class Solution:
    def countHillValley(self, nums: List[int]) -> int:
        nums.append(math.inf)
        a = []
        for i in range(len(nums) - 1):
            if nums[i] != nums[i + 1]:
                a.append(nums[i])
        ans = 0
        for i in range(1, len(a) - 1):
            if a[i - 1] > a[i] < a[i + 1] or a[i - 1] < a[i] > a[i + 1]:
                ans += 1
        return ans


# 2211 - Count Collisions on a Road - MEDIUM
class Solution:
    def countCollisions(self, directions: str) -> int:
        s = []
        co = 0
        for ch in directions:
            if ch == "L":
                if not s:
                    continue
                elif s[-1] == "S":
                    co += 1
                elif s[-1] == "R":
                    while s and s[-1] == "R":
                        s.pop()
                        co += 1
                    co += 1
                    s.append("S")
            elif ch == "R":
                s.append(ch)
            else:
                while s and s[-1] == "R":
                    s.pop()
                    co += 1
                s.append("S")
        return co

    # All the cars that move to the middle will eventually collide
    def countCollisions(self, directions: str) -> int:
        return sum(d != "S" for d in directions.lstrip("L").rstrip("R"))


# 2212 - Maximum Points in an Archery Competition - MEDIUM
# TODO
class Solution:
    # bitmasking. Enumerate the regions on which Bob wins,
    # which has a total of 2^12 different cases
    # O(2 ^ 12 * 12) / O(C)
    def maximumBobPoints(self, num: int, ali: List[int]) -> List[int]:
        ans = []
        mx = 0
        for i in range(1 << len(ali)):
            score, arrow, bob = 0, 0, [0] * 12
            for j in range(len(ali)):
                # if i & (1 << j):
                if i >> j & 1 == 1:
                    score += j
                    arrow += ali[j] + 1
                    bob[j] = ali[j] + 1
            if arrow > num:
                continue
            if score > mx:
                mx = score
                bob[0] += num - arrow  # has remaining arrow
                ans = bob
        return ans

    # O(2 * 12 * numArrows) / O(12 * numArrows)
    # There are total 12 * numArrows states, each state need at most 2 case (Lose or Win) to compute
    def maximumBobPoints(self, numArrows: int, ali: List[int]) -> List[int]:
        @functools.lru_cache(None)
        def dp(k, numArrows):
            if k == 12 or numArrows <= 0:
                return 0
            mx = dp(k + 1, numArrows)  # Bob Lose
            if numArrows > ali[k]:
                mx = max(mx, dp(k + 1, numArrows - ali[k] - 1) + k)  # Bob Win
            return mx

        # backtracking
        ans = [0] * 12
        remain = numArrows
        for k in range(12):
            # It means that section k was chosen where bob wins.
            # If dp(k, numArrows) == dp(k+1, numArrows),
            # then that would mean that maxScore didn't change
            # and hence bob didn't win at section k.
            # Else, it would mean that the maxScore changed,
            # implying that bob won at section k
            if dp(k, numArrows) != dp(k + 1, numArrows):  # If Bob win
                ans[k] = ali[k] + 1
                numArrows -= ans[k]
                remain -= ans[k]

        ans[
            0
        ] += remain  # In case of having remain arrows then it means in all sections Bob always win
        # then we can distribute the remain to any section, here we simple choose first section.
        return ans

    def maximumBobPoints(self, numArrows: int, ali: List[int]) -> List[int]:
        ans = 0
        plan = [0] * 10

        def search(i, arrows, score, cur_plan):
            nonlocal ans, plan
            if i == len(ali):
                if score > ans:
                    ans = score
                    plan = cur_plan[:]
                return
            if ali[i] + 1 <= arrows:
                cur_plan.append(ali[i] + 1)
                search(i + 1, arrows - ali[i] - 1, score + i, cur_plan)
                cur_plan.pop()
            cur_plan.append(0)
            search(i + 1, arrows, score, cur_plan)
            cur_plan.pop()

        search(1, numArrows, 0, [])
        return [numArrows - sum(plan)] + plan

    # TLE, knapsack problem with path reduction
    def maximumBobPoints(self, numArrows: int, ali: List[int]) -> List[int]:
        f = [[0] * (numArrows + 1) for _ in range(12)]
        ans = [0] * 12
        for i in range(1, 12):
            a = ali[i]
            for j in range(1, numArrows + 1):
                if j < a + 1:
                    f[i][j] = f[i - 1][j]
                else:
                    f[i][j] = max(f[i - 1][j - a - 1] + i, f[i - 1][j])
        for i in range(11, 0, -1):
            if f[i][numArrows] > f[i - 1][numArrows]:
                ans[i] = ali[i] + 1
                numArrows -= ali[i] + 1
        ans[0] = numArrows
        return ans


# 2215 - Find the Difference of Two Arrays - EASY
class Solution:
    def findDifference(self, nums1, nums2):
        s1 = set(nums1)
        s2 = set(nums2)
        a = set()
        b = set()
        for n in nums1:
            if n not in s2:
                a.add(n)
        for n in nums2:
            if n not in s1:
                b.add(n)
        return [list(a), list(b)]

    def findDifference(self, nums1, nums2):
        s1, s2 = set(nums1), set(nums2)
        return [list(s1 - s2), list(s2 - s1)]

    def findDifference(self, nums1, nums2):
        return [list((s1 := set(nums1)) - (s2 := set(nums2))), list(s2 - s1)]


# 2216 - Minimum Deletions to Make Array Beautiful - MEDIUM
class Solution:
    # O(n) / O(n)
    # using the stack to simulate the process.
    # if the stack size is even, can add any value
    # if the stack size is odd, can not add the value the same as the top of stack
    # no need for a stack, use a variable to represent the parity of the stack
    def minDeletion(self, nums: List[int]) -> int:
        a = []
        for n in nums:
            if len(a) % 2 == 0 or n != a[-1]:
                a.append(n)
        return len(nums) - (len(a) - len(a) % 2)

    def minDeletion(self, a: List[int]) -> int:
        b = []
        for n in a:
            if len(b) % 2 == 1 and b[-1] == n:
                b.pop()
            b.append(n)
        if len(b) % 2 == 1:
            b.pop()
        return len(a) - len(b)

    def minDeletion(self, nums: List[int]) -> int:
        ans = []
        for n in nums:
            if len(ans) % 2 == 0 or ans[-1] != n:
                ans.append(n)
        if len(ans) % 2 == 1:
            ans.pop()
        return len(nums) - len(ans)

    # O(n) / O(1), greedy
    def minDeletion(self, nums: List[int]) -> int:
        flag = 0
        ans = 0
        for i in range(len(nums) - 1):
            if i % 2 == flag and nums[i] == nums[i + 1]:
                ans += 1
                flag = 1 - flag
        if (len(nums) - ans) % 2 == 1:
            ans += 1
        return ans

    # if the number can be the second of the pair, keep it
    # skip each pair
    def minDeletion(self, nums: List[int]) -> int:
        ans = i = 0
        while i < len(nums) - 1:
            if nums[i] == nums[i + 1]:
                ans += 1
            else:
                i += 1
            i += 1
        if (len(nums) - ans) % 2:
            ans += 1
        return ans

    # using a variable 'pre' to record the last element with even index.
    def minDeletion(self, nums: List[int]) -> int:
        ans = 0
        pre = -1
        for n in nums:
            if n == pre:
                ans += 1
            else:
                pre = n if pre < 0 else -1
        return ans + (pre >= 0)

    def minDeletion(self, nums: List[int]) -> int:
        ans = 0
        l = None
        for n in nums:
            if l is None:
                l = n
            elif l != n:
                l = None
                ans += 2
        return len(nums) - ans


# 2217 - Find Palindrome With Fixed Length - MEDIUM
class Solution:
    # O(n * L) / O(n * L)
    def kthPalindrome(self, queries: List[int], intLength: int) -> List[int]:
        base = 10 ** ((intLength - 1) // 2)
        ans = [-1] * len(queries)
        for i, q in enumerate(queries):
            if q <= 9 * base:
                s = str(base + q - 1)
                s += s[-2::-1] if intLength % 2 else s[::-1]
                ans[i] = int(s)
        return ans

    def kthPalindrome(self, queries: List[int], l: int) -> List[int]:
        base = 10 ** ((l - 1) // 2)
        ans = [q - 1 + base for q in queries]
        for i, a in enumerate(ans):
            a = str(a) + str(a)[-1 - l % 2 :: -1]
            ans[i] = int(a) if len(a) == l else -1
        return ans


# 2220 - Minimum Bit Flips to Convert Number - EASY
class Solution:
    def minBitFlips(self, start: int, goal: int) -> int:
        s = bin(start)[2:]  # bin() -> O(logn)
        g = bin(goal)[2:]
        if len(s) > len(g):
            g = "0" * (len(s) - len(g)) + g
        if len(s) < len(g):
            s = "0" * (len(g) - len(s)) + s
        ans = 0
        for i in range(len(s)):
            if s[i] != g[i]:
                ans += 1
        return ans

    # O(logM) / O(1), M = max(start, goal)
    def minBitFlips(self, start: int, goal: int) -> int:
        ans = 0
        xor = start ^ goal
        while xor:
            ans += xor & 1
            xor >>= 1
        return ans

    def minBitFlips(self, start: int, goal: int) -> int:
        ans = 0
        xor = start ^ goal
        while xor:
            ans += 1
            xor &= xor - 1
        return ans

    # python3.10: int.bit_count()
    def minBitFlips(self, start: int, goal: int) -> int:
        return (start ^ goal).bit_count()


# 2221. Find Triangular Sum of an Array - MEDIUM
class Solution:
    # O(n ^ 2) / O(1), in place
    def triangularSum(self, nums: List[int]) -> int:
        n = len(nums)
        while n > 1:
            for i in range(n - 1):
                nums[i] = (nums[i] + nums[i + 1]) % 10
            n -= 1
        return nums[0]


# 2222. Number of Ways to Select Buildings - MEDIUM
class Solution:
    def numberOfWays(self, s: str) -> int:
        ans = n0 = n1 = n01 = n10 = 0
        for ch in s:
            if ch == "1":
                n10 += n0
                ans += n01
                n1 += 1
            else:
                n01 += n1
                ans += n10
                n0 += 1
        return ans

    def numberOfWays(self, s: str) -> int:
        ans = n0 = 0
        t0 = s.count("0")
        for i, ch in enumerate(s):
            if ch == "1":
                ans += n0 * (t0 - n0)  # (left '0') * (right '0')
            else:
                n1 = i - n0
                ans += n1 * (len(s) - t0 - n1)  # (left '1') * (right '1')
                n0 += 1
        return ans

    # eg: 101, c: '101', b = '10', a = '1'
    def numberOfWays(self, s: str) -> int:
        def f(t: str) -> int:
            a = b = c = 0
            for ch in s:
                if ch == t[2]:
                    c += b
                if ch == t[1]:
                    b += a
                if ch == t[0]:
                    a += 1
            return c

        return f("101") + f("010")


# 2223 - Sum of Scores of Built Strings - HARD
class Solution:
    def sumScores(self, s: str) -> int:
        visited = set()
        res = 0
        for start in range(1, len(s)):
            if start in visited:
                continue
            count = 0
            for i in range(start, len(s)):
                if s[i] != s[i - start]:
                    break
                count += 1
            res += count
            for k in range(2 * start, count + 1, start):
                res += count - (k - start)
                visited.add(k)
        return res + len(s)

    # https://oi-wiki.org/string/z-func/
    def sumScores(self, s: str) -> int:
        n = len(s)
        z = [0] * n
        ans, l, r = n, 0, 0
        for i in range(1, n):
            z[i] = max(min(z[i - l], r - i + 1), 0)
            while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                l, r = i, i + z[i]
                z[i] += 1
            ans += z[i]
        return ans

    def sumScores(self, s: str) -> int:
        a = p = 0
        n = len(s)
        nxt = [0] * n
        nxt[0] = n
        T = s
        for i in range(1, n):
            if i >= p or i + nxt[i - a] >= p:
                if i > p:
                    p = i
                while p < n and T[p] == T[p - i]:
                    p += 1
                nxt[i] = p - i
                a = i
            else:
                nxt[i] = nxt[i - a]
        return sum(nxt)

    def sumScores(self, s: str) -> int:
        def z_function(S):
            # https://github.com/cheran-senthil/PyRival/blob/master/pyrival/strings/z_algorithm.py
            # https://cp-algorithms.com/string/z-function.html
            n = len(S)
            Z = [0] * n
            l = r = 0
            for i in range(1, n):
                z = Z[i - l]
                if i + z >= r:
                    z = max(r - i, 0)
                    while i + z < n and S[z] == S[i + z]:
                        z += 1
                    l, r = i, i + z
                Z[i] = z
            Z[0] = n
            return Z

        lst = z_function(s)
        return sum(lst)


# 2224 - Minimum Number of Operations to Convert Time - EASY
class Solution:
    def convertTime(self, a: str, b: str) -> int:
        c = int(a[:2]) * 60 + int(a[3:])
        d = int(b[:2]) * 60 + int(b[3:])
        e = d - c
        ans = 0
        for i in [60, 15, 5, 1]:
            ans += e // i
            e %= i
        return ans


# 2225 - Find Players With Zero or One Losses - MEDIUM
class Solution:
    # O(nlogn) / O(n)
    def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
        win = set()
        lose = collections.defaultdict(int)
        for w, l in matches:
            win.add(w)
            lose[l] += 1
        aw = []
        al = []
        for w in win:
            if lose[w] == 0:
                aw.append(w)
        for l in lose:
            if lose[l] == 1:
                al.append(l)
        return [sorted(aw), sorted(al)]

    def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
        freq = collections.defaultdict(int)
        for w, l in matches:
            if w not in freq:
                freq[w] = 0
            freq[l] += 1
        ans = [[], []]
        for k, v in freq.items():
            if v < 2:
                ans[v].append(k)
        ans[0].sort()
        ans[1].sort()
        return ans


# 2226 - Maximum Candies Allocated to K Children - MEDIUM
class Solution:
    def maximumCandies(self, candies: List[int], k: int) -> int:
        def check(i: int) -> bool:
            res = 0
            for c in candies:
                res += c // i
            return res >= k

        l = 1
        r = max(candies) + 1
        while l < r:
            mid = l + (r - l) // 2
            if check(mid):
                l = mid + 1
            else:
                r = mid
        return l - 1

    def maximumCandies(self, candies: List[int], k: int) -> int:
        fn = lambda t: sum(x // t for x in candies) < k
        return bisect.bisect_left(range(1, max(candies) + 1), True, key=fn)

    def maximumCandies(self, candies: List[int], k: int) -> int:
        fn = lambda x: -sum(v // (x + 1) for v in candies)
        return bisect.bisect_right(range(sum(candies) // k), -k, key=fn)


# 2227 - Encrypt and Decrypt Strings - HARD
class Encrypter:
    def __init__(self, keys: List[str], values: List[str], dictionary: List[str]):
        self.k2v = {k: v for k, v in zip(keys, values)}
        self.cnt = collections.Counter(self.encrypt(d) for d in dictionary)

    def encrypt(self, word1: str) -> str:
        ans = ""
        for c in word1:
            if c not in self.k2v:
                return ""
            ans += self.k2v[c]
        return ans

    def decrypt(self, word2: str) -> int:
        return self.cnt[word2]


class Encrypter:
    def __init__(self, keys: List[str], values: List[str], dictionary: List[str]):
        self.k2v = {k: v for k, v in zip(keys, values)}
        self.v2k = v2k = collections.defaultdict(list)
        for v, k in zip(values, keys):
            v2k[v].append(k)
        self.r = dict()
        for d in dictionary:
            r = self.r
            for c in d:
                if c not in r:
                    r[c] = dict()
                r = r[c]
            r["end"] = 1

    def encrypt(self, word1: str) -> str:
        return "".join(self.k2v[i] for i in word1)

    def decrypt(self, word2: str) -> int:
        dq = collections.deque([self.r])
        for i in range(0, len(word2), 2):
            w = word2[i : i + 2]
            if w not in self.v2k:
                return 0
            for _ in range(len(dq)):
                r = dq.popleft()
                for c in self.v2k[w]:
                    if c in r:
                        dq.append(r[c])
        return sum("end" in r for r in dq)


# 2231 - Largest Number After Digit Swaps by Parity - EASY
class Solution:
    # do not need to care about specific indices
    def largestInteger(self, num: int):
        arr = [int(i) for i in str(num)]
        odd = []
        even = []
        for i in arr:
            if i % 2 == 0:
                even.append(i)
            else:
                odd.append(i)
        odd.sort()
        even.sort()
        ans = 0
        for i in range(len(str(num))):
            if arr[i] % 2 == 0:
                ans = ans * 10 + even.pop()
            else:
                ans = ans * 10 + odd.pop()
        return ans

    def largestInteger(self, num: int) -> int:
        s = str(num)
        o, e = [], []
        for ch in s:
            if int(ch) % 2:
                o.append(ch)
            else:
                e.append(ch)
        o.sort()
        e.sort()
        ss = ""
        for ch in s:
            if int(ch) % 2:
                ss += o.pop()
            else:
                ss += e.pop()
        return int(ss)


# 2232 - Minimize Result by Adding Parentheses to Expression - MEDIUM
class Solution:
    def minimizeResult(self, expression: str) -> str:
        n1, n2 = expression.split("+")
        m = 1e99
        ans = None
        for i in range(len(n1)):
            a = 1 if i == 0 else int(n1[:i])
            s1 = str(a) if i != 0 else ""
            b = int(n1[i:])
            for j in range(len(n2)):
                c = int(n2[: j + 1])
                d = 1 if j == len(n2) - 1 else int(n2[j + 1 :])
                s2 = str(d) if j != len(n2) - 1 else ""
                p = a * (b + c) * d
                if p < m:
                    m = p
                    ans = "%s(%d+%d)%s" % (s1, b, c, s2)
        return ans

    def minimizeResult(self, expression: str) -> str:
        a, b = expression.split("+")
        m = math.inf
        ans = None
        for i in range(len(a)):
            for j in range(1, len(b) + 1):
                s = "" if i == 0 else a[:i] + "*"
                s += "("
                s += a[i:]
                s += "+"
                s += b[:j]
                s += ")"
                if j != len(b):
                    s += "*" + b[j:]
                cur = eval(s)
                if cur < m:
                    m = cur
                    ans = s
        return ans.replace("*", "")


# 2233 - Maximum Product After K Increments - MEDIUM
class Solution:
    def maximumProduct(self, nums: List[int], k: int) -> int:
        cnt = collections.Counter(nums)
        pq = [(n, v) for n, v in cnt.items()]
        heapq.heapify(pq)
        while k:
            if k >= pq[0][1]:
                n, v = heapq.heappop(pq)
                k -= v
                n += 1
                if pq and pq[0][0] == n:
                    _, vv = heapq.heappop(pq)
                    heapq.heappush(pq, (n, v + vv))
                else:
                    heapq.heappush(pq, (n, v))
            else:
                n, v = heapq.heappop(pq)
                if pq and pq[0][0] == n + 1:
                    nn, vv = heapq.heappop(pq)
                    pq.append((nn, vv + k))
                    pq.append((n, v - k))
                else:
                    pq.append((n + 1, k))
                    pq.append((n, v - k))
                break
        ans = 1
        for i in range(len(pq)):
            ans *= pq[i][0] ** pq[i][1]
        return ans % (10**9 + 7)

    def maximumProduct(self, nums: List[int], k: int) -> int:
        cnt = collections.Counter(nums)
        keys = sorted(list(cnt.keys()))
        i = keys[0]
        while k > 0:
            if k > cnt[i]:
                k -= cnt[i]
                cnt[i + 1] += cnt[i]
                cnt[i] = 0
                i += 1
            else:
                cnt[i + 1] += k
                cnt[i] -= k
                k = 0
        mod = 10**9 + 7
        ans = 1
        for i in cnt.keys():
            if cnt[i] > 0:
                ans *= i ** cnt[i]
                ans %= mod
        return ans

    def maximumProduct(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        heapq.heapify(nums)
        while k:
            heapq.heapreplace(nums, nums[0] + 1)
            k -= 1
        ans = 1
        for n in nums:
            ans = ans * n % mod
        return ans

    def maximumProduct(self, nums: List[int], k: int) -> int:
        heapq.heapify(nums)
        for _ in range(k):
            heapq.heapreplace(nums, nums[0] + 1)
        return functools.reduce(lambda x, y: x * y % 1000000007, nums)


# 2234 - Maximum Total Beauty of the Gardens - HARD
class Solution:
    # O(n * logn) / O(1)
    def maximumBeauty(
        self, f: List[int], newFlowers: int, target: int, full: int, partial: int
    ) -> int:
        f.sort()
        n = len(f)
        if f[0] >= target:
            return n * full
        leftFlowers = newFlowers
        for i in range(n):
            leftFlowers -= max(target - f[i], 0)
            f[i] = min(f[i], target)
        ans = x = sumFlowers = 0
        for i in range(n + 1):
            if leftFlowers >= 0:
                while x < i and f[x] * x - sumFlowers <= leftFlowers:
                    sumFlowers += f[x]
                    x += 1
                beauty = (n - i) * full
                if x:  # for division
                    beauty += min((leftFlowers + sumFlowers) // x, target - 1) * partial
                ans = max(ans, beauty)
            if i < n:
                leftFlowers += target - f[i]
        return ans

    def maximumBeauty(
        self, f: List[int], newFlowers: int, target: int, full: int, partial: int
    ) -> int:
        f = [min(target, x) for x in f]
        f.sort()
        n = len(f)
        if f[0] == target:
            return full * n
        if newFlowers >= target * n - sum(f):
            return max(full * n, full * (n - 1) + partial * (target - 1))
        cost = [0]
        for i in range(1, n):
            pre = cost[-1]
            cost.append(pre + i * (f[i] - f[i - 1]))
        j = n - 1
        while f[j] == target:
            j -= 1
        ans = 0
        while newFlowers >= 0:
            idx = min(j, bisect.bisect_right(cost, newFlowers) - 1)
            bar = f[idx] + (newFlowers - cost[idx]) // (idx + 1)
            ans = max(ans, bar * partial + (n - j - 1) * full)
            newFlowers -= target - f[j]
            j -= 1
        return ans


# 2239 - Find Closest Number to Zero - EASY
class Solution:
    def findClosestNumber(self, nums: List[int]) -> int:
        return sorted(nums, key=lambda x: (abs(x), -x))[0]

        return max([-abs(n), n] for n in nums)[1]
        return -min([abs(n), -n] for n in nums)[1]


# 2240 - Number of Ways to Buy Pens and Pencils - MEDIUM
class Solution:
    def waysToBuyPensPencils(self, t: int, c1: int, c2: int) -> int:
        ans = 0
        for i in range(0, t + 1, c1):
            ans += (t - i) // c2 + 1
        return ans

        return sum(((t - i * c1) // c2 + 1) for i in range(t // c1 + 1))
        return sum(((t - i) // c2 + 1) for i in range(0, t + 1, c1))

    def waysToBuyPensPencils(self, t: int, c1: int, c2: int) -> int:
        ans = 0
        while t >= 0:
            ans += t // c2 + 1
            t -= c1
        return ans


# 2241 - Design an ATM Machine - MEDIUM
class ATM:
    def __init__(self):
        self.b = [0, 0, 0, 0, 0]
        self.m = [20, 50, 100, 200, 500]

    def deposit(self, banknotesCount: List[int]) -> None:
        for i in range(5):
            self.b[i] += banknotesCount[i]
        return

    def withdraw(self, amount: int) -> List[int]:
        ans = [0, 0, 0, 0, 0]
        for i in range(4, -1, -1):
            if amount >= self.m[i] * self.b[i]:
                amount -= self.m[i] * self.b[i]
                ans[i] = self.b[i]
            else:
                ans[i] = amount // self.m[i]
                amount %= self.m[i]

        # for i in range(4, -1, -1):
        #     v = min(self.b[i], amount // self.m[i])
        #     ans[i] = v
        #     amount -= v * self.m[i]

        if amount == 0:
            for i in range(5):
                self.b[i] -= ans[i]
        return ans if amount == 0 else [-1]


# 2243 - Calculate Digit Sum of a String - EASY
class Solution:
    def digitSum(self, s: str, k: int) -> str:
        while len(s) > k:
            new = ""
            for i in range(0, len(s), k):
                n = int(s[i : i + k])
                t = 0
                while n:
                    t += n % 10
                    n //= 10
                new += str(t)
            s = new
        return s


# 2244 - Minimum Rounds to Complete All Tasks - MEDIUM
class Solution:
    def minimumRounds(self, tasks: List[int]) -> int:
        cnt = collections.Counter(tasks)
        ans = 0
        for v in cnt.values():
            m = v % 6
            if m == 0 or m == 3:
                ans += v // 3
            elif m == 5:
                ans += v // 3 + 1
            elif m == 4:
                ans += (v - 4) // 3 + 2
            elif m == 2:
                ans += (v - 2) // 3 + 1
            else:
                if v < 6:
                    return -1
                else:
                    ans += (v - 7) // 3 + 3
        return ans

    def minimumRounds(self, tasks: List[int]) -> int:
        cnt = collections.Counter(tasks)
        ans = 0
        for v in cnt.values():
            if v == 1:
                return -1
            elif v == 2:
                ans += 1
            elif v % 3 == 0:
                ans += v // 3
            else:
                ans += v // 3 + 1
        return ans

    def minimumRounds(self, tasks: List[int]) -> int:
        freq = collections.Counter(tasks).values()
        return -1 if 1 in freq else sum((a + 2) // 3 for a in freq)


# 2245 - Maximum Trailing Zeros in a Cornered Path - MEDIUM
class Solution:
    # O(mn) / O(mn)
    def maxTrailingZeros(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        t = [[0] * (n + 1) for _ in range(m + 1)]
        f = [[0] * (n + 1) for _ in range(m + 1)]
        s = [[[0, 0] for _ in range(n + 1)] for _ in range(m + 1)]

        for i in range(m):
            for j in range(n):
                x = grid[i][j]
                two = five = 0
                while x % 2 == 0:  # suppose time complexity is constant
                    two += 1
                    x //= 2
                while x % 5 == 0:
                    five += 1
                    x //= 5
                s[i + 1][j + 1][0] = two
                s[i + 1][j + 1][1] = five
                t[i + 1][j + 1] = two + t[i + 1][j] + t[i][j + 1] - t[i][j]
                f[i + 1][j + 1] = five + f[i + 1][j] + f[i][j + 1] - f[i][j]

        ans = 0
        for i in range(m):
            for j in range(n):
                u2 = t[i][j + 1] - t[i][j]
                d2 = t[-1][j + 1] - t[-1][j] - (t[i + 1][j + 1] - t[i + 1][j])
                u5 = f[i][j + 1] - f[i][j]
                d5 = f[-1][j + 1] - f[-1][j] - (f[i + 1][j + 1] - f[i + 1][j])

                l2 = t[i + 1][j] - t[i][j]
                r2 = t[i + 1][-1] - t[i][-1] - (t[i + 1][j + 1] - t[i][j + 1])
                l5 = f[i + 1][j] - f[i][j]
                r5 = f[i + 1][-1] - f[i][-1] - (f[i + 1][j + 1] - f[i][j + 1])

                ul = min(u2 + l2 + s[i + 1][j + 1][0], u5 + l5 + s[i + 1][j + 1][1])
                ur = min(u2 + r2 + s[i + 1][j + 1][0], u5 + r5 + s[i + 1][j + 1][1])
                dl = min(d2 + l2 + s[i + 1][j + 1][0], d5 + l5 + s[i + 1][j + 1][1])
                dr = min(d2 + r2 + s[i + 1][j + 1][0], d5 + r5 + s[i + 1][j + 1][1])

                mx = max(ul, ur, dl, dr)
                ans = max(ans, mx)
        return ans

    # compute 'prefix sum' separately by row and column.
    def maxTrailingZeros(self, grid: List[List[int]]) -> int:
        def get_25(v):
            r = [0, 0]
            while v % 2 == 0:
                v = v // 2
                r[0] += 1
            while v % 5 == 0:
                v = v // 5
                r[1] += 1
            return r

        m = len(grid)
        n = len(grid[0])
        pu = [[[0, 0] for _ in range(n + 1)] for _ in range(m + 1)]  # up
        pl = [[[0, 0] for _ in range(n + 1)] for _ in range(m + 1)]  # left

        for i in range(m):
            for j in range(n):
                x = get_25(grid[i][j])
                pu[i + 1][j + 1][0] = pu[i][j + 1][0] + x[0]
                pu[i + 1][j + 1][1] = pu[i][j + 1][1] + x[1]
                pl[i + 1][j + 1][0] = pl[i + 1][j][0] + x[0]
                pl[i + 1][j + 1][1] = pl[i + 1][j][1] + x[1]

        ans = 0
        for i in range(m):
            for j in range(n):
                t, f = pu[i + 1][j + 1]
                ul = min(t + pl[i + 1][j][0], f + pl[i + 1][j][1])
                ur = min(
                    t + pl[i + 1][-1][0] - pl[i + 1][j + 1][0],
                    f + pl[i + 1][-1][1] - pl[i + 1][j + 1][1],
                )
                t = pu[-1][j + 1][0] - pu[i][j + 1][0]
                f = pu[-1][j + 1][1] - pu[i][j + 1][1]
                dl = min(t + pl[i + 1][j][0], f + pl[i + 1][j][1])
                dr = min(
                    t + pl[i + 1][-1][0] - pl[i + 1][j + 1][0],
                    f + pl[i + 1][-1][1] - pl[i + 1][j + 1][1],
                )
                ans = max(ans, ul, ur, dl, dr)
        return ans


# 2246 - Longest Path With Different Adjacent Characters - HARD
class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        n = len(parent)
        g = [[] for _ in range(n)]
        for i in range(1, n):
            g[parent[i]].append(i)
        ans = 0

        def dfs(x: int) -> int:
            nonlocal ans
            max_len = 0
            for y in g[x]:
                length = dfs(y) + 1
                if s[y] != s[x]:
                    ans = max(ans, max_len + length)
                    max_len = max(max_len, length)
            return max_len

        dfs(0)
        return ans + 1

    def longestPath(self, parent: List[int], s: str) -> int:
        children = [[] for _ in range(len(s))]
        for i, p in enumerate(parent):
            if p >= 0:
                children[p].append(i)
        ans = 0

        def dfs(i: int) -> int:
            nonlocal ans
            candi = [0, 0]
            for j in children[i]:
                cur = dfs(j)
                if s[i] != s[j]:
                    candi.append(cur)
            candi = heapq.nlargest(2, candi)
            ans = max(ans, candi[0] + candi[1] + 1)
            return max(candi) + 1

        dfs(0)
        return ans

    def longestPath(self, parent: List[int], s: str) -> int:
        children = collections.defaultdict(list)
        for i in range(len(parent)):
            if parent[i] != -1:
                children[parent[i]].append(i)
        ans = 0

        def dfs(node):
            """Return longest path of all the paths start from node"""
            nonlocal ans
            longest = 1
            first, second = 0, 0
            for child in children[node]:
                d = dfs(child)
                if s[child] != s[node]:
                    longest = max(longest, d + 1)
                    if d > first:
                        second = first
                        first = d
                    elif d > second:
                        second = d
            ans = max(ans, first + second)
            return longest

        dfs(0)
        return ans + 1

    def longestPath(self, parent: List[int], s: str) -> int:
        g = [[] for _ in range(len(parent))]
        for i, j in enumerate(parent):
            if i > 0:
                g[j].append(i)
        ans = 1

        def dfs(u, p):
            nonlocal ans
            ch = [0, 0]
            for c in g[u]:
                if c != p:
                    x = dfs(c, u)
                    if s[c] != s[u]:
                        ch.append(x)
            ch.sort()
            ans = max(ans, ch[-1] + ch[-2] + 1)
            return ch[-1] + 1

        dfs(0, -1)
        return ans

    def longestPath(self, p: List[int], s: str) -> int:
        children = [[] for _ in range(len(p))]
        for i, a in enumerate(p):
            if a != -1:
                children[a].append(i)
        self.ans = 1

        def dfs(x):
            arr = []
            for c in children[x]:
                ns = dfs(c)
                if s[c] != s[x]:
                    arr.append(ns)
            if len(arr) == 0:
                return 1
            arr.sort()
            arr = [0] + arr
            self.ans = max(self.ans, 1 + arr[-1] + arr[-2])
            return arr[-1] + 1

        dfs(0)
        return self.ans


# 2248 - Intersection of Multiple Arrays - EASY
class Solution:
    def intersection(self, nums: List[List[int]]) -> List[int]:
        a = set(nums[0])
        for i in range(1, len(nums)):
            b = set(nums[i])
            a.intersection_update(b)
        return sorted(list(a))

    def intersection(self, nums: List[List[int]]) -> List[int]:
        s = set(nums[0])
        for t in nums[1:]:
            s &= set(t)
        return sorted(s)

    def intersection(self, nums: List[List[int]]) -> List[int]:
        flat = [x for n in nums for x in n]
        cnt = collections.Counter(flat)
        ans = [i for i in cnt.keys() if cnt[i] == len(nums)]
        return sorted(ans)


# 2249 - Count Lattice Points Inside a Circle - MEDIUM
class Solution:
    def countLatticePoints(self, circles: List[List[int]]) -> int:
        s = set()
        ans = 0
        for x, y, r in circles:
            d = r**2
            for i in range(x - r, x + r + 1):
                for j in range(y - r, y + r + 1):
                    if (i, j) in s:
                        continue
                    else:
                        # if (i - x)**2 + (j - y)**2 <= d: # slower if the power is small
                        if (i - x) * (i - x) + (j - y) * (j - y) <= d:
                            ans += 1
                            s.add((i, j))
        return ans


# 2250 - Count Number of Rectangles Containing Each Point - MEDIUM
class Solution:
    # the range of 'h' is 1 <= h <= 100, much smaller than the range of 'l'
    def countRectangles(
        self, rectangles: List[List[int]], points: List[List[int]]
    ) -> List[int]:
        ans = []
        d = collections.defaultdict(list)
        for l, h in rectangles:  # O(n)
            d[h].append(l)
        for h in d:  # O(n * llogl)
            d[h].sort()
        for x, y in points:  # O(n * C * logl)
            count = 0
            for h in range(y, 101):
                j = bisect.bisect_left(d[h], x)
                count += len(d[h]) - j
            ans.append(count)
        return ans

    def countRectangles(
        self, rectangles: List[List[int]], points: List[List[int]]
    ) -> List[int]:
        rectangles.sort(key=lambda r: -r[1])
        ans = [0] * len(points)
        i = 0
        xs = []
        for (x, y), id in sorted(
            zip(points, range(len(points))), key=lambda x: -x[0][1]
        ):
            start = i
            while i < len(rectangles) and rectangles[i][1] >= y:
                xs.append(rectangles[i][0])
                i += 1
            if start < i:
                xs.sort()  # 只有在 xs 插入了新元素时才排序
            ans[id] = i - bisect.bisect_left(xs, x)
        return ans

    def countRectangles(
        self, rectangles: List[List[int]], points: List[List[int]]
    ) -> List[int]:
        rectangles.sort(key=lambda r: -r[0])
        n = len(points)
        ans = [0] * n
        cnt = [0] * (max(r[1] for r in rectangles) + 1)
        i = 0
        for (x, y), id in sorted(zip(points, range(n)), key=lambda x: -x[0][0]):
            while i < len(rectangles) and rectangles[i][0] >= x:
                cnt[rectangles[i][1]] += 1
                i += 1
            ans[id] = sum(cnt[y:])
        return ans

    # O(nlogn + mlogm + mlogn) / O(m), n = len(rectangles), m = len(points)
    def countRectangles(
        self, rectangles: List[List[int]], points: List[List[int]]
    ) -> List[int]:
        rectangles.sort(key=lambda r: -r[1])
        ans = [0] * len(points)
        i = 0
        sl = sortedcontainers.SortedList()
        for (x, y), id in sorted(
            zip(points, range(len(points))), key=lambda x: -x[0][1]
        ):
            while i < len(rectangles) and rectangles[i][1] >= y:
                sl.add(rectangles[i][0])
                i += 1
            ans[id] = i - sl.bisect_left(x)
        return ans

    def countRectangles(
        self, rectangles: List[List[int]], points: List[List[int]]
    ) -> List[int]:
        rectangles.sort()
        points = sorted([[x, y, i] for i, (x, y) in enumerate(points)])
        sl = sortedcontainers.SortedList()
        ans = [0] * len(points)
        r = len(rectangles) - 1
        for x, y, i in reversed(points):
            while r >= 0 and rectangles[r][0] >= x:
                sl.add(rectangles[r][1])
                r -= 1
            ans[i] = len(sl) - sl.bisect_left(y)
        return ans


# 2255 - Count Prefixes of a Given String - EASY
class Solution:
    def countPrefixes(self, words: List[str], s: str) -> int:
        ans = 0
        cnt = collections.Counter(words)
        for i in range(len(s)):
            if s[: i + 1] in cnt:
                ans += cnt[s[: i + 1]]
        return ans

    def countPrefixes(self, words: List[str], s: str) -> int:
        return sum(s.startswith(w) for w in words)


# 2256 - Minimum Average Difference - MEDIUM
class Solution:
    def minimumAverageDifference(self, nums: List[int]) -> int:
        n = len(nums)
        p = [0]
        for i in range(n):
            p.append(p[i] + nums[i])
        ans = -1
        d = math.inf
        for i in range(n):
            l = int(p[i + 1] / (i + 1))
            r = 0
            if i != n - 1:
                r = int((p[-1] - p[i + 1]) / (n - i - 1))
            if abs(l - r) < d:
                ans = i
                d = abs(l - r)
        return ans

    def minimumAverageDifference(self, nums: List[int]) -> int:
        pre = 0
        suf = sum(nums)
        n = len(nums)
        mindiff = math.inf
        for i in range(n - 1):
            pre += nums[i]
            suf -= nums[i]
            d = abs(int(pre / (i + 1)) - int(suf / (n - i - 1)))
            if d < mindiff:
                mindiff = d
                ans = i
        if int(pre + nums[-1]) / n < mindiff:
            ans = n - 1
        return ans


# 2257 - Count Unguarded Cells in the Grid - MEDIUM
class Solution:
    def countUnguarded(
        self, m: int, n: int, guards: List[List[int]], walls: List[List[int]]
    ) -> int:
        # guard: 3 / wall:2 / can be guarded: 1
        g = [[0] * n for _ in range(m)]
        for r, c in walls:
            g[r][c] = 2
        dq = collections.deque([])
        for r, c in guards:
            g[r][c] = 3
            dq.append((r, c, 1, 0))
            dq.append((r, c, -1, 0))
            dq.append((r, c, 0, 1))
            dq.append((r, c, 0, -1))
        while dq:
            for _ in range(len(dq)):
                x, y, dx, dy = dq.popleft()
                nx = x + dx
                ny = y + dy
                if 0 <= nx < m and 0 <= ny < n and g[nx][ny] < 2:
                    g[nx][ny] = 1
                    dq.append((nx, ny, dx, dy))
        return sum(v == 0 for r in g for v in r)

    def countUnguarded(
        self, m: int, n: int, guards: List[List[int]], walls: List[List[int]]
    ) -> int:
        f = [[0] * n for _ in range(m)]
        for x, y in walls:
            f[x][y] = -1
        for x, y in guards:
            f[x][y] = -2
        d = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        for x, y in guards:
            for dx, dy in d:
                nx, ny = x, y
                while True:
                    nx += dx
                    ny += dy
                    if 0 <= nx < m and 0 <= ny < n:
                        if f[nx][ny] < 0:
                            break
                        if f[nx][ny] == 0:
                            f[nx][ny] = 1
                    else:
                        break
        return sum(v == 0 for r in f for v in r)


# 2258 - Escape the Spreading Fire - HARD
class Solution:
    def maximumMinutes(self, g: List[List[int]]) -> int:
        m = len(g)
        n = len(g[0])
        d = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        fires = [[i, j, 0] for i in range(m) for j in range(n) if g[i][j] == 1]
        inf = 10**10
        g = [[inf if v < 2 else -1 for v in r] for r in g]

        def bfs(queue: List[List[int]], seen: List[List[int]]):
            for i, j, t in queue:
                if seen[i][j] < inf:
                    continue
                seen[i][j] = t
                for di, dj in d:
                    x, y = i + di, j + dj
                    if (
                        0 <= x < m
                        and 0 <= y < n
                        and seen[x][y] >= inf
                        and t + 1 < g[x][y]
                    ):
                        queue.append([x, y, t + 1])

        def die(t: int) -> bool:
            seen = [[math.inf] * n for _ in range(m)]
            bfs([[0, 0, t]], seen)
            return seen[-1][-1] > g[-1][-1]

        bfs(fires, g)
        g[-1][-1] += 1
        # return bisect_left(range(10**9 + 1), True, key=die) - 1
        ans = bisect.bisect_left(range(m * n + 1), 1, key=die) - 1
        return ans if ans < m * n else 10**9

    # no binary search, just BFS
    def maximumMinutes(self, g: List[List[int]]) -> int:
        m = len(g)
        n = len(g[0])
        f = [[math.inf] * n for _ in range(m)]
        s = collections.deque([])
        for i, r in enumerate(g):
            for j, v in enumerate(r):
                if v == 1:
                    s.append((i, j))
                    f[i][j] = 0
        while s:
            for _ in range(len(s)):
                x, y = s.popleft()
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if (
                        0 <= nx < m
                        and 0 <= ny < n
                        and f[nx][ny] == math.inf
                        and g[nx][ny] != 2
                    ):
                        s.append((nx, ny))
                        f[nx][ny] = f[x][y] + 1

        my = collections.deque([(0, 0)])
        p = [[math.inf] * n for _ in range(m)]
        p[0][0] = 0
        step = 0
        while my:
            for _ in range(len(my)):
                x, y = my.popleft()
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if (
                        0 <= nx < m
                        and 0 <= ny < n
                        and g[nx][ny] != 2
                        and p[nx][ny] == math.inf
                    ):
                        if (
                            f[nx][ny] == math.inf
                            or step + 1 < f[nx][ny]
                            or (step + 1 == f[nx][ny] and nx == m - 1 and ny == n - 1)
                        ):
                            p[nx][ny] = p[x][y] + 1
                            my.append((nx, ny))
            step += 1

        if p[-1][-1] == math.inf:
            return -1
        if f[-1][-1] == math.inf:
            return 10**9
        if f[-1][-1] == p[-1][-1]:
            return 0
        # !
        diff = f[-1][-1] - p[-1][-1]
        # if m > 1 and n > 1:
        d1 = f[-1][-2] - p[-1][-2]
        d2 = f[-2][-1] - p[-2][-1]
        if d1 > diff or d2 > diff:
            return diff
        return diff - 1

    def maximumMinutes(self, g: List[List[int]]) -> int:
        m = len(g)
        n = len(g[0])
        f = [[-1] * n for _ in range(m)]
        p = [[-1] * n for _ in range(m)]

        def bfs(dq: collections.deque, board: List[List[int]], is_fire=True):
            while dq:
                r, c, step = dq.popleft()
                board[r][c] = step
                for nr, nc in [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]]:
                    if (
                        0 <= nc < n
                        and 0 <= nr < m
                        and board[nr][nc] < 0
                        and g[nr][nc] != 2
                        and (
                            is_fire
                            or f[nr][nc] < 0
                            or f[nr][nc] > step + 1
                            or (f[nr][nc] == step + 1 and nr == m - 1 and nc == n - 1)
                        )
                    ):
                        dq.append((nr, nc, step + 1))
            return

        bfs(
            collections.deque(
                [(r, c, 0) for r in range(m) for c in range(n) if g[r][c] == 1]
            ),
            f,
        )
        bfs(collections.deque([(0, 0, 0)]), p, is_fire=False)

        if p[-1][-1] < 0:
            return -1
        if f[-1][-1] < 0:
            return 1000000000
        if f[-1][-1] == p[-1][-1]:
            return 0
        diff = f[-1][-1] - p[-1][-1]
        # if m > 1 and n > 1:
        d1 = f[-1][-2] - p[-1][-2]
        d2 = f[-2][-1] - p[-2][-1]
        if d1 > diff or d2 > diff:
            return diff
        return diff - 1


# 2259 - Remove Digit From Number to Maximize Result - EASY
class Solution:
    def removeDigit(self, number: str, digit: str) -> str:
        ans = ""
        for i in range(len(number)):
            if number[i] == digit:
                s = number[:i] + number[i + 1 :]
                if s > ans:
                    ans = s
        return ans


# 2260 - Minimum Consecutive Cards to Pick Up - MEDIUM
class Solution:
    def minimumCardPickup(self, cards: List[int]) -> int:
        ans = math.inf
        d = {}
        for i, v in enumerate(cards):
            if v in d and i - d[v] < ans:
                ans = i - d[v] + 1
            d[v] = i
        return -1 if ans == math.inf else ans


# 2261 - K Divisible Elements Subarrays - MEDIUM
class Solution:
    def countDistinct(self, nums: List[int], k: int, p: int) -> int:
        n = len(nums)
        s = set()
        for i in range(n):
            arr = []
            cnt = 0
            for j in range(i, n):
                if nums[j] % p == 0:
                    cnt += 1
                if cnt > k:
                    break
                arr.append(nums[j])
                s.add(tuple(arr))
        return len(s)


# 2262 - Total Appeal of A String - HARD
class Solution:
    # O(n) / O(26)
    def appealSum(self, s: str) -> int:
        ans = last = 0
        pos = [-1] * 26
        for i, c in enumerate(s):
            idx = ord(c) - ord("a")
            last += i - pos[idx]
            pos[idx] = i
            ans += last
        return ans

    def appealSum(self, s: str) -> int:
        ans = last = 0
        d = {}
        for i, c in enumerate(s):
            if c in d:
                last += i - d[c]
            else:
                last += i + 1
            d[c] = i
            ans += last
        return ans

    def appealSum(self, s: str) -> int:
        n = len(s)
        record = {}
        ans = 0
        for i, c in enumerate(s):
            ans += (i - record.get(c, -1)) * (n - i)
            record[c] = i
        return ans

    def appealSum(self, s: str) -> int:
        dp = [0 for _ in range(len(s) + 1)]
        d = {}
        for i, c in enumerate(s):
            dp[i + 1] = dp[i] + i - d.get(c, -1)
            d[c] = i
        return sum(dp)

    def appealSum(self, s: str) -> int:
        last = {}
        ans = 0
        for i, c in enumerate(s):
            last[c] = i + 1
            ans += sum(last.values())
        return ans


# 2264 - Largest 3-Same-Digit Number in String - EASY
class Solution:
    def largestGoodInteger(self, num: str) -> str:
        ans = ""
        for i in range(1, len(num) - 1):
            if num[i - 1] == num[i] == num[i + 1]:
                cur = num[i - 1 : i + 2]
                if ans == "":
                    ans = cur
                elif int(cur) > int(ans):
                    ans = cur
        return ans

    def largestGoodInteger(self, num: str) -> str:
        ans = ""
        for i in range(len(num) - 2):
            if num[i : i + 3] == num[i] * 3:
                ans = max(ans, num[i] * 3)
        return ans

    def largestGoodInteger(self, num: str) -> str:
        for i in "9876543210":
            x = i * 3
            if x in num:
                return x
        return ""


# 2265 - Count Nodes Equal to Average of Subtree - MEDIUM
class Solution:
    def averageOfSubtree(self, root: Optional[TreeNode]) -> int:
        self.ans = 0

        def post(root: TreeNode) -> Tuple[int, int]:
            if not root:
                return 0, 0
            l, lch = post(root.left)
            r, rch = post(root.right)
            t = l + r + root.val
            tch = lch + rch + 1
            if root.val == t // tch:
                self.ans += 1
            return t, tch

        post(root)
        return self.ans


# 2266 - Count Number of Texts - MEDIUM
class Solution:
    def countTexts(self, pressedKeys: str) -> int:
        mod = 1_000_000_007

        @functools.lru_cache(None)
        def dfs(n: int, k: int) -> int:
            """Return number of possible text of n repeated k times."""
            if n < 0:
                return 0
            if n == 0:
                return 1
            ans = 0
            for x in range(1, k + 1):
                ans = (ans + dfs(n - x, k)) % mod
            return ans

        ans = 1
        for key, grp in itertools.groupby(pressedKeys):
            if key in "79":
                k = 4
            else:
                k = 3
            ans = (ans * dfs(len(list(grp)), k)) % mod
        return ans

    def countTexts(self, pressedKeys: str) -> int:
        @functools.lru_cache(None)
        def dfs(length: int, k: int) -> int:
            if length == 0:
                return 1
            cur = 0
            for i in range(k):
                if i + 1 <= length:
                    cur = (cur + dfs(length - i - 1, k)) % mod

                    # cur += dfs(length - i - 1, k)
                    # original version:
                    #   if there is no modulo operation,
                    #   then 'cur' is very large and slow to compute
                    #   TLE in leetcode.cn,
                    #   MLE in leetcode.com.

            return cur

        pressedKeys += "#"
        pre = pressedKeys[0]
        cnt = ans = 1
        mod = 1000000000 + 7
        for i in range(1, len(pressedKeys)):
            if pressedKeys[i] == pre:
                cnt += 1
            else:
                if pre in "79":
                    ans = (ans * dfs(cnt, 4)) % mod
                else:
                    ans = (ans * dfs(cnt, 3)) % mod
                pre = pressedKeys[i]
                cnt = 1
        return ans

    def countTexts(self, pressedKeys: str) -> int:
        m = [3] * 10
        m[7] = m[9] = 4
        dp = [1]
        mod = 1000000007
        for i in range(len(pressedKeys)):
            v = int(pressedKeys[i])
            cur = 0
            for j in range(m[v]):
                if i < j or pressedKeys[i - j] != pressedKeys[i]:
                    break
                cur += dp[i - j]
            cur %= mod
            dp.append(cur)
        return dp[-1]

    def countTexts(self, s: str) -> int:
        mod = int(1e9 + 7)
        s += "#"
        p = "*"
        l = 0

        def fn(p: int, l: int) -> int:
            if p == "7" or p == "9":
                a, b, c, d = 1, 0, 0, 0
                for _ in range(1, l):
                    a, b, c, d = (a + b + c + d) % mod, a, b, c
                return (a + b + c + d) % mod
            else:
                a, b, c = 1, 0, 0
                for _ in range(1, l):
                    a, b, c = (a + b + c) % mod, a, b
                return (a + b + c) % mod

        ans = 1
        for c in s:
            if c == p:
                l += 1
            else:
                ans = ans * fn(p, l) % mod
                p = c
                l = 1
        return ans


# 2267 - Check if There Is a Valid Parentheses String Path - HARD
class Solution:
    # O(m * n * (m+n)) / O(m * n * (m+n)), (m+n): the number of bracket
    def hasValidPath(self, grid: List[List[str]]) -> bool:
        @functools.lru_cache(None)
        def dfs(i, j, left) -> bool:
            if i == m - 1 and j == n - 1:
                return left == 0
            l = r = False
            for x, y in [[i + 1, j], [i, j + 1]]:
                if x < m and y < n:
                    if grid[x][y] == "(":
                        # original version:
                        # l = dfs(x, y, left + 1)
                        l = l or dfs(x, y, left + 1)
                        # l |= dfs(x, y, left + 1) # extremely slow
                    elif left > 0:
                        r = r or dfs(x, y, left - 1)
            return l or r

        m = len(grid)
        n = len(grid[0])
        if grid[0][0] == ")" or grid[-1][-1] == "(" or (not (m + n) & 1):
            return False
        return dfs(0, 0, 1)

    def hasValidPath(self, grid: List[List[str]]) -> bool:
        @functools.lru_cache(None)
        def dfs(i: int, j: int, diff: int) -> bool:
            if diff < 0:
                return False
            if (i, j) == (m - 1, n - 1):
                return diff == 0
            r = False
            for x, y in [[i + 1, j], [i, j + 1]]:
                if x < m and y < n:
                    r = r or dfs(x, y, diff + (1 if grid[x][y] == "(" else -1))
            return r

        m = len(grid)
        n = len(grid[0])
        if grid[0][0] == ")" or grid[-1][-1] == "(" or (not (m + n) & 1):
            return False
        # dfs.cache_clear()
        return dfs(0, 0, 1)

    def hasValidPath(self, grid: List[List[str]]) -> bool:
        @functools.lru_cache(None)
        def dfs(r: int, c: int, cnt: int) -> bool:
            if r >= m or c >= n or cnt < 0:
                return False
            if grid[r][c] == "(":
                cnt += 1
            else:
                cnt -= 1
            if r == m - 1 and c == n - 1:
                return cnt == 0
            return dfs(r + 1, c, cnt) or dfs(r, c + 1, cnt)

        m = len(grid)
        n = len(grid[0])
        if grid[0][0] == ")" or grid[-1][-1] == "(" or (m + n - 1) % 2:
            return False
        return dfs(0, 0, 0)

    def hasValidPath(self, grid: List[List[str]]) -> bool:
        r = len(grid)
        c = len(grid[0])
        if (r + c) % 2 == 0 or grid[0][0] == ")" or grid[r - 1][c - 1] == "(":
            return False
        vis = [[[False] * (r + c) for _ in range(c)] for _ in range(r)]
        vis[0][0][1] = True
        dq = collections.deque([[0, 0, 1]])
        while dq:
            x, y, cnt = dq.popleft()
            if x == r - 1 and y == c - 1 and cnt == 0:
                return True
            for nx, ny in [[x + 1, y], [x, y + 1]]:
                if nx < r and ny < c:
                    nt = cnt + 1 if grid[nx][ny] == "(" else cnt - 1
                    if nt >= 0 and not vis[nx][ny][nt]:
                        vis[nx][ny][nt] = True
                        dq.append([nx, ny, nt])
        return False


# 2269 - Find the K-Beauty of a Number - EASY
class Solution:
    def divisorSubstrings(self, num: int, k: int) -> int:
        s = str(num)
        ans = 0
        for i in range(k, len(s) + 1):
            if int(s[i - k : i]) != 0 and num % int(s[i - k : i]) == 0:
                ans += 1
        return ans


# 2270 - Number of Ways to Split Array - MEDIUM
class Solution:
    def waysToSplitArray(self, nums: List[int]) -> int:
        sub = sum(nums)
        ans = pre = 0
        for i in range(len(nums) - 1):
            pre += nums[i]
            sub -= nums[i]
            if pre >= sub:
                ans += 1
        return ans


# 2271 - Maximum White Tiles Covered by a Carpet - MEDIUM
class Solution:
    def maximumWhiteTiles(self, tiles: List[List[int]], carpetLen: int) -> int:
        tiles.sort()
        ans = cover = r = 0
        n = len(tiles)
        for l in range(n):
            while r < n and tiles[r][1] - tiles[l][0] < carpetLen:
                cover += tiles[r][1] - tiles[r][0] + 1
                r += 1

            # if r < n and tiles[r][0] - tiles[l][0] < carpetLen:
            #     ans = max(ans, cover + tiles[l][0] + carpetLen - tiles[r][0])

            if r < n:
                # ans = max(ans, cover + max(0, tiles[l][0] - tiles[r][0] + carpetLen))
                ans = max(ans, cover + max(0, carpetLen - (tiles[r][0] - tiles[l][0])))
            else:
                ans = max(ans, cover)
            cover -= tiles[l][1] - tiles[l][0] + 1
        return ans

    def maximumWhiteTiles(self, tiles: List[List[int]], carpetLen: int) -> int:
        tiles.sort()
        ans = cover = l = 0
        for tl, tr in tiles:
            cover += tr - tl + 1
            while tiles[l][1] < tr - carpetLen + 1:
                cover -= tiles[l][1] - tiles[l][0] + 1
                l += 1
            ans = max(ans, cover - max(tr - carpetLen + 1 - tiles[l][0], 0))
        return ans


# 2273 - Find Resultant Array After Removing Anagrams - EASY
class Solution:
    def removeAnagrams(self, words: List[str]) -> List[str]:
        last = sorted(words[0])
        ans = [words[0]]
        for i in range(1, len(words)):
            new = sorted(words[i])
            if new != last:
                ans.append(words[i])
                last = new
        return ans

    def removeAnagrams(self, words: List[str]) -> List[str]:
        ans = [words[0]]
        for i in range(1, len(words)):
            cnt = [0] * 26
            for c in words[i]:
                cnt[ord(c) - ord("a")] += 1
            for c in ans[-1]:
                cnt[ord(c) - ord("a")] -= 1
            if cnt != [0] * 26:
                ans.append(words[i])
        return ans


# 2274 - Maximum Consecutive Floors Without Special Floors - MEDIUM
class Solution:
    # O(nlogn) / O(logn)
    def maxConsecutive(self, bottom: int, top: int, special: List[int]) -> int:
        ans = max(top - special[-1], special[0] - bottom)
        special.sort()
        for i in range(1, len(special)):
            ans = max(ans, special[i] - special[i - 1] - 1)
        return ans


# 2275 - Largest Combination With Bitwise AND Greater Than Zero - MEDIUM
class Solution:
    def largestCombination(self, candidates: List[int]) -> int:
        d = collections.defaultdict(int)
        for can in candidates:
            i = 0
            while can != 0:
                if can & 1 == 1:
                    d[i] += 1
                i += 1
                can >>= 1
        return max(d.values())

    def largestCombination(self, c: List[int]) -> int:
        bit = 1
        ans = 0
        while bit <= 10**7:
            cnt = 0
            for val in c:
                if val & bit:
                    cnt += 1
            ans = max(ans, cnt)
            bit <<= 1
        return ans

    # 2 ** 23 < 10 ** 7 < 2 ** 24
    def largestCombination(self, candidates: List[int]) -> int:
        return max(sum((x >> i) & 1 for x in candidates) for i in range(24))


# 2278 - Percentage of Letter in String - EASY
class Solution:
    def percentageLetter(self, s: str, letter: str) -> int:
        return s.count(letter) * 100 // len(s)


# 2279 - Maximum Bags With Full Capacity of Rocks - MEDIUM
class Solution:
    def maximumBags(
        self, capacity: List[int], rocks: List[int], additionalRocks: int
    ) -> int:
        for i in range(len(capacity)):
            capacity[i] -= rocks[i]
        capacity.sort()
        ans = 0
        for c in capacity:
            if c > additionalRocks:
                break
            else:
                additionalRocks -= c
                ans += 1
        return ans


# 2280 - Minimum Lines to Represent a Line Chart - MEDIUM
class Solution:
    def minimumLines(self, stock: List[List[int]]) -> int:
        stock.sort()
        if len(stock) == 1:
            return 0
        s = set()
        for i in range(1, len(stock)):
            dy = stock[i][1] - stock[i - 1][1]
            dx = stock[i][0] - stock[i - 1][0]
            g = math.gcd(dy, dx)
            b = stock[i][1] - (dy / dx) * stock[i][0]
            if (dy / g, dx / g, b) not in s:
                s.add((dy / g, dx / g, b))
                # s.add((k, b)) # Using k obtained by division will have precision problems
        return len(s)

    # itertools.pairwise, New in version 3.10
    def pairwise(self, iterable):
        # pairwise('ABCDEFG') --> AB BC CD DE EF FG
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    def minimumLines(self, stockPrices: List[List[int]]) -> int:
        stockPrices.sort()
        ans = 0
        pre_dy = 1  # 1 / 0, infinite slope, different with any line in test case
        pre_dx = 0
        for (x1, y1), (x2, y2) in self.pairwise(stockPrices):
            dy, dx = y2 - y1, x2 - x1
            # different k, no float point precision error
            if dy * pre_dx != pre_dy * dx:
                ans += 1
                pre_dy, pre_dx = dy, dx
        return ans


# 2283 - Check if Number Has Equal Digit Count and Digit Value - EASY
class Solution:
    def digitCount(self, num: str) -> bool:
        cnt = collections.Counter(num)
        for i, n in enumerate(num):
            if int(n) != cnt[str(i)]:
                return False
        return True

    def digitCount(self, num: str) -> bool:
        cnt = collections.Counter(num)
        return all(cnt[str(i)] == int(num[i]) for i in range(len(num)))


# 2284 - Sender With Largest Word Count - MEDIUM
class Solution:
    def largestWordCount(self, messages: List[str], senders: List[str]) -> str:
        cnt = collections.defaultdict(int)
        for m, s in zip(messages, senders):
            cnt[s] += len(m.split())
        m = max(cnt.values())
        names = [k for k in cnt if cnt[k] == m]
        return sorted(names)[-1]

    def largestWordCount(self, messages: List[str], senders: List[str]) -> str:
        cnt = collections.defaultdict(int)
        for i, m in enumerate(messages):
            cnt[senders[i]] += m.count(" ") + 1
        ans = ""
        for k, v in list(cnt.items()):
            if v > cnt[ans] or v == cnt[ans] and k > ans:
                ans = k
        return ans


# 2285 - Maximum Total Importance of Roads - MEDIUM
class Solution:
    def maximumImportance(self, n: int, roads: List[List[int]]) -> int:
        d = collections.defaultdict(int)
        for a, b in roads:
            d[a] += 1
            d[b] += 1
        arr = sorted(d.values(), reverse=True)
        ans = 0
        for i, v in enumerate(arr):
            ans += (n - i) * v
        return ans

    def maximumImportance(self, n: int, roads: List[List[int]]) -> int:
        arr = [0] * n
        for a, b in roads:
            arr[a] += 1
            arr[b] += 1
        arr.sort()
        return sum(v * i for i, v in enumerate(arr, start=1))


# 2286 - Booking Concert Tickets in Groups - HARD
class BookMyShow:
    # update single value
    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self.f = [0] * 4 * n
        self.min = [0] * 4 * n

    # self.add(1, 1, n, idx, val)
    def add(self, k: int, l: int, r: int, idx: int, val: int):
        if l == r:
            self.f[k] += val
            self.min[k] += val
            return
        m = (l + r) >> 1
        if idx <= m:
            self.add(k + k, l, m, idx, val)
        else:
            self.add(k + k + 1, m + 1, r, idx, val)
        self.f[k] = self.f[k + k] + self.f[k + k + 1]
        self.min[k] = min(self.min[k + k], self.min[k + k + 1])
        return

    # self.calc(1, 1, n, L, R)
    def calc(self, k: int, l: int, r: int, L: int, R: int) -> int:
        # calc arr[L, R]
        if L <= l and r <= R:
            return self.f[k]
        m = (l + r) >> 1
        ret = 0
        if L <= m:
            ret += self.calc(k + k, l, m, L, R)
        if m < R:
            ret += self.calc(k + k + 1, m + 1, r, L, R)
        return ret

    # return the minimum index of val in range [1, R], if not exist: return 0
    def index(self, k: int, l: int, r: int, R: int, val: int) -> int:
        if self.min[k] > val:
            return 0
        if l == r:
            return l
        m = (l + r) >> 1
        if self.min[k + k] <= val:
            return self.index(k + k, l, m, R, val)
        if R > m:
            return self.index(k + k + 1, m + 1, r, R, val)
        return 0

    def gather(self, k: int, maxRow: int) -> List[int]:
        i = self.index(1, 1, self.n, maxRow + 1, self.m - k)
        if i == 0:
            return []
        seats = self.calc(1, 1, self.n, i, i)
        self.add(1, 1, self.n, i, k)
        return [i - 1, seats]

    def scatter(self, k: int, maxRow: int) -> bool:
        left = (maxRow + 1) * self.m - self.calc(1, 1, self.n, 1, maxRow + 1)
        if left < k:
            return False
        i = self.index(1, 1, self.n, maxRow + 1, self.m - 1)
        while True:
            left_seats = self.m - self.calc(1, 1, self.n, i, i)
            if k <= left_seats:
                self.add(1, 1, self.n, i, k)
                return True
            k -= left_seats
            self.add(1, 1, self.n, i, left_seats)
            i += 1


# 2287 - Rearrange Characters to Make Target String - EASY
class Solution:
    def rearrangeCharacters(self, s: str, target: str) -> int:
        cnt = collections.Counter(s)
        d = collections.Counter(target)
        return min(cnt[n] // d[n] for n in d)


# 2288 - Apply Discount to Prices - MEDIUM
class Solution:
    def discountPrices(self, sentence: str, discount: int) -> str:
        ans = []
        s = sentence.split()
        for w in s:
            if w[0] == "$" and w[1:].isdigit():
                price = "%.2f" % (int(w[1:]) * (100 - discount) / 100)
                ans.append("$" + price)
            else:
                ans.append(w)
        return " ".join(ans)


# 2289 - Steps to Make Array Non-decreasing - MEDIUM
class Solution:
    # O(n) / O(n)
    def totalSteps(self, nums: List[int]) -> int:
        st = []  # monotonic stack, (num, max_t)
        ans = 0
        for v in nums:
            cur = 0
            while st and st[-1][0] <= v:
                cur = max(cur, st.pop()[1])
            if st:
                cur += 1
            else:
                cur = 0  # v is the biggest
            ans = max(ans, cur)
            st.append((v, cur))
        return ans

    def totalSteps(self, nums: List[int]) -> int:
        st = []
        f = [0] * len(nums)
        for i, v in enumerate(nums):
            cur = 0
            while st and nums[st[-1]] <= v:
                cur = max(cur, f[st.pop()])
            if st:
                f[i] = cur + 1
            st.append(i)
        return max(f)


# 2290 - Minimum Obstacle Removal to Reach Corner - HARD
class Solution:
    # O(m * n) / O(m * n), 0-1 BFS
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        dis = [[math.inf] * n for _ in range(m)]
        dis[0][0] = 0
        dq = collections.deque([(0, 0)])
        while dq:
            x, y = dq.popleft()
            for nx, ny in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
                if 0 <= nx < m and 0 <= ny < n:
                    g = grid[x][y]
                    if dis[x][y] + g < dis[nx][ny]:
                        dis[nx][ny] = dis[x][y] + g
                        if g == 0:
                            dq.appendleft((nx, ny))
                        else:
                            dq.append((nx, ny))
        return dis[m - 1][n - 1]

    # dijkstra, TODO
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        INF = 10**9
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        Row = len(grid)
        Col = len(grid[0])

        dist = [[INF for _ in range(Col)] for _ in range(Row)]
        minHeap = []

        heapq.heappush(minHeap, (0, 0, 0))
        dist[0][0] = 0

        while minHeap:
            d, r, c = heapq.heappop(minHeap)
            if dist[r][c] < d:
                continue
            for di in range(4):
                (dr, dc) = dirs[di]
                nr = r + dr
                nc = c + dc
                if (
                    0 <= nr < Row
                    and 0 <= nc < Col
                    and dist[r][c] + grid[nr][nc] < dist[nr][nc]
                ):
                    dist[nr][nc] = dist[r][c] + grid[nr][nc]
                    heapq.heappush(minHeap, (dist[nr][nc], nr, nc))

        res = dist[Row - 1][Col - 1]
        return res

    # dijkstra, TODO
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        def dijkstra(st, ed):
            dist = [[math.inf] * m for _ in range(n)]
            ok = [[False] * m for _ in range(n)]
            pq = [(0, st)]
            dist[0][0] = 0
            while pq:
                _, (x, y) = heapq.heappop(pq)
                if (x, y) == ed:
                    return dist[x][y]
                if ok[x][y]:
                    continue
                ok[x][y] = True
                for dx, dy in zip(dir_x, dir_y):
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < n
                        and 0 <= ny < m
                        and dist[nx][ny] > dist[x][y] + grid[nx][ny]
                    ):
                        dist[nx][ny] = dist[x][y] + grid[nx][ny]
                        heapq.heappush(pq, (dist[nx][ny], (nx, ny)))
            return -1

        n, m = len(grid), len(grid[0])
        dir_x, dir_y = (-1, 1, 0, 0), (0, 0, -1, 1)
        return dijkstra((0, 0), (n - 1, m - 1))


# 2293 - Min Max Game - EASY
class Solution:
    def minMaxGame(self, nums: List[int]) -> int:
        while len(nums) > 1:
            new = []
            for i in range(len(nums) // 2):
                if i & 1:
                    new.append(max(nums[2 * i], nums[2 * i + 1]))
                else:
                    new.append(min(nums[2 * i], nums[2 * i + 1]))
            nums = new
        return nums[0]


# 2294 - Partition Array Such That Maximum Difference Is K - MEDIUM
class Solution:
    def partitionArray(self, nums: List[int], k: int) -> int:
        nums.sort()
        ans = 1
        p = nums[0]
        for i in range(1, len(nums)):
            if nums[i] - p > k:
                ans += 1
                p = nums[i]
            else:
                pass
        return ans

    def partitionArray(self, nums: List[int], k: int) -> int:
        nums.sort()
        ans = 1
        p = nums[0]
        for n in nums:
            if n - p > k:
                ans += 1
                p = n
        return ans


# 2295 - Replace Elements in an Array - MEDIUM
class Solution:
    def arrayChange(self, nums: List[int], operations: List[List[int]]) -> List[int]:
        d = {v: i for i, v in enumerate(nums)}
        for a, b in operations:
            d[b] = d[a]
            d[a] = -1
        for k, v in d.items():
            if v != -1:
                nums[v] = k
        return nums

    def arrayChange(self, nums: List[int], operations: List[List[int]]) -> List[int]:
        d = {v: i for i, v in enumerate(nums)}
        for a, b in operations:
            i = d[a]
            nums[i] = b
            d[b] = i
            del d[a]
        return nums

    # the idea is similar to Disjoint Set(or Union-Find)
    def arrayChange(self, nums: List[int], operations: List[List[int]]) -> List[int]:
        m = {}
        for a, b in reversed(operations):  # reverse!
            m[a] = m.get(b, b)
        return [m.get(n, n) for n in nums]


# 2296 - Design a Text Editor - HARD
class TextEditor:
    # O(n ^ 2), O(n) for all operations
    def __init__(self):
        self.arr = ""
        self.p = 0

    def addText(self, text: str) -> None:
        self.arr = self.arr[: self.p] + text + self.arr[self.p :]
        self.p += len(text)
        return

    def deleteText(self, k: int) -> int:
        if self.p >= k:
            self.arr = self.arr[: self.p - k] + self.arr[self.p :]
            self.p -= k
            return k
        else:
            self.arr = self.arr[self.p :]
            r = self.p
            self.p = 0
            return r

    def cursorLeft(self, k: int) -> str:
        if self.p >= k + 10:
            self.p -= k
            return self.arr[self.p - 10 : self.p]
        elif self.p >= k:
            self.p -= k
            return self.arr[: self.p]
        else:
            self.p = 0
            return ""

    def cursorRight(self, k: int) -> str:
        if self.p + k <= len(self.arr):
            self.p += k
        else:
            self.p = len(self.arr)
        if self.p >= 10:
            return self.arr[self.p - 10 : self.p]
        else:
            return self.arr[: self.p]


class TextEditor:
    def __init__(self):
        self.s = ""
        self.cur = 0

    def addText(self, text: str) -> None:
        self.s = self.s[: self.cur] + text + self.s[self.cur :]
        self.cur += len(text)
        return

    def deleteText(self, k: int) -> int:
        new = max(0, self.cur - k)
        r = k if self.cur - k >= 0 else self.cur
        self.s = self.s[:new] + self.s[self.cur :]
        self.cur = new
        return r

    def cursorLeft(self, k: int) -> str:
        self.cur = max(0, self.cur - k)
        start = max(0, self.cur - 10)
        return self.s[start : self.cur]

    def cursorRight(self, k: int) -> str:
        self.cur = min(len(self.s), self.cur + k)
        start = max(0, self.cur - 10)
        return self.s[start : self.cur]


# doubly linked list
class DLLNode:
    __slots__ = ("pre", "nxt", "ch")

    def __init__(self, ch="", pre: "DLLNode" = None, nxt: "DLLNode" = None):
        self.pre = pre
        self.nxt = nxt
        self.ch = ch

    def insert(self, node: "DLLNode") -> "DLLNode":
        node.pre = self
        node.nxt = self.nxt
        node.pre.nxt = node
        node.nxt.pre = node
        return node

    def remove(self) -> None:
        self.pre.nxt = self.nxt
        self.nxt.pre = self.pre
        return


class TextEditor:
    # O(n), O(n) for add, and O(k) for other operations
    def __init__(self):
        self.root = self.cur = DLLNode()
        self.root.pre = self.root
        self.root.nxt = self.root

    def addText(self, text: str) -> None:
        for ch in text:
            self.cur = self.cur.insert(DLLNode(ch))
        return

    def deleteText(self, k: int) -> int:
        k0 = k
        while k and self.cur != self.root:
            self.cur = self.cur.pre
            self.cur.nxt.remove()
            k -= 1
        return k0 - k

    def text(self) -> str:
        s = []
        k = 10
        cur = self.cur
        while k and cur != self.root:
            s.append(cur.ch)
            cur = cur.pre
            k -= 1
        return "".join(reversed(s))

    def cursorLeft(self, k: int) -> str:
        while k and self.cur != self.root:
            self.cur = self.cur.pre
            k -= 1
        return self.text()

    def cursorRight(self, k: int) -> str:
        while k and self.cur.nxt != self.root:
            self.cur = self.cur.nxt
            k -= 1
        return self.text()


# 对顶栈
# Always maintain the left part of string in left stack
# and right part of the string in right stack which are divided by the cursor
class TextEditor:
    # O(n), O(n) for add, and O(k) for other operations
    def __init__(self):
        self.left = []
        self.right = []

    def addText(self, text: str) -> None:
        self.left.extend(list(text))
        return

    def deleteText(self, k: int) -> int:
        k0 = k
        while k and self.left:
            self.left.pop()
            k -= 1
        return k0 - k

    def cursorLeft(self, k: int) -> str:
        while k and self.left:
            self.right.append(self.left.pop())
            k -= 1
        return "".join(self.left[-10:])

    def cursorRight(self, k: int) -> str:
        while k and self.right:
            self.left.append(self.right.pop())
            k -= 1
        return "".join(self.left[-10:])


# 2299 - Strong Password Checker II - EASY
class Solution:
    def strongPasswordCheckerII(self, password: str) -> bool:
        a = b = c = d = e = False
        if len(password) >= 8:
            a = True
        pre = "aa"
        f = True
        for ch in password:
            if ord("a") <= ord(ch) <= ord("z"):
                b = True
            if ord("A") <= ord(ch) <= ord("Z"):
                c = True
            if ch.isdigit():
                d = True
            if ch in "!@#$%^&*()-+":
                e = True
            if pre == ch:
                f = False
            pre = ch
        return a == b == c == d == e == f == True
