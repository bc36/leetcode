import collections, bisect, itertools, functools, math, heapq, re
from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 2000 - Reverse Prefix of Word - EASY
class Solution:
    def reversePrefix(self, word: str, ch: str) -> str:
        idx = -1
        for i in range(len(word)):
            if word[i] == ch:
                idx = i
                break
        return word[: idx + 1][::-1] + word[idx + 1 :]

    def reversePrefix(self, word: str, ch: str) -> str:
        i = word.find(ch) + 1
        return word[:i][::-1] + word[i:]


# 2006 - Count Number of Pairs With Absolute Difference K - EASY
class Solution:
    def countKDifference(self, nums: List[int], k: int) -> int:
        seen, counter = collections.defaultdict(int), 0
        for num in nums:
            counter += seen[num - k] + seen[num + k]
            seen[num] += 1
        return counter

    def countKDifference(self, nums: List[int], k: int) -> int:
        ans, dic = 0, {}
        for n in nums:
            if n - k in dic:
                ans += dic[n - k]
            if n + k in dic:
                ans += dic[n + k]
            dic[n] = dic.get(n, 0) + 1
        return ans


# 2011 - Final Value of Variable After Performing Operations - EASY
class Solution:
    def finalValueAfterOperations(self, operations: List[str]) -> int:
        return sum(1 if "+" in v else -1 for v in operations)


# 2013 - Detect Squares - MEDIUM
class DetectSquares:
    def __init__(self):
        self.samex = collections.defaultdict(collections.Counter)

    def add(self, point: List[int]) -> None:
        x, y = point
        self.samex[x][y] += 1
        return

    def count(self, point: List[int]) -> int:
        x, y = point
        ans = 0

        if x not in self.samex:
            return 0
        samex = self.samex[x]
        for k, diffy in self.samex.items():
            if k != x:
                d = k - x
                ans += diffy[y] * samex[y + d] * diffy[y + d]
                ans += diffy[y] * samex[y - d] * diffy[y - d]
        return ans


# 2016 - Maximum Difference Between Increasing Elements - EASY
class Solution:
    def maximumDifference(self, nums: List[int]) -> int:
        ans, mi = 0, math.inf
        for n in nums:
            mi = min(mi, n)
            ans = max(ans, n - mi)
        return ans if ans > 0 else -1

    def maximumDifference(self, nums: List[int]) -> int:
        ans, premin = -1, nums[0]
        for i in range(1, len(nums)):
            if nums[i] > premin:
                ans = max(ans, nums[i] - premin)
            else:
                premin = nums[i]
        return ans


# 2022 - Convert 1D Array Into 2D Array - EASY
class Solution:
    def construct2DArray(self, original: List[int], m: int, n: int) -> List[List[int]]:
        if len(original) != m * n:
            return []
        ans = []
        for i in range(0, len(original), n):
            ans.append([x for x in original[i : i + n]])
        return ans

    def construct2DArray(self, original: List[int], m: int, n: int) -> List[List[int]]:
        return (
            [original[i : i + n] for i in range(0, len(original), n)]
            if len(original) == m * n
            else []
        )


# 2024 - Maximize the Confusion of an Exam - MEDIUM
class Solution:
    def maxConsecutiveAnswers(self, s: str, k: int) -> int:
        d = collections.defaultdict(int)
        l = 0
        for r in range(len(s)):
            d[s[r]] += 1
            if min(d["T"], d["F"]) > k:
                d[s[l]] -= 1
                l += 1
        return len(s) - l
        # see lc1438, similar reason
        # the length of window will not decrease
        return r - l + 1

    # the length of window will decrease
    def maxConsecutiveAnswers(self, s: str, k: int) -> int:
        d = collections.defaultdict(int)
        l = ans = 0
        for r in range(len(s)):
            d[s[r]] += 1
            while min(d["T"], d["F"]) > k:
                d[s[l]] -= 1
                l += 1
            ans = max(ans, r - l + 1)
        return ans


# 2027 - Minimum Moves to Convert String - EASY
class Solution:
    def minimumMoves(self, s: str) -> int:
        ans = t = 0
        for c in s:
            if c == "X":
                t += 1
            else:
                if t > 0:
                    t += 1
            if t == 3:
                t = 0
                ans += 1
        return ans + int(t > 0)

    def minimumMoves(self, s: str) -> int:
        covered = -1
        ans = 0
        for i, c in enumerate(s):
            if c == "X" and i > covered:
                ans += 1
                covered = i + 2
        return ans


# 2028 - Find Missing Observations - MEDIUM
class Solution:
    def missingRolls(self, rolls: List[int], mean: int, n: int) -> List[int]:
        m = len(rolls)
        t = (m + n) * mean - sum(rolls)
        if t > n * 6 or t < n:
            return []
        a = [0] * n
        while t:
            if t > n:
                for i in range(n):
                    a[i] += 1
                t -= n
            else:
                for i in range(t):
                    a[i] += 1
                t -= t
        return a

    def missingRolls(self, rolls: List[int], mean: int, n: int) -> List[int]:
        x = mean * (n + len(rolls)) - sum(rolls)
        if not n <= x <= n * 6:
            return []
        d, mod = divmod(x, n)
        return [d + 1] * mod + [d] * (n - mod)


# 2029 - Stone Game IX - MEDIUM
class Solution:
    def stoneGameIX(self, stones: List[int]) -> bool:
        d = [0, 0, 0]
        for n in stones:
            d[n % 3] += 1
        if d[0] % 2 == 0:
            return d[1] != 0 and d[2] != 0
        return d[2] > d[1] + 2 or d[1] > d[2] + 2


# 2032 - Two Out of Three - EASY
class Solution:
    def twoOutOfThree(
        self, nums1: List[int], nums2: List[int], nums3: List[int]
    ) -> List[int]:
        a = set(nums1)
        b = set(nums2)
        c = set(nums3)
        d = a.intersection(b)
        e = a.intersection(c)
        f = b.intersection(c)
        return list(d.union(e).union(f))

    def twoOutOfThree(
        self, nums1: List[int], nums2: List[int], nums3: List[int]
    ) -> List[int]:
        return list(
            (set(nums1) & set(nums2))
            | (set(nums1) & set(nums3))
            | (set(nums2) & set(nums3))
        )

    def twoOutOfThree(
        self, nums1: List[int], nums2: List[int], nums3: List[int]
    ) -> List[int]:
        a = [False] * 101
        b = [False] * 101
        c = [False] * 101
        for v in nums1:
            a[v] = True
        for v in nums2:
            b[v] = True
        for v in nums3:
            c[v] = True
        return list(i for i in range(101) if a[i] + b[i] + c[i] >= 2)


# 2034 - Stock Price Fluctuation - MEDIUM
class StockPrice:
    def __init__(self):
        self.maxPrice = []
        self.minPrice = []
        self.timePrice = {}
        self.maxTimestamp = 0

    def update(self, timestamp: int, price: int) -> None:
        heapq.heappush(self.maxPrice, (-price, timestamp))
        heapq.heappush(self.minPrice, (price, timestamp))
        self.timePrice[timestamp] = price
        self.maxTimestamp = max(self.maxTimestamp, timestamp)

    def current(self) -> int:
        return self.timePrice[self.maxTimestamp]

    def maximum(self) -> int:
        while True:
            price, timestamp = self.maxPrice[0]
            if -price == self.timePrice[timestamp]:
                return -price
            heapq.heappop(self.maxPrice)

    def minimum(self) -> int:
        while True:
            price, timestamp = self.minPrice[0]
            if price == self.timePrice[timestamp]:
                return price
            heapq.heappop(self.minPrice)


# 2037 - Minimum Number of Moves to Seat Everyone - EASY
class Solution:
    def minMovesToSeat(self, seats: List[int], students: List[int]) -> int:
        return sum(abs(x - y) for x, y in zip(sorted(seats), sorted(students)))


# 2038 - Remove Colored Pieces if Both Neighbors are the Same Color - MEDIUM
class Solution:
    def winnerOfGame(self, c: str) -> bool:
        a = b = 0
        for i in range(1, len(c) - 1):
            if c[i - 1] == c[i] == c[i + 1] == "A":
                a += 1
            elif c[i - 1] == c[i] == c[i + 1] == "B":
                b += 1
        return a > b


# 2039 - The Time When the Network Becomes Idle - MEDIUM
class Solution:
    # O(m + n) / O(m + n), m = len(edges)
    def networkBecomesIdle(self, edges: List[List[int]], p: List[int]) -> int:
        n = len(p)
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        vis = [False] * n
        vis[0] = True
        q = collections.deque([0])
        ans = 0
        d = 1
        while q:
            for _ in range(len(q)):
                u = q.popleft()
                for v in g[u]:
                    if vis[v]:
                        continue
                    vis[v] = True
                    q.append(v)
                    ans = max(ans, (d * 2 - 1) // p[v] * p[v] + d * 2 + 1)
            d += 1
        return ans

    def networkBecomesIdle(self, edges: List[List[int]], p: List[int]) -> int:
        g = {}
        for u, v in edges:
            g.setdefault(u, []).append(v)
            g.setdefault(v, []).append(u)
        dist = [-1] * len(g)
        dist[0] = 0
        val = 0
        q = [0]
        while q:
            val += 1
            new = []
            for u in q:
                for v in g[u]:
                    if dist[v] == -1:
                        dist[v] = val
                        new.append(v)
            q = new
        ans = 0
        for i in range(1, len(p)):
            d = 2 * dist[i]
            s = 0
            if d <= p[i]:
                s = d + 1
            else:
                s = d + d - (d - 1) % p[i]
            ans = max(ans, s)
        return ans


# 2042 - Check if Numbers Are Ascending in a Sentence - EASY
class Solution:
    def areNumbersAscending(self, s: str) -> bool:
        pre = -1
        for w in s.split(" "):
            if w[0].isdigit():
                x = int(w)
                if x <= pre:
                    return False
                pre = x
        return True


# 2043 - Simple Bank System - MEDIUM
class Bank:
    def __init__(self, balance: List[int]):
        self.b = balance
        self.n = len(balance)

    def transfer(self, a1: int, a2: int, money: int) -> bool:
        if a1 > self.n or a2 > self.n or self.b[a1 - 1] < money:
            return False
        self.b[a1 - 1] -= money
        self.b[a2 - 1] += money
        return True

    def deposit(self, a: int, money: int) -> bool:
        if a > self.n:
            return False
        self.b[a - 1] += money
        return True

    def withdraw(self, a: int, money: int) -> bool:
        if a > self.n or self.b[a - 1] < money:
            return False
        self.b[a - 1] -= money
        return True


# 2044 - Count Number of Maximum Bitwise-OR Subsets - MEDIUM
class Solution:
    # O(2^n * n) / O(1)
    def countMaxOrSubsets(self, nums: List[int]) -> int:
        def backtrack(arr: List[int], path: List[int]):
            t = 0
            for n in path:
                t |= n
            if t > self.mx:
                self.mx = t
                self.ans = 1
            elif t == self.mx:
                self.ans += 1
            for i in range(len(arr)):
                backtrack(arr[i + 1 :], path + [arr[i]])
            return

        self.mx = 0
        self.ans = 0
        backtrack(nums, [])
        return self.ans

    # O(2^n * n) / O(1), bitmask
    def countMaxOrSubsets(self, nums: List[int]) -> int:
        ans = 0
        mx = 0
        for n in nums:
            mx |= n
        for state in range(1, 1 << len(nums)):
            t = 0
            for i in range(len(nums)):
                if state & 1:
                    t |= nums[i]
                state >>= 1
            if t == mx:
                ans += 1
        return ans

    def countMaxOrSubsets(self, nums: List[int]) -> int:
        target = 0
        for num in nums:
            target |= num

        @functools.lru_cache(None)
        def dfs(i, cur):
            if cur == target:
                return 2 ** (len(nums) - i)
            elif i == len(nums):
                return 0
            return dfs(i + 1, cur) + dfs(i + 1, cur | nums[i])

        return dfs(0, 0)

    # Similar to knapsack problem, but use bitwise-or sum instead of math sum.
    # O(mn) / O(m), m = max(nums)
    def countMaxOrSubsets(self, nums: List[int]) -> int:
        cnt = collections.Counter([0])
        target = 0
        for n in nums:
            target |= n
            for k, v in list(cnt.items()):
                cnt[k | n] += v
        return cnt[target]

    def countMaxOrSubsets(self, nums: List[int]) -> int:
        dp = collections.Counter([0])
        for n in nums:
            for k, v in list(dp.items()):
                dp[k | n] += v
        return dp[max(dp)]


# 2045 - Second Minimum Time to Reach Destination - HARD
class Solution:
    def secondMinimum(
        self, n: int, edges: List[List[int]], time: int, change: int
    ) -> int:
        g = [[] for _ in range(n + 1)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)
        dist = [-1 for _ in range(n + 1)]
        dist[1] = 0
        dq = collections.deque([1])
        while dq:
            u = dq.popleft()
            for v in g[u]:
                if dist[v] == -1:
                    dq.append(v)
                    dist[v] = dist[u] + 1
        exist = False
        dq.append(n)
        while not exist and dq:
            u = dq.popleft()
            for v in g[u]:
                if dist[v] == dist[u]:
                    exist = True
                    break
                elif dist[v] == dist[u] - 1:
                    dq.append(v)
        d = dist[n] + 1 if exist else dist[n] + 2
        ans = 0
        for i in range(d):
            ans += time
            if (ans // change) % 2 == 1 and i != d - 1:
                ans += change - ans % change
        return ans

    # TODO
    def secondMinimum(
        self, n: int, edges: List[List[int]], time: int, change: int
    ) -> int:
        dis, dis2 = [float("inf")] * (n + 1), [float("inf")] * (n + 1)
        graph = collections.defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        dis[1] = 0
        dq = collections.deque([(0, 1)])
        while dq:
            cost, node = dq.popleft()
            for nei in graph[node]:
                new_cost = cost + time
                if (cost // change) % 2 == 1:
                    new_cost += change - (cost % change)
                if dis[nei] > new_cost:
                    dis2[nei], dis[nei] = dis[nei], new_cost
                    dq.append((new_cost, nei))
                elif dis[nei] < new_cost < dis2[nei]:
                    dis2[nei] = new_cost
                    dq.append((new_cost, nei))
        return dis2[n]


# 2047 - Number of Valid Words in a Sentence - EASY
class Solution:
    def countValidWords(self, sentence: str) -> int:
        return sum(
            bool(re.match(r"[a-z]*([a-z]-[a-z]+)?[!.,]?$", w)) for w in sentence.split()
        )

    def countValidWords(self, sentence: str) -> int:
        def valid(s: str) -> bool:
            hasHyphens = False
            for i, ch in enumerate(s):
                if ch.isdigit() or ch in "!.," and i < len(s) - 1:
                    return False
                if ch == "-":
                    if (
                        hasHyphens
                        or i == 0
                        or i == len(s) - 1
                        or not s[i - 1].islower()
                        or not s[i + 1].islower()
                    ):
                        return False
                    hasHyphens = True
            return True

        return sum(valid(s) for s in sentence.split())


# 2049 - Count Nodes With the Highest Score - MEDIUM
class Solution:
    def countHighestScoreNodes(self, parents: List[int]) -> int:
        g = collections.defaultdict(list)
        for i, v in enumerate(parents):
            g[v].append(i)

        def post(r):
            nonlocal ans, mx
            # if len(g[r]) == 0:
            #     if n - 1 > mx:
            #         ans, mx = 1, n - 1
            #     elif n - 1 == mx:
            #         ans += 1
            #     return 1
            p = 1
            top = n - 1
            child = 0
            for node in g[r]:
                ch = post(node)
                child += ch
                top -= ch
                p *= max(ch, 1)
            p *= max(top, 1)
            if p > mx:
                ans = 1
                mx = p
            elif p == mx:
                ans += 1
            return child + 1

        ans = mx = 0
        n = len(parents)
        post(0)
        return ans

    def countHighestScoreNodes(self, parents: List[int]) -> int:
        n = len(parents)
        g = collections.defaultdict(list)
        for i, p in enumerate(parents):
            g[p].append(i)
        mx, ans = 0, 0

        def dfs(node):
            left = dfs(g[node][0]) if g[node] else 0
            right = dfs(g[node][1]) if len(g[node]) == 2 else 0
            nonlocal mx, ans
            if (
                score := max(1, (n - left - right - 1)) * max(1, left) * max(1, right)
            ) > mx:
                mx, ans = score, 1
            elif score == mx:
                ans += 1
            return left + right + 1

        dfs(0)
        return ans

    def countHighestScoreNodes(self, parents: List[int]) -> int:
        n = len(parents)
        children = [[] for _ in range(n)]
        for node, p in enumerate(parents):
            if p != -1:
                children[p].append(node)
        maxScore, cnt = 0, 0

        def dfs(node: int) -> int:
            score = 1
            size = n - 1
            for ch in children[node]:
                sz = dfs(ch)
                score *= sz
                size -= sz
            if node != 0:
                score *= size
            nonlocal maxScore, cnt
            if score == maxScore:
                cnt += 1
            elif score > maxScore:
                maxScore, cnt = score, 1
            return n - size

        dfs(0)
        return cnt


# 2055 - Plates Between Candles - MEDIUM
class Solution:
    def platesBetweenCandles(self, s: str, queries: List[List[int]]) -> List[int]:
        n = len(s)
        preSum = [0] * n
        summ = 0
        left = [0] * n
        l = -1
        for i, ch in enumerate(s):
            if ch == "*":
                summ += 1
            else:
                l = i
            preSum[i] = summ
            left[i] = l
        right = [0] * n
        r = -1
        for i in range(n - 1, -1, -1):
            if s[i] == "|":
                r = i
            right[i] = r
        ans = [0] * len(queries)
        for i, (x, y) in enumerate(queries):
            x, y = right[x], left[y]
            if x >= 0 and y >= 0 and x < y:
                ans[i] = preSum[y] - preSum[x]
        return ans

    # O(n + Qlogn) / O(n + Q)
    def platesBetweenCandles(self, s: str, queries: List[List[int]]) -> List[int]:
        arr = [i for i, c in enumerate(s) if c == "|"]
        ans = []
        for a, b in queries:
            i = bisect.bisect_left(arr, a)
            j = bisect.bisect_right(arr, b) - 1
            ans.append((arr[j] - arr[i]) - (j - i) if i < j else 0)
        return ans


# 2078 - Two Furthest Houses With Different Colors - EASY
class Solution:
    # O(n**2) / O(n)
    def maxDistance(self, colors: List[int]) -> int:
        ans = 0
        d = {}
        for i, v in enumerate(colors):
            if v not in d:
                d[v] = [i, i]
            else:
                d[v][1] = i
        for c1, v1 in d.items():
            for c2, v2 in d.items():
                if c1 == c2:
                    continue
                ans = max(
                    ans,
                    abs(v1[0] - v2[1]),
                    abs(v1[1] - v2[0]),
                )
        return ans

    # O(n**2) / O(1)
    def maxDistance(self, c: List[int]) -> int:
        ans = 0
        for i in range(len(c) - 1):
            for j in range(i + 1, len(c)):
                if c[i] != c[j] and j - i > ans:
                    ans = j - i
        return ans

    # O(n) / O(1)
    def maxDistance(self, colors: List[int]) -> int:
        n = len(colors)
        if colors[0] != colors[-1]:
            return n - 1
        l = 1
        r = n - 1
        while colors[l] == colors[0]:
            l += 1
        while colors[r] == colors[-1]:
            r -= 1
        return max(n - 1 - l, r)


# 2079 - Watering Plants - MEDIUM
class Solution:
    def wateringPlants(self, plants: List[int], capacity: int) -> int:
        c = capacity
        ans = 0
        for i, p in enumerate(plants):
            if c < p:
                c = capacity
                ans += i * 2
            ans += 1
            c -= p
        return ans


# 2080 - Range Frequency Queries - MEDIUM
class RangeFreqQuery:
    # O(n + qlogn) / O(n)
    def __init__(self, arr: List[int]):
        self.d = collections.defaultdict(list)
        for i, v in enumerate(arr):
            self.d[v].append(i)

    def query(self, left: int, right: int, value: int) -> int:
        l = bisect.bisect_left(self.d[value], left)
        r = bisect.bisect_right(self.d[value], right)
        return r - l


# 2085 - Count Common Words With One Occurrence - EASY
class Solution:
    def countWords(self, words1: List[str], words2: List[str]) -> int:
        c1 = collections.Counter(words1)
        c2 = collections.Counter(words2)
        return sum(c2[k] == 1 for k, v in c1.items() if v == 1)


# 2089 - Find Target Indices After Sorting Array - EASY
class Solution:
    def targetIndices(self, nums: List[int], target: int) -> List[int]:
        return [i for i, v in enumerate(sorted(nums)) if v == target]

    def targetIndices(self, nums: List[int], target: int) -> List[int]:
        less = equal = 0
        for v in nums:
            if v < target:
                less += 1
            elif v == target:
                equal += 1
        return list(range(less, less + equal))


# 2090 - K Radius Subarray Averages - MEDIUM
class Solution:
    def getAverages(self, nums: List[int], k: int) -> List[int]:
        p = [0]
        for v in nums:
            p.append(p[-1] + v)
        ans = [-1] * len(nums)
        for i in range(k, len(nums) - k):
            ans[i] = (p[i + k + 1] - p[i - k]) // (2 * k + 1)
        return ans

    def getAverages(self, nums: List[int], k: int) -> List[int]:
        ans = [-1] * len(nums)
        summ = 0
        for i, v in enumerate(nums):
            summ += v
            if i > 2 * k - 1:
                ans[i - k] = summ // (2 * k + 1)
                summ -= nums[i - 2 * k]
        return ans


# 2091 - Removing Minimum and Maximum From Array - MEDIUM
class Solution:
    def minimumDeletions(self, nums: List[int]) -> int:
        px = nums.index(max(nums))
        pi = nums.index(min(nums))
        l = min(px, pi)
        r = max(px, pi)
        return min(l + 1 + len(nums) - r, r + 1, len(nums) - l)


# 2092 - Find All People With Secret - HARD
class Solution:
    # O(mlogm + m) / O(m)
    def findAllPeople(self, n: int, m: List[List[int]], firstPerson: int) -> List[int]:
        m.sort(key=lambda x: x[2])
        arr = [False] * n
        arr[0] = arr[firstPerson] = True
        l = 0
        while l < len(m):
            r = l
            e = collections.defaultdict(list)
            vis = set()
            while r < len(m) and m[l][2] == m[r][2]:
                x = m[r][0]
                y = m[r][1]
                e[x].append(y)
                e[y].append(x)
                if arr[x]:
                    vis.add(x)
                if arr[y]:
                    vis.add(y)
                r += 1
            q = list(vis)
            while q:
                new = []
                for v in q:
                    for nei in e[v]:
                        if nei not in vis:
                            arr[nei] = True
                            vis.add(nei)
                            new.append(nei)
                q = new
            l = r
        return [i for i, v in enumerate(arr) if v]

    def findAllPeople(self, n: int, m: List[List[int]], firstPerson: int) -> List[int]:
        m.sort(key=lambda x: x[2])
        arr = [False] * n
        arr[0] = arr[firstPerson] = True
        l = 0
        while l < len(m):
            r = l
            e = collections.defaultdict(list)
            pool = set()
            while r < len(m) and m[l][2] == m[r][2]:
                x = m[r][0]
                y = m[r][1]
                e[x].append(y)
                e[y].append(x)
                pool.add(x)
                pool.add(y)
                r += 1
            q = [v for v in pool if arr[v]]
            while q:
                new = []
                for v in q:
                    for nei in e[v]:
                        if not arr[nei]:
                            arr[nei] = True
                            new.append(nei)
                q = new
            l = r
        return [i for i, v in enumerate(arr) if v]

    def findAllPeople(self, n: int, m: List[List[int]], firstPerson: int) -> List[int]:
        class UnionFind:
            def __init__(self, n: int) -> None:
                self.p = [i for i in range(n)]

            def find(self, x: int) -> int:
                """path compression"""
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
                return

            def disconnect(self, x: int) -> None:
                self.p[x] = x
                return

        uf = UnionFind(n)
        uf.union(0, firstPerson)

        # method 1
        d = collections.defaultdict(list)
        for x, y, t in m:
            d[t].append((x, y))
        for t, experts in sorted(d.items()):
            for x, y in experts:
                uf.union(x, y)
            for x, y in experts:
                if uf.find(x) != uf.find(0):
                    uf.disconnect(x)
                    uf.disconnect(y)
        return [i for i in range(n) if uf.find(i) == uf.find(0)]

        # method 2
        m.sort(key=lambda x: x[2])
        for _, grp in itertools.groupby(m, key=lambda x: x[2]):
            pool = set()
            for x, y, _ in grp:
                uf.union(x, y)
                pool.add(x)
                pool.add(y)
            for p in pool:
                if uf.find(p) != uf.find(0):
                    uf.disconnect(p)
        return [i for i in range(n) if uf.find(i) == uf.find(0)]


# 2094 - Finding 3-Digit Even Numbers - EASY
class Solution:
    def findEvenNumbers(self, digits: List[int]) -> List[int]:
        ans = []
        cnt = collections.Counter(digits)
        for n in range(100, 1000, 2):
            x = n
            # a * 100 + b * 10 + c
            x, c = divmod(x, 10)
            x, b = divmod(x, 10)
            _, a = divmod(x, 10)
            cnt[a] -= 1
            cnt[b] -= 1
            cnt[c] -= 1
            if cnt[a] >= 0 and cnt[b] >= 0 and cnt[c] >= 0:
                ans.append(n)
            cnt[a] += 1
            cnt[b] += 1
            cnt[c] += 1
        return ans

    def findEvenNumbers(self, digits: List[int]) -> List[int]:
        ans = []
        cnt = collections.Counter(digits)
        for n in range(100, 1000, 2):
            f = True
            x = n
            d = collections.defaultdict(int)
            for _ in range(3):
                x, m = divmod(x, 10)
                d[m] += 1
                if d[m] > cnt[m]:
                    f = False
                    break
            if f:
                ans.append(n)
        return ans

    def findEvenNumbers(self, digits: List[int]) -> List[int]:
        ans = []
        cnt = collections.Counter(digits)
        for i in range(100, 1000, 2):
            m = collections.Counter([int(d) for d in str(i)])
            if all(cnt[d] >= m[d] for d in m.keys()):
                ans.append(i)
        return ans


# 2095 - Delete the Middle Node of a Linked List - MEDIUM
class Solution:
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        n = 0
        h = head
        while h:
            n += 1
            h = h.next
        if n == 1:
            return None
        h = head
        pre = None
        for _ in range(n // 2):
            pre = h
            h = h.next
        pre.next = h.next
        return head

    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(next=head)
        slow = dummy
        fast = dummy
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        slow.next = slow.next.next
        return dummy.next


# 2096 - Step-By-Step Directions From a Binary Tree Node to Another - MEDIUM
class Solution:
    def getDirections(
        self, root: Optional[TreeNode], startValue: int, destValue: int
    ) -> str:
        def dfs(root: TreeNode, x: int, arr: List[int]) -> bool:
            if not root:
                return False
            if root.val == x:
                return True
            if dfs(root.left, x, arr):
                arr.append("L")
                return True
            if dfs(root.right, x, arr):
                arr.append("R")
                return True
            return False

        a1 = []
        a2 = []
        dfs(root, startValue, a1)
        dfs(root, destValue, a2)
        while a1 and a2 and a1[-1] == a2[-1]:
            a1.pop()
            a2.pop()
        return "U" * len(a1) + "".join(reversed(a2))

    def getDirections(
        self, root: Optional[TreeNode], startValue: int, destValue: int
    ) -> str:
        def dfs(root: TreeNode) -> None:
            """record each father node"""
            nonlocal s, t
            if root.val == startValue:
                s = root
            if root.val == destValue:
                t = root
            if root.left:
                fa[root.left] = root
                dfs(root.left)
            if root.right:
                fa[root.right] = root
                dfs(root.right)
            return

        def path(cur: TreeNode) -> List[str]:
            """the path from root to leave"""
            p = []
            while cur != root:
                f = fa[cur]
                if cur == f.left:
                    p.append("L")
                else:
                    p.append("R")
                cur = f
            return p[::-1]

        fa = {}
        s = None
        t = None
        dfs(root)
        p1 = path(s)
        p2 = path(t)
        i = 0
        while i < min(len(p1), len(p2)):
            if p1[i] == p2[i]:
                i += 1
            else:
                break
        return "U" * (len(p1) - i) + "".join(p2[i:])

    # LCA 的迭代思路
    def getDirections(
        self, root: Optional[TreeNode], startValue: int, destValue: int
    ) -> str:
        def dfs(root: TreeNode) -> None:
            if root.left:
                fa[root.left.val] = (root.val, "L")
                dfs(root.left)
            if root.right:
                fa[root.right.val] = (root.val, "R")
                dfs(root.right)
            return

        fa = {}
        dfs(root)
        fathers = set()
        x = startValue
        while x != root.val:
            fathers.add(x)
            x = fa[x][0]
        fathers.add(root.val)
        down = collections.deque()
        x = destValue
        while x not in fathers:
            f, p = fa[x]
            down.appendleft(p)
            x = f
        up = 0
        y = startValue
        while y != x:
            up += 1
            y = fa[y][0]
        return up * "U" + "".join(down)
