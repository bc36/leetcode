import bisect, collections, functools, heapq, itertools, math, operator, string
from typing import List, Optional, Tuple
import sortedcontainers


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 1302 - Deepest Leaves Sum - MEDIUM
class Solution:
    def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
        def dfs(root: TreeNode, lv: int) -> None:
            nonlocal ans, mx
            if not root:
                return
            if lv > mx:
                mx = lv
                ans = root.val
            elif lv == mx:
                ans += root.val

            dfs(root.left, lv + 1)
            dfs(root.right, lv + 1)
            return

        ans = mx = 0
        dfs(root, 0)
        return ans

    def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
        q = [root]
        while q:
            new = []
            summ = 0
            while q:
                n = q.pop()
                summ += n.val
                if n.left:
                    new.append(n.left)
                if n.right:
                    new.append(n.right)
            q = new
            ans = summ
        return ans


# 1305 - All Elements in Two Binary Search Trees - MEDIUM
class Solution:
    # O((m+n) * log(m+n)) / O(m + n)
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        def dfs(root: TreeNode):
            if root:
                arr.append(root.val)
                dfs(root.left)
                dfs(root.right)

        arr = []
        dfs(root1)
        dfs(root2)
        return sorted(arr)

    # O((m+n) * 2) / O(m + n)
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        def inorder(node: TreeNode, l: List[int]):
            if not node:
                return
            inorder(node.left, l)
            l.append(node.val)
            inorder(node.right, l)
            return

        l1, l2, ans = [], [], []
        inorder(root1, l1)
        inorder(root2, l2)
        i = j = 0
        while i < len(l1) or j < len(l2):
            if i < len(l1) and (j == len(l2) or l1[i] <= l2[j]):
                ans.append(l1[i])
                i += 1
            else:
                ans.append(l2[j])
                j += 1
        return ans


# 1306 - Jump Game III - MEDIUM
class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        visited, self.ans = set(), False

        def dfs(idx: int):
            visited.add(idx)
            if 0 <= idx + arr[idx] < len(arr) and idx + arr[idx] not in visited:
                dfs(idx + arr[idx])
            if 0 <= idx - arr[idx] < len(arr) and idx - arr[idx] not in visited:
                dfs(idx - arr[idx])
            if not arr[idx]:
                self.ans = True
            return

        dfs(start)
        return self.ans

    def canReach(self, arr: List[int], start: int) -> bool:
        dq, seen = collections.deque([start]), {start}
        while dq:
            cur = dq.popleft()
            if arr[cur] == 0:
                return True
            for child in cur - arr[cur], cur + arr[cur]:
                if 0 <= child < len(arr) and child not in seen:
                    seen.add(child)
                    dq.append(child)
        return False


# 1314 - Matrix Block Sum - MEDIUM
class Solution:
    def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        f = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                f[i][j] = sum(mat[i][max(j - k, 0) : min(j + k + 1, n)])
        ans = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                ans[i][j] = sum(
                    [f[p][j] for p in range(max(i - k, 0), min(i + k + 1, m))]
                )
        return ans

    def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        for i in range(m):
            for j in range(1, n):
                mat[i][j] += mat[i][j - 1]
        for i in range(1, m):
            for j in range(n):
                mat[i][j] += mat[i - 1][j]
        ans = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                r1 = max(0, i - k)
                c1 = max(0, j - k)
                r2 = min(m - 1, i + k)
                c2 = min(n - 1, j + k)
                ans[i][j] = (
                    mat[r2][c2]
                    - (mat[r2][c1 - 1] if c1 > 0 else 0)
                    - (mat[r1 - 1][c2] if r1 > 0 else 0)
                    + (mat[r1 - 1][c1 - 1] if r1 > 0 and c1 > 0 else 0)
                )
        return ans

    def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        f = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                f[i + 1][j + 1] = mat[i][j] + f[i][j + 1] + f[i + 1][j] - f[i][j]
        ans = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                r1 = max(0, i - k)
                c1 = max(0, j - k)
                r2 = min(m - 1, i + k)
                c2 = min(n - 1, j + k)
                ans[i][j] = (
                    f[r2 + 1][c2 + 1] - f[r2 + 1][c1] - f[r1][c2 + 1] + f[r1][c1]
                )
        return ans

    def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        f = [[0] * (n + 1) for _ in range(m + 1)]
        for i, j in itertools.product(range(m), range(n)):
            f[i + 1][j + 1] = mat[i][j] + f[i][j + 1] + f[i + 1][j] - f[i][j]
        ans = [[0] * n for _ in range(m)]
        for i, j in itertools.product(range(m), range(n)):
            r1, c1, r2, c2 = (
                max(0, i - k),
                max(0, j - k),
                min(m, i + k + 1),
                min(n, j + k + 1),
            )
            ans[i][j] = f[r2][c2] - f[r2][c1] - f[r1][c2] + f[r1][c1]
        return ans


# 1325 - Delete Leaves With a Given Value - MEDIUM
class Solution:
    def removeLeafNodes(self, root: TreeNode, target: int) -> TreeNode:
        def postorder(root):
            if not root:
                return None
            if postorder(root.left) and root.left.val == target:
                root.left = None
            if postorder(root.right) and root.right.val == target:
                root.right = None
            if not root.left and not root.right:
                return True
            return False

        postorder(root)
        return None if root.val == target and root.right == root.left == None else root

    def removeLeafNodes(self, root: TreeNode, target: int) -> TreeNode:
        if root.left:
            root.left = self.removeLeafNodes(root.left, target)
        if root.right:
            root.right = self.removeLeafNodes(root.right, target)
        return None if root.left == root.right and root.val == target else root

    def removeLeafNodes(self, root, target):
        if root:
            root.left = self.removeLeafNodes(root.left, target)
            root.right = self.removeLeafNodes(root.right, target)
            if root.val != target or root.left or root.right:
                return root


# 1326 - Minimum Number of Taps to Open to Water a Garden - HARD
class Solution:
    # O(nlogn) / O(n)
    def minTaps(self, n: int, ranges: List[int]) -> int:
        # arr = sorted((max(0, i - v), min(i + v, n)) for i, v in enumerate(ranges))
        arr = sorted((i - v, i + v) for i, v in enumerate(ranges))
        ans = l = r = reach = 0
        while r < n:
            while l < n + 1 and arr[l][0] <= r:
                reach = max(reach, arr[l][1])
                l += 1
            if r == reach:
                return -1
            r = reach
            ans += 1
        return ans

    def minTaps(self, n: int, ranges: List[int]) -> int:
        arr = sorted((i - v, i + v) for i, v in enumerate(ranges))
        ans = l = r = 0
        maxHeap = []
        while r < n:
            while l < n + 1 and arr[l][0] <= r:
                heapq.heappush(maxHeap, -arr[l][1])
                l += 1
            if not maxHeap:
                return -1
            r = -heapq.heappop(maxHeap)
            ans += 1
        return ans

    # O(n) / O(n), Jump Game II now
    def minTaps(self, n: int, ranges: List[int]) -> int:
        canReach = [0] * (n + 1)
        for i, v in enumerate(ranges):
            l = max(0, i - v)
            r = min(n, i + v)
            canReach[l] = max(canReach[l], r)
        ans = start = end = 0
        while end < n:
            ans += 1
            nxt = max(canReach[i] for i in range(start, end + 1))
            start, end = end, nxt
            if start == end:
                return -1
        return ans

    def minTaps(self, n: int, ranges: List[int]) -> int:
        canReach = [0] * (n + 1)
        for i, v in enumerate(ranges):
            l = max(0, i - v)
            canReach[l] = max(canReach[l], i + v)
        ans = furthest = currEnd = 0
        for i in range(n):
            furthest = max(furthest, canReach[i])
            if i == currEnd:
                if furthest == currEnd:
                    return -1
                currEnd = furthest
                ans += 1
        return ans

    def minTaps(self, n: int, ranges: List[int]) -> int:
        canReach = [0] * (n + 1)
        for i, v in enumerate(ranges):
            l = max(0, i - v)
            canReach[l] = max(canReach[l], i + v)
        ans = furthest = currEnd = 0
        for i in range(n):
            furthest = max(furthest, canReach[i])
            if i == furthest:
                return -1
            if i == currEnd:
                currEnd = furthest
                ans += 1
        return ans


# 1331 - Rank Transform of an Array - EASY
class Solution:
    def arrayRankTransform(self, arr: List[int]) -> List[int]:
        ranks = {v: i for i, v in enumerate(sorted(set(arr)), start=1)}
        return [ranks[v] for v in arr]


# 1332 - Remove Palindromic Subsequences - EASY
class Solution:
    def removePalindromeSub(self, s: str) -> int:
        return 1 if s == s[::-1] else 2


# 1333 - Filter Restaurants by Vegan-Friendly, Price and Distance - MEDIUM
class Solution:
    def filterRestaurants(
        self,
        restaurants: List[List[int]],
        veganFriendly: int,
        maxPrice: int,
        maxDistance: int,
    ) -> List[int]:
        return [
            i
            for i, _, v, p, d in sorted(restaurants, key=lambda x: (-x[1], -x[0]))
            if v >= veganFriendly and p <= maxPrice and d <= maxDistance
        ]


# 1335 - Minimum Difficulty of a Job Schedule - HARD
class Solution:
    # O(n^2 * d) / O(nd)
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        if len(jobDifficulty) < d:
            return -1

        @functools.lru_cache(None)
        def dfs(i: int, day: int) -> int:
            if day == 1:
                return max(jobDifficulty[i:])
            difficulty = 0
            cur = math.inf
            for j in range(i, len(jobDifficulty) - day + 1):
                difficulty = max(difficulty, jobDifficulty[j])
                cur = min(cur, difficulty + dfs(j + 1, day - 1))
            return cur

        return dfs(0, d)

    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        n = len(jobDifficulty)
        if n < d:
            return -1

        @functools.lru_cache(None)
        def dfs(i: int, day: int) -> int:
            if day == 1:
                return max(jobDifficulty[: i + 1])
            cur = math.inf
            difficulty = 0
            for j in range(i, day - 2, -1):
                difficulty = max(difficulty, jobDifficulty[j])
                cur = min(cur, dfs(j - 1, day - 1) + difficulty)
            return cur

        return dfs(n - 1, d)


# 1342 - Number of Steps to Reduce a Number to Zero - EASY
class Solution:
    def numberOfSteps(self, num: int) -> int:
        step = 0
        while num:
            if num & 1:
                num -= 1
            else:
                num //= 2
            step += 1
        return step


# 1343 - Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold - MEDIUM
class Solution:
    def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
        ans = 0
        cur = sum(arr[: k - 1])
        for i in range(k - 1, len(arr)):
            cur += arr[i]
            ans += cur >= k * threshold
            cur -= arr[i - k + 1]
        return ans


# 1345 - Jump Game IV - HARD
class Solution:
    # O(n) / O(n)
    def minJumps(self, arr: List[int]) -> int:
        n = len(arr)
        g = collections.defaultdict(list)
        # save left and right endpoints of the interval with the same value appearing consecutively
        for i in range(n):
            if i in (0, n - 1):
                g[arr[i]].append(i)
            elif arr[i] != arr[i - 1] or arr[i] != arr[i + 1]:
                g[arr[i]].append(i)
        visited = [True] + [False] * (n - 1)
        dq = collections.deque([(0, 0)])
        while dq:
            i, step = dq.popleft()
            for j in g.get(arr[i], []) + [i - 1, i + 1]:
                if 0 <= j < n and not visited[j]:
                    if j == n - 1:
                        return step + 1
                    visited[j] = True
                    dq.append((j, step + 1))
            g[arr[i]] = []  # has visited
        return 0

    def minJumps(self, arr: List[int]) -> int:
        g = collections.defaultdict(list)
        shorter = []
        size = 0
        # remove the consecutive repeated value in the 'arr'
        for i, v in enumerate(arr):
            if 0 < i < len(arr) - 1 and v == arr[i - 1] and v == arr[i + 1]:
                continue
            else:
                g[v].append(size)
                shorter.append(v)
                size += 1
        arr = shorter
        visited = {0}
        dq = collections.deque([(0, 0)])
        while dq:
            idx, step = dq.popleft()
            if idx == size - 1:
                return step
            value = arr[idx]
            for j in g[value] + [idx - 1, idx + 1]:
                if 0 <= j < size and j not in visited:
                    dq.append((j, step + 1))
                    visited.add(j)
            del g[value]
        return 0

    def minJumps(self, arr: List[int]) -> int:
        g = collections.defaultdict(list)
        n = len(arr)
        for i in range(n):
            g[arr[i]].append(i)
        dq = collections.deque([(0, 0)])
        seen = {0}
        while dq:
            i, step = dq.popleft()
            if i == n - 1:
                return step
            for nxt in g[arr[i]] + [i - 1, i + 1]:
                if 0 <= nxt < n and nxt not in seen:
                    seen.add(nxt)
                    dq.append((nxt, step + 1))
            del g[arr[i]]
        return -1

    def minJumps(self, arr: List[int]) -> int:
        g = collections.defaultdict(list)
        for i in range(len(arr)):
            g[arr[i]].append(i)
        dq = collections.deque([0])
        seen = {0}
        s = 0
        while dq:
            for _ in range(len(dq)):
                i = dq.popleft()
                if i == len(arr) - 1:
                    return s
                for j in [i - 1, i + 1] + g[arr[i]]:
                    if 0 <= j < len(arr) and j not in seen:
                        dq.append(j)
                        seen.add(j)
                g[arr[i]] = []
            s += 1
        return -1


# 1347 - Minimum Number of Steps to Make Two Strings Anagram - MEDIUM
class Solution:
    def minSteps(self, s: str, t: str) -> int:
        d = collections.Counter(s) - collections.Counter(t)
        return sum(abs(v) for v in d.values())


# 1349 - Maximum Students Taking Exam - HARD
class Solution:
    # 状态压缩, 状压dp
    def maxStudents(self, seats: List[List[str]]) -> int:
        # 二维打家劫舍 从最底层开始 顺序从左到右 从下到上的状态
        # 每行进入的时候都有一个 mask 代表上一行选择后屏蔽了哪些位置 由于列数小于 8 用int状态压缩 mask
        # 不用判断横轴末尾, 直接越界才刷新状态
        m, n = len(seats), len(seats[0])

        @functools.lru_cache(None)
        def dfs(x, y, pre, cur):
            if x == -1:
                return 0
            if y >= n:
                return dfs(x - 1, 0, cur, 0)
            if seats[x][y] == "#" or 1 << y + 1 & pre:
                return dfs(x, y + 1, pre, cur)
            # 如果选择坐该座位, 则占该座位 (1 << y), 并在后续的 dfs 搜索中 +1
            # 并且贪心地尽可能占据隔一排的位置 (1 << y + 2), 若违法则被上述 if 跳过, 直到越界退出搜索
            new = cur | 1 << y | 1 << y + 2
            return max(dfs(x, y + 1, pre, cur), 1 + dfs(x, y + 2, pre, new))

        return dfs(m - 1, 0, 0, 0)

    # 每一行只与上一行有关, 同时每一行最多 2^8 种状态, 我们自然想到进行状态压缩DP
    # dp[row][state] = max(dp[row - 1][last] + state.count())
    # O(m * 2^n * 2^n) / O(m * 2^n)
    def maxStudents(self, seats: List[List[str]]) -> int:
        m, n = len(seats), len(seats[0])
        dp = [[0] * (1 << n) for _ in range(m + 1)]
        # arr = [
        #     functools.reduce(
        #         lambda a, b: a | 1 << b, [0] + [j for j, c in enumerate(s) if c == "#"]
        #     )
        #     for s in seats
        # ]
        # arr = [functools.reduce(lambda a, b: a << 1 | (b == "#"), s, 0) for s in seats]
        arr = [sum((c == "#") << j for j, c in enumerate(s)) for s in seats]

        for row in range(m - 1, -1, -1):
            for j in range(1 << n):
                if not j & j << 1 and not j & j >> 1 and not j & arr[row]:
                    for k in range(1 << n):
                        if not j & k << 1 and not j & k >> 1:
                            dp[row][j] = max(dp[row][j], dp[row + 1][k] + j.bit_count())
        return max(dp[0])


# 1373 - Maximum Sum BST in Binary Tree - HARD
class Solution:
    def maxSumBST(self, root: Optional[TreeNode]) -> int:
        def dfs(root: Optional[TreeNode]) -> Tuple[int, int, int]:
            if not root:
                return 0, math.inf, -math.inf
            lv, lmi, lmx = dfs(root.left)
            rv, rmi, rmx = dfs(root.right)
            if root.val <= lmx or root.val >= rmi:
                return 0, -math.inf, math.inf
            s = lv + rv + root.val  # 这棵子树的所有节点值之和
            nonlocal ans
            ans = max(ans, s)
            return s, min(lmi, root.val), max(rmx, root.val)

        ans = 0  # 二叉搜索树可以为空
        dfs(root)
        return ans

    def maxSumBST(self, root: Optional[TreeNode]) -> int:
        def dfs(root):
            """后序遍历, 返回值 = (子树所有节点值之和, 子树包含的最大值, 子树包含的最小值, 递归上来的答案"""
            if not root:
                return 0, math.inf, -math.inf, 0
            lv, lmi, lmx, lans = dfs(root.left)
            rv, rmi, rmx, rans = dfs(root.right)
            if lmx >= root.val or rmi <= root.val:
                return 0, -math.inf, math.inf, max(lans, rans)
            t = lv + rv + root.val
            mx = max(root.val, lmx, rmx)
            mi = min(root.val, lmi, rmi)
            return t, mi, mx, max(t, lans, rans)

        return dfs(root)[3]


# 1374 - Generate a String With Characters That Have Odd Counts - EASY
class Solution:
    def generateTheString(self, n: int) -> str:
        if n & 1:
            return "a" * n
        return "a" + "b" * (n - 1)


# 1375 - Number of Times Binary String Is Prefix-Aligned - MEDIUM
class Solution:
    # O(n) / O(n)
    def numTimesAllBlue(self, flips: List[int]) -> int:
        n = len(flips)
        vis = [False] * n
        ans = p = mx = 0
        for x in flips:
            vis[x - 1] = True
            mx = max(mx, x)
            while p < n and vis[p]:
                p += 1
            ans += mx == p
        return ans

    # O(n) / O(1)
    def numTimesAllBlue(self, flips: List[int]) -> int:
        ans = mx = 0
        for i, x in enumerate(flips):
            mx = max(mx, x)
            ans += mx == i + 1
        return ans


# 1376 - Time Needed to Inform All Employees - MEDIUM
class Solution:
    def numOfMinutes(
        self, n: int, headID: int, manager: List[int], informTime: List[int]
    ) -> int:
        sub = collections.defaultdict(list)
        for i, v in enumerate(manager):
            sub[v].append(i)

        def dfs(x: int) -> int:
            if informTime[x] == 0:
                return 0
            return informTime[x] + max(dfs(i) for i in sub[x])

        return dfs(headID)

    def numOfMinutes(
        self, n: int, headID: int, manager: List[int], informTime: List[int]
    ) -> int:
        @functools.lru_cache(None)
        def dfs(x: int) -> int:
            if manager[x] < 0:
                return informTime[x]
            return dfs(manager[x]) + informTime[x]

        return max(dfs(i) for i in range(n))

    def numOfMinutes(
        self, n: int, headID: int, manager: List[int], informTime: List[int]
    ) -> int:
        def dfs(x: int) -> int:
            if manager[x] >= 0:
                informTime[x] += dfs(manager[x])
                manager[x] = -1  # visited
            return informTime[x]

        return max(dfs(i) for i in range(n))


# 1377 - Frog Position After T Seconds - HARD
class Solution:
    def frogPosition(
        self, n: int, edges: List[List[int]], t: int, target: int
    ) -> float:
        g = [[] for _ in range(n + 1)]
        g[1] = [0]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)
        ans = 0

        def dfs(x: int, fa: int, t: int, p: int) -> True:
            # 恰好到达 / target 是叶子停在原地
            if x == target and (t == 0 or len(g[x]) == 1):
                nonlocal ans
                ans = 1 / p
                return True
            if x == target or t == 0:
                return False
            for y in g[x]:
                if y != fa and dfs(y, x, t - 1, p * (len(g[x]) - 1)):
                    return True
            return False

        dfs(1, 0, t, 1)
        return ans

    def frogPosition(
        self, n: int, edges: List[List[int]], t: int, target: int
    ) -> float:
        g = collections.defaultdict(list)
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)
        dq = collections.deque([(1, 1.0)])
        vis = [False] * (n + 1)
        vis[1] = True
        while dq and t >= 0:
            for _ in range(len(dq)):
                x, p = dq.popleft()
                cnt = len(g[x]) - int(x != 1)
                if x == target:
                    return p if cnt * t == 0 else 0
                for y in g[x]:
                    if not vis[y]:
                        vis[y] = True
                        dq.append((y, p / cnt))
            t -= 1
        return 0


# 1379 - Find a Corresponding Node of a Binary Tree in a Clone of That Tree - EASY
class Solution:
    def getTargetCopy(
        self, original: TreeNode, cloned: TreeNode, target: TreeNode
    ) -> TreeNode:
        def dfs(root: TreeNode, val: int) -> TreeNode:
            if not root or root.val == val:
                return root
            if root.left and root.left.val == val:
                return root.left
            if root.right and root.right.val == val:
                return root.right
            return dfs(root.left, val) or dfs(root.right, val)

        return dfs(cloned, target.val)

    def getTargetCopy(
        self, original: TreeNode, cloned: TreeNode, target: TreeNode
    ) -> TreeNode:
        if original is None or original is target:
            return cloned
        return self.getTargetCopy(
            original.left, cloned.left, target
        ) or self.getTargetCopy(original.right, cloned.right, target)


# 1380 - Lucky Numbers in a Matrix - EASY
class Solution:
    def luckyNumbers(self, matrix: List[List[int]]) -> List[int]:
        ans = []
        colmax = {}
        m, n = len(matrix), len(matrix[0])
        for j in range(n):
            for i in range(m):
                if matrix[i][j] > colmax.get(j, 0):
                    colmax[j] = matrix[i][j]
        s = set(colmax.values())
        for i in range(m):
            rowmin = math.inf
            for j in range(n):
                if matrix[i][j] < rowmin:
                    rowmin = matrix[i][j]
            if rowmin in s:
                ans.append(rowmin)
        return ans

    def luckyNumbers(self, matrix: List[List[int]]) -> List[int]:
        minRow = [min(row) for row in matrix]
        maxCol = [max(col) for col in zip(*matrix)]
        ans = []
        for i, row in enumerate(matrix):
            for j, x in enumerate(row):
                if x == minRow[i] == maxCol[j]:
                    ans.append(x)
        return ans

    def luckyNumbers(self, matrix: List[List[int]]) -> List[int]:
        ans = []
        cols = list(zip(*matrix))
        for rows in matrix:
            num = min(rows)
            c = rows.index(num)
            if max(cols[c]) == num:
                ans.append(num)
        return ans


# 1381 - Design a Stack With Increment Operation - MEDIUM
class CustomStack:
    def __init__(self, maxSize: int):
        self.m = maxSize
        self.l = 0
        self.s = []

    def push(self, x: int) -> None:
        if self.l < self.m:
            self.s.append(x)
            self.l += 1

    def pop(self) -> int:
        if self.s:
            r = self.s.pop()
            self.l -= 1
            return r
        return -1

    # O(k)
    def increment(self, k: int, val: int) -> None:
        i = 0
        while i < k and i < self.l:
            self.s[i] += val
            i += 1


class CustomStack:
    def __init__(self, maxSize: int):
        self.stk = [0] * maxSize
        self.add = [0] * maxSize
        self.top = -1

    def push(self, x: int) -> None:
        if self.top < len(self.stk) - 1:
            self.top += 1
            self.stk[self.top] = x

    def pop(self) -> int:
        if self.top == -1:
            return -1
        ret = self.stk[self.top] + self.add[self.top]
        if self.top != 0:
            self.add[self.top - 1] += self.add[self.top]
        self.add[self.top] = 0
        self.top -= 1
        return ret

    def increment(self, k: int, val: int) -> None:
        l = min(k - 1, self.top)
        if l >= 0:
            self.add[l] += val


# 1385 - Find the Distance Value Between Two Arrays - EASY
class Solution:
    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        return sum(1 - any(abs(x - y) <= d for y in arr2) for x in arr1)
        return sum(all(abs(x - y) > d for y in arr2) for x in arr1)


# 1388 - Pizza With 3n Slices - HARD
class Solution:
    # 转化成: 给一个长度为 3n 的环状序列, 你可以在其中选择 n 个数, 并且任意两个数不能相邻, 求这 n 个数的最大值
    # dp[i][j] 前 i 个数中选择了 j 个不相邻的数的最大和
    # 动态规划的解决方法和 213. 打家劫舍 II 较为相似
    # 首先考虑该序列不是环状时的解决方法
    # 当该序列是环状序列时, 普通序列中的第一个和最后一个数不能同时选, 需要对普通序列进行两遍动态
    # O(n^2) / O(n^2)
    def maxSizeSlices(self, slices: List[int]) -> int:
        def calc(nums: List[int]) -> int:
            m = len(nums)
            f = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    f[i][j] = max(
                        f[i - 1][j], (f[i - 2][j - 1] if i >= 2 else 0) + nums[i - 1]
                    )
            return f[m][n]

        n = len(slices) // 3
        return max(calc(slices[:-1]), calc(slices[1:]))


# 1396 - Design Underground System - MEDIUM
class UndergroundSystem:
    def __init__(self):
        self.p = {}
        self.ids = {}

    def checkIn(self, id: int, stationName: str, t: int) -> None:
        self.ids[id] = (stationName, t)
        return

    def checkOut(self, id: int, stationName: str, t: int) -> None:
        start, pre = self.ids.pop(id)
        if (start, stationName) in self.p:
            total, cnt = self.p.pop((start, stationName))
        else:
            total, cnt = 0, 0
        total += t - pre
        cnt += 1
        self.p[(start, stationName)] = (total, cnt)
        return

    def getAverageTime(self, startStation: str, endStation: str) -> float:
        t, c = self.p[(startStation, endStation)]
        return t / c


class UndergroundSystem:
    def __init__(self):
        self.ids = {}
        self.p = collections.defaultdict(int)
        self.freq = collections.defaultdict(int)

    def checkIn(self, id, stationName, t):
        self.ids[id] = (stationName, t)

    def checkOut(self, id, stationName, t):
        name, pre = self.ids.pop(id)
        self.p[(name, stationName)] += t - pre
        self.freq[(name, stationName)] += 1

    def getAverageTime(self, startStation, endStation):
        return (
            self.p[(startStation, endStation)] / self.freq[(startStation, endStation)]
        )
