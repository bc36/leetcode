import collections, itertools, functools, math, heapq
from typing import List, Optional


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
        # rg = sorted([(max(0, i - v), min(i + v, n)) for i, v in enumerate(ranges)])
        rg = sorted([[i - v, i + v] for i, v in enumerate(ranges)])
        ans = l = r = reach = 0
        while r < n:
            while l < n + 1 and rg[l][0] <= r:
                reach = max(reach, rg[l][1])
                l += 1
            if r == reach:
                return -1
            r = reach
            ans += 1
        return ans

    def minTaps(self, n: int, ranges: List[int]) -> int:
        arr = sorted([(i - v, i + v) for i, v in enumerate(ranges)])
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
        max_range = [0] * (n + 1)
        for i, v in enumerate(ranges):
            l = max(0, i - v)
            r = min(n, i + v)
            max_range[l] = max(max_range[l], r)
        start = end = step = 0
        while end < n:
            step += 1
            nxt = max(max_range[i] for i in range(start, end + 1))
            start, end = end, nxt
            if start == end:
                return -1
        return step

    def minTaps(self, n: int, ranges: List[int]) -> int:
        jump = [0] * (n + 1)
        for i in range(n + 1):
            l = max(0, i - ranges[i])
            r = min(n, i + ranges[i])
            jump[l] = max(jump[l], r)
        ans = furthest = currEnd = 0
        for i in range(n):
            furthest = max(furthest, jump[i])
            if i == currEnd:
                if furthest <= currEnd:
                    return -1
                currEnd = furthest
                ans += 1
        return ans

    def minTaps(self, n: int, ranges: List[int]) -> int:
        reach = [0] * (n + 1)
        for i, width in enumerate(ranges):
            start = max(0, i - width)
            end = min(n, i + width)
            reach[start] = max(reach[start], end)
        ans = furthest = currEnd = 0
        for i in range(n):
            furthest = max(furthest, reach[i])
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


# 1374 - Generate a String With Characters That Have Odd Counts - EASY
class Solution:
    def generateTheString(self, n: int) -> str:
        if n & 1:
            return "a" * n
        return "a" + "b" * (n - 1)


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
