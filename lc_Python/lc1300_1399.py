import collections, itertools, functools, math
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 1306 - Jump Game III - MEDIUM
class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        visited, self.ans = set(), False

        def dfs(idx: int):
            visited.add(idx)
            if 0 <= idx + arr[idx] < len(
                    arr) and idx + arr[idx] not in visited:
                dfs(idx + arr[idx])
            if 0 <= idx - arr[idx] < len(
                    arr) and idx - arr[idx] not in visited:
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
        sums = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                sums[i][j] = sum(mat[i][max(j - k, 0):min(j + k + 1, n)])
        ans = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                ans[i][j] = sum([
                    sums[p][j] for p in range(max(i - k, 0), min(i + k + 1, m))
                ])
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
                r1, c1, r2, c2 = max(0, i - k), max(0, j - k), min(
                    m - 1, i + k), min(n - 1, j + k)
                ans[i][j] = mat[r2][c2] - (
                    mat[r2][c1 - 1]
                    if c1 > 0 else 0) - (mat[r1 - 1][c2] if r1 > 0 else 0) + (
                        mat[r1 - 1][c1 - 1] if r1 > 0 and c1 > 0 else 0)
        return ans

    def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        ps = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                ps[i +
                   1][j +
                      1] = mat[i][j] + ps[i][j + 1] + ps[i + 1][j] - ps[i][j]
        ans = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                r1, c1, r2, c2 = max(0, i - k), max(0, j - k), min(
                    m - 1, i + k), min(n - 1, j + k)
                ans[i][j] = ps[r2 + 1][c2 + 1] - ps[r2 + 1][c1] - ps[r1][
                    c2 + 1] + ps[r1][c1]
        return ans

    def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        ps = [[0] * (n + 1) for _ in range(m + 1)]
        for i, j in itertools.product(range(m), range(n)):
            ps[i + 1][j +
                      1] = mat[i][j] + ps[i][j + 1] + ps[i + 1][j] - ps[i][j]
        ans = [[0] * n for _ in range(m)]
        for i, j in itertools.product(range(m), range(n)):
            r1, c1, r2, c2 = max(0, i - k), max(0,
                                                j - k), min(m, i + k + 1), min(
                                                    n, j + k + 1)
            ans[i][j] = ps[r2][c2] - ps[r2][c1] - ps[r1][c2] + ps[r1][c1]
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
# 1332 - Remove Palindromic Subsequences - EASY
class Solution:
    def removePalindromeSub(self, s: str) -> int:
        return 1 if s == s[::-1] else 2
            
# 1345 - Jump Game IV - HARD
class Solution:
    # O(n) / O(n)
    def minJumps(self, arr: List[int]) -> int:
        # +1 / -1 / same value
        n, idx = len(arr), {}  # defaultdict(list)
        for i in range(n):
            # save left and right endpoints of the interval with the same value appearing consecutively
            if i in (0, n - 1):
                idx[arr[i]] = idx.get(arr[i], []) + [i]
            elif arr[i] != arr[i - 1] or arr[i] != arr[i + 1]:
                idx[arr[i]] = idx.get(arr[i], []) + [i]
        visited = [True] + [False] * (n - 1)
        queue = [(0, 0)]  # deque
        while queue:
            i, step = queue.pop(0)
            for j in (idx.get(arr[i], []) + [i - 1, i + 1]):
                if 0 <= j < n and not visited[j]:
                    if j == n - 1:
                        return step + 1
                    visited[j] = True
                    queue.append((j, step + 1))
            idx[arr[i]] = []  # has visited
        return 0

    def minJumps(self, arr: List[int]) -> int:
        graph = collections.defaultdict(list)
        for i, v in enumerate(arr):
            graph[v].append(i)
        visited, n = {0}, len(arr)
        dq = collections.deque([(0, 0)])
        while dq:
            i, step = dq.popleft()
            for j in graph[arr[i]]:
                if j not in visited:
                    if j == n - 1:
                        return step + 1
                    visited.add(j)
                    dq.append((j, step + 1))
            for j in [i - 1, i + 1]:
                if 0 <= j < n and j not in visited:
                    if j == n - 1:
                        return step + 1
                    visited.add(j)
                    dq.append((j, step + 1))
            graph[arr[i]] = []
        return 0