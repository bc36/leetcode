import collections
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


# 1345 - Jump Game IV - HARD
class Solution:
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