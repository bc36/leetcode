import collections
from typing import List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 938 - Range Sum of BST - EASY
# dfs
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        ans = []

        # preorder
        def dfs(root: TreeNode, ans: List[int]):
            if not root:
                return
            # process
            if root.val >= low and root.val <= high:
                ans.append(root.val)
            # left node and right node
            if root.left:
                dfs(root.left, ans)
            if root.right:
                dfs(root.right, ans)
            return

        dfs(root, ans)
        return sum(ans)


class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        # preorder
        def dfs(root: TreeNode, ans: int):
            if not root:
                return 0
            # process
            val, left, right = 0, 0, 0
            if root.val >= low and root.val <= high:
                val = root.val
            # left node and right node
            if root.left:
                left = dfs(root.left, ans)
            if root.right:
                right = dfs(root.right, ans)
            return val + left + right

        return dfs(root, 0)

# since its a binary search tree which means that left.val < root.val < right.val
# so we can speed up by jump some unqualified node (the value greater than high or smalller than low)
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        if not root:
            return 0
        if root.val > high:
            return self.rangeSumBST(root.left, low, high)
        if root.val < low:
            return self.rangeSumBST(root.right, low, high)
        return root.val + self.rangeSumBST(
            root.left, low, high) + self.rangeSumBST(root.right, low, high)


# bfs
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        dq = collections.deque()
        ans = 0
        if root:
            dq.append(root)
        while dq:
            # process a layer of nodes
            size = len(dq)
            for i in range(size):
                # get one node to process from left side -> FIFO
                tmpNode = dq.popleft()
                # add the qualified value
                if tmpNode.val >= low and tmpNode.val <= high:
                    ans += tmpNode.val
                # add new children node to dq
                if tmpNode.left:
                    dq.append(tmpNode.left)
                if tmpNode.right:
                    dq.append(tmpNode.right)
        return ans