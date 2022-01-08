from typing import List
import collections, queue


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 1428 - Leftmost Column with at Least a One - MEDIUM


# 1446 - Consecutive Characters - EASY
class Solution:
    def maxPower(self, s: str) -> int:
        tmp, ans = 1, 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                tmp += 1
                ans = max(tmp, ans)
            else:
                tmp = 1
        return ans


class Solution:
    def maxPower(self, s: str) -> int:
        i, ans = 0, 1
        while i < len(s):
            j = i
            while j < len(s) and s[i] == s[j]:
                j += 1
            ans = max(ans, j - i)
            i = j
        return ans


# 1448 - Count Good Nodes in Binary Tree - MEDIUM
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        def inorder(root, premax):
            if not root:
                return
            if root.val >= premax:
                self.cnt += 1
                premax = root.val
            inorder(root.left, premax)
            inorder(root.right, premax)
            return

        self.cnt = 0
        inorder(root, float('-inf'))
        return self.cnt

    def goodNodes(self, root: TreeNode) -> int:
        def inorder(root, premax):
            if root.val >= premax: self.cnt += 1
            if root.left: inorder(root.left, max(premax, root.val))
            if root.right: inorder(root.right, max(premax, root.val))
            return

        self.cnt = 0
        inorder(root, root.val)
        return self.cnt

    def goodNodes(self, root: TreeNode) -> int:
        def inorder(root, premax):
            if not root: return 0
            premax = max(root.val, premax)
            return (root.val >= premax) + inorder(root.left, premax) + inorder(
                root.right, premax)

        return inorder(root, root.val)