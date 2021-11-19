import collections
from typing import List


# Definition for a Node.
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# 1650 - Lowest Common Ancestor of a Binary Tree III - MEDIUM
class Solution:
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        path = set()
        while p:
            path.add(p)
            p = p.parent
        while q not in path:
            q = q.parent
        return q


# like running in a cycle
class Solution:
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        p1, p2 = p, q
        while p1 != p2:
            p1 = p1.parent if p1.parent else q
            p2 = p2.parent if p2.parent else p

        return p1


# 1676 - Lowest Common Ancestor of a Binary Tree IV - MEDIUM
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode',
                             nodes: 'List[TreeNode]') -> 'TreeNode':
        nodes = set(nodes)

        def lca(root):
            """Return LCA of nodes."""
            if not root or root in nodes:
                return root
            left, right = lca(root.left), lca(root.right)
            if left and right:
                return root
            return left or right

        return lca(root)