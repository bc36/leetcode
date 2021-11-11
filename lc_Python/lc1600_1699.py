# Definition for a Node.
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None

# 1650 - Lowest Common Ancestor of a Binary Tree III - MEDIUM
class Solution:
    def lowestCommonAncestor(self, p: Node, q: Node) -> Node:
        return