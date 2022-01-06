import collections, itertools
from typing import List


class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# 1609 - Even Odd Tree - MEDIUM
class Solution:
    # bfs
    def isEvenOddTree(self, root: TreeNode) -> bool:
        dq = collections.deque([root])
        is_even = True
        while dq:
            pre = None
            for _ in range(len(dq)):
                n = dq.popleft()
                if is_even:
                    if n.val % 2 == 0: return False
                    if pre and pre.val >= n.val: return False
                else:
                    if n.val % 2 == 1: return False
                    if pre and pre.val <= n.val: return False
                if n.left: dq.append(n.left)
                if n.right: dq.append(n.right)
                pre = n
            is_even = not is_even  # bool value cannot use '~' to inverse
        return True

    def isEvenOddTree(self, root: TreeNode) -> bool:
        l, nodes = 0, [root]
        while nodes:
            nxt, cur = [], float('inf') if l % 2 else 0
            for n in nodes:
                if (l % 2 == n.val % 2) or (l % 2 and cur <= n.val) or (
                    (not l % 2) and cur >= n.val):
                    return False
                cur = n.val
                if n.left:
                    nxt.append(n.left)
                if n.right:
                    nxt.append(n.right)
            nodes = nxt
            l += 1
        return True


# 1614 - Maximum Nesting Depth of the Parentheses - EASY
class Solution:
    def maxDepth(self, s: str) -> int:
        ans = left = 0
        for ch in s:
            if ch == '(':
                left += 1
                ans = max(ans, left)
            elif ch == ')':
                left -= 1
        return ans


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