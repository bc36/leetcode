import collections, itertools, heapq, functools, math
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


# 1629 - Slowest Key - EASY
class Solution:
    def slowestKey(self, rT: List[int], keys: str) -> str:
        ans, time = keys[0], rT[0]
        for i in range(len(rT) - 1):
            if rT[i + 1] - rT[i] > time or rT[i + 1] - rT[i] == time and keys[
                    i + 1] > ans:
                time = rT[i + 1] - rT[i]
                ans = keys[i + 1]
        return ans


# 1648 - Sell Diminishing-Valued Colored Balls - MEDIUM
class Solution:
    # O(nlogC) / O(1), C = max(inventory)
    def maxProfit(self, inventory: List[int], orders: int) -> int:
        l = 0
        r = max(inventory)
        T = -1
        while l < r:
            mid = (l + r) // 2
            count = sum(n - mid for n in inventory if n >= mid)
            if count <= orders:
                r = mid
            else:
                T = mid + 1
                l = mid + 1
        fn = lambda x, y: (x + y) * (y - x + 1) // 2
        rest = orders - sum(n - T for n in inventory if n >= T)
        ans = 0
        for n in inventory:
            if n >= T:
                if rest > 0:
                    ans += fn(T, n)
                    rest -= 1
                else:
                    ans += fn(T + 1, n)
        return ans % (10**9 + 7)

    # O(nlogn) / O(1)
    def maxProfit(self, inv: List[int], orders: int) -> int:
        fn = lambda s, e: (e + s) * (e - s + 1) // 2
        inv.sort(reverse=True)
        inv.append(0)
        ans = 0
        cnt = 1  # the number of maximum values
        for i in range(len(inv) - 1):
            if inv[i] > inv[i + 1]:
                if cnt * (inv[i] - inv[i + 1]) < orders:
                    ans += cnt * fn(inv[i + 1] + 1, inv[i])
                    orders -= cnt * (inv[i] - inv[i + 1])
                else:
                    whole, remaining = divmod(orders, cnt)
                    ans += cnt * fn(inv[i] - whole + 1, inv[i])
                    ans += remaining * (inv[i] - whole)
                    break
            cnt += 1
        return ans % (10**9 + 7)


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


# 1672 - Richest Customer Wealth - EASY
class Solution:
    def maximumWealth(self, accounts: List[List[int]]) -> int:
        # return max(sum(a) for a in accounts)
        return max(map(sum, accounts))


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


# 1688 - Count of Matches in Tournament - EASY
class Solution:
    def numberOfMatches(self, n: int) -> int:
        ans = 0
        while n > 1:
            if n & 1:
                ans += (n - 1) // 2
                n += 1
            else:
                ans += n // 2
            n //= 2
        return ans

    def numberOfMatches(self, n: int) -> int:
        return n - 1
