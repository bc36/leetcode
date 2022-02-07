from typing import List
import collections, heapq, functools, itertools, math


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 1405 - Longest Happy String - MEDIUM
class Solution:
    def longestDiverseString(self, a: int, b: int, c: int) -> str:
        ans, pq = "", []
        if a: pq.append([-a, 'a'])
        if b: pq.append([-b, 'b'])
        if c: pq.append([-c, 'c'])
        heapq.heapify(pq)
        while pq:
            n = len(ans)
            cur = heapq.heappop(pq)
            if n >= 2 and ans[-1] == ans[-2] == cur[1]:
                if not pq:
                    break
                nxt = heapq.heappop(pq)
                ans += nxt[1]
                if nxt[0] + 1 < 0:
                    nxt[0] += 1
                    heapq.heappush(pq, nxt)
                heapq.heappush(pq, cur)
            else:
                ans += cur[1]
                if cur[0] + 1 < 0:
                    cur[0] += 1
                    heapq.heappush(pq, cur)
        return ans

    def longestDiverseString(self, a: int, b: int, c: int) -> str:
        def generate(a, b, c, aa, bb, cc):
            if a < b:
                return generate(b, a, c, bb, aa, cc)
            if b < c:
                return generate(a, c, b, aa, cc, bb)
            if b == 0:
                return min(2, a) * aa
            usedA = min(2, a)
            usedB = 1 if a - usedA >= b else 0
            return usedA * aa + usedB * bb + generate(a - usedA, b - usedB, c,
                                                      aa, bb, cc)

        return generate(a, b, c, 'a', 'b', 'c')

    def longestDiverseString(self,
                             a: int,
                             b: int,
                             c: int,
                             aa='a',
                             bb='b',
                             cc='c') -> str:
        if a < b: return self.longestDiverseString(b, a, c, bb, aa, cc)
        elif b < c: return self.longestDiverseString(a, c, b, aa, cc, bb)
        elif b == 0: return min(2, a) * aa
        usedA = min(2, a)
        usedB = 1 if a - usedA >= b else 0
        return usedA * aa + usedB * bb + self.longestDiverseString(
            a - usedA, b - usedB, c, aa, bb, cc)


# 1414 - Find the Minimum Number of Fibonacci Numbers Whose Sum Is K - MEDIUM
class Solution:
    def findMinFibonacciNumbers(self, k: int) -> int:
        s = [1, 1]
        while s[-1] <= k:
            s.append(s[-2] + s[-1])
        ans = 0
        i = len(s) - 1
        while k != 0:
            if k >= s[i]:
                k -= s[i]
                ans += 1
            i -= 1
        return ans

    def findMinFibonacciNumbers(self, k: int) -> int:
        if k == 0:
            return 0
        f1 = f2 = 1
        while f1 + f2 <= k:
            f1, f2 = f2, f1 + f2
        return self.findMinFibonacciNumbers(k - f2) + 1


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
            if root.val >= premax:
                self.cnt += 1
            if root.left:
                inorder(root.left, max(premax, root.val))
            if root.right:
                inorder(root.right, max(premax, root.val))
            return

        self.cnt = 0
        inorder(root, root.val)
        return self.cnt

    def goodNodes(self, root: TreeNode) -> int:
        def inorder(root, premax):
            if not root:
                return 0
            premax = max(root.val, premax)
            return (root.val >= premax) + inorder(root.left, premax) + inorder(
                root.right, premax)

        return inorder(root, root.val)


# 1460 - Make Two Arrays Equal by Reversing Sub-arrays - EASY
class Solution:
    def canBeEqual(self, target: List[int], arr: List[int]) -> bool:
        return collections.Counter(target) == collections.Counter(arr)
        return sorted(target) == sorted(arr)
