import bisect, collections, functools, heapq, itertools, math, operator, string
from typing import List, Optional, Tuple
import sortedcontainers


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 2706 - Buy Two Chocolates - EASY
class Solution:
    functools.reduce(
        operator.mul,
    )
    # O(nlogn) / O(1)
    def buyChoco(self, prices: List[int], money: int) -> int:
        prices.sort()
        return money if prices[0] + prices[1] > money else money - prices[0] - prices[1]

    # O(nlogn) / O(1)
    def buyChoco(self, prices: List[int], money: int) -> int:
        v = sum(heapq.nsmallest(2, prices))
        return money if v > money else money - v


# 2707 - Extra Characters in a String - MEDIUM
class Solution:
    # O(n^3) / O(n)
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        d = set(dictionary)

        @functools.lru_cache(None)
        def dfs(i: int) -> int:
            if i < 0:
                return 0
            cur = dfs(i - 1) + 1
            for j in range(i + 1):
                if s[j : i + 1] in d:
                    cur = min(cur, dfs(j - 1))
            return cur

        return dfs(len(s) - 1)

    # O(n^3) / O(n)
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dictionary = set(dictionary)
        f = [0] + [n] * n
        for i in range(n):
            f[i + 1] = f[i] + 1
            for j in range(i + 1):
                if s[j : i + 1] in dictionary:
                    f[i + 1] = min(f[i + 1], f[j])
        return f[n]


# 2708 - Maximum Strength of a Group - MEDIUM
class Solution:
    # O(n) / O(n)
    def maxStrength(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        pos = [v for v in nums if v > 0]
        neg = [v for v in nums if v < 0]
        # 这个表达式整合了奇数(不小于3)和偶数(不小于2)个负数, 0 或 1 个负数则不选负数
        # 注意只要数组元素不止1个, 答案最小也是0
        negP = (
            functools.reduce(operator.mul, neg) // (max(neg) ** (len(neg) % 2))
            if len(neg) >= 2
            else 0
        )
        if not pos:
            return negP
        posP = functools.reduce(operator.mul, pos)
        return posP * negP if negP else posP

    # O(n) / O(n), dp
    def maxStrength(self, nums: List[int]) -> int:
        mx = mi = nums[0]
        for x in nums[1:]:
            tmp = mx
            mx = max(mx, x, mx * x, mi * x)
            mi = min(mi, x, tmp * x, mi * x)
        return mx

    # O(2**n * n) / O(1)
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        ans = -math.inf
        for i in range(1, 1 << n):
            p = 1
            for j in range(n):
                if (i >> j) & 1:
                    p *= nums[j]
            ans = max(ans, p)
        return ans


# 2709 - Greatest Common Divisor Traversal - HARD
@functools.lru_cache(None)
def get_prime_factor(n: int) -> set:
    if n == 1:
        return set()
    ans = set()
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            n //= i
            ans.add(i)
            ans = ans.union(get_prime_factor(n))
            return ans
    ans.add(n)
    return ans


class Solution:
    # O(nlogU) / O(n), 边很少, log(U)
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        if 1 in nums:
            return False

        class UnionFind:
            def __init__(self, n: int) -> None:
                self.p = [i for i in range(n)]
                self.sz = [1] * n

            def find(self, x: int) -> int:
                if self.p[x] != x:
                    self.p[x] = self.find(self.p[x])
                return self.p[x]

            def union(self, x: int, y: int) -> None:
                """x's root = y"""
                px = self.find(x)
                py = self.find(y)
                if px == py:
                    return
                self.p[px] = py
                return

        uf = UnionFind(len(nums) + max(nums) + 1)
        for x in nums:
            for y in get_prime_factor(x):
                uf.union(x, y)
        return len(set(uf.find(x) for x in nums)) == 1
        root = uf.find(nums[0])
        return all(uf.find(x) == root for x in nums)
