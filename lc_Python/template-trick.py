import bisect, collections, copy, functools, heapq, math
from typing import List

MOD = 10**9 + 7

"""
Trick:
    @functools.lru_cache(None)
        for some problems that input can be used for other test cases,
        put the cache outside the class Solution,
        each instance can reuse cache and speed up
"""


"""
Directory: (abcdefghijklmnopqrstuvwxyz)
    binary
    math related
    loop
    segment tree
    set
    union-find, disjoint set
"""


"""binary"""

# pay attension to Operation Priority: Addition(+), Subtraction(-) higher than Bitwise shift operator(<<, >>)
def str2binary(s: str):
    n = 0
    for c in s:
        n |= 1 << ord(c) - ord("a")
    return n


"""loop"""


def loop(grid: List[List[int]]) -> None:
    for r in grid:
        print(r)
    for c in zip(*grid):
        print(c)
    return


"""math related"""


def gcd(a: int, b: int) -> int:
    return gcd(b, a % b) if b else a


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


@functools.lru_cache(None)
def get_factors(n: int) -> collections.defaultdict(int):
    if n == 1:
        return collections.defaultdict(int)
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            tmp = copy.deepcopy(get_factors(n // i))
            tmp[i] += 1
            return tmp
    tmp = collections.defaultdict(int)
    tmp[n] = 1
    return tmp


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


@functools.lru_cache(None)
def getcomb(m: int, n: int) -> int:
    return math.comb(m, n) % MOD


"""set"""


def set_operation() -> None:
    a = set()
    b = set()
    if a | b == a.union(b):
        pass
    if a & b == a.intersection(b):
        pass
    if a - b == a.difference(b):
        pass
    if a ^ b == a.symmetric_difference(b):
        pass
    if (len(a & b) == 0) == a.isdisjoint(b):
        pass
    if (a <= b) == a.issubset(b):
        pass
    if (a >= b) == a.issuperset(b):
        pass

    # >>> a = {'foo', 'bar', 'baz'}
    # >>> b = {'foo', 'bar', 'baz'}
    # >>> a.issubset(b)
    # True
    # >>> a <= b
    # True
    # >>> a < b
    # False

    # frozenset: TypeError: unhashable type: 'set'
    words = []
    s = set(frozenset(w) for w in words)
    return


"""TODO: segment tree"""

"""union-find, disjoint set"""


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = [i for i in range(n)]
        self.sz = [1] * n
        self.part = n
        # self.rank = [1] * n

    def find(self, x: int) -> int:
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x: int, y: int) -> None:
        """x's root -> y"""
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return
        # if self.sz[px] >= self.sz[py]:
        #     px, py = py, px
        self.p[px] = py
        self.sz[py] += self.sz[px]
        self.part -= 1
        return
