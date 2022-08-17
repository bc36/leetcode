import bisect, collections, copy, functools, heapq, itertools, math, random, string
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
    dp
    math related
    loop
    permutation
    segment tree
    set
    union-find, disjoint set
"""


"""binary"""

# be careful of Operation Priority: Addition(+), Subtraction(-) higher than Bitwise shift operator(<<, >>)
def str2binary(s: str):
    n = 0
    for c in s:
        n |= 1 << ord(c) - ord("a")
    return n


"""dp - digit DP"""


def countSpecialNumbers(n: int) -> int:
    s = str(n)

    @functools.lru_cache(None)
    def fn(i: int, mask: int, is_limit: bool, is_num: bool) -> int:
        if i == len(s):
            return int(is_num)
        ans = 0
        if not is_num:
            ans = fn(i + 1, mask, False, False)
        up = int(s[i]) if is_limit else 9
        for d in range(0 if is_num else 1, up + 1):
            if mask >> d & 1 == 0:
                ans += fn(i + 1, mask | (1 << d), is_limit and d == up, True)
        return ans

    return fn(0, 0, True, False)


"""loop"""


def loop(grid: List[List[int]]) -> None:
    for r in grid:
        print(r)
    for c in zip(*grid):
        print(c)
    return


"""permutation"""


def fn() -> None:
    # 1. itertools.permutations
    for lst in itertools.permutations(range(1, 4), 3):
        # do sth
        pass

    # 2. math.perm
    n = 3
    k = 2
    p = math.perm(n, k)
    return


"""math related"""


def ceil(x: int, y: int) -> int:
    return (x + y - 1) // y
    return (x - 1) // y + 1


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
        # self.sz[px] = 0  # sum(self.sz) == n
        return

    def find(self, x: int) -> int:
        """path compression"""
        need_compress = []
        while self.p[x] != x:
            need_compress.append(x)
            x = self.p[x]
        while need_compress:
            self.p[need_compress.pop()] = x
        return x
