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
    permutation
    segment tree
    set
    transpose
    trie
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


def mathRelated() -> None:
    math.factorial(5) == 1 * 2 * 3 * 4 * 5  # n!
    math.perm(5, 2)  # n! // (n - k)!
    math.comb(5, 2)  # n! // (k! * (n - k)!)
    return


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


# n in 10 base convert to other base
def n2xbase(n: int, b: int) -> int:
    if 0 <= n < b:
        return n
    return n2xbase(n // b, b) * 10 + n % b


def n2xbase(n: int, b: int) -> str:
    s = ""
    while n:
        s += str(n % b)
        n //= b
    return s[::-1]


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

"""transpose"""


def transpose(matrix: List[List[int]]) -> List[List[int]]:
    def loop(grid: List[List[int]]) -> None:
        for r in grid:
            print(r)
        for c in zip(*grid):
            print(c)
        return

    return list(zip(*matrix))


"""trie"""


def build(words: List[List[int]]) -> None:
    trie = {}
    for w in words:
        r = trie
        for c in w:
            if c not in r:
                r[c] = {}
            r = r[c]
        r["end"] = True
    return


# wc 311 T4, 2416
class Solution:
    # 1.3s
    def sumPrefixScores(self, words: List[str]) -> List[int]:
        Trie = lambda: collections.defaultdict(Trie)
        CNT = "#"

        trie = Trie()
        for w in words:
            r = trie
            for ch in w:
                r = r[ch]
                r[CNT] = r.get(CNT, 0) + 1
        ans = []
        for w in words:
            r = trie
            score = 0
            for ch in w:
                r = r[ch]
                score += r[CNT]
            ans.append(score)
        return ans


# 小慢 2s
def build(words: List[List[int]]) -> None:
    trie = {}
    for w in words:
        r = trie
        for c in w:
            if c not in r:
                r[c] = {}
                r[(c, "#")] = 1  # 打标记, 当前多有少个以 word[: i + 1] 为前缀, wc 311 T4
            else:
                r[(c, "#")] += 1
            r = r[c]
        r["end"] = True
    return


# 小慢 2s
def build(words: List[List[int]]) -> None:
    trie = [None] * 27  # 最后一位用于计数
    trie[26] = 0
    for w in words:
        r = trie
        for ch in w:
            c = ord(ch) - ord("a")
            if r[c] is None:
                r[c] = [None] * 27
                r[c][26] = 0
            r = r[c]
            r[26] += 1
    return


# 1s
def build(words: List[List[int]]) -> None:
    trie = {}
    for w in words:
        r = trie
        for c in w:
            if c not in r:
                r[c] = {}
                r[c]["cnt"] = 1  # 打标记, 当前多有少个以 word[: i + 1] 为前缀, wc 311 T4
            else:
                r[c]["cnt"] += 1
            r = r[c]
        r["end"] = True
    return


def build(words: List[List[int]]) -> None:
    trie = {}
    for w in words:
        r = trie
        for c in w:
            if c not in r:
                r[c] = {}
            r = r[c]
            r["cnt"] = r.get("cnt", 0) + 1  # 打标记, 当前多有少个以 word[: i + 1] 为前缀, wc 311 T4
        r["end"] = True
    return


"""union-find, disjoint set"""


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = [i for i in range(n)]
        self.sz = [1] * n
        self.part = n
        self.rank = [1] * n

    def find(self, x: int) -> int:
        """path compression"""
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

    def disconnect(self, x: int) -> None:
        self.p[x] = x
        # self.rank[x] = 1
        # self.part += 1
        return


"""
区间运算:
add:              query: O(n), update: O(1)
prefix sum:       query: O(1), update: O(n)
difference array: query: O(1), update: O(n)

前缀和数组 <-> 差分数组
积分      <-> 导数
差分数组: 
    它可以维护多次对序列的一个区间加上一个数，并在最后询问某一位的数或是多次询问某一位的数
    注意修改操作一定要在查询操作之前

多次 query and update
BIT:              query: O(logn), update: O(logn)

segment tree:     query: O(logn), update: O(logn)



拓扑排序: topological sort

# 拓扑排序找环两种方法:
# 1. 检查排序后每个点入度是否都为 0
# 2. 检查排序后长度是否为 n, (方便一点)

# BFS:
# 1. BFS先排序, 而后将所有 孤立点 加入序列尾部 (较麻烦) O(2n)

# 2. 直接排序, 入度为 0 的点入队孤立点 穿插在序列中 O(n)
#    孤立点会进入队列一次然后 append 到 arr 中


# DFS:
# 1. vis 数组, 枚举点, 前向搜索, 找一个结果, 然后验证, O(2n)
# 2. vis 打标记, dfs 中已 vis 则退出, O(n)



~n, 用取反打标记, 因为要区分 0 , 0 的补码为 -1



求区间最大值:
线段树 / ST表 / 单调队列
"""
