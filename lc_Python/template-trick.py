import bisect, collections, copy, functools, heapq, itertools, math, random, string
from typing import List

MOD = 10**9 + 7
MOD = 998244353

"""
Trick:
    @functools.lru_cache(None)
    def fn():
        pass

    # fn.cache_info()
    fn.cache_clear()
    return

    For some problems that input can be used for other test cases,
    put the cache outside the class Solution,
    each instance can reuse cache and speed up
"""


"""
Directory: (abcdefghijklmnopqrstuvwxyz)
    binary
    BIT
    dp
    low bit
    math related
    permutation
    segment tree
    set
    st, sparse table
    string hash
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


"""BIT"""


class BIT:
    __slots__ = "size", "bit", "tree"

    def __init__(self, n: int):
        self.size = n
        self.bit = n.bit_length()
        self.tree = dict()

    def add(self, index: int, delta: int) -> None:
        while index <= self.size:
            self.tree[index] = self.tree.get(index, 0) + delta
            index += index & -index
        return

    def query(self, index: int) -> int:
        res = 0
        while index > 0:
            res += self.tree.get(index, 0)
            index -= index & -index
        return res


class BIT:
    def __init__(self, n):
        self.n = n
        self.a = [0] * (n + 1)

    @staticmethod
    def lowbit(x):
        return x & (-x)

    def query(self, x):
        ret = 0
        while x > 0:
            ret += self.a[x]
            x -= BIT.lowbit(x)
        return ret

    def update(self, x, dt):
        while x <= self.n:
            self.a[x] += dt
            x += BIT.lowbit(x)


"""dp - digit DP"""


def countSpecialNumbers(n: int) -> int:
    s = str(n)

    @functools.lru_cache(None)
    def dfs(i: int, mask: int, is_limit: bool, is_num: bool) -> int:
        if i == len(s):
            return int(is_num)
        ans = 0
        if not is_num:
            ans = dfs(i + 1, mask, False, False)
        bound = int(s[i]) if is_limit else 9
        for d in range(0 if is_num else 1, bound + 1):
            if mask >> d & 1 == 0:
                ans += dfs(i + 1, mask | (1 << d), is_limit and d == bound, True)
        return ans

    return dfs(0, 0, True, False)


"""low bit"""


def low_bit(x: int) -> None:
    """
    如何求 x 最低位的 1
    1.
    x      = 1011000
    ~x     = 0100111
    ~x + 1 = 0101000
    ~x + 1 = -x 补码性质
    得到 low_bit = x ^ -x
    去掉 low_bit -> x -= x & (-x)

    2.
    x     = 1011000
    x - 1 = 1010111
    去掉 low_bit -> x &= x - 1
    """
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


"""TODO: segment tree
1. 每个区间拆分成 O(logn) 个区间
2. O(n) 个区间, 可以拼成任意区间

lazy:
    1. 如果当前节点对应区间被查询区间完整包含, 停止递归, 对于更新操作, 需要记录更新内容 lazy tag, 在这里打住
    2. 后续某个更新操作需要递归的话, 带着 lazy tag 继续递归

"""


class SegmentTree:
    def __init__(self, data, merge=sum):
        """
        data: 传入的数组
        merge: 处理的业务逻辑, 例如求和/最大值/最小值, lambda 表达式
        """
        self.data = data
        self.n = len(data)
        self.tree = [None] * (4 * self.n)  # 索引 i 的左孩子索引为 2i+1, 右孩子为 2i+2
        self._merge = merge
        if self.n:
            self._build(0, 0, self.n - 1)

    def query(self, ql, qr) -> int:
        """
        返回区间[ql, ... , qr] 的值
        """
        return self._query(0, 0, self.n - 1, ql, qr)

    def update(self, index, value) -> None:
        # 将 data 数组 index 位置的值更新为 value, 然后递归更新线段树中被影响的各节点的值
        self.data[index] = value
        self._update(0, 0, self.n - 1, index)
        return

    def _build(self, tree_index, l, r) -> None:
        """
        递归创建线段树
        tree_index: 线段树节点在数组中位置
        l, r: 该节点表示的区间的左, 右边界
        """
        if l == r:
            self.tree[tree_index] = self.data[l]
            return
        mid = (l + r) // 2  # 区间中点, 对应左孩子区间结束, 右孩子区间开头
        left = 2 * tree_index + 1  # tree_index 的左右子树索引
        right = 2 * tree_index + 2
        self._build(left, l, mid)
        self._build(right, mid + 1, r)
        self.tree[tree_index] = self._merge(self.tree[left], self.tree[right])
        return

    def _query(self, tree_index, l, r, ql, qr) -> int:
        """
        递归查询区间 [ql, ... ,qr] 的值
        tree_index : 某个根节点的索引
        l, r : 该节点表示的区间的左右边界
        ql, qr: 待查询区间的左右边界
        """
        if l == ql and r == qr:
            return self.tree[tree_index]
        mid = (l + r) // 2  # 区间中点, 对应左孩子区间结束,右孩子区间开头
        left = 2 * tree_index + 1
        right = 2 * tree_index + 2
        if qr <= mid:
            return self._query(left, l, mid, ql, qr)  # 查询区间全在左子树
        elif ql > mid:
            return self._query(right, mid + 1, r, ql, qr)  # 查询区间全在右子树
        # 查询区间一部分在左子树一部分在右子树
        return self._merge(
            self._query(left, l, mid, ql, mid),
            self._query(right, mid + 1, r, mid + 1, qr),
        )

    def _update(self, tree_index, l, r, index) -> None:
        """
        tree_index: 某个根节点索引
        l, r : 此根节点代表区间的左右边界
        index : 更新的值的索引
        """
        if l == r == index:
            self.tree[tree_index] = self.data[index]
            return
        mid = (l + r) // 2
        left = 2 * tree_index + 1
        right = 2 * tree_index + 2
        if index > mid:
            self._update(right, mid + 1, r, index)  # 要更新的区间在右子树
        else:
            self._update(left, l, mid, index)  # 要更新的区间在左子树 index <= mid
        # 里面的小区间变化了, 包裹的大区间也要更新
        self.tree[tree_index] = self._merge(self.tree[left], self.tree[right])
        return


"""st, sparse table"""


class SparseTable:
    def __init__(self, data, merge_method):
        self.note = [0] * len(data)
        self.merge_method = merge_method
        l, r, v = 1, 2, 0
        while True:
            for i in range(l, r):
                if i >= len(self.note):
                    break
                self.note[i] = v
            else:
                l *= 2
                r *= 2
                v += 1
                continue
            break
        self.ST = [[0] * len(data) for _ in range(self.note[-1] + 1)]
        self.ST[0] = data
        for i in range(1, len(self.ST)):
            for j in range(len(data) - (1 << i) + 1):
                self.ST[i][j] = merge_method(
                    self.ST[i - 1][j], self.ST[i - 1][j + (1 << (i - 1))]
                )

    def query(self, l, r):
        pos = self.note[r - l + 1]
        return self.merge_method(self.ST[pos][l], self.ST[pos][r - (1 << pos) + 1])


"""string hash
字符串哈希, 定义一个把字符串映射到整数的函数 f 这个 f 称为是 Hash 函数, 希望这个函数 f 可以方便地帮我们判断两个字符串是否相等
Hash 的核心思想在于，将输入映射到一个值域较小、可以方便比较的范围
通常采用的多项式 Hash 的方法,  MOD需要选择一个素数(至少要比最大的字符要大), base 可以任意选择

py 切片较快, 大部分情况可以直接比较切片
"""


def string_hash(arr: List[int]) -> None:
    n = len(arr)
    base = 131  # 哈希指数, 是一个经验值, 可以取 1331 等等
    mod = 998244353
    p = [0] * 4001
    h = [0] * 4001
    p[0] = 1
    for i in range(1, n + 1):
        p[i] = (p[i - 1] * base) % mod
        h[i] = (h[i - 1] * base + ord(arr[i - 1])) % mod

    def getHash(l: int, r: int) -> int:
        return (h[r] - h[l - 1] * p[r - l + 1]) % mod

    return


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


def build(words: List[List[int]]) -> None:
    # 小慢 2s
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

    # 小慢 2s
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

    # 1s
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
        self.part = n  # 连通块数量
        self.rank = [1] * n  # 按秩合并

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

    def is_connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


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
