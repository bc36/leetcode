import bisect, collections, copy, functools, heapq, itertools, math, operator, random, string
from typing import List, Tuple, Optional, cast

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
    factors
    inv - Modular Multiplicative Inverse - 逆元
    math related
    minimum cost flow
    prime
    set
    st, sparse table
    str2binary
    transpose
"""


"""factors"""


def find_all_primes(n: int) -> None:
    d = collections.defaultdict(int)  # frequence
    arr = []  # kind, which is set
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            arr.append(i)
            while n % i == 0:
                n //= i
                d[i] += 1
        if n > 1:
            arr.append(n)
            d[n] += 1
    return


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


"""
inv

逆元: 
在数论中, 如果 ab === 1 (mod p), 我们就说 a 和 b 在模 p 意义下互为乘法逆元, 记作 a = inv(b)
inv(a) 其实可以看做模 p 意义下的 1 / a, 那么在模 p 意义下, a / b 就可以变形为 a * inv(b) (mod p)

三种方法计算逆元: 拓展欧几里得(线性同余方程), 费马小定理, 线性递推
"""


def inv(a: int, p: int) -> int:
    def exgcd(a: int, b: int, x: int, y: int) -> int:
        # 扩展欧几里得算法只要求 gcd(a, p) = 1
        if b == 0:
            x = 1
            y = 0
            return a
        d = exgcd(b, a % b, y, x)
        y -= (a // b) * x
        return d

    x = y = 0
    if exgcd(a, p, x, y) != -1:  # 无解的情形
        return -1
    return (x % p + p) % p


def inv(a: int, p: int) -> int:
    # 使用 费马小定理 需要限制 p 是一个素数
    # inv(a) = a**(p-2) mod p
    def qpow(a: int, n: int, p: int) -> int:
        ans = 1
        while n:
            if n & 1:
                ans = (ans * a) % p
            a = (a * a) % p
            n >>= 1
        return ans

    return qpow(a, p - 2, p)


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


def powerOfTwo(x: int) -> bool:
    # 是否是 2 的幂次, 位运算
    return (x & (x - 1)) == 0


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


"""minimum cost flow"""


class Edge:
    __slots__ = ("fromV", "toV", "cap", "cost", "flow")

    def __init__(self, fromV: int, toV: int, cap: int, cost: int, flow: int) -> None:
        self.fromV = fromV
        self.toV = toV
        self.cap = cap
        self.cost = cost
        self.flow = flow


class MinCostMaxFlow:
    """最小费用流的连续最短路算法复杂度为流量*最短路算法复杂度"""

    __slots__ = (
        "_n",
        "_start",
        "_end",
        "_edges",
        "_reGraph",
        "_dist",
        "_visited",
        "_curEdges",
    )

    def __init__(self, n: int, start: int, end: int):
        """
        Args:
            n (int): 包含虚拟点在内的总点数
            start (int): (虚拟)源点
            end (int): (虚拟)汇点
        """
        assert 0 <= start < n and 0 <= end < n
        self._n = n
        self._start = start
        self._end = end
        self._edges: List["Edge"] = []
        self._reGraph: List[List[int]] = [[] for _ in range(n + 10)]  # 残量图存储的是边的下标

        self._dist = [math.inf] * (n + 10)
        self._visited = [False] * (n + 10)
        self._curEdges = [0] * (n + 10)

    def addEdge(self, fromV: int, toV: int, cap: int, cost: int) -> None:
        """原边索引为i 反向边索引为i^1"""
        self._edges.append(Edge(fromV, toV, cap, cost, 0))
        self._edges.append(Edge(toV, fromV, 0, -cost, 0))
        len_ = len(self._edges)
        self._reGraph[fromV].append(len_ - 2)
        self._reGraph[toV].append(len_ - 1)

    def work(self) -> Tuple[int, int]:
        """
        Returns:
            Tuple[int, int]: [最大流,最小费用]
        """
        maxFlow, minCost = 0, 0
        while self._spfa():
            # !如果流量限定为1, 那么一次dfs只会找到一条费用最小的增广流
            # !如果流量限定为INF, 那么一次dfs不只会找到一条费用最小的增广流
            flow = self._dfs(self._start, self._end, math.inf)
            maxFlow += flow
            minCost += flow * self._dist[self._end]
        return maxFlow, minCost

    def slope(self) -> List[Tuple[int, int]]:
        """
        Returns:
            List[Tuple[int, int]]: 流量为a时,最小费用是b
        """
        res = [(0, 0)]
        flow, cost = 0, 0
        while self._spfa():
            deltaFlow = self._dfs(self._start, self._end, math.inf)
            flow += deltaFlow
            cost += deltaFlow * self._dist[self._end]
            res.append((flow, cost))  # type: ignore
        return res

    def _spfa(self) -> bool:
        """spfa沿着最短路寻找增广路径 有负cost的边不能用dijkstra"""
        n, start, end, edges, reGraph, visited = (
            self._n,
            self._start,
            self._end,
            self._edges,
            self._reGraph,
            self._visited,
        )

        self._curEdges = [0] * n
        self._dist = dist = [math.inf] * n
        dist[start] = 0
        q = collections.deque([start])

        while q:
            cur = q.popleft()
            visited[cur] = False
            for edgeIndex in reGraph[cur]:
                edge = edges[edgeIndex]
                cost, remain, next = edge.cost, edge.cap - edge.flow, edge.toV
                if remain > 0 and dist[cur] + cost < dist[next]:
                    dist[next] = dist[cur] + cost
                    if not visited[next]:
                        visited[next] = True
                        if q and dist[q[0]] > dist[next]:
                            q.appendleft(next)
                        else:
                            q.append(next)

        return dist[end] != math.inf

    def _dfs(self, cur: int, end: int, flow: int) -> int:
        if cur == end:
            return flow

        visited, reGraph, curEdges, edges, dist = (
            self._visited,
            self._reGraph,
            self._curEdges,
            self._edges,
            self._dist,
        )

        visited[cur] = True
        res = flow
        index = curEdges[cur]
        while res and index < len(reGraph[cur]):
            edgeIndex = reGraph[cur][index]
            next, remain = (
                edges[edgeIndex].toV,
                edges[edgeIndex].cap - edges[edgeIndex].flow,
            )
            if (
                remain > 0
                and not visited[next]
                and dist[next] == dist[cur] + edges[edgeIndex].cost
            ):
                delta = self._dfs(next, end, remain if remain < res else res)
                res -= delta
                edges[edgeIndex].flow += delta
                edges[edgeIndex ^ 1].flow -= delta
            curEdges[cur] += 1
            index = curEdges[cur]

        visited[cur] = False
        return flow - res


class MincostFlow:
    class Edge:
        def __init__(self, u, v, cap, cost, rev=None):
            self.u = u
            self.v = v
            self.cap = cap
            self.cost = cost
            self.rev = rev

    def __init__(self, n):
        self.__n = n
        self.__edges = []

    def add_edge(self, u, v, cap, cost):
        e1 = self.Edge(u, v, cap, cost)
        e2 = self.Edge(v, u, 0, -cost)
        e1.rev = e2
        e2.rev = e1
        self.__edges.append(e1)
        self.__edges.append(e2)

    def build_graph(self):
        self.__graph = [[] for i in range(self.__n + 1)]
        for e in self.__edges:
            self.__graph[e.u].append(e)

    def slope(self, s, t):
        res = [[0, 0]]
        h = self.__build_height(s, t)
        while True:
            path = self.__shortest_path(s, t, h)
            if path is None:
                break
            flow = float("inf")
            cost = 0
            for e in path:
                flow = min(flow, e.cap)
            for e in path:
                cost += flow * e.cost
                e.cap -= flow
                e.rev.cap += flow

            flow += res[-1][0]
            cost += res[-1][1]
            res.append([flow, cost])
        return res

    def max_flow(self, s, t):
        # [cap: max_flow, cost: min_cost]
        return (self.slope(s, t))[-1]

    def __build_height(self, s, t):
        d = [math.inf for _ in range(self.__n + 1)]
        d[s] = 0
        q = collections.deque([s])
        vis = {s}
        while len(q) != 0:
            u = q.popleft()
            vis.remove(u)
            for e in self.__graph[u]:
                if e.cap > 0 and d[u] + e.cost < d[e.v]:
                    d[e.v] = d[u] + e.cost
                    if e.v not in vis:
                        vis.add(e.v)
                        q.append(e.v)
        return d

    def __shortest_path(self, s, t, h):
        d = [math.inf for _ in range(self.__n + 1)]
        back = [None for _ in range(self.__n + 1)]
        d[s] = 0
        pq = [(d[i], i) for i in range(self.__n + 1)]
        heapq.heapify(pq)
        vis = set()
        while len(pq) != 0:
            _, u = heapq.heappop(pq)
            if u in vis:
                continue
            if d[u] == math.inf:
                break
            vis.add(u)
            for e in self.__graph[u]:
                if e.cap > 0 and d[u] + (e.cost + h[u] - h[e.v]) < d[e.v]:
                    d[e.v] = d[u] + (e.cost + h[u] - h[e.v])
                    back[e.v] = e
                    heapq.heappush(pq, (d[e.v], e.v))
        if d[t] == math.inf:
            return None
        else:
            for i in range(self.__n + 1):
                h[i] += d[i]
            res = []
            v = t
            while back[v] is not None:
                res.append(back[v])
                v = back[v].u
            return res[::-1]


class MCFGraph:
    class Edge(collections.NamedTuple):
        src: int
        dst: int
        cap: int
        flow: int
        cost: int

    class _Edge:
        def __init__(self, dst: int, cap: int, cost: int) -> None:
            self.dst = dst
            self.cap = cap
            self.cost = cost
            self.rev: Optional[MCFGraph._Edge] = None

    def __init__(self, n: int) -> None:
        self._n = n
        self._g: List[List[MCFGraph._Edge]] = [[] for _ in range(n)]
        self._edges: List[MCFGraph._Edge] = []

    def add_edge(self, src: int, dst: int, cap: int, cost: int) -> int:
        assert 0 <= src < self._n
        assert 0 <= dst < self._n
        assert 0 <= cap
        m = len(self._edges)
        e = MCFGraph._Edge(dst, cap, cost)
        re = MCFGraph._Edge(src, 0, -cost)
        e.rev = re
        re.rev = e
        self._g[src].append(e)
        self._g[dst].append(re)
        self._edges.append(e)
        return m

    def get_edge(self, i: int) -> Edge:
        assert 0 <= i < len(self._edges)
        e = self._edges[i]
        re = cast(MCFGraph._Edge, e.rev)
        return MCFGraph.Edge(re.dst, e.dst, e.cap + re.cap, re.cap, e.cost)

    def edges(self) -> List[Edge]:
        return [self.get_edge(i) for i in range(len(self._edges))]

    def flow(self, s: int, t: int, flow_limit: Optional[int] = None) -> Tuple[int, int]:
        return self.slope(s, t, flow_limit)[-1]

    def slope(
        self, s: int, t: int, flow_limit: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        assert 0 <= s < self._n
        assert 0 <= t < self._n
        assert s != t
        if flow_limit is None:
            flow_limit = cast(int, sum(e.cap for e in self._g[s]))

        dual = [0] * self._n
        prev: List[Optional[Tuple[int, MCFGraph._Edge]]] = [None] * self._n

        def refine_dual() -> bool:
            pq = [(0, s)]
            visited = [False] * self._n
            dist: List[Optional[int]] = [None] * self._n
            dist[s] = 0
            while pq:
                dist_v, v = heapq.heappop(pq)
                if visited[v]:
                    continue
                visited[v] = True
                if v == t:
                    break
                dual_v = dual[v]
                for e in self._g[v]:
                    w = e.dst
                    if visited[w] or e.cap == 0:
                        continue
                    reduced_cost = e.cost - dual[w] + dual_v
                    new_dist = dist_v + reduced_cost
                    dist_w = dist[w]
                    if dist_w is None or new_dist < dist_w:
                        dist[w] = new_dist
                        prev[w] = v, e
                        heapq.heappush(pq, (new_dist, w))
            else:
                return False
            dist_t = dist[t]
            for v in range(self._n):
                if visited[v]:
                    dual[v] -= cast(int, dist_t) - cast(int, dist[v])
            return True

        flow = 0
        cost = 0
        prev_cost_per_flow: Optional[int] = None
        result = [(flow, cost)]
        while flow < flow_limit:
            if not refine_dual():
                break
            f = flow_limit - flow
            v = t
            while prev[v] is not None:
                u, e = cast(Tuple[int, MCFGraph._Edge], prev[v])
                f = min(f, e.cap)
                v = u
            v = t
            while prev[v] is not None:
                u, e = cast(Tuple[int, MCFGraph._Edge], prev[v])
                e.cap -= f
                assert e.rev is not None
                e.rev.cap += f
                v = u
            c = -dual[s]
            flow += f
            cost += f * c
            if c == prev_cost_per_flow:
                result.pop()
            result.append((flow, cost))
            prev_cost_per_flow = c
        return result


"""prime"""


def isPrime(x: int) -> bool:
    if x == 1:
        return False
    for i in range(2, int(x**0.5) + 1):
        if x % i == 0:
            return False
    return True


def isPrime(n: int) -> bool:
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return n >= 2  # 1 不是质数


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


"""st, sparse table
也用到了 binary lifting 思想
https://cp-algorithms.com/data_structures/sparse-table.html
"""


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


class SparseTable:
    def __init__(self, data: list, func=operator.or_):
        # ST表 稀疏表, O(nlgn) 预处理, O(1) 查询区间最值/或和/gcd
        # 下标从 0 开始
        self.func = func
        self.st = st = [list(data)]
        i, N = 1, len(st[0])
        while 2 * i <= N + 1:
            pre = st[-1]
            st.append([func(pre[j], pre[j + i]) for j in range(N - 2 * i + 1)])
            i <<= 1

    def query(self, begin: int, end: int):  # 查询闭区间[begin, end]的最大值
        lg = (end - begin + 1).bit_length() - 1
        return self.func(self.st[lg][begin], self.st[lg][end - (1 << lg) + 1])


# lc 2736
class SparseTable:
    def __init__(self, data, merge_method):
        self.note = [0] * (len(data) + 1)
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


"""str2binary"""


# be careful of Operation Priority: Addition(+), Subtraction(-) higher than Bitwise shift operator(<<, >>)
def str2binary(s: str):
    n = 0
    for c in s:
        n |= 1 << ord(c) - ord("a")
    return n


"""transpose"""


def transpose(matrix: List[List[int]]) -> List[List[int]]:
    def loop(grid: List[List[int]]) -> None:
        for r in grid:
            print(r)
        for c in zip(*grid):  # unpack
            print(c)
        return

    return list(zip(*matrix))


"""
区间运算:
add:              query: O(n), update: O(1)
prefix sum:       query: O(1), update: O(n)
difference array: query: O(1), update: O(n)

前缀和数组 <-> 差分数组
积分      <-> 导数
差分数组: 
    它可以维护多次对序列的一个区间加上一个数, 并在最后询问某一位的数或是多次询问某一位的数
    注意修改操作一定要在查询操作之前

多次 query and update
BIT:              query: O(logn), update: O(logn)

segment tree:     query: O(logn), update: O(logn)


区间求和问题:
数组不变, 区间查询: 前缀和 > 树状数组、线段树; 
数组单点修改, 区间查询: 树状数组 > 线段树; 
数组区间修改, 单点查询: 差分 > 线段树; 
数组区间修改, 区间查询: 线段树. 
java 线段树板子: https://mp.weixin.qq.com/s?__biz=MzU4NDE3MTEyMA==&mid=2247490329&idx=1&sn=6d448a53cd722bbd990fda82bd262857


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


0 - 1 背包倒序转移原因:
    空间优化时, 防止覆盖掉前面已经计算过的值
"""


# PEP 350 Codetags
# https://peps.python.org/pep-0350/
# TODO To do: Informal tasks/features that are pending completion.
# FIXME (XXX) Fix me: Areas of problematic or ugly code needing refactoring or cleanup.
# BUG Bugs: Reported defects tracked in bug database.
# HACK Hacks: Temporary code to force inflexible functionality, or simply a test change, or workaround a known problem.
# NOTE Notes: Sections where a code reviewer found something that needs discussion or further investigation.


# 带余取模
# (a + b) % m = ((a % m) + (b % m)) % m
# (a * b) % m = ((a % m) * (b % m)) % m
# 这两个恒等式, 可以随意地对代码中的加法和乘法的结果取模
