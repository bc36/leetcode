import bisect, collections, copy, functools, heapq, itertools, math, random, string
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
    binar search
    BIT
    dfs
    dijkstra
    dp
    euler - sieve of Euler - 欧拉筛/线性筛
    eratosthenes - sieve of Eratosthenes - 埃式筛
    inv - Modular Multiplicative Inverse - 逆元
    low bit
    math related
    minimum cost flow
    permutation
    segment tree
    set
    st, sparse table
    string hash
    str2binary
    transpose
    trie
    union-find, disjoint set
"""


"""binary search

https://codeforces.com/blog/entry/75879

Suppose you have a predicate P(n) which goes from being false to being true as 'n' increases, and you want to find the least 'n' for which it is true.
There are two things to remember so you never get a binary search wrong:

1) Remember the invariant you are maintaining!
At the end, you'll have l = r, P(l-1) false and P(l) true,
so a good invariant is to say that P(l-1) should always be false and P(r) should always be true.
With this, you can initialize the variables appropriately.

Now let's look at the iteration steps:

    while (l < r) {
        int mid = (l+r)/2;
        if (P(mid))
            r = mid; // Note that P(r) = P(mid) is true, so the invariant is maintained.
        else
            l = mid+1; // Note that P(l-1) = P(mid+1-1) is false, so the invariant is maintained.
    }

2) Both updates must decrease the length of the interval [l,r], and we must round up or down to ensure that.
Let's check the above code is correct:
Since l < r, we have that (as real numbers) l < (l+r)/2 < r, and therefore l <= (l+r)/2 < r after rounding down.
Therefore, r = mid decreases 'r' and l = mid + 1 increases 'l'.

Let's do the same for a predicate P(n) that goes from being true to being false as 'n' increases.
Suppose we want to find the largest 'n' for which P(n) is true.
Then at the end, we will have l = r, P(l) true and P(l+1) false.
Therefore, the invariant we will maintain is that P(l) should always be true and P(r+1) should always be false.

How does the code look like in this case?

    while (l < r) {
        int mid = ????;
        if (P(mid))
            l = mid; // Note that P(l) = P(mid) is true, so the invariant is maintained.
        else
            r = mid-1; // Note that P(r+1) = P(mid-1+1) is false, so the invariant is maintained.
    }

Now, it is still true that (as real numbers) l < (l+r)/2 < r.
But if we want l = mid to increase 'l', then we cannot round the division down.
Rounding it up (by doing (l+r+1)/2) is fine, because then l < (l+r+1)/2 <= r, and therefore r = mid - 1 decreases 'r' and l = mid increases 'l'.

try it on
lc 1552 https://leetcode.cn/problems/magnetic-force-between-two-balls/
lc 1802 https://leetcode.cn/problems/maximum-value-at-a-given-index-in-a-bounded-array/
lc 2517 https://leetcode.cn/problems/maximum-tastiness-of-candy-basket/

最小值最大化 / 左侧最后一个满足条件的值
while l < r:
    m = l + r + 1 >> 1
    l = m
    r = m - 1
return l

最大化最小值 / 左侧第一个满足条件的值
while l < r:
    m = l + r >> 1
    l = m + 1
    r = m
return l

"""

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
    def __init__(self, n: int):
        self.tree = [0] * n

    def add(self, x: int) -> None:
        while x < len(self.tree):
            self.tree[x] += 1
            x += x & -x
        return

    def query(self, x: int) -> int:
        res = 0
        while x > 0:
            res += self.tree[x]
            x &= x - 1
        return res


class BIT:
    def __init__(self, n: int):
        self.n = n
        self.a = [0] * (n + 1)

    @staticmethod
    def lowbit(x: int) -> int:
        return x & (-x)

    def query(self, x: int) -> int:
        ret = 0
        while x > 0:
            ret += self.a[x]
            x -= BIT.lowbit(x)
        return ret

    def update(self, x: int, dt: int) -> None:
        while x <= self.n:
            self.a[x] += dt
            x += BIT.lowbit(x)
        return


"""dfs"""


def example():
    g = []

    # 统计子树大小 / 统计子树点权和, 无向图
    def dfs(x: int, fa: int) -> int:
        sz = 1
        for y in g[x]:
            if y != fa:
                sz += dfs(y, x)
        return sz

    dfs(0, -1)

    return


"""dijkstra"""
# 返回从 start 到每个点的最短路
def dijkstra(g: List[List[Tuple[int]]], start: int) -> List[int]:
    dist = [math.inf] * len(g)
    dist[start] = 0
    h = [(0, start)]
    while h:
        d, x = heapq.heappop(h)
        if d > dist[x]:
            continue
        for y, w in g[x]:
            new = dist[x] + w
            if new < dist[y]:
                dist[y] = new
                heapq.heappush(h, (new, y))
    return dist


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


"""euler

欧拉筛 / 线性筛

埃氏筛法仍有优化空间, 它会将一个合数重复多次标记, 比如 12 被 2, 3 同时划掉

每一个合数都是被最小的质因子筛掉 -> 时间复杂度 O(n)

2 划掉 4(乘2)
3 划掉 6(乘2), 9(乘3)
4 划掉 8(乘2), 不能划掉 12, 因为 3 已经超过了 4 的最小质因子(2)
5 划掉 10(乘2), 15(乘3), 25(乘5)

每个数 x, 乘以 <= lpf[x] 的质数, lpf[x] 为 x 的最小的质因子

因为取模操作是算术操作中最慢的, 数据范围小时, 不如埃氏筛快
"""


def euler(n: int) -> List[int]:
    primes = []
    is_prime = [True] * (n + 1)
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
        for p in primes:
            if i * p >= n:
                break
            is_prime[i * p] = False
            if i % p == 0:  # p 是 lpf[i]
                break
    return primes


primes = euler(10**6 + 1)


"""eratosthenes

枚举质数, 划掉质数的倍数

时间复杂度: O(n * loglogn), n * (1/2 + 1/3 + 1/5 + ...) -> n * 素数倒数之和 -> n * O(loglogn)
空间复杂度: O(n / logn), [2, n] 范围内素数个数 
"""


def eratosthenes(n: int) -> List[int]:
    primes = []
    is_prime = [True] * (n + 1)
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, n + 1, i):  # 注意是 *, 不是 +, 比 i 小的 i 的倍数已经被枚举过了
                is_prime[j] = False
    return primes


primes = eratosthenes(10**6 + 1)


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


def powerOfTwo(x: int) -> bool:
    # 是否是 2 的幂次, 位运算
    return (x & (x - 1)) == 0


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
字符串哈希, 定义一个把字符串映射到整数的函数 f 这个 f 称为是 Hash 函数
希望这个函数 f 可以方便地帮我们判断两个字符串是否相等
Hash 的核心思想在于, 将输入映射到一个值域较小、可以方便比较的范围
通常采用的多项式 Hash 的方法,  MOD 需要选择一个素数(至少要比最大的字符要大), base 可以任意选择

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
        # self.part = n  # 连通块数量
        # self.rank = [1] * n  # 按秩合并

    def find(self, x: int) -> int:
        """path compression"""
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x: int, y: int) -> None:
        """x's root = y"""
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return
        # if self.sz[px] >= self.sz[py]:
        #     px, py = py, px
        self.p[px] = py
        self.sz[py] += self.sz[px]
        # self.part -= 1
        # self.sz[px] = 0  # sum(self.sz) == n
        return

    def find(self, x: int) -> int:
        cp = x
        while x != self.p[x]:
            x = self.p[x]
        while cp != x:
            self.p[cp], cp = x, self.p[cp]
        return x

    def union(self, x: int, y: int) -> None:
        """x = y's root"""
        if self.find(y) != self.find(x):
            self.p[self.find(y)] = self.find(x)
            self.sz[self.find(x)] += self.sz[self.find(y)]
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

    def get_size(self, x: int) -> int:
        return self.sz[self.find(x)]


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
