"""disjoint set, union-find algorithm

数据结构
"""


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
