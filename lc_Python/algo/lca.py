"""lca

倍增算法是最经典的 LCA 求法
https://oi-wiki.org/graph/lca/#%E5%80%8D%E5%A2%9E%E7%AE%97%E6%B3%95

"""


class LCA:
    def __init__(self, g, root):
        self.n = len(g)
        self.root = root
        self.num = (self.n).bit_length()
        self.depth = [0] * self.n
        self.parent = [[-1] * self.n for i in range(self.num)]

        s = [root]
        while s:
            v = s.pop()
            for u, _ in g[v]:
                if u == self.parent[0][v]:
                    continue
                self.parent[0][u] = v
                self.depth[u] = self.depth[v] + 1
                s.append(u)

        for k in range(self.num - 1):
            for v in range(self.n):
                if self.parent[k][v] == -1:
                    self.parent[k + 1][v] = -1
                else:
                    self.parent[k + 1][v] = self.parent[k][self.parent[k][v]]

    def getLCA(self, u, v):
        if self.depth[u] > self.depth[v]:
            u, v = v, u
        for k in range(self.num):
            if ((self.depth[v] - self.depth[u]) >> k) & 1:
                v = self.parent[k][v]
        if u == v:
            return u

        for k in reversed(range(self.num)):
            if self.parent[k][u] != self.parent[k][v]:
                u = self.parent[k][u]
                v = self.parent[k][v]
        return self.parent[0][u]
