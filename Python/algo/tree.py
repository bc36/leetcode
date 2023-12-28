# 利用父节点
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
