"""树状数组

https://cp-algorithms.com/data_structures/fenwick.html
https://oi-wiki.org/ds/fenwick/

树状数组是一种支持 单点修改 和 区间查询 的, 代码量小的数据结构
gcd, max 这些信息不可差分, 所以不能用普通树状数组处理 -> 两个树状数组?
"""


class Fenwick:
    __slots__ = "tree"

    def __init__(self, n: int):
        self.tree = [0] * n

    # 把下标为 i 的元素增加 1
    def add(self, i: int) -> None:
        while i < len(self.tree):
            self.tree[i] += 1
            i += i & -i

    # 返回下标在 [1, i] 的元素之和
    def pre(self, i: int) -> int:
        res = 0
        while i > 0:
            res += self.tree[i]
            i &= i - 1  # i -= i & -i
        return res

    def lowbit(self, x: int) -> int:
        return x & -x
