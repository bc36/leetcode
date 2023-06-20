"""树状数组

https://cp-algorithms.com/data_structures/fenwick.html
https://oi-wiki.org/ds/fenwick/
https://zhuanlan.zhihu.com/p/93795692

Fenwick tree is also called Binary Indexed Tree, or just BIT abbreviated.
树状数组是一种支持 单点修改 和 区间查询 的, 代码量小的数据结构. 多用于高效计算数列的前缀和 / 区间和

时间复杂度: O(logn) 时间内得到任意前缀和, O(logn) 时间内支持动态单点值的修改
空间复杂度: O(n)

二维偏序问题一般用树状数组解决
https://zhuanlan.zhihu.com/p/112504092

题单
https://leetcode.cn/tag/binary-indexed-tree/problemset/
"""


class BIT:
    # 下标从 1 开始, bit = BIT(n + 1)
    __slots__ = "size", "bit", "tree"

    def __init__(self, n: int):
        self.size = n
        self.bit = n.bit_length()
        self.tree = dict()

    def add(self, i: int, d: int = 1) -> None:
        while i <= self.size:
            self.tree[i] = self.tree.get(i, 0) + d
            i += i & -i
        return

    def query(self, i: int) -> int:
        res = 0
        while i > 0:
            res += self.tree.get(i, 0)
            i -= i & -i
        return res

    def rsum(self, l: int, r: int) -> int:
        return self.query(r) - self.query(l - 1)


class BIT:
    # 下标从 1 开始, bit = BIT(n + 1)
    def __init__(self, n: int):
        self.tree = [0] * n

    def add(self, i: int, d: int = 1) -> None:
        """将下标 i 上的数加 d"""
        while i < len(self.tree):
            self.tree[i] += d
            i += i & -i
        return

    def query(self, i: int) -> int:
        """返回闭区间 [1, i] 的元素和"""
        res = 0
        while i > 0:
            res += self.tree[i]
            i &= i - 1
        return res

    def rsum(self, l: int, r: int) -> int:
        """range sum, 返回闭区间 [l, r] 的元素和"""
        return self.query(r) - self.query(l - 1)


# 注意 tree 与 lowbit 实现方式的 tree 长的不一样
class BIT:
    # 下标从 1 开始, bit = BIT(n + 1)
    def __init__(self, n: int):
        self.tree = [0] * n

    def add(self, i: int, d: int = 1) -> None:
        """将下标 i 上的数加 d"""
        while i < len(self.tree):
            self.tree[i] += d
            i = i | (i + 1)
        return

    def query(self, i: int) -> int:
        """返回闭区间 [1, i] 的元素和"""
        res = 0
        while i > 0:
            res += self.tree[i]
            i = (i & (i + 1)) - 1
        return res

    def rsum(self, l: int, r: int) -> int:
        """range sum, 返回闭区间 [l, r] 的元素和"""
        return self.query(r) - self.query(l - 1)


class BIT:
    # 下标从 1 开始, bit = BIT(n + 1)
    def __init__(self, n: int):
        self.tree = [0] * n

    @staticmethod
    def lowbit(x: int) -> int:
        return x & (-x)

    def add(self, i: int, d: int = 1) -> None:
        """将下标 i 上的数加 d"""
        while i < len(self.tree):
            self.tree[i] += d
            i += BIT.lowbit(i)
        return

    def query(self, i: int) -> int:
        """返回闭区间 [1, i] 的元素和"""
        res = 0
        while i > 0:
            res += self.tree[i]
            i -= BIT.lowbit(i)
        return res

    def rsum(self, l: int, r: int) -> int:
        """range sum, 返回闭区间 [l, r] 的元素和"""
        return self.query(r) - self.query(l - 1)
