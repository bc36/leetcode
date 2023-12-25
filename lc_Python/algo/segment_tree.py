"""线段树

数据结构

https://cp-algorithms.com/data_structures/segment_tree.html
https://oi-wiki.org/ds/seg/
https://www.acwing.com/blog/content/1684/
https://www.desgard.com/algo/docs/part3/ch02/3-segment-tree-range/

A Segment Tree is a data structure that stores information about array intervals as a tree. 
This allows answering range queries over an array efficiently, while still being flexible enough to allow quick modification of the array.
线段树是算法竞赛中常用的用来维护 区间信息 的数据结构. 通过二分区段的方式, 将每一个区段的答案记录在二叉树每一个节点上.

时间复杂度: O(logn) 的时间复杂度内实现单点修改, 区间修改, 区间查询(区间求和, 求区间最大值, 求区间最小值)等操作
空间复杂度: O(4n)

核心: 递归

query: 从整个/全部/最大/根区间中搜索目标区间, 搜索过程中, 递归区间(块/段/segment)逐渐缩小到正好能拼凑出目标区间.
       以 m = (l + r) / 2 为分界, 继续从当前区间的 左/右 子区间(子树) 中寻找合适的区间(块/段/segment).
       若当前区间被目标区间完全包含, 直接返回.

单点更新: 思路同上, 注意在更新之后, 逐级更新根节点的值(归的部分)

区间更新: 需要额外的空间存储要更新的信息. 常用 lazy 标记(懒标记), 所谓的懒更新. 更新时需要标记下传.

lazy 标记: 将此区间标记, 表示这个区间的值已更新, 但它的子区间却没有更新, 更新的信息就是标记里存的值.
          每次执行修改时, 我们通过打标记的方法表明该节点对应的区间在某一次操作中被更改, 但不更新该节点的子节点的信息.
          实质性的修改则在下一次访问带有标记的节点时才进行


题单
https://leetcode.cn/tag/segment-tree/problemset/

板子题, 单点更新 + 区间查询 LC 307  https://leetcode.cn/problems/range-sum-query-mutable/description/
板子题, 区间更新 + 区间查询 LC 1109 https://leetcode.cn/problems/corporate-flight-bookings/description/
"""

import collections
from typing import List


class SegmentTree:
    """基本款, 根节点下标 1, 管辖范围 1 - n, 按题意 更新/min/max/相乘 自行修改 update 逻辑"""

    def __init__(self, nums: List[int]):
        n = len(nums)
        self.t = [0] * (n * 4)
        self.build(nums, 1, 1, n)

    def build(self, nums: List[int], o: int, l: int, r: int) -> None:
        if l == r:
            self.t[o] = nums[l - 1]
            return
        m = l + r >> 1
        self.build(nums, o << 1, l, m)
        self.build(nums, o << 1 | 1, m + 1, r)
        self.t[o] = self.t[o << 1] + self.t[o << 1 | 1]
        return

    def update(self, o: int, l: int, r: int, idx: int, val: int) -> None:
        """给 idx 下标位置 += val, self.update(1, 1, n, idx, val), l, r 表示当前节点 o 所对应区间"""
        if l == r:
            self.t[o] = val
            return
        m = l + r >> 1
        if idx <= m:
            self.update(o << 1, l, m, idx, val)
        else:
            self.update(o << 1 | 1, m + 1, r, idx, val)
        self.t[o] = self.t[o << 1] + self.t[o << 1 | 1]  # push up
        return

    def query(self, o: int, l: int, r: int, L: int, R: int) -> int:
        """返回 [L, R] 闭区间内元素和, self.query(1, 1, n, L, R), l, r 表示当前节点 o 所对应区间"""
        if L <= l and r <= R:
            return self.t[o]
        res = 0
        m = l + r >> 1
        if L <= m:
            res += self.query(o << 1, l, m, L, R)
        if R > m:
            res += self.query(o << 1 | 1, m + 1, r, L, R)
        return res


class SegmentTree:
    """基本款 其二, 根节点下标 0, 单点更新, 区间查询"""

    def __init__(self, nums: List[int]):
        n = len(nums)
        self.t = [0] * (n * 4)
        self.build(nums, 0, 0, n - 1)

    def build(self, nums: List[int], o: int, l: int, r: int):
        if l == r:
            self.t[o] = nums[l]
            return
        m = l + r >> 1
        self.build(nums, o * 2 + 1, l, m)
        self.build(nums, o * 2 + 2, m + 1, r)
        self.t[o] = self.t[o * 2 + 1] + self.t[o * 2 + 2]
        return

    def update(self, o: int, l: int, r: int, idx: int, val: int):
        """将 idx 下标位置更新为 val, self.update(0, 0, n - 1, idx, val)"""
        if l == r:
            self.t[o] = val
            return
        m = l + r >> 1
        if idx <= m:
            self.update(o * 2 + 1, l, m, idx, val)
        else:
            self.update(o * 2 + 2, m + 1, r, idx, val)
        self.t[o] = self.t[o * 2 + 1] + self.t[o * 2 + 2]
        return

    def query(self, o: int, l: int, r: int, L: int, R: int) -> int:
        """返回 [L, R] 闭区间内元素和, self.query(0, 0, self.n - 1, L, R)"""
        if L <= l and r <= R:
            return self.t[o]
        m = l + r >> 1
        res = 0
        if L <= m:
            res += self.query(o * 2 + 1, l, m, L, R)
        if R > m:
            res += self.query(o * 2 + 2, m + 1, r, L, R)
        return res

    # def query(self, o: int, l: int, r: int, L: int, R: int) -> int:
    #     if R < l or L > r:
    #         return 0
    #     elif L <= l and r <= R:
    #         return self.t[o]
    #     m = l + r >> 1
    #     return self.query(o * 2 + 1, l, m, L, R) + self.query(o * 2 + 2, m + 1, r, L, R)

    # def query(self, o: int, l: int, r: int, L: int, R: int) -> int:
    #     if L == l and R == r:
    #         return self.t[o]
    #     m = l + r >> 1
    #     if R <= m:
    #         return self.query(o * 2 + 1, l, m, L, R)
    #     if m < L:
    #         return self.query(o * 2 + 2, m + 1, r, L, R)
    #     return self.query(o * 2 + 1, l, m, L, m) + self.query(
    #         o * 2 + 2, m + 1, r, m + 1, R
    #     )


class SegmentTree:
    """区间更新款, 根节点下标 1, 管辖范围 1 - n"""

    def __init__(self, nums: List[int]):
        n = len(nums)
        self.t = [0] * (4 * n)
        self.lazy = [0] * (4 * n)
        self.build(nums, 1, 1, n)

    def build(self, nums: List[int], o: int, l: int, r: int) -> None:
        if l == r:
            self.t[o] = nums[l - 1]
            return
        m = l + r >> 1
        self.build(nums, o << 1, l, m)
        self.build(nums, o << 1 | 1, m + 1, r)
        self.t[o] = self.t[o << 1] + self.t[o << 1 | 1]
        return

    def pushdown(self, o: int, cnt: int) -> None:
        """o 为该节点, cnt 为该节点管辖范围内有多少点, 所有 push down 总和为 o.lazy * cnt"""
        if self.lazy[o] != 0:
            # 更新对左/右子区间的影响
            self.t[o << 1] += self.lazy[o] * (cnt - cnt // 2)
            self.t[o << 1 | 1] += self.lazy[o] * (cnt // 2)
            # 更新对左/右儿子的标记的影响
            self.lazy[o << 1] += self.lazy[o]
            self.lazy[o << 1 | 1] += self.lazy[o]
            self.lazy[o] = 0
        return

    def range_update(self, o: int, l: int, r: int, L: int, R: int, val: int) -> None:
        """Range updates (Lazy Propagation)"""
        if L <= l and r <= R:
            self.t[o] += val * (r - l + 1)
            self.lazy[o] += val  # 如果到了最后一层子树, 那么懒标记就挂着, 反正不会再往下 push 了
            return
        self.pushdown(o, r - l + 1)
        m = l + r >> 1
        if L <= m:
            self.range_update(o << 1, l, m, L, R, val)
        if m < R:
            self.range_update(o << 1 | 1, m + 1, r, L, R, val)
        self.t[o] = self.t[o << 1] + self.t[o << 1 | 1]  # push up
        return

    def query(self, o: int, l: int, r: int, L: int, R: int) -> int:
        """返回 [L, R] 闭区间内元素和, self.query(1, 1, n, L, R)"""
        if L <= l and r <= R:
            return self.t[o]
        self.pushdown(o, r - l + 1)
        res = 0
        m = l + r >> 1
        if L <= m:
            res += self.query(o << 1, l, m, L, R)
        if m < R:
            res += self.query(o << 1 | 1, m + 1, r, L, R)
        return res

    # def query(self, o: int, l: int, r: int, L: int, R: int) -> int:
    #     """同上, 另一种写法"""
    #     if r < L or R < l:
    #         return 0
    #     if L <= l and r <= R:
    #         return self.t[o]
    #     self.pushdown(o, r - l + 1)
    #     m = l + r >> 1
    #     return self.query(o << 1, l, m, L, R) + self.query(o << 1 | 1, m + 1, r, L, R)

