import bisect, collections, functools, heapq, itertools, math, operator, string, sys
from typing import List, Optional, Tuple
import sortedcontainers


# 3000 - Maximum Area of Longest Diagonal Rectangle - EASY
class Solution:
    def areaOfMaxDiagonal(self, dimensions: List[List[int]]) -> int:
        return max((x * x + y * y, x * y) for x, y in dimensions)[1]


# 3001 - Minimum Moves to Capture The Queen - MEDIUM
class Solution:
    def minMovesToCaptureTheQueen(
        self, a: int, b: int, c: int, d: int, e: int, f: int
    ) -> int:
        # 首先注意到答案至多为 2, 因为
        # - 当车和皇后既不在同一行又不在同一列时, 车有至少两种走法吃到皇后(先走行再走列, 或者先走列再走行), 由于象只有一个, 因此肯定有一种走法不会撞到象
        # - 当车和皇后在同一行或同一列时, 就算它们之间有象, 也可以先用一步把象移走, 第二步就可以用车吃到皇后
        # 因此只要判断车和象能否一步吃到皇后即可, 枚举车或象走的方向, 再枚举走的步数, 看能否在离开棋盘或者撞到其它棋子之前吃到皇后
        if a == e and (c != a or (b - d) * (f - d) > 0):
            return 1
        if b == f and (d != b or (a - c) * (e - c) > 0):
            return 1
        if c + d == e + f and (a + b != c + d or (c - a) * (e - a) > 0):
            return 1
        if c - d == e - f and (a - b != c - d or (c - a) * (e - a) > 0):
            return 1
        return 2

    def minMovesToCaptureTheQueen(
        self, a: int, b: int, c: int, d: int, e: int, f: int
    ) -> int:
        def ok(a: int, b: int, c: int) -> bool:
            return not min(a, c) < b < max(a, c)

        if (
            a == e
            and (c != e or ok(b, d, f))
            or b == f
            and (d != f or ok(a, c, e))  # 车 和 后 同一条直线, 象是否挡在中间
            or c + d == e + f
            and (a + b != e + f or ok(c, a, e))
            or c - d == e - f
            and (a - b != e - f or ok(c, a, e))  # 象 和 后 同一条斜线, 车是否挡在中间
        ):
            return 1
        return 2

    def minMovesToCaptureTheQueen(
        self, a: int, b: int, c: int, d: int, e: int, f: int
    ) -> int:
        def bet(a, b, c):
            return a <= b <= c or c <= b <= a

        if a == e and (a != c or (not bet(b, d, f))):
            return 1
        if b == f and (b != d or (not bet(a, c, e))):
            return 1

        if d - c == f - e and (d - c != b - a or (not bet(d, b, f))):
            return 1
        if d + c == e + f and (d + c != a + b or (not bet(d, b, f))):
            return 1

        return 2


# 3002 - Maximum Size of a Set After Removals - MEDIUM
class Solution:
    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        s1 = set(nums1)
        s2 = set(nums2)
        common = len(s1 & s2)
        n1 = len(s1)
        n2 = len(s2)
        ans = n1 + n2 - common
        half = len(nums1) // 2
        if n1 > half:
            # 说明 nums1 中没什么重复元素, 在移除 nums1 中的所有重复出现的元素之后, 还需要再移除一部分 = n1 - half
            # 从哪一部分移除比较好呢? 从公共部分中选较好
            # 如果交集较少, 先用交集抵掉 n1 - half 中的一部分, 然后再移除单独出现在 nums1 中的
            # 如果交集较多, 则交集部分可以cover还需要移除的数量
            x = min(n1 - half, common)
            ans -= n1 - half - x
            common -= x  # 注意要更新交集大小, 因为用交集相抵的部分已经不在 nums1 中出现了, 成了仅在 nums2 出现的元素
        if n2 > half:
            n2 -= min(n2 - half, common)
            ans -= n2 - half
        return ans

    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        # 反向思考或许容易一些
        # 简单转化一下题意: 给定两个长度为 n 的数组 nums1 和 nums2, 从每个数组中选出至多 n / 2 个数, 使得选出的数种类最多
        # 容易看出, 选重复的数是没有意义的, 因此首先把 nums1 和 nums2 变成集合进行去重
        # 接下来, 如果有一个数只在 nums1 不在 nums2, 那么选它一定不亏, 必选: 同理只在 nums2 不在 nums1 里的数也必选
        # 最后就只剩下又在 nums1 又在 nums2 里的数了, 这些数谁选都可以, 只要不超过选数的限额即可
        n = len(nums1)
        s1 = set(nums1)
        s2 = set(nums2)

        a = b = 0  # 选出每个集合独有的数
        for x in s1:
            if a < n // 2 and x not in s2:
                a += 1
        for x in s2:
            if b < n // 2 and x not in s1:
                b += 1
        common = s1 & s2  # 选出两个集合都有的数, 只要不超过选数限额即可, 谁选都一样
        for x in common:
            if a < n // 2:
                a += 1
            elif b < n // 2:
                b += 1
        return a + b

    def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1) // 2
        s1 = set(nums1)
        s2 = set(nums2)
        a = len([x for x in s1 if x not in s2])
        b = len([x for x in s2 if x not in s1])
        c = len([x for x in s1 if x in s2])
        d = n - min(a, n) + n - min(b, n)
        return min(a, n) + min(b, n) + min(c, d)
