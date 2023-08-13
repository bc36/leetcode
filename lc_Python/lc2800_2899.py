import bisect, collections, functools, heapq, itertools, math, operator, string
from typing import List, Optional, Tuple
import sortedcontainers


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 2800 - Shortest String That Contains Three Strings - MEDIUM
class Solution:
    # O(n^2) / O(n)
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(s: str, t: str) -> str:
            # 先特判完全包含的情况
            if t in s:
                return s
            if s in t:
                return t
            for i in range(min(len(s), len(t)), 0, -1):
                # 枚举：s 的后 i 个字母和 t 的前 i 个字母是一样的
                if s[-i:] == t[:i]:
                    return s + t[i:]
            return s + t

        return min(
            (merge(merge(a, b), c) for a, b, c in itertools.permutations((a, b, c))),
            key=lambda s: (len(s), s),
        )

        ans = ""
        for a, b, c in itertools.permutations((a, b, c)):
            s = merge(merge(a, b), c)
            if ans == "" or len(s) < len(ans) or len(s) == len(ans) and s < ans:
                ans = s
        return ans


# 2801 - Count Stepping Numbers in Range - HARD
class Solution:
    # 数位dp
    # n = len(high)
    # 时间复杂度 = O(状态个数) * O(单个状态需要的时间)
    #          = O(10n) * O(10)
    #          = O(100n)
    # https://leetcode.cn/problems/count-stepping-numbers-in-range/solution/shu-wei-dp-tong-yong-mo-ban-by-endlessch-h8fj/

    # 620ms
    def countSteppingNumbers(self, low: str, high: str) -> int:
        mod = 10**9 + 7

        def calc(s: str) -> int:
            @functools.cache
            def f(i: int, pre: int, isLimit: bool, isNum: bool) -> int:
                """表示构造第 i 位及其之后数位的合法方案数"""
                if i == len(s):
                    return int(isNum)  # isNum 为 True 表示得到了一个合法数字
                res = 0
                if not isNum:  # 可以跳过当前数位
                    res = f(i + 1, pre, False, False)
                down = 0 if isNum else 1  # 如果前面没有填数字, 必须从 1 开始, 因为不能有前导零
                # 如果前面填的数字都和 s 的一样, 那么这一位至多填 s[i], 否则就超过 s 了
                up = int(s[i]) if isLimit else 9
                for d in range(down, up + 1):  # 枚举要填入的数字 d
                    if not isNum or abs(d - pre) == 1:  # 第一位数字随便填, 其余必须相差 1
                        res += f(i + 1, d, isLimit and d == up, True)
                return res % mod

            return f(0, 0, True, False)

        return (calc(high) - calc(str(int(low) - 1))) % mod

    def countSteppingNumbers(self, low: str, high: str) -> int:
        mod = 10**9 + 7

        def calc(s: str) -> int:
            @functools.cache
            def f(i: int, pre: int, isLimit: bool, isNum: bool) -> int:
                if i == len(s):
                    return int(isNum)
                res = 0
                if not isNum:
                    res = f(i + 1, -1, False, False)
                down = 0 if isNum else 1
                up = int(s[i]) if isLimit else 9
                for d in range(down, up + 1):
                    if pre != -1 and abs(d - pre) != 1:
                        continue
                    res += f(i + 1, d, isLimit and d == up, True)
                return res % mod

            return f(0, -1, True, False)

        return (calc(high) - calc(str(int(low) - 1))) % mod

    # 280ms
    def countSteppingNumbers(self, low: str, high: str) -> int:
        mod = 10**9 + 7
        # 把 high 和 low 通过补 0 补到一样长, 就可以一起计算了
        low = "0" * (len(high) - len(low)) + low

        @functools.cache
        def f(i: int, pre: int, isUpLimit: bool, isDownLimit: bool, isNum: bool) -> int:
            if i == len(high):
                return isNum
            res = 0
            down = int(low[i]) if isDownLimit else 0
            up = int(high[i]) if isUpLimit else 9
            for d in range(down, up + 1):  # 枚举要填入的数字 d
                if not isNum or d - pre in (-1, 1):
                    res += f(
                        i + 1,
                        d,
                        isUpLimit and d == up,
                        isDownLimit and d == down,
                        isNum or d > 0,
                    )
            return res % mod

        return f(0, 0, True, True, False)
