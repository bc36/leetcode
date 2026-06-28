import bisect, collections, copy, functools, heapq, itertools, math, operator, random, string
from typing import Callable, List, Literal, Optional, Tuple
import sortedcontainers


# 3606 - Coupon Code Validator - EASY
class Solution:
    def validateCoupons(
        self, code: List[str], businessLine: List[str], isActive: List[bool]
    ) -> List[str]:
        ans = []
        s = set(string.ascii_letters + string.digits + "_")
        business = {"electronics", "grocery", "pharmacy", "restaurant"}
        for c, b, i in zip(code, businessLine, isActive):
            # 这是 Python 的一个常见坑:
            # any([]) -> False
            # all([]) -> True
            if any(x not in s for x in c) or not c:
                continue
            if b not in business:
                continue
            if not i:
                continue
            ans.append((c, b))
        ans.sort(key=lambda x: (x[1], x[0]))
        return [x for x, _ in ans]


# 3612 - Process String with Special Operations I - MEDIUM
class Solution:
    def processStr(self, s: str) -> str:
        ans = []
        for c in s:
            if c == "*":
                if ans:
                    ans.pop()
            elif c == "#":
                ans += ans
            elif c == "%":
                ans = ans[::-1]
            else:
                ans.append(c)
        return "".join(ans)


# 3614 - Process String with Special Operations II - HARD
class Solution:
    def processStr(self, s: str, k: int) -> str:
        n = len(s)
        size = [0] * n
        sz = 0
        for i, c in enumerate(s):
            if c == "*":
                sz = max(sz - 1, 0)
            elif c == "#":
                sz *= 2
            elif c != "%":  # c 是字母
                sz += 1
            size[i] = sz

        if k >= size[-1]:
            return "."

        for i in range(n - 1, -1, -1):
            c = s[i]
            sz = size[i]
            if c == "#":
                if k >= sz // 2:  # k 在复制后的右半边
                    k -= sz // 2
            elif c == "%":
                k = sz - 1 - k  # 反转前的下标为 sz-1-k 的字母就是答案
            elif c != "*" and k == sz - 1:  # 找到答案
                return c
