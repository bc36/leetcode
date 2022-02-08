import bisect, collections, functools, random, math, itertools, heapq
from typing import List, Optional


# 2164 - Sort Even and Odd Indices Independently - EASY
class Solution:
    def sortEvenOdd(self, nums: List[int]) -> List[int]:
        even = sorted(nums[::2])
        odd = sorted(nums[1::2])[::-1]
        nums[::2] = even
        nums[1::2] = odd
        return nums


# 2165 - Smallest Value of the Rearranged Number - MEDIUM
class Solution:
    def smallestNumber(self, num: int) -> int:
        if num > 0:
            l = list(str(num))
            zero = l.count('0')
            n = sorted(l)
            f = False
            ans = 0
            for i in n:
                if i == '0':
                    continue
                else:
                    if not f:
                        ans += int(i)
                        ans *= 10**zero
                        f = True
                    else:
                        ans = ans * 10 + int(i)
            return ans
        elif num < 0:
            l = list(str(-num))
            zero = l.count('0')
            n = sorted(l, reverse=True)
            ans = 0
            for i in n:
                if i == '0':
                    continue
                else:
                    ans = ans * 10 + int(i)
            ans *= 10**zero
            return -ans
        else:
            return 0


# 2166 - Design Bitset - MEDIUM
class Bitset:
    def __init__(self, size: int):
        self.a = ["0"] * size
        self.b = ["1"] * size
        self.size = size
        self.cnt = 0

    def fix(self, idx: int) -> None:
        if self.a[idx] == "0":
            self.a[idx] = "1"
            self.b[idx] = "0"
            self.cnt += 1

    def unfix(self, idx: int) -> None:
        if self.a[idx] == "1":
            self.a[idx] = "0"
            self.b[idx] = "1"
            self.cnt -= 1

    def flip(self) -> None:
        self.cnt = self.size - self.cnt
        self.a, self.b = self.b, self.a

    def all(self) -> bool:
        return self.cnt == self.size

    def one(self) -> bool:
        return self.cnt > 0

    def count(self) -> int:
        return self.cnt

    def toString(self) -> str:
        return "".join(self.a)


class Bitset:
    def __init__(self, size: int):
        self.arr = [0] * size
        self.ones = 0
        self.reverse = 0  # flag

    def fix(self, idx: int) -> None:
        if self.reverse ^ self.arr[idx] == 0:
            self.ones += 1
            self.arr[idx] ^= 1
        # if self.reverse:
        #     if self.arr[idx] == 1:
        #         self.ones += 1
        #     self.arr[idx] = 0
        # else:
        #     if self.arr[idx] == 0:
        #         self.ones += 1
        #     self.arr[idx] = 1

    def unfix(self, idx: int) -> None:
        if self.reverse ^ self.arr[idx] == 1:
            self.ones -= 1
            self.arr[idx] ^= 1
        # if self.reverse:
        #     if self.arr[idx] == 0:
        #         self.ones -= 1
        #     self.arr[idx] = 1
        # else:
        #     if self.arr[idx] == 1:
        #         self.ones -= 1
        #     self.arr[idx] = 0

    def flip(self) -> None:
        self.reverse ^= 1
        self.ones = len(self.arr) - self.ones

    def all(self) -> bool:
        return self.ones == len(self.arr)

    def one(self) -> bool:
        return self.ones > 0

    def count(self) -> int:
        return self.ones

    def toString(self) -> str:
        ans = ''
        for i in self.arr:
            ans += str(i ^ self.reverse)
        return ans


class Bitset:
    def __init__(self, size: int):
        self.c = [0] * size
        self.n = 0
        self.f = 0
        self.s = size

    def fix(self, idx: int) -> None:
        if self.c[idx] ^ self.f:
            self.n -= 1
        self.c[idx] = self.f ^ 1
        self.n += 1

    def unfix(self, idx: int) -> None:
        if self.c[idx] ^ self.f:
            self.n -= 1
        self.c[idx] = self.f

    def flip(self) -> None:
        self.f ^= 1
        self.n = self.s - self.n

    def all(self) -> bool:
        return self.s == self.n

    def one(self) -> bool:
        return self.n > 0

    def count(self) -> int:
        return self.n

    def toString(self) -> str:
        ans = ''
        for i in self.c:
            ans += str(i ^ self.f)
        return ans


# 2167 - Minimum Time to Remove All Cars Containing Illegal Goods - HARD
class Solution:
    # dp[i] = dp[i-1] if s[i] == '0' else min(dp[i-1]+2, i+1)
    def minimumTime(self, s: str) -> int:
        n = ans = len(s)
        pre = 0
        for idx, char in enumerate(s):
            if char == '1':
                pre = min(pre + 2, idx + 1)
            ans = min(ans, pre + n - idx - 1)
        return ans

    def minimumTime(self, s: str) -> int:
        n = len(s)
        suf = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            if s[i] == '0':
                suf[i] = suf[i + 1]
            else:
                suf[i] = min(suf[i + 1] + 2, n - i)
        ans = suf[0]
        pre = 0
        for i, ch in enumerate(s):
            if ch == '1':
                pre = min(pre + 2, i + 1)
                ans = min(ans, pre + suf[i + 1])
        return ans
