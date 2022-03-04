import bisect, collections, functools, random, math, itertools, heapq
from typing import List, Optional


# 2104 - Sum of Subarray Ranges - MEDIUM
class Solution:
    # O(n ^ 2) / O(2)
    def subArrayRanges(self, nums: List[int]) -> int:
        ans = 0
        for i in range( len(nums)):
            mi = math.inf
            mx = -math.inf
            for j in range(i,  len(nums)):
                mi = min(mi, nums[j])
                mx = max(mx, nums[j])
                ans += mx - mi
        return ans


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


# 2169 - Minimum Operations to Make the Array Alternating - MEDIUM
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        cnt0 = collections.Counter(nums[::2])
        cnt1 = collections.Counter(nums[1::2])
        # cnt0 = sorted(cnt0.items(), key=lambda x: x[1], reverse=True)
        cnt0 = sorted(cnt0.items(), key=lambda x: -x[1])
        cnt1 = sorted(cnt1.items(), key=lambda x: -x[1])
        if cnt0[0][0] != cnt1[0][0]:
            return n - cnt0[0][1] - cnt1[0][1]
        else:
            a = n - cnt0[0][1] - (0 if len(cnt1) == 1 else cnt1[1][1])
            b = n - cnt1[0][1] - (0 if len(cnt0) == 1 else cnt0[1][1])
            return min(a, b)

    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        cnt0 = collections.Counter(nums[::2]).most_common()
        cnt1 = collections.Counter(nums[1::2]).most_common()

        e0, ecnt0 = cnt0[0] if cnt0 else (0, 0)
        _, ecnt1 = cnt0[1] if len(cnt0) > 1 else (0, 0)
        o0, ocnt0 = cnt1[0] if cnt1 else (0, 0)
        _, ocnt1 = cnt1[1] if len(cnt1) > 1 else (0, 0)
        if e0 != o0:
            return n - ecnt0 - ocnt0
        else:
            return min(n - ecnt1 - ocnt0, n - ecnt0 - ocnt1)

    def minimumOperations(self, nums: List[int]) -> int:
        c1 = collections.Counter(nums[::2])
        c2 = collections.Counter(nums[1::2])
        m1 = c1.most_common(2)
        m2 = c2.most_common(2)

        m1.append((None, 0))
        m2.append((None, 0))
        # or
        # if len(set(nums)) == 1:
        #     return len(nums) // 2

        ans = 0
        for k1, v1 in m1:
            for k2, v2 in m2:
                if k1 != k2:
                    ans = max(ans, v1 + v2)
        return len(nums) - ans


# 2170 - Removing Minimum Number of Magic Beans - MEDIUM
class Solution:
    def minimumRemoval(self, beans: List[int]) -> int:
        beans.sort(reverse=True)
        presum = [beans[0]] + [0] * (len(beans) - 1)
        for i in range(1, len(beans)):
            presum[i] = presum[i - 1] + beans[i]
        ans = math.inf
        left = 0
        for i in range(len(beans)):
            if i:
                left += (beans[i - 1] - beans[i]) * (i)
            right = presum[-1] - presum[i]
            ans = min(ans, right + left)
        return ans

    # min(pick) => max(save)
    def minimumRemoval(self, beans: List[int]) -> int:
        beans.sort()
        n = len(beans)
        total = remain = 0
        for i in range(n):
            total += beans[i]
            remain = max(remain, (n - i) * beans[i])
        return total - remain

    def minimumRemoval(self, beans: List[int]) -> int:
        beans.sort()
        n = len(beans)
        if n == 1:
            return 0
        total = sum(beans)
        ans = total
        for i in range(n):
            ans = min(total - beans[i] * (n - i), ans)
        return ans

    def minimumRemoval(self, beans: List[int]) -> int:
        beans.sort(reverse=True)
        ans = 0
        for i in range(len(beans)):
            ans = max(ans, beans[i] * (i + 1))
        return sum(beans) - ans


# 2185 - Counting Words With a Given Prefix - EASY
class Solution:
    def prefixCount(self, words: List[str], pref: str) -> int:
        return len([word for word in words if word.startswith(pref)])


# 2186 - Minimum Number of Steps to Make Two Strings Anagram II - MEDIUM
class Solution:
    def minSteps(self, s: str, t: str) -> int:
        cnt1 = collections.Counter(s)
        cnt2 = collections.Counter(t)
        ans = 0
        for i in range(26):
            ch = chr(ord('a') + i)
            ans += abs(cnt1[ch] - cnt2[ch])
        return ans

    def minSteps(self, s: str, t: str) -> int:
        arr = [0] * 26
        for ch in s:
            arr[ord(ch) - ord('a')] += 1
        for ch in t:
            arr[ord(ch) - ord('a')] -= 1
        ans = 0
        for i in arr:
            ans += abs(i)
        return ans


# 2187 - Minimum Time to Complete Trips - MEDIUM
class Solution:
    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        def check(t: int) -> bool:
            total = 0
            for period in time:
                total += t // period
            return total >= totalTrips

        l, r = 1, totalTrips * min(time)
        while l < r:
            mid = l + (r - l) // 2
            if check(mid):
                r = mid
            else:
                l = mid + 1
        return l

    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        cnt = collections.Counter(time)
        l, r = 1, totalTrips * min(time)
        while l < r:
            mid = (l + r) // 2
            total = 0
            for i, v in cnt.items():
                total += (mid // i) * v
            if total >= totalTrips:
                r = mid
            else:
                l = mid + 1
        return l


# 2188 - Minimum Time to Finish the Race - HARD
class Solution:
    def minimumFinishTime(self, tires: List[List[int]], changeTime: int,
                          numLaps: int) -> int:
        # minimum time to complete x consecutive circles with one tire (up to 17 circles)
        min_sec = [math.inf] * 18
        for f, r in tires:
            x, time, sum = 1, f, 0
            while time <= changeTime + f:
                sum += time
                min_sec[x] = min(min_sec[x], sum)
                time *= r
                x += 1

        f = [0] * (numLaps + 1)
        f[0] = -changeTime
        for i in range(1, numLaps + 1):
            f[i] = changeTime + min(f[i - j] + min_sec[j]
                                    for j in range(1, min(18, i + 1)))
        return f[numLaps]
