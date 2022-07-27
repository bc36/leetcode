import bisect, collections, functools, math, itertools, heapq
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 2100 - Find Good Days to Rob the Bank - MEDIUM
class Solution:
    def goodDaysToRobBank(self, s: List[int], time: int) -> List[int]:
        if time == 0:
            return [i for i in range(len(s))]
        n = len(s)
        f = [False] * n
        b = [False] * n
        count = 1
        for i in range(1, n):
            if s[i - 1] >= s[i]:
                f[i] = True if count >= time else False
                count += 1
            else:
                f[i] = False
                count = 1
        count = 1
        for i in range(n - 2, -1, -1):
            if s[i] <= s[i + 1]:
                b[i] = True if count >= time else False
                count += 1
            else:
                b[i] = False
                count = 1
        return [i for i in range(n) if f[i] and b[i]]

    def goodDaysToRobBank(self, s: List[int], t: int) -> List[int]:
        n = len(s)
        l = [0] * n
        r = [0] * n
        for i in range(1, n):
            if s[i - 1] >= s[i]:
                l[i] = l[i - 1] + 1
            if s[n - i - 1] <= s[n - i]:
                r[n - i - 1] = r[n - i] + 1
        return [i for i in range(t, n - t) if l[i] >= t and r[i] >= t]


# 2103 - Rings and Rods - EASY
class Solution:
    def countPoints(self, rings: str) -> int:
        d = collections.defaultdict(set)
        for i in range(0, len(rings), 2):
            d[rings[i + 1]].add(rings[i])
        return sum(1 for v in d.values() if len(v) == 3)

    def countPoints(self, rings: str) -> int:
        d = collections.defaultdict(set)
        for c, i in zip(rings[::2], rings[1::2]):
            d[i].add(c)
        return sum(1 for v in d.values() if len(v) == 3)


# 2104 - Sum of Subarray Ranges - MEDIUM
class Solution:
    # O(n ^ 2) / O(2)
    def subArrayRanges(self, nums: List[int]) -> int:
        ans = 0
        for i in range(len(nums)):
            mx = mi = nums[i]
            for j in range(i + 1, len(nums)):
                mx = max(mx, nums[j])
                mi = min(mi, nums[j])
                ans += mx - mi
        return ans

    # O(n) / O(n)
    # why monotonic stack: sum(ranges) = sum(maxes) - sum(mines)
    def subArrayRanges(self, nums: List[int]) -> int:
        ans = 0
        arr = [-math.inf] + nums + [-math.inf]
        s = []
        for i in range(len(arr)):
            while s and arr[s[-1]] > arr[i]:
                j = s.pop()
                # s[-1] ... j ... i
                # left: j - s[-1]
                # right: i - j
                ans -= arr[j] * (i - j) * (j - s[-1])
            s.append(i)
        arr = [math.inf] + nums + [math.inf]
        s = []
        for i in range(len(arr)):
            while s and arr[s[-1]] < arr[i]:
                j = s.pop()
                ans += arr[j] * (i - j) * (j - s[-1])
            s.append(i)
        return ans

    def subArrayRanges(self, nums: List[int]) -> int:
        ans = 0
        arr = nums + [-math.inf]
        s = [-1]
        for i in range(len(arr)):
            while s and arr[s[-1]] > arr[i]:
                j = s.pop()
                ans -= arr[j] * (i - j) * (j - s[-1])
            s.append(i)
        arr = nums + [math.inf]
        s = [-1]
        for i in range(len(arr)):
            while s and arr[s[-1]] < arr[i]:
                j = s.pop()
                ans += arr[j] * (i - j) * (j - s[-1])
            s.append(i)
        return ans


# 2105 - Watering Plants II - MEDIUM
class Solution:
    def minimumRefill(self, plants: List[int], capacityA: int, capacityB: int) -> int:
        ans = i = 0
        ca = capacityA
        cb = capacityB
        j = len(plants) - 1
        while i < j:
            if ca < plants[i]:
                ans += 1
                ca = capacityA
            if cb < plants[j]:
                ans += 1
                cb = capacityB
            ca -= plants[i]
            cb -= plants[j]
            i += 1
            j -= 1
        if i == j:
            if ca < plants[i] and cb < plants[j]:
                ans += 1
        return ans


# 2106 - Maximum Fruits Harvested After at Most K Steps - HARD
class Solution:
    # O(n) / O(1), cover: b - a + min(sp - a, b - sp)
    def maxTotalFruits(self, f: List[List[int]], sp: int, k: int) -> int:
        ans = l = summ = 0
        cal = lambda l, r: f[r][0] - f[l][0] + min(abs(sp - f[l][0]), abs(sp - f[r][0]))
        for r in range(len(f)):
            while l <= r and cal(l, r) > k:
                summ -= f[l][1]
                l += 1
            summ += f[r][1]
            ans = max(ans, summ)
        return ans

    def maxTotalFruits(self, f: List[List[int]], sp: int, k: int) -> int:
        dq = collections.deque([])
        ans = i = 0
        while i < len(f) and f[i][0] <= sp:  # left part
            if sp - f[i][0] <= k:
                ans += f[i][1]
                dq.append((f[i][0], f[i][1]))
            i += 1
        tmp = ans
        while i < len(f) and f[i][0] - sp <= k:  # right part
            while (
                dq
                and dq[0][0] < sp
                and f[i][0] - dq[0][0] + min(sp - dq[0][0], f[i][0] - sp) > k
            ):
                tmp -= dq[0][1]
                dq.popleft()
            tmp += f[i][1]
            ans = max(ans, tmp)
            i += 1
        return ans

    # TLE
    def maxTotalFruits(self, f: List[List[int]], sp: int, k: int) -> int:
        ans = 0
        for i in range(len(f)):
            j = i
            summ = 0
            while j < len(f) and 2 * max(sp - f[i][0], 0) + max(f[j][0] - sp, 0) <= k:
                summ += f[j][1]
                ans = max(ans, summ)
                j += 1
            summ -= f[i][1]
        for i in range(len(f)):
            j = i
            summ = 0
            while j < len(f) and max(sp - f[i][0], 0) + 2 * max(f[j][0] - sp, 0) <= k:
                summ += f[j][1]
                ans = max(ans, summ)
                j += 1
            summ -= f[i][1]
        return ans

    # TLE
    def maxTotalFruits(self, f: List[List[int]], sp: int, k: int) -> int:
        ans = 0
        for i in range(len(f)):
            j = i
            summ = 0
            while j < len(f) and 2 * max(sp - f[i][0], 0) + max(f[j][0] - sp, 0) <= k:
                summ += f[j][1]
                ans = max(ans, summ)
                j += 1
            summ -= f[i][1]
        for i in range(len(f) - 1, -1, -1):
            j = i
            summ = 0
            while j >= 0 and 2 * max(f[i][0] - sp, 0) + 2 * max(sp - f[j][0], 0) <= k:
                summ += f[j][1]
                ans = max(ans, summ)
                j -= 1
            summ -= f[i][1]
        return ans


# 2108 - Find First Palindromic String in the Array - EASY
class Solution:
    def firstPalindrome(self, words: List[str]) -> str:
        for w in words:
            if w == w[::-1]:
                return w
        return ""


# 2109 - Adding Spaces to a String - MEDIUM
class Solution:
    def addSpaces(self, s: str, spaces: List[int]) -> str:
        ans = ""
        spaces = spaces[::-1]
        for i, c in enumerate(s):
            if spaces and i == spaces[-1]:
                ans += " "
                spaces.pop()
            ans += c
        return ans


# 2110 - Number of Smooth Descent Periods of a Stock - MEDIUM
class Solution:
    def getDescentPeriods(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [1] * n
        for i in range(1, n):
            if prices[i - 1] - 1 == prices[i]:
                dp[i] += dp[i - 1]
        return sum(dp)

    def getDescentPeriods(self, prices: List[int]) -> int:
        ans = cur = 1
        for i in range(1, len(prices)):
            if prices[i] == prices[i - 1] - 1:
                cur += 1
            else:
                cur = 1
            ans += cur
        return ans

    def getDescentPeriods(self, prices: List[int]) -> int:
        ans = pre = cur = 0
        for p in prices:
            if pre - 1 == p:
                cur += 1
            else:
                cur = 1
            ans += cur
            pre = p
        return ans


# 2111 - Minimum Operations to Make the Array K-Increasing - HARD
class Solution:
    # O(n * log(n/k)) / O(n / k)
    def kIncreasing(self, arr: List[int], k: int) -> int:
        group = collections.defaultdict(list)
        for i, v in enumerate(arr):
            group[i % k].append(v)
        ans = 0
        # LIS: longest increasing subsequence, lc300
        for g in group.values():
            a = []
            for v in g:
                pos = bisect.bisect_right(a, v)
                if pos == len(a):
                    a.append(v)
                else:
                    a[pos] = v
            ans += len(g) - len(a)
        return ans

    def kIncreasing(self, arr: List[int], k: int) -> int:
        def LIS(nums: List[int]) -> int:
            a = []
            for v in nums:
                pos = bisect.bisect_right(a, v)
                if pos == len(a):
                    a.append(v)
                else:
                    a[pos] = v
            return len(a)

        ans = 0
        for i in range(k):
            group = []
            for x in range(i, len(arr), k):
                group.append(arr[x])
            ans += len(group) - LIS(group)
        return ans


# 2119 - A Number After a Double Reversal - EASY
class Solution:
    def isSameAfterReversals(self, num: int) -> bool:
        n = int(str(num)[::-1])
        n = int(str(n)[::-1])
        return n == num

    def isSameAfterReversals(self, num: int) -> bool:
        if num != 0 and num % 10 == 0:
            return False
        return True
        # return num == 0 or num % 10 != 0


# 2120 - Execution of All Suffix Instructions Staying in a Grid - MEDIUM
class Solution:
    def executeInstructions(self, n: int, startPos: List[int], s: str) -> List[int]:
        ans = []
        d = {"R": (0, 1), "L": (0, -1), "U": (-1, 0), "D": (1, 0)}
        for i in range(len(s)):
            x, y = startPos
            j = i
            while j < len(s):
                dx, dy = d[s[j]]
                x += dx
                y += dy
                if not 0 <= x < n or not 0 <= y < n:
                    break
                j += 1
            ans.append(j - i)
        return ans

    def executeInstructions(self, n: int, startPos: List[int], s: str) -> List[int]:
        ans = []
        d = {"R": (0, 1), "L": (0, -1), "U": (-1, 0), "D": (1, 0)}
        for i in range(len(s)):
            x, y = startPos
            cnt = 0
            for j in range(i, len(s)):
                dx, dy = d[s[j]]
                x += dx
                y += dy
                if not 0 <= x < n or not 0 <= y < n:
                    break
                cnt += 1
            ans.append(cnt)
        return ans


# 2121 - Intervals Between Identical Elements - MEDIUM
class Solution:
    # O(n) / O(n), prefix sum + suffix sum
    def getDistances(self, arr: List[int]) -> List[int]:
        ans = [0] * len(arr)
        p = collections.defaultdict(int)
        for i, v in enumerate(arr):
            if v in p:
                summ, x, cnt = p[v]
                p[v] = (cnt * (i - x) + summ, i, cnt + 1)
            else:
                p[v] = (0, i, 1)
            ans[i] += p[v][0]
        s = collections.defaultdict(int)
        n = len(arr) - 1
        # for i, v in zip(reversed(range(len(arr))), reversed(arr)):
        for i, v in enumerate(arr[::-1]):
            i = n - i
            if v in s:
                summ, x, cnt = s[v]
                s[v] = (cnt * (x - i) + summ, i, cnt + 1)
            else:
                s[v] = (0, i, 1)
            ans[i] += s[v][0]
        return ans

    def getDistances(self, arr: List[int]) -> List[int]:
        d = collections.defaultdict(list)
        for i in range(len(arr)):
            d[arr[i]].append(i)
        ans = [0] * len(arr)
        for v in d.values():
            pre = 0
            s = sum(v)
            for i in range(len(v)):
                # left: v[i] * i - pre
                # right: (s - pre) - v[i] * (len(v) - i)
                ans[v[i]] = v[i] * i - pre + (s - pre) - v[i] * (len(v) - i)
                pre += v[i]
        return ans

    def getDistances(self, arr: List[int]) -> List[int]:
        d = collections.defaultdict(list)
        for i, v in enumerate(arr):
            d[v].append(i)
        ans = [0] * len(arr)
        for v in d.values():
            sub = sum(v)
            pre = 0
            cnt = len(v)
            for i in v:
                ans[i] = sub - i * cnt - pre
                cnt -= 2
                sub -= i
                pre += i
        return ans

    def getDistances(self, arr: List[int]) -> List[int]:
        d = collections.defaultdict(list)  # total_abs, pre_idx, pre, sub
        for i, v in enumerate(arr):
            if len(d[v]) == 0:
                d[v] = [0, i, 0, 1]
            else:
                d[v][0] += i - d[v][1]
                d[v][3] += 1
        ans = []
        for i, v in enumerate(arr):
            x = d[v][0] + (i - d[v][1]) * (d[v][2] - d[v][3])
            ans.append(x)
            d[v][0] = x
            d[v][1] = i
            d[v][2] += 1
            d[v][3] -= 1
        return ans

    def getDistances(self, arr: List[int]) -> List[int]:
        group = collections.defaultdict(list)
        for i, v in enumerate(arr):
            group[v].append(i)
        ans = [0] * len(arr)
        for g in group.values():
            summ = sum(i - g[0] for i in g)
            ans[g[0]] = summ
            for i in range(1, len(g)):
                summ += i * (g[i] - g[i - 1]) - (len(g) - i) * (g[i] - g[i - 1])
                ans[g[i]] = summ
        return ans


# 2122 - Recover the Original Array - HARD
class Solution:
    # O(n ^ 2) / O(n), two pointers
    def recoverArray(self, nums: List[int]) -> List[int]:
        nums.sort()
        n = len(nums)
        for i in range(1, n):
            if nums[i] == nums[i - 1]:  # skip same k cuz same num
                continue
            d = nums[i] - nums[0]
            if d & 1:
                continue
            k = d // 2
            vis = [False] * n
            vis[i] = True
            ans = [nums[0] + k]
            l = 1
            r = i + 1
            while r < n:
                while vis[l]:  # skip the elements seen in 'higher'
                    l += 1
                while r < n and nums[r] - nums[l] < 2 * k:
                    r += 1
                if r == n or nums[r] - nums[l] > 2 * k:
                    break
                vis[r] = True
                ans.append((nums[l] + nums[r]) // 2)
                l += 1
                r += 1
            if len(ans) == n // 2:
                return ans

    # hashmap
    def recoverArray(self, nums: List[int]) -> List[int]:
        nums.sort()
        for i in range(1, len(nums)):
            if (nums[i] - nums[0]) % 2 == 1 or nums[i] == nums[i - 1]:
                continue
            k = (nums[i] - nums[0]) // 2
            cnt = collections.Counter(nums)
            f = True
            ans = []
            for v in nums:
                if cnt[v] == 0:
                    continue
                if cnt[v + k * 2] == 0:
                    f = False
                    break
                cnt[v] -= 1
                cnt[v + k * 2] -= 1
                ans.append(v + k)
            if f:
                return ans

    # delay delete, require array sorted, hashmap is more useful
    def recoverArray(self, nums: List[int]) -> List[int]:
        nums.sort()
        for n in nums:
            d = n - nums[0]
            if d and d % 2 == 0:
                dq = collections.deque()
                ans = []
                for n in nums:
                    if dq and n == dq[0]:
                        ans.append(dq.popleft() - d // 2)
                    else:
                        dq.append(n + d)
                if not dq:
                    return ans


# 2124 - Check if All A's Appears Before All B's - EASY
class Solution:
    def checkString(self, s: str) -> bool:
        if len(set(s)) == 1:
            return True
        a = len(s) - 1 - s[::-1].find("a")
        b = s.find("b")
        return True if a < b else False

    def checkString(self, s: str) -> bool:
        return s.find("ba") == -1
        return "".join(sorted(s)) == s


# 2125 - Number of Laser Beams in a Bank - MEDIUM
class Solution:
    def numberOfBeams(self, bank: List[str]) -> int:
        ans = pre = 0
        for s in bank:
            cur = s.count("1")
            if cur != 0:
                ans += pre * cur
                pre = cur
        return ans


# 2126 - Destroying Asteroids - MEDIUM
class Solution:
    def asteroidsDestroyed(self, mass: int, asteroids: List[int]) -> bool:
        asteroids.sort()
        for v in asteroids:
            if v > mass:
                return False
            mass += v
        return True


# 2134 - Minimum Swaps to Group All 1's Together II - MEDIUM
class Solution:
    # maximum 1
    def minSwaps(self, nums: List[int]) -> int:
        ones = sum(nums)
        mx = c = sum(nums[:ones])
        nums = nums * 2
        l = 0
        for r in range(ones, len(nums)):
            c -= nums[l]
            l += 1
            c += nums[r]
            mx = max(mx, c)
        return ones - mx

    def minSwaps(self, nums: List[int]) -> int:
        ones = nums.count(1)
        n = len(nums)
        cur = mx = 0
        for i in range(n * 2):
            if i >= ones and nums[i % n - ones]:
                cur -= 1
            if nums[i % n] == 1:
                cur += 1
            mx = max(cur, mx)
        return ones - mx

    # minimum 0
    def minSwaps(self, nums: List[int]) -> int:
        ones = sum(nums)
        window = sum(nums[:ones])
        ans = ones - window
        for i in range(len(nums)):
            window -= nums[i]
            i = (i + ones) % len(nums)
            window += nums[i]
            ans = min(ans, ones - window)
        return ans


# 2135 - Count Words Obtained After Adding a Letter - MEDIUM
class Solution:
    def wordCount(self, startWords: List[str], targetWords: List[str]) -> int:
        d = collections.defaultdict(set)
        for s in startWords:
            d[len(s)].add("".join(sorted(s)))
        ans = 0
        for t in targetWords:
            st = "".join(sorted(t))
            # do not operate each set from d[len(t) - 1], TLE
            for i in range(len(t)):
                if st[:i] + st[i + 1 :] in d[len(t) - 1]:
                    ans += 1
                    break
        return ans

    # Since 'No letter occurs more than once in any string of startWords or targetWords'.
    # if occurs more than once, use above solution.
    def wordCount(self, startWords: List[str], targetWords: List[str]) -> int:
        def s2b(s: str):
            n = 0
            for c in s:
                n |= 1 << (ord(c) - ord("a"))
            return n

        s = set()
        ans = 0
        for v in startWords:
            s.add(s2b(v))
        for t in targetWords:
            x = s2b(t)
            for c in t:
                # if x - (1 << (ord(c) - 97)) in s:
                #     ans += 1
                #     break
                if x ^ s2b(c) in s:
                    ans += 1
                    break
        return ans

    # frozenset: TypeError: unhashable type: 'set'
    def wordCount(self, startWords: List[str], targetWords: List[str]) -> int:
        s = set(frozenset(w) for w in startWords)
        ans = 0
        for t in targetWords:
            st = set(t)
            for c in t:
                if st.difference(c) in s:
                    ans += 1
                    break
        return ans


# 2151 - Maximum Good People Based on Statements - HARD
class Solution:
    # O(2^n * n^2) / O(1)
    def maximumGood(self, statements: List[List[int]]) -> int:
        def check(s):
            cnt = 0
            for i, v in enumerate(statements):
                if (s >> i) & 1:
                    for j in range(len(v)):
                        if v[j] < 2 and v[j] != (s >> j) & 1:
                            return 0
                    cnt += 1
            return cnt

        ans = 0
        for s in range(1 << len(statements)):
            ans = max(ans, check(s))
        return ans

    def maximumGood(self, statements: List[List[int]]) -> int:
        def check(i: int) -> int:
            cnt = 0
            for j, s in enumerate(statements):
                if (i >> j) & 1:
                    if any(v < 2 and v != (i >> k) & 1 for k, v in enumerate(s)):
                        return 0
                    cnt += 1
            return cnt

        return max(check(i) for i in range(1, 1 << len(statements)))

    def maximumGood(self, statements: List[List[int]]) -> int:
        def check(st):
            st = set(st)
            for m in st:
                for i, s in enumerate(statements[m]):
                    if (s == 0 and i in st) or (s == 1 and i not in st):
                        return False
            return True

        n = len(statements)
        nums = list(range(n))
        for i in range(n + 1, 0, -1):
            for comb in itertools.combinations(nums, i):
                if check(comb):
                    return i
        return 0


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
            zero = l.count("0")
            n = sorted(l)
            f = False
            ans = 0
            for i in n:
                if i == "0":
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
            zero = l.count("0")
            n = sorted(l, reverse=True)
            ans = 0
            for i in n:
                if i == "0":
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
        ans = ""
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
        ans = ""
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
            if char == "1":
                pre = min(pre + 2, idx + 1)
            ans = min(ans, pre + n - idx - 1)
        return ans

    def minimumTime(self, s: str) -> int:
        n = len(s)
        suf = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            if s[i] == "0":
                suf[i] = suf[i + 1]
            else:
                suf[i] = min(suf[i + 1] + 2, n - i)
        ans = suf[0]
        pre = 0
        for i, ch in enumerate(s):
            if ch == "1":
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
            ch = chr(ord("a") + i)
            ans += abs(cnt1[ch] - cnt2[ch])
        return ans

    def minSteps(self, s: str, t: str) -> int:
        arr = [0] * 26
        for ch in s:
            arr[ord(ch) - ord("a")] += 1
        for ch in t:
            arr[ord(ch) - ord("a")] -= 1
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
    def minimumFinishTime(
        self, tires: List[List[int]], changeTime: int, numLaps: int
    ) -> int:
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
            f[i] = changeTime + min(
                f[i - j] + min_sec[j] for j in range(1, min(18, i + 1))
            )
        return f[numLaps]


# 2190 - Most Frequent Number Following Key In an Array - EASY
class Solution:
    def mostFrequent(self, nums: List[int], key: int) -> int:
        dic = collections.defaultdict(int)
        for i in range(1, len(nums)):
            if nums[i - 1] == key:
                dic[nums[i]] += 1
        ans = t = 0
        for num in dic:
            if dic[num] > t:
                ans = num
                t = dic[num]
        return ans

    def mostFrequent(self, nums: List[int], key: int) -> int:
        cnt = collections.Counter()
        for i in range(len(nums) - 1):
            if nums[i] == key:
                cnt[nums[i + 1]] += 1
        return cnt.most_common(1)[0][0]


# 2191 - Sort the Jumbled Numbers - MEDIUM
class Solution:
    def sortJumbled(self, mapping: List[int], nums: List[int]) -> List[int]:
        arr = []
        # m = {i: n for i, n in enumerate(mapping)}
        for n in nums:
            s = list(str(n))
            t = 0
            x = n
            for i in range(len(s)):
                t = t * 10 + mapping[int(s[i]) % 10]
            arr.append((x, t))
        arr.sort(key=lambda x: x[1])
        return [i[0] for i in arr]


# 2192 - All Ancestors of a Node in a Directed Acyclic Graph - MEDIUM
class Solution:
    def getAncestors(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        g = collections.defaultdict(list)
        for o, i in edges:
            g[i].append(o)
        ans = []
        for i in range(n):
            seen = set()
            dq = collections.deque([i])
            while dq:
                n = dq.popleft()
                for nxt in g[n]:
                    if nxt not in seen:
                        dq.append(nxt)
                        seen.add(nxt)
            ans.append(sorted(seen))
        return ans


# 2194 - Cells in a Range on an Excel Sheet - EASY
class Solution:
    def cellsInRange(self, s: str) -> List[str]:
        ans = []
        for i in range(ord(s[0]), ord(s[3]) + 1):
            for j in range(int(s[1]), int(s[4]) + 1):
                ans.append(chr(i) + str(j))
        return ans


# 2195 - Append K Integers With Minimal Sum - MEDIUM
class Solution:
    def minimalKSum(self, nums: List[int], k: int) -> int:
        nums.sort()
        ans = k * (k + 1) // 2
        last = k
        pre = -1
        for num in nums:
            if num == pre:  # be careful!
                continue
            if num <= last:
                ans += last + 1 - num
                last += 1
                pre = num
            else:
                break
        return ans

    def minimalKSum(self, nums: List[int], k: int) -> int:
        ans = 0
        nums += [0, 2e9 + 1]  # larger than 2*k
        nums.sort()
        for i in range(1, len(nums)):
            fill = nums[i] - nums[i - 1] - 1
            if fill < 0:  # repeat
                continue
            if fill >= k:
                ans += (nums[i - 1] * 2 + 1 + k) * k // 2
                break
            ans += (nums[i - 1] + nums[i]) * fill // 2
            k -= fill
        return ans


# 2196 - Create Binary Tree From Descriptions - MEDIUM
class Solution:
    def createBinaryTree(self, d: List[List[int]]) -> Optional[TreeNode]:
        tree = collections.defaultdict(list)
        ind = set()
        for p, c, i in d:
            tree[p].append((c, i))
            ind.add(c)
        root = None
        for p, _, _ in d:
            if p not in ind:
                root = TreeNode(p)
                break
        dq = collections.deque([root])
        while dq:
            n = dq.popleft()
            for ch, i in tree[n.val]:
                if i == 1:
                    n.left = TreeNode(ch)
                    dq.append(n.left)
                else:
                    n.right = TreeNode(ch)
                    dq.append(n.right)
        return root

    def createBinaryTree(self, r: List[List[int]]) -> Optional[TreeNode]:
        # build tree
        # and calculate the indegree
        # indegree = 0, root
        ind = collections.Counter()
        # node value -> node
        mp = dict()
        for parent, child, isLeft in r:
            if parent not in mp:
                mp[parent] = TreeNode(parent)
            if child not in mp:
                mp[child] = TreeNode(child)
            ind[child] += 1
            if isLeft:
                mp[parent].left = mp[child]
            else:
                mp[parent].right = mp[child]
        for parent, child, isLeft in r:
            if ind[parent] == 0:
                return mp[parent]
        return None

    def createBinaryTree(self, r: List[List[int]]) -> Optional[TreeNode]:
        nodes = {}
        hasParent = set()
        for (parent, child, left) in r:
            if parent not in nodes:
                nodes[parent] = TreeNode(parent)
            if child not in nodes:
                nodes[child] = TreeNode(child)
            if left:
                nodes[parent].left = nodes[child]
            else:
                nodes[parent].right = nodes[child]
            hasParent.add(child)
        for node in nodes:
            if node not in hasParent:
                return nodes[node]


# 2197 - Replace Non-Coprime Numbers in Array - HARD
class Solution:
    def replaceNonCoprimes(self, nums: List[int]) -> List[int]:
        def gcd(x: int, y: int) -> int:
            while y:
                x, y = y, x % y
            return x

        def lcm(x: int, y: int) -> int:
            res = (x * y) // gcd(x, y)
            return res

        q = collections.deque()
        for i in range(len(nums)):
            q.append(nums[i])
            while len(q) >= 2 and gcd(q[-1], q[-2]) > 1:
                fir = q.pop()
                sec = q.pop()
                q.append(lcm(fir, sec))
        return list(q)

    def replaceNonCoprimes(self, nums: List[int]) -> List[int]:
        s = [nums[0]]
        for num in nums[1:]:
            s.append(num)
            while len(s) > 1:
                x, y = s[-1], s[-2]
                g = math.gcd(x, y)
                if g == 1:
                    break
                s.pop()
                s[-1] *= x // g
        return s

    def replaceNonCoprimes(self, nums: List[int]) -> List[int]:
        ans = [1]
        for n in nums:
            g = math.gcd(ans[-1], n)
            if g <= 1:
                ans.append(n)
                continue
            while g > 1:
                n = ans.pop() * n // g
                g = math.gcd(ans[-1], n)
            ans.append(n)
        return ans[1:]

    def replaceNonCoprimes(self, nums: List[int]) -> List[int]:
        ans = []
        pre = nums[0]
        for cur in nums[1:]:
            if (g := math.gcd(cur, pre)) > 1:
                cur = cur * pre // g
                while ans and (gg := math.gcd(cur, ans[-1])) > 1:
                    last = ans.pop()
                    cur = cur * last // gg
                pre = cur
            else:
                ans.append(pre)
                pre = cur
        ans.append(pre)
        return ans

    def replaceNonCoprimes(self, nums: List[int]) -> List[int]:
        s = []
        for num in nums:
            while s and (g := math.gcd(num, s[-1])) > 1:
                num = s.pop() // g * num
            s.append(num)
        return s
