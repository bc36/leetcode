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


# 3005 - Count Elements With Maximum Frequency - EASY
class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        cnt = collections.Counter(nums)
        mx = max(cnt.values())
        return mx * sum(v == mx for v in cnt.values())
        return sum(v for v in cnt.values() if v == mx)

    def maxFrequencyElements(self, nums: List[int]) -> int:
        ans = mx = 0
        cnt = collections.Counter()
        for x in nums:
            cnt[x] += 1
            if cnt[x] > mx:
                mx = ans = cnt[x]
            elif cnt[x] == mx:
                ans += cnt[x]
        return ans


# 3006 - Find Beautiful Indices in the Given Array I - MEDIUM
# 题意同 LC 3008, 但数据范围不同
# 1 <= k <= s.length <= 10**5
# 1 <= a.length, b.length <= 10

# LC 3008 数据范围
# 1 <= k <= s.length <= 5 * 10**5
# 1 <= a.length, b.length <= 5 * 10**5


# 3007 - Maximum Number That Sum of the Prices Is Less Than or Equal to K - MEDIUM
class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        def countDigitOne(n: int) -> int:
            """类似 LC233"""
            s = bin(n)[2:]

            @functools.cache
            def dfs(i: int, cnt: int, is_limit: bool) -> int:
                if i == len(s):
                    return cnt
                res = 0
                up = int(s[i]) if is_limit else 1
                for d in range(up + 1):
                    res += dfs(
                        i + 1,
                        cnt + (d == 1 and (len(s) - i) % x == 0),
                        is_limit and d == up,
                    )
                return res

            return dfs(0, 0, True)

        return bisect.bisect_left(range(10**16), k + 1, key=countDigitOne) - 1
        return bisect.bisect_right(range(10**16), k, key=countDigitOne) - 1

    def findMaximumNumber(self, k: int, x: int) -> int:
        def count(num: int) -> int:
            @functools.cache
            def f(i: int, cnt: int, is_limit: bool) -> int:
                if i == 0:
                    return cnt
                res = 0
                up = num >> (i - 1) & 1 if is_limit else 1
                for d in range(up + 1):
                    res += f(i - 1, cnt + (d == 1 and i % x == 0), is_limit and d == up)
                return res

            return f(num.bit_length(), 0, True)

        # <= k 转换成 >= k+1 的数再减一
        # 原理见 https://www.bilibili.com/video/BV1AP41137w7/
        return bisect.bisect_left(range((k + 1) << x), k + 1, key=count) - 1
        return bisect.bisect_right(range((k + 1) << x), k, key=count) - 1

    # https://leetcode.cn/problems/maximum-number-that-sum-of-the-prices-is-less-than-or-equal-to-k/solutions/2603673/er-fen-da-an-shu-wei-dpwei-yun-suan-pyth-tkir/
    # https://leetcode.cn/problems/maximum-number-that-sum-of-the-prices-is-less-than-or-equal-to-k/solutions/2603671/xiao-yang-xiao-en-er-fen-shu-xue-er-jin-c7iwg/


# 3008 - Find Beautiful Indices in the Given Array II - HARD
class KMP:
    def __init__(self):
        return

    @staticmethod
    def prefix_function(s: str) -> List[int]:
        """calculate the longest common true prefix and true suffix for s [:i] and s [:i]"""
        pi = [0] * len(s)
        for i in range(1, len(s)):
            j = pi[i - 1]
            while j > 0 and s[i] != s[j]:
                j = pi[j - 1]
            if s[i] == s[j]:
                j += 1
            pi[i] = j  # pi[i]<=i
        # pi[0] = 0
        return pi

    def find(self, s1: str, s2: str):
        """find the index position of s2 in s1"""
        n, m = len(s1), len(s2)
        pi = self.prefix_function(s2 + "#" + s1)
        ans = []
        for i in range(m + 1, m + n + 1):
            if pi[i] == m:
                ans.append(i - m - m)
        return ans


class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        lst1 = KMP().find(s, a)
        lst2 = KMP().find(s, b)
        ans = []
        for i in lst1:
            j = bisect.bisect_left(lst2, i)
            for x in [j - 1, j, j + 1]:
                if 0 <= x < len(lst2) and abs(lst2[x] - i) <= k:
                    ans.append(i)
                    break
        return ans


def prep(p):
    pi = [0] * len(p)
    j = 0
    for i in range(1, len(p)):
        while j != 0 and p[j] != p[i]:
            j = pi[j - 1]
        if p[j] == p[i]:
            j += 1
        pi[i] = j
    return pi


class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        pa = prep(a + "#" + s)
        pb = prep(b + "#" + s)
        ia = [i - len(a) * 2 for i in range(len(pa)) if pa[i] == len(a)]
        ib = [i - len(b) * 2 for i in range(len(pb)) if pb[i] == len(b)]
        ans = []
        for i in ia:
            p = bisect.bisect(ib, i)
            for p1 in range(p - 1, p + 2):
                if 0 <= p1 < len(ib) and abs(ib[p1] - i) <= k:
                    ans.append(i)
                    break
        return ans


class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # KMP
        def partial(s):
            g, pi = 0, [0] * len(s)
            for i in range(1, len(s)):
                while g and (s[g] != s[i]):
                    g = pi[g - 1]
                pi[i] = g = g + (s[g] == s[i])

            return pi

        def match(s, pat):
            pi = partial(pat)

            g, idx = 0, []
            for i in range(len(s)):
                while g and pat[g] != s[i]:
                    g = pi[g - 1]
                g += pat[g] == s[i]
                if g == len(pi):
                    idx.append(i + 1 - g)
                    g = pi[g - 1]

            return idx

        i1 = match(s, a)
        i2 = match(s, b)
        # i1.sort()
        # i2.sort()

        ans = []
        for i in i1:
            l = bisect.bisect_left(i2, i)
            r = bisect.bisect_right(i2, i) - 1
            if (
                0 <= l < len(i2)
                and abs(i - i2[l]) <= k
                or 0 <= r < len(i2)
                and abs(i - i2[r]) <= k
            ):
                ans.append(i)
        return ans


class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        pos_a = self.kmp(s, a)
        pos_b = self.kmp(s, b)

        ans = []
        for i in pos_a:
            bi = bisect.bisect_left(pos_b, i)
            if (
                bi < len(pos_b)
                and pos_b[bi] - i <= k
                or bi > 0
                and i - pos_b[bi - 1] <= k
            ):
                ans.append(i)
        return ans

        ans = []
        j, m = 0, len(pos_b)
        for i in pos_a:
            while j < m and pos_b[j] < i - k:
                j += 1
            if j < m and pos_b[j] <= i + k:
                ans.append(i)
        return ans

    def kmp(self, text: str, pattern: str) -> List[int]:
        m = len(pattern)
        pi = [0] * m
        c = 0
        for i in range(1, m):
            v = pattern[i]
            while c and pattern[c] != v:
                c = pi[c - 1]
            if pattern[c] == v:
                c += 1
            pi[i] = c

        res = []
        c = 0
        for i, v in enumerate(text):
            v = text[i]
            while c and pattern[c] != v:
                c = pi[c - 1]
            if pattern[c] == v:
                c += 1
            if c == len(pattern):
                res.append(i - m + 1)
                c = pi[c - 1]
        return res


# 3010 - Divide an Array Into Subarrays With Minimum Cost I - EASY
class Solution:
    # O(n) / O(1)
    def minimumCost(self, nums: List[int]) -> int:
        return nums[0] + sum(heapq.nsmallest(2, nums[1:]))


# 3011 - Find if Array Can Be Sorted - MEDIUM
class Solution:
    # O(nlogn) / O(n)
    def canSortArray(self, nums: List[int]) -> bool:
        arr = nums[::]
        i = 0
        while i < len(nums):
            j = i
            b = nums[i].bit_count()
            while j < len(nums) and b == nums[j].bit_count():
                j += 1
            arr[i:j] = sorted(nums[i:j])
            i = j
        return arr == sorted(nums)

    # O(nlogn) / O(1)
    def canSortArray(self, nums: List[int]) -> bool:
        n = len(nums)
        i = 0
        while i < n:
            j = i
            ones = nums[i].bit_count()
            i += 1
            while i < n and ones == nums[i].bit_count():
                i += 1
            nums[j:i] = sorted(nums[j:i])
        return all(x <= y for x, y in itertools.pairwise(nums))


# 3012 - Minimize Length of Array Using Operations - MEDIUM
class Solution:
    # 1. 小数 % 大数 = 小数, 相当于直接删除大的数, 所以要找数组内最小值, 将其他的数都消去
    # 2. 重点: 如果 x % 最小值 m = y != 0, 则此时 y 一定小于 当前最小值 m, 并且 y 会成为数组内新的最小值, 且唯一
    # O(n) / O(1)
    def minimumArrayLength(self, nums: List[int]) -> int:
        m = min(nums)
        return 1 if any(x % m for x in nums) else (sum(x == m for x in nums) + 1) // 2
        return 1 if any(x % m for x in nums) else nums.count(m) - nums.count(m) // 2
        return (
            sum(x == m for x in nums) + 1 >> 2 if all(x % m == 0 for x in nums) else 1
        )


# 3013 - Divide an Array Into Subarrays With Minimum Cost II - HARD
class Solution:
    # 由于 nums[0] 必然是第一个子数组的代价, 只需考虑剩下的 k - 1 个数.
    # 题意可以转化为: 在长度为dist + 1的滑动窗口中, 维护最小的 k - 1 个元素的累加和.
    # 所有滑动窗口中累加和的最小值 + nums[0] 即为答案
    # 维护两个个有序结构, 一个维护 k − 1 个最小的, 一个维护剩余子数组
    # O(nlogn) / O(n)
    def minimumCost(self, nums: List[int], k: int, dist: int) -> int:
        init = sorted(nums[1 : dist + 2])
        s1 = sortedcontainers.SortedList(init[: k - 1])
        s2 = sortedcontainers.SortedList(init[k - 1 :])
        ans = cur = sum(init[: k - 1])
        for i in range(1, len(nums) - dist - 1):
            if nums[i] in s1:
                cur -= nums[i]
                s1.remove(nums[i])
            else:
                s2.remove(nums[i])

            if len(s1) == 0 or s1[-1] >= nums[i + dist + 1]:
                s1.add(nums[i + dist + 1])
                cur += nums[i + dist + 1]
            else:
                s2.add(nums[i + dist + 1])

            if len(s1) > k - 1:
                cur -= s1[-1]
                s2.add(s1.pop())
            elif len(s1) < k - 1:
                cur += s2[0]
                s1.add(s2.pop(0))

            ans = min(ans, cur)
        return ans + nums[0]

    # 由于第一个子数组的开头一定是 nums[0], 因此把数组分成 k 个子数组
    # 等价于 从 nums[1] 到 nums[n - 1] 里选出 (k - 1) 个数作为子数组的开头
    # 枚举最后一个子数组的开头 nums[i], 则需要从 nums[i - delta] 到 nums[i - 1] 中再选出 (k - 2) 个数作为其它子数组的开头
    # 这显然是一个长度为 delta 的滑动窗口. 为了最小化答案, 应该选择最小的 (k - 2) 个数
    def minimumCost(self, nums: List[int], k: int, dist: int) -> int:
        self.cur = 0  # s1 中所有数的和
        s1 = sortedcontainers.SortedList()  # s1 保存前 k - 2 小值
        s2 = sortedcontainers.SortedList()  # s2 保存其它值

        def adjust() -> None:
            while len(s1) < k - 2 and len(s2) > 0:
                self.cur += s2[0]
                s1.add(s2.pop(0))
            while len(s1) > k - 2:
                self.cur -= s1[-1]
                s2.add(s1.pop())

        def add(x: int) -> None:
            if len(s2) > 0 and x >= s2[0]:
                s2.add(x)
            else:
                self.cur += x
                s1.add(x)
            adjust()

        def delete(x: int) -> None:
            if x in s1:
                self.cur -= x
                s1.remove(x)
            else:
                s2.remove(x)
            adjust()

        for i in range(1, k - 1):
            add(nums[i])

        ans = self.cur + nums[k - 1]
        for i in range(k, len(nums)):
            t = i - dist - 1
            if t > 0:
                delete(nums[t])
            add(nums[i - 1])
            ans = min(ans, self.cur + nums[i])
        return ans + nums[0]

    def minimumCost(self, nums: List[int], k: int, dist: int) -> int:
        k -= 1
        self.sum = sum(nums[: dist + 2])
        L = sortedcontainers.SortedList(nums[1 : dist + 2])
        R = sortedcontainers.SortedList()

        def L2R() -> None:
            x = L.pop()
            self.sum -= x
            R.add(x)

        def R2L() -> None:
            x = R.pop(0)
            self.sum += x
            L.add(x)

        while len(L) > k:
            L2R()

        ans = self.sum
        for i in range(dist + 2, len(nums)):
            # 移除 out
            out = nums[i - dist - 1]
            if out in L:
                self.sum -= out
                L.remove(out)
            else:
                R.remove(out)
            # 添加 in
            in_val = nums[i]
            if in_val < L[-1]:
                self.sum += in_val
                L.add(in_val)
            else:
                R.add(in_val)
            # 维护大小
            if len(L) == k - 1:
                R2L()
            elif len(L) == k + 1:
                L2R()
            ans = min(ans, self.sum)
        return ans


# 3014 - Minimum Number of Pushes to Type Word I - EASY
class Solution:
    # 题意同 lc 3016, 但限制和数据范围不同
    def minimumPushes(self, word: str) -> int:
        x, m = divmod(len(word), 8)
        return (x * 4 + m) * (x + 1)

    def minimumPushes(self, word: str) -> int:
        l = len(word)
        ans = 0
        t = 1
        while l > 0:
            if l >= 8:
                ans += 8 * t
            else:
                ans += l * t
            l -= 8
            t += 1
        return ans
