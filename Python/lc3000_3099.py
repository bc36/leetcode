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


# 3015 - Count the Number of Houses at a Certain Distance I - MEDIUM
# 题意同 lc 3017, 但数据范围不同
# 2 <= n <= 100
# 1 <= x, y <= n


# lc 3017 数据范围不
# 2 <= n <= 10**5
# 1 <= x, y <= n
class Solution:
    # O(n ^ 2) / O(n)
    def countOfPairs(self, n: int, x: int, y: int) -> List[int]:
        x -= 1
        y -= 1
        ans = [0] * n
        for i in range(n):
            for j in range(i):
                ans[
                    min(
                        abs(i - j),
                        abs(i - x) + 1 + abs(j - y),
                        abs(i - y) + 1 + abs(j - x),
                    )
                    - 1
                ] += 1
        return [x * 2 for x in ans]


# 3016 - Minimum Number of Pushes to Type Word II - MEDIUM
class Solution:
    def minimumPushes(self, word: str) -> int:
        cnt = collections.Counter(word)
        arr = sorted((v for v in cnt.values()), reverse=True)
        return sum(v * (i // 8 + 1) for i, v in enumerate(arr))

    def minimumPushes(self, word: str) -> int:
        cnt = collections.Counter(word)
        h = [1] * 8
        ans = 0
        for x in sorted(cnt.values(), reverse=True):
            v = heapq.heappop(h)
            ans += x * v
            v += 1
            heapq.heappush(h, v)
        return ans


# 3017 - Count the Number of Houses at a Certain Distance II - HARD
class Solution:
    # 寻找某一个 分界点, 在 分界点 右侧都需要经过 新加的边, 才会式路径更短
    def countOfPairs(self, n: int, x: int, y: int) -> List[int]:
        # 调整 x, y 的大小关系
        if x > y:
            x, y = y, x
        ans = [0] * (n + 1)
        x -= 1
        y -= 1
        for i in range(n):
            # 不需要中介的情况
            if abs(i - x) + 1 > abs(i - y):
                ans[1] += 1
                ans[n - i] -= 1
            # 需要中介的情况
            else:
                # 找到分界点
                d = abs(i - x) + 1
                sep = i + d + (y - i - d) // 2
                # 分界点左侧是直接走, 距离从 1 到 sep - i
                ans[1] += 1
                ans[sep - i + 1] -= 1
                # 分界点及其右侧与 y 左侧, 通过中介, 距离从 d + 1 到 d + y - (sep + 1)
                ans[d + 1] += 1
                ans[d + y - sep] -= 1
                # y 及其右侧, 距离从 d 到 d + (n - 1) - y
                ans[d] += 1
                ans[d + n - y] -= 1
        ans = list(itertools.accumulate(ans))
        return [x * 2 for x in ans[1:]]

    def countOfPairs(self, n: int, x: int, y: int) -> List[int]:
        if x > y:
            x, y = y, x
        x -= 1
        y -= 1
        ans = [0] * (n + 1)
        l = y - x + 1

        def add(frm: int, to: int) -> None:
            if frm > to:
                return
            ans[frm] += 1
            ans[to + 1] -= 1

        for i in range(n):
            if i < x:
                add(1, i)
                add(1, x - i)
                add(x - i + 1, x - i + (l // 2))
                add(x - i + 1, x - i + ((l - 1) // 2))
                toy = min(x - i + 1, y - i)
                add(toy + 1, toy + n - 1 - y)
            if i >= x and i <= y:
                tox = min(i - x, y - i + 1)
                toy = min(y - i, i - x + 1)
                add(tox + 1, tox + x)
                add(toy + 1, toy + n - 1 - y)
                add(1, (l // 2))
                add(1, ((l - 1) // 2))
            if i > y:
                tox = min(i - x, i - y + 1)
                add(tox + 1, tox + x)
                add(i - y + 1, i - y + (l // 2))
                add(i - y + 1, i - y + ((l - 1) // 2))
                add(1, i - y)
                add(1, n - 1 - i)
        return list(itertools.accumulate(ans))[1:]


# 3019 - Number of Changing Keys - EASY
class Solution:
    def countKeyChanges(self, s: str) -> int:
        return sum(x != y for x, y in itertools.pairwise(s.lower()))


# 3020 - Find the Maximum Number of Elements in Subset - MEDIUM
class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        ans = 1
        cnt = collections.Counter(nums)
        for x in set(nums):
            if x == 1:
                ans = max(ans, cnt[1] - (cnt[1] % 2 == 0))
                continue
            cur = 0
            while cnt[x] >= 2:
                x *= x
                cur += 2
            ans = max(ans, cur + (1 if cnt[x] >= 1 else -1))
        return ans

    def maximumLength(self, nums: List[int]) -> int:
        cnt = collections.Counter(nums)
        ans = cnt[1] - (cnt[1] % 2 ^ 1)
        del cnt[1]
        for x in cnt:
            res = 0
            while cnt[x] > 1:
                res += 2
                x *= x
            ans = max(ans, res + (1 if x in cnt else -1))  # 保证 res 是奇数
        return ans


# 3021 - Alice and Bob Playing Flower Game - MEDIUM
class Solution:
    # 两数和为奇数
    # n 中的奇数 * m 中的偶数 + n 中的偶数 * m 中的奇数
    def flowerGame(self, n: int, m: int) -> int:
        return n * m // 2


# 3022 - Minimize OR of Remaining Elements Using Operations - HARD
class Solution:
    # 思考如何在计算新的 位 的时候, 如何携带高位的信息
    # 注意这两种解法的区别

    # O(nlogU) / O(1), U = max(nums)
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        ans = 0  # 所有为 0 的位构成的数字为去掉的数字, 我们要最大化这个去掉的数字
        for i in range(29, -1, -1):
            cur = ans | (1 << i)
            val = -1  # -1 在 python 里满足任何数和它进行与运算都是它本身
            cnt = len(
                nums
            )  # 最终找到了 cnt 个 与和 为 0 的子数组. 需要进行的操作数为 n - cnt
            for x in nums:
                val &= x & cur
                if val == 0:
                    cnt -= 1
                    val = -1
            if cnt <= k:
                ans = cur
        return (1 << 30) - 1 - ans

    # O(nlogU) / O(1), U = max(nums)
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        ans = mask = 0
        for b in range(max(nums).bit_length() - 1, -1, -1):
            mask |= 1 << b
            cnt = 0  # 操作次数
            and_res = -1  # -1 的二进制全为 1
            for x in nums:
                and_res &= x & mask
                if and_res:
                    cnt += 1  # 合并 x
                else:
                    and_res = -1  # 准备合并下一段
            if cnt > k:
                ans |= 1 << b  # 答案的这个比特位必须是 1
                mask ^= 1 << b  # 后面不考虑这个比特位
        return ans

    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        mask = (1 << 30) - 1

        def check(m: int) -> bool:
            x = mask
            cnt = 0
            for v in nums:
                x &= v
                if x & m == x:  # 经过与操作, 这一段可以合并
                    cnt += 1
                    if cnt >= n - k:
                        return True
                    x = mask
            return False

        return bisect.bisect_left(range(mask), True, key=check)


# 3065. Minimum Operations to Exceed Threshold Value I - EASY
class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        return sum(v < k for v in nums)


# 3066. Minimum Operations to Exceed Threshold Value II - MEDIUM
class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        ans = 0
        while nums and nums[0] < k:
            x = heapq.heappop(nums)
            y = heapq.heappop(nums)
            heapq.heappush(nums, x * 2 + y)
            ans += 1
        return ans

    def minOperations(self, nums: List[int], k: int) -> int:
        ans = 0
        heapq.heapify(nums)
        while nums[0] < k:
            x = heapq.heappop(nums)
            heapq.heapreplace(nums, x * 2 + nums[0])
            ans += 1
        return ans


# 3067. Count Pairs of Connectable Servers in a Weighted Tree Network - MEDIUM
class Solution:
    def countPairsOfConnectableServers(
        self, edges: List[List[int]], signalSpeed: int
    ) -> List[int]:
        n = len(edges)
        g = [[] for _ in range(n + 1)]
        for x, y, w in edges:
            g[x].append((y, w))
            g[y].append((x, w))

        def dfs(x: int, fa: int, d: int):
            nonlocal cnt
            if d % signalSpeed == 0:
                cnt += 1
            for y, w in g[x]:
                if y != fa:
                    dfs(y, x, d + w)
            return

        ans = [0] * (n + 1)
        for x in range(n + 1):
            cur = 0
            for y, w in g[x]:
                cnt = 0
                dfs(y, x, w)
                ans[x] += cnt * cur
                cur += cnt
        return ans

    def countPairsOfConnectableServers(
        self, edges: List[List[int]], signalSpeed: int
    ) -> List[int]:
        n = len(edges)
        g = [[] for _ in range(n + 1)]
        for x, y, w in edges:
            g[x].append((y, w))
            g[y].append((x, w))

        def dfs(x: int, fa: int, s: int) -> int:
            cnt = 0 if s % signalSpeed else 1
            for y, w in g[x]:
                if y != fa:
                    cnt += dfs(y, x, s + w)
            return cnt

        ans = [0] * (n + 1)
        for i, gi in enumerate(g):
            s = 0
            for y, w in gi:
                cnt = dfs(y, i, w)
                ans[i] += cnt * s
                s += cnt
        return ans


# 3068. Find the Maximum Sum of Node Values - HARD
class Solution:
    # 1. 由于一个数异或两次(偶数次) k 后保持不变, 所以对于一条从 x 到 y 的简单路径, 我们把路径上的所有边操作后, 路径上除了 x 和 y 的其它节点都恰好操作两次,
    #    所以只有 nums[x] 和 nums[y] 都异或了 k, 其余元素不变, 所以题目中的操作可以作用在任意两个数上. 所以不需要建树, edges 是多余的
    # 2. 无论操作多少次, 总是有偶数个元素异或了 k, 其余元素不变

    # 状态机 DP
    # 定义 f[i][0] 表示选择 nums 的前 i 数中的偶数个元素异或 k 得到的最大元素和
    # 定义 f[i][1] 表示选择 nums 的前 i 数中的奇数个元素异或 k 得到的最大元素和
    def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
        f0, f1 = 0, -math.inf
        for x in nums:
            f0, f1 = max(f0 + x, f1 + (x ^ k)), max(f1 + x, f0 + (x ^ k))
        return f0

    def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
        return functools.reduce(
            lambda x, y: (max(x[0] + y, x[1] + (y ^ k)), max(x[1] + y, x[0] + (y ^ k))),
            nums,
            (0, -math.inf),
        )[0]

    def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
        ans = sum(nums)
        arr = sorted(((t ^ k) - t for t in nums), reverse=True)
        for i in range(0, len(arr) - 1, 2):
            t = arr[i] + arr[i + 1]
            if t > 0:
                ans += t
            else:
                break
        return ans

    def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
        ans = sum(nums)
        arr = sorted((i ^ k) - i for i in nums)
        while len(arr) >= 2 and arr[-1] + arr[-2] >= 0:
            ans += arr[-1] + arr[-2]
            arr.pop()
            arr.pop()
        return ans


# 3069 - Distribute Elements Into Two Arrays I - EASY
class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        a = [nums[0]]
        b = [nums[1]]
        for i in range(2, len(nums)):
            if a[-1] > b[-1]:
                a.append(nums[i])
            else:
                b.append(nums[i])
        return a + b


# 3070 - Count Submatrices with Top-Left Element and Sum Less Than k - MEDIUM
class Solution:
    def countSubmatrices(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        f = [[0] * (n + 1) for _ in range(m + 1)]
        ans = 0
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                f[i + 1][j + 1] = f[i + 1][j] + f[i][j + 1] - f[i][j] + v
                ans += f[i + 1][j + 1] <= k
        return ans
        return sum(f[i + 1][j + 1] <= k for i in range(m) for j in range(n))


# 3071 - Minimum Operations to Write the Letter Y on a Grid - MEDIUM
class Solution:
    def minimumOperationsToWriteY(self, grid: List[List[int]]) -> int:
        n = len(grid)
        y = [0] * 3
        o = [0] * 3  # other
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if (i < n // 2 and (i == j or i + j == n - 1)) or (
                    i >= n // 2 and j == n // 2
                ):
                    y[v] += 1
                else:
                    o[v] += 1
        return min(n * n - y[i] - o[j] for i in range(3) for j in range(3) if i != j)


# 3072 - Distribute Elements Into Two Arrays II - HARD
class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        l1 = [nums[0]]
        s1 = sortedcontainers.SortedList([nums[0]])
        l2 = [nums[1]]
        s2 = sortedcontainers.SortedList([nums[1]])
        for x in nums[2:]:
            p = s1.bisect_right(x)
            q = s2.bisect_right(x)
            if len(l1) - p > len(l2) - q:
                l1.append(x)
                s1.add(x)
            elif len(l1) - p < len(l2) - q:
                l2.append(x)
                s2.add(x)
            elif len(l1) <= len(l2):
                l1.append(x)
                s1.add(x)
            else:
                l2.append(x)
                s2.add(x)
        return l1 + l2


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
            i &= i - 1
        return res


class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        sorted_nums = sorted(set(nums))  # 将元素值用下表映射, 否则值域太大了
        m = len(sorted_nums)
        a = [nums[0]]
        b = [nums[1]]
        t1 = Fenwick(m + 1)
        t2 = Fenwick(m + 1)
        t1.add(bisect.bisect_left(sorted_nums, nums[0]) + 1)
        t2.add(bisect.bisect_left(sorted_nums, nums[1]) + 1)
        for x in nums[2:]:
            v = bisect.bisect_left(sorted_nums, x) + 1
            gc1 = len(a) - t1.pre(v)  # greaterCount(a, v)
            gc2 = len(b) - t2.pre(v)  # greaterCount(b, v)
            if gc1 > gc2 or gc1 == gc2 and len(a) <= len(b):
                a.append(x)
                t1.add(v)
            else:
                b.append(x)
                t2.add(v)
        return a + b


class Fenwick:
    __slots__ = "tree"

    def __init__(self, n: int):
        self.tree = [0] * n

    # 把下标为 i 的元素增加 v
    def add(self, i: int, v: int) -> None:
        while i < len(self.tree):
            self.tree[i] += v
            i += i & -i

    # 返回下标在 [1, i] 的元素之和
    def pre(self, i: int) -> int:
        res = 0
        while i > 0:
            res += self.tree[i]
            i &= i - 1
        return res


class Solution:
    # 一棵树状数组, 把元素 v 添加到 t2 的操作, 可以改成把元素 v 在 t1 中的出现次数减一
    # 用一棵树状数组维护 a 和 b 元素出现次数的差值
    def resultArray(self, nums: List[int]) -> List[int]:
        sorted_nums = sorted(set(nums))
        m = len(sorted_nums)
        a = [nums[0]]
        b = [nums[1]]
        t = Fenwick(m + 1)
        t.add(m - bisect.bisect_left(sorted_nums, nums[0]), 1)
        t.add(m - bisect.bisect_left(sorted_nums, nums[1]), -1)
        for x in nums[2:]:
            v = m - bisect.bisect_left(sorted_nums, x)
            d = t.pre(v - 1)  # 转换成 < v 的元素个数之差
            if d > 0 or d == 0 and len(a) <= len(b):
                a.append(x)
                t.add(v, 1)
            else:
                b.append(x)
                t.add(v, -1)
        return a + b


# 3074 - Apple Redistribution into Boxes - MEDIUM
class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        s = sum(apple)
        for i, c in enumerate(sorted(capacity, reverse=True)):
            s -= c
            if s <= 0:
                return i + 1


# 3075 - Maximize Happiness of Selected Children - MEDIUM
class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        return sum(
            h - i for i, h in enumerate(sorted(happiness, reverse=True)[:k]) if i < h
        )


# 3076 - Shortest Uncommon Substring in an Array - MEDIUM
class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        cnt = collections.Counter()
        for w in arr:
            for i in range(len(w)):
                for j in range(i, len(w)):
                    cnt[w[i : j + 1]] += 1
        ans = [""] * len(arr)
        for k, w in enumerate(arr):
            for i in range(len(w)):
                for j in range(i, len(w)):
                    cnt[w[i : j + 1]] -= 1
            z = "z" * 20
            for l in range(1, len(w) + 1):
                for i in range(0, len(w) - l + 1):
                    u = w[i : i + l]
                    if (len(u) < len(z) or len(u) == len(z) and u < z) and cnt[u] == 0:
                        z = u
            for i in range(len(w)):
                for j in range(i, len(w)):
                    cnt[w[i : j + 1]] += 1
            if z != "z" * 20:
                ans[k] = z
        return ans


# 3077 - Maximum Strength of K Disjoint Subarrays - HARD
class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        dp0 = [-math.inf] * (k + 1)
        dp1 = [-math.inf] * (k + 1)
        dp0[0] = dp1[0] = 0
        for v in nums:
            # dp0 表示最后一个元素在第 i 个数组中的状态
            # 两种转移过来: 前一个位置是否截断
            # dp1 表示该位置第 i 个数组已经截断的情况下的最大价值
            for i in range(k, 0, -1):
                dp0[i] = max(dp0[i], dp1[i - 1])
                dp0[i] += (1 if i % 2 else -1) * v * (k - i + 1)
                dp1[i] = max(dp1[i], dp0[i])
        return dp1[-1]

    # 定义 f[i][j] 表示从 nums[0] 到 nums[j-1]] 中选出 i 个不相交非空连续子数组的最大能量值
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        s = list(itertools.accumulate(nums, initial=0))
        f = [[0] * (n + 1) for _ in range(k + 1)]
        for i in range(1, k + 1):
            f[i][i - 1] = mx = -math.inf
            w = (k - i + 1) * (1 if i % 2 else -1)
            # j 不能太小也不能太大, 要给前面留 i-1 个数, 后面留 k-i 个数
            for j in range(i, n - k + i + 1):
                mx = max(mx, f[i - 1][j - 1] - s[j - 1] * w)
                f[i][j] = max(f[i][j - 1], s[j] * w + mx)
        return f[k][n]


# 3079 - Find the Sum of Encrypted Integers - EASY
class Solution:
    def sumOfEncryptedInt(self, nums: List[int]) -> int:
        return sum(map(lambda v: int(max(str(v)) * len(str(v))), nums))


# 3080 - Mark Elements on Array by Performing Queries - MEDIUM
class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        s = sum(nums)
        marked = [False] * len(nums)
        arr = sorted((v, i) for i, v in enumerate(nums))
        ans = []
        p = 0
        for idx, k in queries:
            if not marked[idx]:
                marked[idx] = True
                s -= nums[idx]
            while p < len(nums) and k:
                if not marked[arr[p][1]]:
                    marked[arr[p][1]] = True
                    s -= arr[p][0]
                    k -= 1
                p += 1
            ans.append(s)
        return ans


# 3081 - Replace Question Marks in String to Minimize Its Value - MEDIUM
class Solution:
    def minimizeStringValue(self, s: str) -> str:
        cnt = collections.Counter(s)
        h = sorted((cnt[c], c) for c in string.ascii_lowercase)
        s = list(s)
        rev = []
        for c in s:
            if c == "?":
                rev.append(h[0][1])
                heapq.heapreplace(h, (h[0][0] + 1, h[0][1]))
        rev.sort(reverse=True)  # to ensure get the lexicographically smallest one
        return "".join(c if c != "?" else rev.pop() for c in s)


# 3082 - Find the Sum of the Power of All Subsequences - HARD
class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        mod = 1000000007
        n = len(nums)
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for v in nums:
            for l in range(n - 1, -1, -1):
                for val in range(v, k + 1):
                    dp[l + 1][val] = (dp[l + 1][val] + dp[l][val - v]) % mod
        ans = 0
        for l in range(1, n + 1):
            # ans += dp[l][k] * 2 ** (n - l)
            ans += dp[l][k] * pow(2, n - l, mod)
            ans %= mod
        return ans

    def sumOfPower(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        dp = collections.Counter()
        dp[(0, 0)] = 1  # 第一个状态是当前和, 第二个状态是当前长度
        for v in nums:
            ndp = collections.Counter()
            for x, length in dp:
                ndp[(x, length)] += dp[(x, length)]
                ndp[(x, length)] %= mod
                if x + v <= k:
                    ndp[(x + v, length + 1)] += dp[(x, length)]
                    ndp[(x + v, length + 1)] %= mod
            dp = ndp
        n = len(nums)
        ans = 0
        # 最后枚举当前长度
        for l in range(1, n + 1):
            ans += dp[(k, l)] * pow(2, n - l, mod) % mod
        return ans % mod

    def sumOfPower(self, nums: List[int], k: int) -> int:
        @functools.cache
        def dfs(i: int, cur: int, length: int) -> int:
            if cur == 0:
                return 2 ** (len(nums) - length)
            if i < 0:
                return 0
            if nums[i] > cur:
                return dfs(i - 1, cur, length)
            return dfs(i - 1, cur, length) + dfs(i - 1, cur - nums[i], length + 1)

        return dfs(len(nums) - 1, k, 0) % (10**9 + 7)


# 3083 - Existence of a Substring in a String and Its Reverse - EASY
class Solution:
    def isSubstringPresent(self, s: str) -> bool:
        rev = s[::-1]
        for i in range(len(s) - 1):
            if s[i : i + 2] in rev:
                return True
        return False

    def isSubstringPresent(self, s: str) -> bool:
        return any(y + x in s for x, y in itertools.pairwise(s))

    def isSubstringPresent(self, s: str) -> bool:
        vis = set()
        for x, y in itertools.pairwise(s):
            vis.add((x, y))
            if (y, x) in vis:
                return True
        return False


# 3084 - Count Substrings Starting and Ending with Given Character - MEDIUM
class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        cnt = s.count(c)
        return cnt * (cnt + 1) // 2

    def countSubstrings(self, s: str, c: str) -> int:
        return math.comb(s.count(c) + 1, 2)


# 3085 - Minimum Deletions to Make String K-Special - MEDIUM
class Solution:
    # O(n + 26^2) / O(26)
    def minimumDeletions(self, word: str, k: int) -> int:
        vals = collections.Counter(word).values()
        ans = math.inf
        for x in vals:
            cur = 0
            for y in vals:
                if y < x:
                    cur += y
                elif y > x + k:
                    cur += y - x - k
            ans = min(ans, cur)
        return ans

    def minimumDeletions(self, word: str, k: int) -> int:
        vals = sorted(collections.Counter(word).values())
        save = max(sum(min(x, v + k) for x in vals[i:]) for i, v in enumerate(vals))
        return len(word) - save


# 3086 - Minimum Moves to Pick K Ones - HARD
class Solution:
    def minimumMoves(self, nums: List[int], k: int, max_changes: int) -> int:
        pos = []
        c = 0  # nums 中连续的 1 长度
        for i, x in enumerate(nums):
            if x == 1:
                pos.append(i)
                c = max(c, 1)
                if i > 0 and nums[i - 1] == 1:
                    if i > 1 and nums[i - 2] == 1:
                        c = 3  # 有 3 个连续的 1
                    else:
                        c = max(c, 2)  # 有 2 个连续的 1

        c = min(c, k)
        if max_changes >= k - c:
            # 其余 k - c 个 1 可以全部用两次操作得到
            return max(c - 1, 0) + (k - c) * 2

        pre_sum = list(itertools.accumulate(pos, initial=0))
        ans = math.inf
        # 除了 max_changes 个数可以用两次操作得到, 其余的 1 只能一步步移动到 pos[i]
        size = k - max_changes
        for right in range(size, len(pos) + 1):
            # s1 + s2 是 j 在 [left, right) 中的所有 pos[j] 到 pos[(left + right) / 2] 的距离之和
            left = right - size
            mid = left + right >> 1
            s1 = pos[mid] * (mid - left) - (pre_sum[mid] - pre_sum[left])
            s2 = pre_sum[right] - pre_sum[mid] - pos[mid] * (right - mid)
            ans = min(ans, s1 + s2)
        return ans + max_changes * 2


# 3090 - Maximum Length Substring With Two Occurrences - EASY
class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        ans = j = 0
        cnt = collections.Counter()
        for i, c in enumerate(s):
            cnt[c] += 1
            while cnt[c] > 2:
                cnt[s[j]] -= 1
                j += 1
            ans = max(ans, i - j + 1)
        return ans


# 3091 - Apply Operations to Make Sum of Array Greater Than or Equal to k - MEDIUM
class Solution:
    def minOperations(self, k: int) -> int:
        return min(i - 1 + (k - 1) // i for i in range(1, k + 1))
        return min(i + (k - 1) // (i + 1) for i in range(k + 1))

    def minOperations(self, k: int) -> int:
        v = math.isqrt(k)
        return v + (k - 1) // v - 1  # 对勾函数

    def minOperations(self, k: int) -> int:
        return math.ceil(2 * math.sqrt(k) - 2)  # 求导


# 3092 - Most Frequent IDs - MEDIUM
class Solution:
    # O(nlogn) / O(n)
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        sl = sortedcontainers.SortedList()
        cnt = collections.defaultdict(int)
        ans = []
        for x, f in zip(nums, freq):
            if cnt[x] in sl:
                sl.discard(cnt[x])  # remove 也可, 多个 cnt[x] 只会移除一个
            cnt[x] += f
            # if cnt[x]:
            #     sl.add(cnt[x])
            # ans.append(sl[-1] if sl else 0)
            sl.add(cnt[x])
            ans.append(sl[-1])
        return ans

    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        cnt = collections.Counter()
        ans = []
        h = []
        for x, f in zip(nums, freq):
            cnt[x] += f
            heapq.heappush(h, (-cnt[x], x))
            while -h[0][0] != cnt[h[0][1]]:  # 堆顶保存的数据已经发生变化
                heapq.heappop(h)
            ans.append(-h[0][0])
        return ans


# 3093 - Longest Common Suffix Queries - HARD
class Node:
    __slots__ = "son", "mi", "i"

    def __init__(self):
        self.son = [None] * 26
        self.mi = math.inf  # minimum length


class Solution:
    def stringIndices(
        self, wordsContainer: List[str], wordsQuery: List[str]
    ) -> List[int]:
        oa = ord("a")
        root = Node()
        for i, s in enumerate(wordsContainer):
            l = len(s)
            cur = root
            if l < cur.mi:
                cur.mi, cur.i = l, i
            for c in map(ord, reversed(s)):
                c -= oa
                if cur.son[c] is None:
                    cur.son[c] = Node()
                cur = cur.son[c]
                if l < cur.mi:
                    cur.mi, cur.i = l, i
        ans = []
        for s in wordsQuery:
            cur = root
            for c in map(ord, reversed(s)):
                c -= oa
                if cur.son[c] is None:
                    break
                cur = cur.son[c]
            ans.append(cur.i)
        return ans

    def stringIndices(
        self, wordsContainer: List[str], wordsQuery: List[str]
    ) -> List[int]:
        trie = {}
        mi = math.inf
        tmp = -1
        for i, w in enumerate(wordsContainer):
            l = len(w)
            if l < mi:
                mi, tmp = l, i
            root = trie
            for c in w[::-1]:
                if c not in root:
                    root[c] = {}
                root = root[c]
                if "#" not in root:
                    root["#"] = (i, l)
                elif l < root["#"][1]:
                    root["#"] = (i, l)
        ans = []
        for w in wordsQuery:
            root = trie
            x = tmp
            for c in w[::-1]:
                if c not in root:
                    # ans.append(x)
                    break
                else:
                    root = root[c]
                    x = root["#"][0]
            # else:
            ans.append(x)
        return ans

    def stringIndices(
        self, wordsContainer: List[str], wordsQuery: List[str]
    ) -> List[int]:
        idx = sorted(range(len(wordsContainer)), key=lambda i: len(wordsContainer[i]))
        nex = [{}]
        f = [idx[0]]
        for i in idx:
            cur = 0
            for c in wordsContainer[i][::-1]:
                if c in nex[cur]:
                    cur = nex[cur][c]
                else:
                    l = len(nex)
                    nex[cur][c] = l
                    nex.append({})
                    f.append(i)
                    cur = l
        ans = []
        for s in wordsQuery:
            cur = 0
            for c in s[::-1]:
                if c in nex[cur]:
                    cur = nex[cur][c]
                else:
                    break
            ans.append(f[cur])
        return ans

    def stringIndices(
        self, wordsContainer: List[str], wordsQuery: List[str]
    ) -> List[int]:
        trie = [None] * 27
        ls = [len(w) for w in wordsContainer]
        for i, w in enumerate(wordsContainer):
            r = trie
            if r[26] is None or ls[r[26]] > ls[i]:
                r[26] = i
            for c in w[::-1]:
                c = ord(c) - ord("a")
                if r[c] is None:
                    r[c] = [None] * 27
                r = r[c]
                if r[26] is None or ls[r[26]] > ls[i]:
                    r[26] = i
        ans = []
        for w in wordsQuery:
            r = trie
            for c in w[::-1]:
                c = ord(c) - ord("a")
                if r[c] is None:
                    break
                r = r[c]
            ans.append(r[26])
        return ans


class Trie:
    def __init__(self):
        self.sons = {}
        self.val = 10**9

    def insert(self, word: str, value: int) -> None:
        tmp = self
        for char in word:
            if value < tmp.val:
                tmp.val = value
            if char not in tmp.sons:
                tmp.sons[char] = Trie()
            tmp = tmp.sons[char]
        if value < tmp.val:
            tmp.val = value

    def search(self, word: str) -> bool:
        tmp = self
        for char in word:
            if char in tmp.sons:
                tmp = tmp.sons[char]
            else:
                break
        return tmp.val


class Solution:
    # 这里将 (长度, 下标) 对用一个整数表示, 由于长度不超过 5000, 索引不超过 10^4, 因此不会发生碰撞, 也不会超过 10^9
    def stringIndices(
        self, wordsContainer: List[str], wordsQuery: List[str]
    ) -> List[int]:
        t = Trie()
        for i, w in enumerate(wordsContainer):
            t.insert(w[::-1], len(w) * 100000 + i)
        return [t.search(w[::-1]) % 100000 for w in wordsQuery]
