import bisect, collections, functools, heapq, itertools, math, operator, string, sys
from typing import List, Optional, Tuple
import sortedcontainers


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


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
                # 枚举: s 的后 i 个字母和 t 的前 i 个字母是一样的
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
            def dfs(i: int, pre: int, isLimit: bool, isNum: bool) -> int:
                """表示构造第 i 位及其之后数位的合法方案数"""
                if i == len(s):
                    return int(isNum)  # isNum 为 True 表示得到了一个合法数字
                res = 0
                if not isNum:  # 可以跳过当前数位
                    res = dfs(i + 1, pre, False, False)
                down = 0 if isNum else 1  # 如果前面没有填数字, 必须从 1 开始, 因为不能有前导零
                # 如果前面填的数字都和 s 的一样, 那么这一位至多填 s[i], 否则就超过 s 了
                up = int(s[i]) if isLimit else 9
                for d in range(down, up + 1):  # 枚举要填入的数字 d
                    if not isNum or abs(d - pre) == 1:  # 第一位数字随便填, 其余必须相差 1
                        res += dfs(i + 1, d, isLimit and d == up, True)
                return res % mod

            return dfs(0, 0, True, False)

        return (calc(high) - calc(str(int(low) - 1))) % mod

    def countSteppingNumbers(self, low: str, high: str) -> int:
        mod = 10**9 + 7

        def calc(s: str) -> int:
            @functools.cache
            def dfs(i: int, pre: int, isLimit: bool, isNum: bool) -> int:
                if i == len(s):
                    return int(isNum)
                res = 0
                if not isNum:
                    res = dfs(i + 1, -1, False, False)
                down = 0 if isNum else 1
                up = int(s[i]) if isLimit else 9
                for d in range(down, up + 1):
                    if pre != -1 and abs(d - pre) != 1:
                        continue
                    res += dfs(i + 1, d, isLimit and d == up, True)
                return res % mod

            return dfs(0, -1, True, False)

        return (calc(high) - calc(str(int(low) - 1))) % mod

    # 280ms
    def countSteppingNumbers(self, low: str, high: str) -> int:
        mod = 10**9 + 7
        # 把 high 和 low 通过补 0 补到一样长, 就可以一起计算了
        low = "0" * (len(high) - len(low)) + low

        @functools.cache
        def dfs(
            i: int, pre: int, isUpLimit: bool, isDownLimit: bool, isNum: bool
        ) -> int:
            if i == len(high):
                return isNum
            res = 0
            down = int(low[i]) if isDownLimit else 0
            up = int(high[i]) if isUpLimit else 9
            for d in range(down, up + 1):  # 枚举要填入的数字 d
                if not isNum or d - pre in (-1, 1):
                    res += dfs(
                        i + 1,
                        d,
                        isUpLimit and d == up,
                        isDownLimit and d == down,
                        isNum or d > 0,
                    )
            return res % mod

        return dfs(0, 0, True, True, False)


# 2815 - Max Pair Sum in an Array - EASY
class Solution:
    def maxSum(self, nums: List[int]) -> int:
        ans = -1
        d = dict()
        for v in nums:
            mx = max(str(v))
            if mx in d:
                ans = max(ans, d[mx] + v)
            d[mx] = max(d.get(mx, 0), v)
        return ans


# 2816 - Double a Number Represented as a Linked List - MEDIUM
class Solution:
    # ValueError: Exceeds the limit (4300) for integer string conversion; use sys.set_int_max_str_digits() to increase the limit
    sys.set_int_max_str_digits(0)

    def doubleIt(self, head: Optional[ListNode]) -> Optional[ListNode]:
        x = 0
        while head:
            x *= 10
            x += head.val
            head = head.next
        x *= 2
        dummy = tmp = ListNode(-1)
        for w in str(x):
            tmp.next = ListNode(int(w))
            tmp = tmp.next
        return dummy.next

    # 本题因为乘数为2, 故低一位的数字可以影响到该位的结果只有0, 1 不会再影响更高一位, 所以可以从前至后遍历
    def doubleIt(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = dummy = ListNode(val=0, next=head)
        while cur.next:
            cur.val = (cur.val * 2 + (cur.next.val >= 5)) % 10
            cur = cur.next
        cur.val = cur.val * 2 % 10
        return dummy if dummy.val else dummy.next

    def doubleIt(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head.val >= 5:
            head = ListNode(0, head)
        cur = head
        while cur:
            cur.val = cur.val * 2 % 10
            if cur.next and cur.next.val >= 5:
                cur.val += 1
            cur = cur.next
        return head


# 2817 - Minimum Absolute Difference Between Elements With Constraint - MEDIUM
class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        sl = sortedcontainers.SortedList()
        ans = math.inf
        for i in range(x, len(nums)):
            sl.add(nums[i - x])
            # p = sl.bisect_left(nums[i])
            p = sl.bisect_right(nums[i])
            if p == len(sl):
                ans = min(ans, nums[i] - sl[-1])
            else:
                ans = min(
                    ans,
                    sl[p] - nums[i],
                    math.inf if p == 0 else nums[i] - sl[p - 1],
                )
            if ans == 0:
                return 0
        return ans

    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        ans = math.inf
        sl = sortedcontainers.SortedList([-math.inf, math.inf])  # 哨兵
        for i in range(x, len(nums)):
            sl.add(nums[i - x])
            p = sl.bisect_left(nums[i])
            # p = sl.bisect_right(nums[i])
            ans = min(ans, sl[p] - nums[i], nums[i] - sl[p - 1])
        return ans


# 2818 - Apply Operations to Maximize Score - HARD
def eratosthenes(n: int) -> List[int]:
    diffPrime = [0] * (n + 1)
    for i in range(2, n + 1):
        if diffPrime[i] == 0:
            for j in range(i, n + 1, i):
                diffPrime[j] += 1
    return diffPrime


diffPrime = eratosthenes(10**5)


class Solution:
    # O(nlogn) / O(n), 单调栈, 计算贡献
    def maximumScore(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ps = [diffPrime[v] for v in nums]
        right = [n] * n
        left = [-1] * n
        # 如何确定栈内是递增还是递减? 思考什么时候栈会被清空.
        # 往左看, 如果遇到一个 ps 极大值, 则会停止, 所以此时栈内是(从头到尾/从底到顶)递减的
        st = []
        for i in range(n):
            while st and ps[st[-1]] < ps[i]:
                st.pop()
            if st:
                left[i] = st[-1]
            st.append(i)
        # 往右看, 如果遇到一个 ps 极大值, 则会停止x
        st = []
        for i in range(n)[::-1]:
            while st and ps[st[-1]] <= ps[i]:  # 根据题意, 注意 '=' 时的处理
                st.pop()
            if st:
                right[i] = st[-1]
            st.append(i)
        ans = 1
        for v, i in sorted(zip(nums, range(n)), reverse=True):
            t = (right[i] - i) * (i - left[i])
            if t >= k:
                ans = ans * pow(v, k, 1000000007) % 1000000007
                break
            ans = ans * pow(v, t, 1000000007) % 1000000007
            k -= t
        return ans

    def maximumScore(self, nums: List[int], k: int) -> int:
        n = len(nums)
        left = [-1] * n  # 质数分数 >= diffPrime[nums[i]] 的左侧最近元素下标
        right = [n] * n  # 质数分数 >  diffPrime[nums[i]] 的右侧最近元素下标
        st = []
        for i, v in enumerate(nums):
            while st and diffPrime[nums[st[-1]]] < diffPrime[v]:
                right[st.pop()] = i  # 此时该 i 也是栈顶元素的右边界
            if st:
                left[i] = st[-1]
            st.append(i)
        ans = 1
        for i, v, l, r in sorted(zip(range(n), nums, left, right), key=lambda z: -z[1]):
            t = (i - l) * (r - i)
            if t >= k:
                ans = ans * pow(v, k, 1000000007) % 1000000007
                break
            ans = ans * pow(v, t, 1000000007) % 1000000007
            k -= t
        return ans


def eratosthenes(n: int) -> List[int]:
    """[2, x] 内的质数"""
    primes = []
    isPrime = [True] * (n + 1)
    for i in range(2, n + 1):
        if isPrime[i]:
            primes.append(i)
            for j in range(i * i, n + 1, i):  # 注意是 *, 不是 +, 比 i 小的 i 的倍数已经被枚举过了
                isPrime[j] = False
    return primes


primes = eratosthenes(10**5)


# wrong
def primeScore(x: int) -> int:
    """这样计算会 TLE, 原因是如果遇到一个大质数, 会一直枚举到它为止, 10**5 内最多9592个质数"""
    if x == 1:
        return 0
    s = 0
    for p in primes:
        if x % p == 0:
            s += 1
            while x % p == 0:
                x //= p
    return s


# right
def primeScore(x: int) -> int:
    """此时则只会枚举到 sqrt(10**5) 内的最大质数, 最多 65 个, 最大为 313"""
    s = 0
    for p in primes:
        if p * p > x:
            break
        if x % p == 0:
            s += 1
            while x % p == 0:
                x //= p
    if x > 1:
        s += 1
    return s


# 2824 - Count Pairs Whose Sum is Less than Target - EASY
class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        ans = 0
        for j in range(1, len(nums)):
            for i in range(j):
                ans += nums[i] + nums[j] < target
        return ans

    def countPairs(self, nums: List[int], target: int) -> int:
        return sum(
            nums[i] + nums[j] < target
            for i, j in itertools.combinations(range(len(nums)), 2)
        )

    def countPairs(self, nums: List[int], target: int) -> int:
        nums.sort()
        ans = l = 0
        r = len(nums) - 1
        while l < r:
            if nums[l] + nums[r] < target:
                ans += r - l
                l += 1
            else:
                r -= 1
        return ans


# 2825 - Make String a Subsequence Using Cyclic Increments - MEDIUM
d = {string.ascii_lowercase[i]: string.ascii_lowercase[(i + 1) % 26] for i in range(26)}


class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        j = 0
        for c in str1:
            # if c == str2[j] or chr((ord(c) - 97 + 1) % 26 + 97) == str2[j]:
            # if c == str2[j] or d[c] == str2[j]:
            if c == str2[j] or (chr(ord(c) + 1) if c != "z" else "a") == str2[j]:
                j += 1
                if j == len(str2):
                    return True
        return False


# 2826 - Sorting Three Groups - MEDIUM
class Solution:
    # O(n^3) / O(1)
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        ans = math.inf
        for i in range(n + 1):
            one = nums[:i].count(1)
            for j in range(i, n + 1):
                two = nums[i:j].count(2)
                three = nums[j:].count(3)
                ans = min(i - one + j - i - two + n - j - three, ans)
        return ans

    # O(n^2) / O(1)
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        f = [1] * n
        for i in range(1, n):
            for j in range(i):
                if nums[i] >= nums[j]:
                    f[i] = max(f[i], f[j] + 1)
        return n - max(f)

    # O(n) / O(1)
    def minimumOperations(self, nums: List[int]) -> int:
        a = b = c = 0
        for x in nums:
            match x:
                case 1:
                    a, b, c = a, min(a, b) + 1, min(a, b, c) + 1
                case 2:
                    a, b, c = a + 1, min(a, b), min(a, b, c) + 1
                case 3:
                    a, b, c = a + 1, min(a, b) + 1, min(a, b, c)
        return min(a, b, c)

    # O(n) / O(1)
    def minimumOperations(self, nums: List[int]) -> int:
        a = b = c = 0
        for x in nums:
            if x == 1:
                a, b, c = a, b + 1, c + 1
            elif x == 2:
                a, b, c = a + 1, b, c + 1
            else:
                a, b, c = a + 1, b + 1, c
            b = min(a, b)
            c = min(b, c)
        return c

    # O(n) / O(1)
    def minimumOperations(self, nums: List[int]) -> int:
        f = [math.inf, 0, 0, 0]
        for x in nums:
            for i in (1, 2, 3):
                f[i] = min(f[i - 1], f[i] + (x != i))
        return f[3]

    # O(n) / O(1)
    def minimumOperations(self, nums: List[int]) -> int:
        f = [0] * 4
        for x in nums:
            for j in range(3, 0, -1):
                f[j] = min(f[k] for k in range(1, j + 1)) + (j != x)
        return min(f[1:])

    # O(nlogn) / O(n)
    def minimumOperations(self, nums: List[int]) -> int:
        from algo.lis import LongestIncreasingSubsequence

        n = len(nums)
        return n - LongestIncreasingSubsequence().definitely_not_reduce(nums)


# 2827 - Number of Beautiful Integers in the Range - HARD
class Solution:
    # 数位dp
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def calc(s: str) -> int:
            @functools.cache
            def dfs(
                i: int, pre: int, isLimit: bool, isNum: bool, e: int, o: int
            ) -> int:
                """表示构造第 i 位及其之后数位的合法方案数"""
                if i == len(s):
                    # isNum 为 True 表示得到了一个合法数字
                    return int(isNum and e == o and pre == 0)
                res = 0
                if not isNum:  # 可以跳过当前数位
                    res = dfs(i + 1, pre, False, False, e, o)
                down = 0 if isNum else 1  # 如果前面没有填数字, 必须从 1 开始, 因为不能有前导零
                # 如果前面填的数字都和 s 的一样, 那么这一位至多填 s[i], 否则就超过 s 了
                up = int(s[i]) if isLimit else 9
                for d in range(down, up + 1):  # 枚举要填入的数字 d
                    res += dfs(
                        i + 1,
                        (pre * 10 + d) % k,
                        isLimit and d == up,
                        True,
                        e + (d & 1),
                        o + (1 - d & 1),
                    )
                return res

            return dfs(0, 0, True, False, 0, 0)

        return calc(str(high)) - calc(str(low - 1))

    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def calc(high: int) -> int:
            s = str(high)

            @functools.cache
            def dfs(i: int, val: int, diff: int, isLimit: bool, isNum: bool) -> int:
                if i == len(s):
                    return int(isNum and val == 0 and diff == 0)
                res = 0
                if not isNum:
                    res = dfs(i + 1, val, diff, False, False)
                d0 = 0 if isNum else 1
                up = int(s[i]) if isLimit else 9
                for d in range(d0, up + 1):
                    res += dfs(
                        i + 1,
                        (val * 10 + d) % k,
                        diff + d % 2 * 2 - 1,
                        isLimit and d == up,
                        True,
                    )
                return res

            return dfs(0, 0, 0, True, False)

        return calc(high) - calc(low - 1)


# 2828 - Check if a String Is an Acronym of Words - EASY
class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # return "".join(w[0] for w in words) == s
        return len(s) == len(words) and all(w[0] == c for w, c in zip(words, s))


# 2829 - Determine the Minimum Sum of a k-avoiding Array - MEDIUM
class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        s = set()
        i = 1
        for _ in range(n):
            while k - i in s:
                i += 1
            s.add(i)
            i += 1
        return sum(s)

    def minimumSum(self, n: int, k: int) -> int:
        m = min(k // 2, n)
        return (m * (m + 1) + (k * 2 + n - m - 1) * (n - m)) // 2


# 2830 - Maximize the Profit as the Salesman - MEDIUM
class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        offers.sort(key=lambda x: x[1])
        f = [0] * (n + 1)
        j = 0
        for i in range(n):
            f[i + 1] = f[i]
            while j < len(offers) and offers[j][1] == i:
                f[i + 1] = max(f[i + 1], f[offers[j][0]] + offers[j][2])
                j += 1
        return f[n]

    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        d = collections.defaultdict(list)
        for s, e, g in offers:
            d[e].append((s, g))
        f = [0] * (n + 1)
        for i in range(n):
            f[i + 1] = f[i]
            for s, g in d[i]:
                f[i + 1] = max(f[i + 1], g + f[s])
        return f[n]


# 2831 - Find the Longest Equal Subarray - MEDIUM
class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        ans = 1
        d = collections.defaultdict(list)
        for i, v in enumerate(nums):
            d[v].append(i)
        for arr in d.values():
            l = cur = 0
            for r in range(1, len(arr)):
                cur += arr[r] - arr[r - 1] - 1
                while cur > k:  # 删太多了
                    cur -= arr[l + 1] - arr[l] - 1
                    l += 1
                ans = max(ans, r - l + 1)
        return ans

    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        pos = [[] for _ in range(len(nums) + 1)]
        for i, x in enumerate(nums):
            pos[x].append(i - len(pos[x]))
        ans = 0
        for arr in pos:
            if len(arr) <= ans:
                continue
            l = 0
            for r, p in enumerate(arr):
                while p - arr[l] > k:
                    l += 1
                ans = max(ans, r - l + 1)
        return ans

    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        pos = [[] for _ in range(len(nums) + 1)]
        for i in range(len(nums)):
            pos[nums[i]].append(i)
        ans = 0
        for pi in pos:
            if len(pi) > ans:
                j = 0
                for i in range(len(pi)):
                    while j < len(pi) and pi[j] - pi[i] - (j - i) <= k:
                        j += 1
                    ans = max(ans, j - i)
        return ans


# 2833 - Furthest Point From Origin - EASY
class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        n = len(moves)
        l = moves.count("L")
        r = moves.count("R")
        return n - l - r + abs(r - l)

    def furthestDistanceFromOrigin(self, moves: str) -> int:
        return abs(moves.count("R") - moves.count("L")) + moves.count("_")


# 2834 - Find the Minimum Possible Sum of a Beautiful Array - MEDIUM
class Solution:
    # 和 2839 题意一样, 数据范围不同
    def minimumPossibleSum(self, n: int, target: int) -> int:
        s = set()
        i = 1
        while len(s) < n:
            while i in s or target - i in s:
                i += 1
            s.add(i)
        return sum(s)

    def minimumPossibleSum(self, n: int, k: int) -> int:
        m = min(k // 2, n)
        return (m * (m + 1) + (k * 2 + n - m - 1) * (n - m)) // 2

    # 那么我们可以发现, 我们可以使用不超过 k / 2 的所有数, 再使用不小于 k 的所有数.
    # 这么做的原因是 i 和 k - i 不能同时出现在数组里, 因此我们都取小的那个, 对于这些数, 我们贪心地取即可
    def minimumPossibleSum(self, n: int, k: int) -> int:
        if n <= k // 2:
            return n * (n + 1) // 2
        return (k // 2) * (k // 2 + 1) // 2 + (2 * k + n - k // 2 - 1) * (
            n - k // 2
        ) // 2


# 2835 - Minimum Operations to Form Subsequence With Target Sum - HARD
class Solution:
    # O(nlogn + log(target)) / O(n)
    def minOperations(self, nums: List[int], target: int) -> int:
        nums.sort()
        total = sum(nums)
        if total < target:
            return -1
        ans = 0
        while target:
            p = nums.pop()
            # 去除之后总和依然大于等于 target, 直接去掉
            if total - p >= target:
                total -= p
            elif p > target:
                # 必须把 p 折半
                ans += 1
                nums.append(p // 2)
                nums.append(p // 2)
            else:
                # 不需要折半, 直接减去
                target -= p
                total -= p
        return ans

    # 注意: 如果一系列的幂次小于 i 的和 大于等于 一个 2 的幂次, 那么这一系列的部分(子序列)一定可以表示这个 2 的幂次
    # O(n + log(target)) / O(1)
    def minOperations(self, nums: List[int], target: int) -> int:
        if sum(nums) < target:
            return -1
        cnt = collections.Counter(nums)
        ans = 0
        # summ 记录所有更小的次幂的数的和
        summ = 0
        for i in range(31):
            # 先加上当前次幂的数值
            summ += cnt[1 << i] << i
            # 注意: target 分解后, 每个 2 的幂次系数只有 1 或 0! 所以在后续寻找中, 找到一个更大的幂次分解一次即可, 不需要多次!
            if target >> i & 1:
                # 如果更小的数之和是大于当前要表示的比特位, 那么不需要进行操作
                if summ >= 1 << i:
                    summ -= 1 << i
                else:
                    # 否则寻找最小的更大的 2 的次幂
                    for j in range(i + 1, 31):
                        if cnt[1 << j]:
                            cnt[1 << j] -= 1
                            summ += 1 << j
                            summ -= 1 << i
                            ans += j - i
                            break
        return ans

    # O(n + log(target)) / O(1)
    def minOperations(self, nums: List[int], target: int) -> int:
        if sum(nums) < target:
            return -1
        cnt = collections.Counter(nums)
        ans = summ = i = 0
        while 1 << i <= target:
            summ += cnt[1 << i] << i
            mask = (1 << (i + 1)) - 1
            i += 1
            if summ >= (target & mask):  # sum 是否大于等于 所有小于 i 幂次的和. target 和 sum 可以都不做减法
                continue
            ans += 1  # 一定要找更大的数操作
            while cnt[1 << i] == 0:
                ans += 1  # 还没找到, 继续找更大的数
                i += 1
        return ans


# 2836 - Maximize Value of Function in a Ball Passing Game - HARD
class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        # 赋值: 初始位置以及起点的得分
        ans = list(range(n))
        pos = list(range(n))
        # 赋值: 走 2 ^ 0 步时的结束位置和得分
        ans2power = receiver
        pos2power = receiver
        for i in range(34):
            if k >> i & 1:
                # 继续往前走 2 ^ i 步
                ans = [ans[i] + ans2power[pos[i]] for i in range(n)]
                pos = [pos[pos2power[i]] for i in range(n)]
            # 从 2 ^ i 步变成 2 ^ (i + 1) 步
            ans2power = [ans2power[i] + ans2power[pos2power[i]] for i in range(n)]
            pos2power = [pos2power[x] for x in pos2power]
        return max(ans)

    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        ans = list(range(n))
        cur = list(range(n))
        f = receiver.copy()
        g = receiver.copy()
        i = 0
        while k:
            if (k >> i) & 1:
                for j in range(n):
                    ans[j] += g[cur[j]]
                cur = [f[cur[j]] for j in range(n)]
                k -= 1 << i
            g = [g[j] + g[f[j]] for j in range(n)]
            f = [f[f[j]] for j in range(n)]
            i += 1
        return max(ans)

    # O(nlogk) / O(nlogk), 树上倍增
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        m = k.bit_length() - 1
        pa = [[(p, p)] + [None] * m for p in receiver]
        for i in range(m):
            for x in range(n):
                p, s = pa[x][i]
                pp, ss = pa[p][i]
                pa[x][i + 1] = (pp, s + ss)  # 合并节点值之和
        ans = 0
        for i in range(n):
            x = summ = i
            for j in range(m + 1):
                if (k >> j) & 1:  # k 的二进制从低到高第 j 位是 1
                    x, s = pa[x][j]
                    summ += s
            ans = max(ans, summ)
        return ans

    # 内向基环树
    # 如果将 receiver 看成每个点的出边, 那么就能得到基环树(森林), 即每个连通分量恰有一个环.
    # 从每个点沿着唯一出边出发, 在有限次后会到环上, 而一个点在环上移动 k 次经过的点标号和在预处理前缀和后 即可 O(1) 求出.
    # 剩下的问题就是怎么求出每个点需要多少步到环上, 以及 k 次后还没到环上的点的答案.
    # 只要从每个环上的点开始 dfs, 并且用栈维护环上的点到当前的点的路径的点标号前缀和就能线性求出所有需要信息
    # O(n) / O(n),
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        # 找环
        cycles: List[List[int]] = []
        vis = [False] * n
        in_cycles = [False] * n
        for i in range(n):
            path = []
            u = i
            while not vis[u]:
                vis[u] = True
                path.append(u)
                u = receiver[u]
            if u in path:
                cycles.append(path[path.index(u) :])
                for root in cycles[-1]:
                    in_cycles[root] = True
        # 反向图
        g = [[] for _ in range(n)]
        for u in range(n):
            g[receiver[u]].append(u)
        # 通过 dfs 确定每个点到环后还需走多少次, 以及特判 k 次后不能到环的点的答案
        tree: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
        ans = [0] * n
        for cycle in cycles:
            for root in cycle:
                stack = []

                # u: 结点, s: 到根点和
                def dfs(u: int, s: int) -> None:
                    s += u
                    if len(stack) < k:
                        tree[root].append((u, k - len(stack)))
                        ans[u] += s
                    else:
                        ans[u] = s - stack[-k]
                    stack.append(s - u)
                    for v in g[u]:
                        if not in_cycles[v]:
                            dfs(v, s)
                    stack.pop()
                    return

                dfs(root, 0)
        # 枚举每个环, 使用前缀和统计环上的贡献
        for cycle in cycles:
            prefix = cycle.copy()
            for i in range(1, len(prefix)):
                prefix[i] += prefix[i - 1]
            for i in range(len(cycle)):
                for u, d in tree[cycle[i]]:
                    ans[u] += d // len(cycle) * prefix[-1]
                    d %= len(cycle)
                    if i + d < len(cycle):
                        ans[u] += prefix[i + d] - prefix[i]
                    else:
                        ans[u] += prefix[-1] - prefix[i] + prefix[i + d - len(cycle)]
        return max(ans)


# 2839 - Check if Strings Can be Made Equal With Operations I - EASY
class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        return collections.Counter(s1[::2]) == collections.Counter(
            s2[::2]
        ) and collections.Counter(s1[1::2]) == collections.Counter(s2[1::2])

        return sorted(s1[::2]) == sorted(s2[::2]) and sorted(s1[1::2]) == sorted(
            s2[1::2]
        )


# 2840 - Check if Strings Can be Made Equal With Operations II - MEDIUM
class Solution:
    def checkStrings(self, s1: str, s2: str) -> bool:
        return collections.Counter(s1[::2]) == collections.Counter(
            s2[::2]
        ) and collections.Counter(s1[1::2]) == collections.Counter(s2[1::2])

        return sorted(s1[::2]) == sorted(s2[::2]) and sorted(s1[1::2]) == sorted(
            s2[1::2]
        )


# 2841 - Maximum Sum of Almost Unique Subarray - MEDIUM
class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        d = {}
        ans = summ = 0
        for i, v in enumerate(nums):
            d[v] = d.get(v, 0) + 1
            summ += v
            if i + 1 >= k and len(d) >= m:
                ans = max(ans, summ)
            if i + 1 >= k:
                x = nums[i + 1 - k]
                d[x] -= 1
                summ -= x
                if d[x] == 0:
                    del d[x]
        return ans

    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        ans = 0
        summ = sum(nums[: k - 1])
        cnt = collections.Counter(nums[: k - 1])
        for out, inn in zip(nums, nums[k - 1 :]):
            summ += inn
            cnt[inn] += 1
            if len(cnt) >= m:
                ans = max(ans, summ)
            summ -= out
            cnt[out] -= 1
            if cnt[out] == 0:
                del cnt[out]
        return ans


# 2842 - Count K-Subsequences of a String With Maximum Beauty - HARD
class Solution:
    def countKSubsequencesWithMaxBeauty(self, s: str, k: int) -> int:
        mod = 10**9 + 7
        ans = 1
        cur = 0
        d = collections.Counter(collections.Counter(s).values())
        for freq, val in sorted(d.items(), key=lambda x: -x[0]):
            if cur + val < k:
                ans = ans * pow(freq, val, mod) % mod
                cur += val
            else:
                ans = ans * pow(freq, k - cur, mod) * math.comb(val, k - cur) % mod
                return ans
        return 0

    def countKSubsequencesWithMaxBeauty(self, s: str, k: int) -> int:
        mod = 10**9 + 7
        ans = 1
        cnt = collections.Counter(collections.Counter(s).values())
        for freq, val in sorted(cnt.items(), reverse=True):
            if val >= k:
                return ans * pow(freq, k, mod) * math.comb(val, k) % mod
            ans *= pow(freq, val, mod)
            k -= val
        return 0

    def countKSubsequencesWithMaxBeauty(self, s: str, k: int) -> int:
        cnt = collections.Counter(s)
        if len(cnt) < k:
            return 0
        mod = 10**9 + 7
        vals = sorted(cnt.values(), reverse=True)
        end = vals[k - 1]
        ans = 1
        for v in vals:
            if v == end:
                break
            k -= 1
            ans = ans * v % mod
        return ans * math.comb(vals.count(end), k) * pow(end, k, mod) % mod


# 2843 - Count Symmetric Integers - EASY
def calc(x: int) -> bool:
    s = list(str(x))
    if len(s) & 1:
        return False
    return sum(map(int, s[: len(s) // 2])) == sum(map(int, s[len(s) // 2 :]))


s = set(i for i in range(1, 10001) if calc(i))


class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        return sum(i in s for i in range(low, high + 1))

    def countSymmetricIntegers(self, low: int, high: int) -> int:
        ans = 0
        for v in range(low, high + 1):
            if v < 100 and v % 11 == 0:
                ans += 1
            elif v > 1000 and v // 1000 + (v % 1000) // 100 == (v % 100) // 10 + v % 10:
                ans += 1
        return ans


# 2844 - Minimum Operations to Make a Special Number - MEDIUM
class Solution:
    def minimumOperations(self, num: str) -> int:
        # last two digit is {00, 25, 50, 75}
        pre = [-1, -1, -1, -1]  # position of 0, 2, 5, 7
        for i in range(len(num))[::-1]:
            if num[i] == "0":
                if pre[0] == -1:
                    pre[0] = i
                elif pre[0] != -1:
                    return len(num) - i - 2
            elif num[i] == "2":
                if pre[2] != -1:
                    return len(num) - i - 2
            elif num[i] == "5":
                if pre[0] != -1:
                    return len(num) - i - 2
                pre[2] = i
            elif num[i] == "7":
                if pre[2] != -1:
                    return len(num) - i - 2
        return len(num) - ("0" in num)

        # 不需要减去全部 0, 只需要考虑类似 "10" -> "0", "210" -> "0" 情况, 所以可以少减去一个 0
        # 如果有多个 0 (即 num.count("0") > 1), 那么在上述的过程中, 一定会先遇到 "00" 的情况
        # 所以如果能运行到下面再 return, num.count("0") 总是等于 0 或 1, 与 "0" in num = True / False 等效
        return len(num) - num.count("0")

    def minimumOperations(self, num: str) -> int:
        n = len(num)

        def f(tail: str) -> int:
            i = num.rfind(tail[1])
            if i == -1:
                return n
            i = num.rfind(tail[0], 0, i)
            if i == -1:
                return n
            return n - i - 2

        return min(n - ("0" in num), f("00"), f("25"), f("50"), f("75"))

    def minimumOperations(self, num: str) -> int:
        f = [-1] * 25
        f[0] = 0
        for c in num:
            d = ord(c) - ord("0")
            g = [-1] * 25
            for i in range(25):
                if f[i] != -1:
                    g[i] = max(f[i], g[i])
                    g[(i * 10 + d) % 25] = max(f[i] + 1, g[(i * 10 + d) % 25])
            f = g
        return len(num) - f[0]


# 2845 - Count of Interesting Subarrays - MEDIUM
class Solution:
    # O(n) / O(n + mod)
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        pre = list(itertools.accumulate(v % modulo == k for v in nums))
        cnt = collections.defaultdict(int)
        ans = 0
        for i in range(len(nums)):
            mod = pre[i] % modulo
            ans += mod == k
            ans += cnt[(mod - k + modulo) % modulo]
            cnt[mod] += 1
        return ans

    # O(n) / O(mod)
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        # cnt = collections.Counter({0: 1})
        cnt = collections.Counter([0])
        ans = mod = 0
        for v in nums:
            mod = (mod + (v % modulo == k)) % modulo
            ans += cnt[(mod - k) % modulo]
            cnt[mod] += 1
        return ans

    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        cnt = collections.defaultdict(int)
        ans = 0
        for x in itertools.accumulate((v % modulo == k for v in nums), initial=0):
            ans += cnt[(x - k) % modulo]
            cnt[x % modulo] += 1
        return ans


# 2846 - Minimum Edge Weight Equilibrium Queries in a Tree - HARD
class Solution:
    # O(nq) / O(U), U = 边数, TLE
    def minOperationsQueries(
        self, n: int, edges: List[List[int]], queries: List[List[int]]
    ) -> List[int]:
        g = [[] for _ in range(n + 1)]
        for x, y, w in edges:
            g[x].append((y, w - 1))
            g[y].append((x, w - 1))

        def bfs(s: int, d: int) -> int:
            q = [(s, [0] * 26)]
            l = 0
            while q:
                new = []
                for x, arr in q:
                    if x == d:
                        return l - max(arr)
                    for y, w in g[x]:
                        cp = arr.copy()
                        cp[w] += 1
                        new.append((y, cp))
                q = new
                l += 1
            return -1

        return [bfs(x, y) for x, y in queries]

    # O((n + q) * U * logn), O(nUlogn), U = 边权种类数
    def minOperationsQueries(
        self, n: int, edges: List[List[int]], queries: List[List[int]]
    ) -> List[int]:
        g = [[] for _ in range(n)]
        for x, y, w in edges:
            g[x].append((y, w - 1))
            g[y].append((x, w - 1))

        m = n.bit_length()
        pa = [[-1] * m for _ in range(n)]
        cnt = [[[0] * 26 for _ in range(m)] for _ in range(n)]
        depth = [0] * n

        def dfs(x: int, fa: int) -> None:
            pa[x][0] = fa
            for y, w in g[x]:
                if y != fa:
                    cnt[y][0][w] = 1
                    depth[y] = depth[x] + 1
                    dfs(y, x)

        dfs(0, -1)

        # 倍增模板
        for i in range(m - 1):
            for x in range(n):
                p = pa[x][i]
                if p != -1:
                    pp = pa[p][i]
                    pa[x][i + 1] = pp
                    for j, (c1, c2) in enumerate(zip(cnt[x][i], cnt[p][i])):
                        cnt[x][i + 1][j] = c1 + c2

        ans = []
        for x, y in queries:
            path_len = depth[x] + depth[y]  # 最后减去 depth[lca] * 2
            cw = [0] * 26
            if depth[x] > depth[y]:
                x, y = y, x

            # 使 y 和 x 在同一深度
            k = depth[y] - depth[x]
            for i in range(k.bit_length()):
                if (k >> i) & 1:  # k 二进制从低到高第 i 位是 1
                    p = pa[y][i]
                    for j, c in enumerate(cnt[y][i]):
                        cw[j] += c
                    y = p

            if y != x:
                for i in range(m - 1, -1, -1):
                    px, py = pa[x][i], pa[y][i]
                    if px != py:
                        for j, (c1, c2) in enumerate(zip(cnt[x][i], cnt[y][i])):
                            cw[j] += c1 + c2
                        x, y = px, py  # 同时上跳 2^i 步
                for j, (c1, c2) in enumerate(zip(cnt[x][0], cnt[y][0])):
                    cw[j] += c1 + c2
                x = pa[x][0]

            lca = x
            path_len -= depth[lca] * 2
            ans.append(path_len - max(cw))
        return ans

    # O((n + q) * M + (n + q) * logn), 来源于统计频率以及寻找 LCA
    def minOperationsQueries(
        self, n: int, edges: List[List[int]], queries: List[List[int]]
    ) -> List[int]:
        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w - 1))
            g[v].append((u, w - 1))

        from algo.lca import LCA

        lca = LCA(g, 0)

        parent = lca.parent[0]
        vals = [[0] * 26 for _ in range(n)]
        stack = [0]
        vis = [0] * n
        vis[0] = 1
        while stack:
            u = stack.pop()
            for v, w in g[u]:
                if not vis[v]:
                    vis[v] = 1
                    stack.append(v)
                    for i in range(26):
                        # 统计
                        vals[v][i] += vals[u][i]
                    vals[v][w] += 1

        ans = []
        for u, v in queries:
            l = lca.getLCA(u, v)
            tmp = [0] * 26
            for i in range(26):
                tmp[i] += vals[u][i] + vals[v][i] - vals[l][i] * 2
            ans.append(sum(tmp) - max(tmp))
        return ans

    # 枚举可能的众数, 把对应数字的边权重都强行赋值为 0 这样要求的就是所有情况中路径长度的最小值.
    # 为此可以通过 26 次建图, 得到 26 个不同的从根节点出发的深度,
    # 再使用 dist(u) + dist(v) - dist(lca(u,v)) * 2 得到新路径的长度.
    # 注意这里不需要查询 26q 次 lca, 因为路径权值的改变不改变 lca, 因此开始预处理即可
    def minOperationsQueries(
        self, n: int, edges: List[List[int]], queries: List[List[int]]
    ) -> List[int]:
        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w))

        from algo.lca import LCA

        lca = LCA(g, 0)

        ans = [math.inf] * len(queries)
        lcas = [lca.getLCA(u, v) for u, v in queries]
        for i in range(1, 27):
            depth = [math.inf] * n
            depth[0] = 0
            stack = [0]
            while stack:
                u = stack.pop()
                for v, w in g[u]:
                    if depth[v] == math.inf:
                        if w == i:
                            depth[v] = depth[u]
                        else:
                            depth[v] = depth[u] + 1
                        stack.append(v)
            for i in range(len(queries)):
                u, v = queries[i]
                l = lcas[i]
                ans[i] = min(depth[u] + depth[v] - 2 * depth[l], ans[i])
        return ans

    def minOperationsQueries(
        self, n: int, edges: List[List[int]], queries: List[List[int]]
    ) -> List[int]:
        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w - 1))
            g[v].append((u, w - 1))
        q = collections.deque([0])
        par = [0] * n
        dep = [0] * n
        summ = [None for _ in range(n)]
        f = [[0] * 14 for _ in range(n)]
        summ[0] = [0] * 26
        while q:
            u = q.popleft()
            f[u][0] = par[u]
            for i in range(1, 14):
                f[u][i] = f[f[u][i - 1]][i - 1]
            for v, w in g[u]:
                if par[u] != v:
                    par[v] = u
                    summ[v] = summ[u].copy()
                    summ[v][w] += 1
                    dep[v] = dep[u] + 1
                    q.append(v)
        ans = []
        for u, v in queries:
            x, y = u, v
            if dep[x] > dep[y]:
                x, y = y, x
            for i in reversed(range(14)):
                if dep[y] - dep[x] >= 1 << i:
                    y = f[y][i]
            for i in reversed(range(14)):
                if f[x][i] != f[y][i]:
                    x, y = f[x][i], f[y][i]
            if x != y:
                x = par[x]
            ans.append(
                dep[u]
                + dep[v]
                - 2 * dep[x]
                - max([summ[u][i] + summ[v][i] - 2 * summ[x][i] for i in range(26)])
            )
        return ans


# 2848 - Points That Intersect With Cars - EASY
class Solution:
    # O(nD) / O(D), D = max(e - s)
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        car = [False] * 101
        for s, e in nums:
            for i in range(s, e + 1):
                car[i] = True
        return sum(car)

    # O(n + M) / O(M), M = max(ends)
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        diff = [0] * 102
        for s, e in nums:
            diff[s] += 1
            diff[e + 1] -= 1
        return sum(v > 0 for v in itertools.accumulate(diff))


# 2849 - Determine if a Cell Is Reachable at a Given Time - MEDIUM
class Solution:
    # O(1) / O(1)
    def isReachableAtTime(self, sx: int, sy: int, fx: int, fy: int, t: int) -> bool:
        if sx == fx and sy == fy:
            return t != 1
        return max(abs(fx - sx), abs(fy - sy)) <= t

    def isReachableAtTime(self, sx: int, sy: int, fx: int, fy: int, t: int) -> bool:
        dx = abs(sx - fx)
        dy = abs(sy - fy)
        return t >= max(dx, dy) if dx or dy else t != 1


# 2850 - Minimum Moves to Spread Stones Over Grid - MEDIUM
class Solution:
    # 更快的做法是最小费用最大流, 即使是 10 x 10 的网格也可以做
    # 建图规则如下:
    # 从每个大于 1 的格子向每个等于 0 的格子连边, 容量为 1, 费用为两个格子之间的曼哈顿距离.
    # 从超级源点向每个大于 1 的格子连边, 容量为格子的值减一 (即移走的石子数), 费用为 0.
    # 从每个等于 0 的格子向超级汇点连边, 容量 1 (即移入的石子数), 费用为 0.
    # 答案为最大流时, 对应的最小费用.

    # 最少步数经常的想法是使用 BFS / Dijkstra 等最短路算法
    # 对于此题, 可能的状态数是 9 个非负整数相加等于 9 的不同方案数, math.comb(17, 8) = 24310
    # O(mn * (mn)!) / O(mn), m = n = 3, 1120ms
    def minimumMoves(self, grid: List[List[int]]) -> int:
        q = [tuple(v for row in grid for v in row)]
        vis = set(q)
        ans = 0
        while q:
            new = []
            for state in q:
                if max(state) == 1:
                    return ans
                state = list(state)
                for i in range(9):
                    if state[i] > 1:
                        x, y = divmod(i, 3)
                        for nx, ny in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
                            if 0 <= nx < 3 and 0 <= ny < 3:
                                ni = nx * 3 + ny
                                state[i] -= 1
                                state[ni] += 1
                                t = tuple(state)
                                if t not in vis:
                                    vis.add(t)
                                    new.append(t)
                                state[i] += 1
                                state[ni] -= 1
            q = new
            ans += 1
        return -1


# 或者反过来预处理, 跑一次最短路即可, 296ms
init = (1,) * 9
q = [init]
vis = {init: 0}
while q:
    new = []
    for pre in q:
        state = list(pre)
        for i in range(9):
            if state[i] > 0:
                x, y = divmod(i, 3)
                for nx, ny in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
                    if 0 <= nx < 3 and 0 <= ny < 3:
                        ni = nx * 3 + ny
                        state[i] -= 1
                        state[ni] += 1
                        t = tuple(state)
                        if t not in vis:
                            vis[t] = vis[pre] + 1
                            new.append(t)
                        state[i] += 1
                        state[ni] -= 1
    q = new


class Solution:
    def minimumMoves(self, grid: List[List[int]]) -> int:
        return vis[tuple(v for row in grid for v in row)]


class Solution:
    # 枚举 from 的所有排列, 与 to 匹配, 累加从 from[i] 到 to[i] 的曼哈顿距离
    #  O(mn * (mn)!) / O(mn), m = n = 3, 600ms
    def minimumMoves(self, grid: List[List[int]]) -> int:
        fromm = []
        to = []
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v > 1:
                    fromm.extend(((i, j),) * (v - 1))
                elif v == 0:
                    to.append((i, j))
        return min(
            sum(abs(x1 - x2) + abs(y1 - y2) for (x1, y1), (x2, y2) in zip(possible, to))
            for possible in itertools.permutations(fromm)
        )


# 2851 - String Transformation - HARD
class Solution:
    def numberOfWays(self, s, t, k):
        n = len(s)
        c = self.kmp_search(s + s[:-1], t)
        m = [[c - 1, c], [n - c, n - 1 - c]]
        m = self.pow(m, k)
        return m[0][s != t]

    # KMP 模板
    def calc_max_match(self, s: str) -> List[int]:
        match = [0] * len(s)
        c = 0
        for i in range(1, len(s)):
            v = s[i]
            while c and s[c] != v:
                c = match[c - 1]
            if s[c] == v:
                c += 1
            match[i] = c
        return match

    # KMP 模板
    # 返回 text 中出现了多少次 pattern（允许 pattern 重叠）
    def kmp_search(self, text: str, pattern: str) -> int:
        match = self.calc_max_match(pattern)
        match_cnt = c = 0
        for i, v in enumerate(text):
            v = text[i]
            while c and pattern[c] != v:
                c = match[c - 1]
            if pattern[c] == v:
                c += 1
            if c == len(pattern):
                match_cnt += 1
                c = match[c - 1]
        return match_cnt

    # 矩阵乘法
    def multiply(self, a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
        c = [[0, 0], [0, 0]]
        for i in range(2):
            for j in range(2):
                c[i][j] = (a[i][0] * b[0][j] + a[i][1] * b[1][j]) % (10**9 + 7)
        return c

    # 矩阵快速幂
    def pow(self, a: List[List[int]], n: int) -> List[List[int]]:
        res = [[1, 0], [0, 1]]
        while n:
            if n % 2:
                res = self.multiply(res, a)
            a = self.multiply(a, a)
            n //= 2
        return res


class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        tmp = t + "#" + s * 2
        n = len(s)
        cnt = 0
        kmp = self.prep(tmp)
        for i in range(len(kmp)):
            if kmp[i] == n and i - n * 2 < n:
                cnt += 1

        grid = [[cnt - 1, cnt], [n - cnt, n - cnt - 1]]
        grid_pow = self.matrix_pow(grid, k)
        return grid_pow[0][0] if s == t else grid_pow[0][1]

    def prep(self, p):
        pi = [0] * len(p)
        j = 0
        for i in range(1, len(p)):
            while j != 0 and p[j] != p[i]:
                j = pi[j - 1]
            if p[j] == p[i]:
                j += 1
            pi[i] = j
        return pi

    def matrix_mul(self, A, B, mod=10**9 + 7):
        n, m = len(A), len(A[0])
        p = len(B[0])
        ans = [[0] * p for _ in range(n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ans[i][k] += A[i][j] * B[j][k]
                    ans[i][k] %= mod
        return ans

    def matrix_pow(self, A, n):
        length = len(A)
        tmp = A
        ans = [[0] * length for _ in range(length)]
        for i in range(length):
            ans[i][i] = 1
        for i in range(60):
            if n % 2:
                ans = self.matrix_mul(ans, tmp)
            tmp = self.matrix_mul(tmp, tmp)
            n //= 2
        return ans
