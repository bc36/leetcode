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


# 2825 - Make String a Subsequence Using Cyclic Increments - MEDIUM
d = {string.ascii_lowercase[i]: string.ascii_lowercase[(i + 1) % 26] for i in range(26)}


class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        j = 0
        for c in str1:
            # if c == str2[j] or chr((ord(c) - 97 + 1) % 26 + 97) == str2[j]:
            if c == str2[j] or d[c] == str2[j]:
                j += 1
                if j == len(str2):
                    return True
        return False


# 2826 - Sorting Three Groups - MEDIUM
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        f = [1] * n
        for i in range(1, n):
            for j in range(i):
                if nums[i] >= nums[j]:
                    f[i] = max(f[i], f[j] + 1)
        return n - max(f)

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

    def minimumOperations(self, nums: List[int]) -> int:
        f = [math.inf, 0, 0, 0]
        for x in nums:
            for i in (1, 2, 3):
                f[i] = min(f[i - 1], f[i] + (x == i))
        return f[3]

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
            def f(i: int, pre: int, isLimit: bool, isNum: bool, e: int, o: int) -> int:
                """表示构造第 i 位及其之后数位的合法方案数"""
                if i == len(s):
                    # isNum 为 True 表示得到了一个合法数字
                    return int(isNum and e == o and pre == 0)
                res = 0
                if not isNum:  # 可以跳过当前数位
                    res = f(i + 1, pre, False, False, e, o)
                down = 0 if isNum else 1  # 如果前面没有填数字, 必须从 1 开始, 因为不能有前导零
                # 如果前面填的数字都和 s 的一样, 那么这一位至多填 s[i], 否则就超过 s 了
                up = int(s[i]) if isLimit else 9
                for d in range(down, up + 1):  # 枚举要填入的数字 d
                    res += f(
                        i + 1,
                        (pre * 10 + d) % k,
                        isLimit and d == up,
                        True,
                        e + (d & 1),
                        o + (1 - d & 1),
                    )
                return res

            return f(0, 0, True, False, 0, 0)

        return calc(str(high)) - calc(str(low - 1))
