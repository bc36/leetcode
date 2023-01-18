import bisect, collections, functools, math, itertools, heapq, string, operator, re
from typing import List, Optional
import sortedcontainers

# 1800 - Maximum Ascending Subarray Sum - EASY
class Solution:
    def maxAscendingSum(self, nums: List[int]) -> int:
        ans = cur = p = 0
        for v in nums:
            if p < v:
                cur += v
            else:
                cur = v
            p = v
            ans = max(ans, cur)
        return ans


# 1801 - Number of Orders in the Backlog - MEDIUM
class Solution:
    def getNumberOfBacklogOrders(self, orders: List[List[int]]) -> int:
        sell = []
        buy = []
        for v in orders:
            if v[2] == 0:
                while sell and v[1] and v[0] >= sell[0][0]:
                    if sell[0][1] > v[1]:
                        sell[0][1] -= v[1]
                        v[1] = 0
                    else:
                        v[1] -= heapq.heappop(sell)[1]
                if v[1] > 0:
                    v[0] *= -1
                    heapq.heappush(buy, v)
            else:
                while buy and v[1] and v[0] <= -buy[0][0]:
                    if buy[0][1] > v[1]:
                        buy[0][1] -= v[1]
                        v[1] = 0
                    else:
                        v[1] -= heapq.heappop(buy)[1]
                if v[1] > 0:
                    heapq.heappush(sell, v)
        return sum(v for _, v, _ in sell + buy) % 1000000007


# 1802 - Maximum Value at a Given Index in a Bounded Array - MEDIUM
class Solution:
    def maxValue(self, n: int, index: int, maxSum: int) -> int:
        def calc(m: int) -> int:
            s = 0
            if index + 1 < m:  # index = m - 1
                s += (m + m - index) * (index + 1) // 2
            else:
                s += (m + 1) * m // 2 + index - m + 1
            if n - index < m:  # index = n - m
                s += (m + m - (n - 1 - index)) * (n - index) // 2
            else:
                s += (m + 1) * m // 2 + n - index - m
            return s - m

        l = 0
        r = maxSum
        while l < r:
            m = (l + r + 1) // 2
            if calc(m) <= maxSum:
                l = m
            else:
                r = m - 1

            # 思考: 有什么区别?
            # m = (l + r) // 2
            # if calc(m) <= maxSum:
            #     l = m + 1
            # else:
            #     r = m

        return l

        # 可以通过的代码
        while l < r:
            m = (l + r) // 2
            if calc(m) <= maxSum:
                l = m + 1
            else:
                r = m
        return l + 1 if calc(l + 1) <= maxSum else l if calc(l) <= maxSum else l - 1


# 1803 - Count Pairs With XOR in a Range - HARD
class Solution:
    # O(nlogU) / O(nlogU), U = max(nums) < 2**15, 5700 ms
    def countPairs(self, nums: List[int], low: int, high: int) -> int:
        class TrieNode:
            def __init__(self) -> None:
                self.ch = [None, None]
                self.sum = 0

        class Trie:
            def __init__(self) -> None:
                self.r = TrieNode()

            def add(self, x: int) -> None:
                root = self.r
                for k in range(14, -1, -1):  # max_bit_length = 14
                    b = x >> k & 1
                    if not root.ch[b]:
                        root.ch[b] = TrieNode()
                    root = root.ch[b]
                    root.sum += 1
                return

            # 表示有多少对数字的异或运算结果小于等于 lmt
            def count(self, x: int, lmt: int) -> int:
                res = 0
                root = self.r
                for k in range(14, -1, -1):
                    b = x >> k & 1
                    if lmt >> k & 1:
                        if root.ch[b]:  # 因为题目是小于号, 所以只有 lmt 对应位为 1 才能计数
                            res += root.ch[b].sum
                        root = root.ch[b ^ 1]
                    else:
                        root = root.ch[b]
                    if root is None:
                        return res
                res += root.sum  # 计算小于等于 lmt 和 小于 lmt 的区别, 上面已经避免了 root 为 None 的情况
                return res

        def f(lmt: int) -> int:
            ans = 0
            tree = Trie()
            for i in range(1, len(nums)):
                tree.add(nums[i - 1])
                ans += tree.count(nums[i], lmt)
            return ans

        return f(high) - f(low - 1)

    # O(nlogU) / O(nlogU), U = max(nums) < 2**15, 3600 ms
    def countPairs(self, nums: List[int], low: int, high: int) -> int:
        class Trie:
            def __init__(self):
                self.ch = [None, None]
                self.cnt = 0

            def add(self, x: int) -> None:
                root = self
                for k in range(14, -1, -1):
                    b = x >> k & 1
                    if root.ch[b] is None:
                        root.ch[b] = Trie()
                    root = root.ch[b]
                    root.cnt += 1
                return

            # 统计有多少数对的异或值小于 lmt
            def count(self, x: int, lmt: int) -> int:
                root = self
                ans = 0
                for k in range(14, -1, -1):
                    if root is None:
                        return ans
                    b = x >> k & 1
                    if lmt >> k & 1:
                        if root.ch[b]:
                            ans += root.ch[b].cnt
                        root = root.ch[b ^ 1]
                    else:
                        root = root.ch[b]
                return ans

        ans = 0
        tree = Trie()
        for x in nums:
            ans += tree.count(x, high + 1) - tree.count(x, low)
            tree.add(x)
        return ans

    # O(n + nlog(U/n)) / O(log(U/n)), U = max(nums), 350 ms
    def countPairs(self, nums: List[int], low: int, high: int) -> int:
        ans = 0
        cnt = collections.Counter(nums)
        high += 1
        while high:
            nxt = collections.Counter()
            for k, v in cnt.items():
                # high % 2 * cnt[(high - 1) ^ k] 相当于 cnt[(high - 1) ^ k] if high % 2 else 0
                # ans += v * (high % 2 * cnt[(high - 1) ^ k] - low % 2 * cnt[(low - 1) ^ k])
                if high & 1:
                    ans += v * cnt[k ^ (high - 1)]
                if low & 1:
                    ans -= v * cnt[k ^ (low - 1)]
                nxt[k >> 1] += v
            cnt = nxt
            low >>= 1
            high >>= 1
        return ans // 2


# 1805 - Number of Different Integers in a String - EASY
class Solution:
    def numDifferentIntegers(self, word: str) -> int:
        s = set()
        x = ""
        for c in word + "#":
            if c.isdigit():
                x += c
            else:
                if x:
                    s.add(int(x))
                x = ""
        return len(s)

    def numDifferentIntegers(self, word: str) -> int:
        return len(set(map(int, re.findall("\d+", word))))


# 1806 - Minimum Number of Operations to Reinitialize a Permutation - MEDIUM
class Solution:
    def reinitializePermutation(self, n: int) -> int:
        p = list(range(n))
        a = [p[n // 2 + (i - 1) // 2] if i % 2 else p[i // 2] for i in range(n)]
        t = 1
        while a != p:
            a = [a[n // 2 + (i - 1) // 2] if i % 2 else a[i // 2] for i in range(n)]
            t += 1
        return t


# 1807 - Evaluate the Bracket Pairs of a String - MEDIUM
class Solution:
    def evaluate(self, s: str, knowledge: List[List[str]]) -> str:
        d = dict(knowledge)
        ans = ""
        i = 0
        while i < len(s):
            if s[i] == "(":
                k = ""
                i += 1
                while i < len(s) and s[i] != ")":
                    k += s[i]
                    i += 1
                ans += d.get(k, "?")
            else:
                ans += s[i]
            i += 1
        return ans

    def evaluate(self, s: str, knowledge: List[List[str]]) -> str:
        d = dict(knowledge)
        s = s.split("(")
        for i in range(len(s)):
            p = s[i].split(")")
            if len(p) == 2:
                s[i] = d.get(p[0], "?") + p[1]
        return "".join(s)


# 1812 - Determine Color of a Chessboard Square - EASY
class Solution:
    def squareIsWhite(self, coordinates: str) -> bool:
        return (ord(coordinates[0]) - 97 + int(coordinates[1])) % 2 == 0


# 1813 - Sentence Similarity III - MEDIUM
class Solution:
    def areSentencesSimilar(self, sentence1: str, sentence2: str) -> bool:
        d1 = collections.deque(sentence1.split())
        d2 = collections.deque(sentence2.split())
        if len(d1) < len(d2):
            return self.areSentencesSimilar(sentence2, sentence1)
        while d2 and d1[0] == d2[0]:
            d1.popleft()
            d2.popleft()
        while d2 and d1[-1] == d2[-1]:
            d1.pop()
            d2.pop()
        return len(d2) == 0

    def areSentencesSimilar(self, sentence1: str, sentence2: str) -> bool:
        sentence1 = sentence1.split()
        sentence2 = sentence2.split()
        m = len(sentence1)
        n = len(sentence2)
        i = j = 0
        while i < m and i < n and sentence1[i] == sentence2[i]:
            i += 1
        while (
            m - 1 - j > -1
            and n - 1 - j > -1
            and sentence1[m - 1 - j] == sentence2[n - 1 - j]
        ):
            j += 1
        return i + j >= min(m, n)


# 1814 - Count Nice Pairs in an Array - MEDIUM
class Solution:
    def countNicePairs(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        cnt = collections.defaultdict(int)
        ans = 0
        for v in nums:
            rev = int(str(v)[::-1])
            ans = (ans + cnt[v - rev]) % mod
            cnt[v - rev] += 1
        return ans

    def countNicePairs(self, nums: List[int]) -> int:
        return sum(
            map(
                lambda x: x * (x - 1) // 2,
                collections.Counter([v - int(str(v)[::-1]) for v in nums]).values(),
            )
        ) % (10**9 + 7)


# 1816 - Truncate Sentence - EASY
class Solution:
    def truncateSentence(self, s: str, k: int) -> str:
        return " ".join(s.split()[:k])


# 1819 - Number of Different Subsequences GCDs - HARD
class Solution:
    # 对于数组 nums 的所有子序列, 其最大公约数一定不超过数组中的最大值 -> 考虑值域
    # O(n + UlogU) / O(U)
    def countDifferentSubsequenceGCDs(self, nums: List[int]) -> int:
        ans = 0
        mx = max(nums)
        has = [False] * (mx + 1)
        for v in nums:
            has[v] = True
        for i in range(1, mx + 1):
            g = 0  # 0 和任何数 x 的最大公约数都是 x
            for j in range(i, mx + 1, i):  # 枚举 i 的倍数
                if has[j]:
                    g = math.gcd(g, j)  # 更新最大公约数
                    if g == i:  # 找到一个合法答案, g 无法继续减小
                        ans += 1
                        break
        return ans


# 1822 - Sign of the Product of an Array - EASY
class Solution:
    def arraySign(self, nums: List[int]) -> int:
        a = 0
        for n in nums:
            if n == 0:
                return 0
            a += 1 if n < 0 else 0
        return -1 if a & 1 else 1

    def arraySign(self, nums: List[int]) -> int:
        a = 1
        for n in nums:
            if n == 0:
                return 0
            a *= 1 if n > 0 else -1
        return a


# 1823 - Find the Winner of the Circular Game - MEDIUM
class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        nxt = (0,)
        arr = [i for i in range(1, n + 1)]
        while len(arr) > 1:
            lost = (nxt + k - 1) % len(arr)
            nxt = lost if (lost != len(arr) - 1) else 0
            del arr[lost]
        return arr[0]

    # Josephus problem
    def findTheWinner(self, n: int, k: int) -> int:
        p = 0
        for i in range(2, n + 1):
            p = (p + k) % i
        return p + 1

    def findTheWinner(self, n: int, k: int) -> int:
        f = [0] * (n + 1)
        f[1] = 1
        for i in range(2, n + 1):
            f[i] = (f[i - 1] + k - 1) % i + 1
        return f[n]

    def findTheWinner(self, n: int, k: int) -> int:
        dp = 1
        for i in range(2, n + 1):
            dp = (dp + k - 1) % i + 1
        return dp


# 1825 - Finding MK Average - HARD
class MKAverage:
    def __init__(self, m: int, k: int):
        self.m = m
        self.k = k
        self.s = 0
        # 维护三个有序集合, 保存末尾最小/大的 k 个数, 以及中间剩余的数
        self.low = sortedcontainers.SortedList()
        self.middle = sortedcontainers.SortedList()
        self.high = sortedcontainers.SortedList()
        self.arr = (
            collections.deque()
        )  # len(arr) = len(low) + len(middle) + len(high), 保持插入顺序

    def addElement(self, num: int) -> None:
        self.arr.append(num)
        # 按大小插入到对应集合
        if self.low and self.low[-1] >= num:
            self.low.add(num)
        elif self.high and self.high[0] <= num:
            self.high.add(num)
        else:
            self.middle.add(num)
            self.s += num
        # low / high 平衡到 middle, 每次插入一个数, 所以可以不用 while
        if len(self.low) > self.k:
            self.s += self.low[-1]
            self.middle.add(self.low.pop())
        if len(self.high) > self.k:
            self.s += self.high[0]
            self.middle.add(self.high.pop(0))
        # 末尾序列总数超过 m, 依次查询应移除的数字在哪个集合中
        if len(self.arr) > self.m:
            x = self.arr.popleft()
            if self.low.bisect_left(x) < self.k:
                self.low.remove(x)
            elif self.middle.bisect_left(x) < len(self.middle):
                self.middle.remove(x)
                self.s -= x
            else:
                self.high.remove(x)
        # 删除最旧的数字后, 重新平衡三个集合, 这里因为在 arr 不足 m 时, 会有累积的不平衡, 所以需要使用 while
        if len(self.arr) == self.m:
            while len(self.low) < self.k:
                self.s -= self.middle[0]
                self.low.add(self.middle.pop(0))
            while len(self.high) < self.k:
                self.s -= self.middle[-1]
                self.high.add(self.middle.pop())
        return

    def calculateMKAverage(self) -> int:
        if len(self.arr) < self.m:
            return -1
        return self.s // (self.m - self.k * 2)


class MKAverage:
    def __init__(self, m: int, k: int):
        self.m = m
        self.k = k
        self.s = 0
        self.sl = None
        self.arr = collections.deque()

    def addElement(self, num: int) -> None:
        self.arr.append(num)
        if len(self.arr) == self.m:
            # 初始化 sl
            self.sl = sortedcontainers.SortedList(self.arr)
            self.s = sum(self.sl[self.k : -self.k])

        if len(self.arr) > self.m:
            # 加入后对区间和的影响, num 会把 sortedList 里的元素挤到左边或者右边
            pos = self.sl.bisect_left(num)
            if pos < self.k:
                # 挤到中间
                self.s += self.sl[self.k - 1]
            elif self.k <= pos <= self.m - self.k:
                self.s += num
            else:
                # 挤到中间
                self.s += self.sl[self.m - self.k]
            self.sl.add(num)

            x = self.arr.popleft()
            pos = self.sl.bisect_left(x)
            if pos < self.k:
                # 左移
                self.s -= self.sl[self.k]
            elif self.k <= pos <= self.m - self.k:
                self.s -= x
            else:
                # 右移
                self.s -= self.sl[self.m - self.k]
            self.sl.remove(x)

    def calculateMKAverage(self) -> int:
        if len(self.arr) < self.m:
            return -1
        return self.s // (self.m - 2 * self.k)


# 1827 - Minimum Operations to Make the Array Increasing - EASY
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        ans = 0
        for i in range(1, len(nums)):
            if nums[i - 1] >= nums[i]:
                ans += nums[i - 1] - nums[i] + 1
                nums[i] = nums[i - 1] + 1
        return ans

    def minOperations(self, nums: List[int]) -> int:
        ans = mx = 0
        for v in nums:
            ans += max(0, mx - v + 1)
            mx = max(mx + 1, v)
        return ans


# 1832 - Check if the Sentence Is Pangram - EASY
class Solution:
    def checkIfPangram(self, sentence: str) -> bool:
        return set(string.ascii_lowercase) == set(sentence)

    def checkIfPangram(self, sentence: str) -> bool:
        return len(set(sentence)) == 26

    def checkIfPangram(self, sentence: str) -> bool:
        m = 0
        for c in sentence:
            m |= 1 << (ord(c) - ord("a"))
        return m == (1 << 26) - 1


# 1837 - Sum of Digits in Base K - EASY
class Solution:
    def sumBase(self, n: int, k: int) -> int:
        x = 0
        while n:
            x += n % k
            n //= k
        return x


# 1844 - Replace All Digits with Characters - EASY
class Solution:
    def replaceDigits(self, s: str) -> str:
        return "".join(
            chr(ord(s[i - 1]) + int(c)) if i & 1 else c for i, c in enumerate(s)
        )

    def replaceDigits(self, s: str) -> str:
        arr = list(s)
        for i in range(1, len(arr), 2):
            arr[i] = chr(ord(arr[i - 1]) + int(arr[i]))
        return "".join(arr)
