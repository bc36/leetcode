import bisect, collections, functools, math, itertools, heapq, string, operator, sortedcontainers
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 2400 - Number of Ways to Reach a Position After Exactly k Steps - MEDIUM
class Solution:
    # 有负数, 写记忆化, 数组麻烦
    def numberOfWays(self, startPos: int, endPos: int, k: int) -> int:
        d = abs(startPos - endPos)
        if d > k or (k - d) & 1:
            return 0
        mod = 10**9 + 7

        @functools.lru_cache(None)
        def f(p: int, k: int) -> int:
            """位置 p, 剩余步数 k, 从 p 往 0 走"""
            if abs(p) > k:
                return 0
            if k == 0 and p == 0:
                return 1
            return (f(p - 1, k - 1) + f(p + 1, k - 1)) % mod

        return f(d, k)

    def numberOfWays(self, startPos: int, endPos: int, k: int) -> int:
        d = abs(startPos - endPos)
        if d > k or d % 2 != k % 2:
            return 0
        mod = 10**9 + 7

        @functools.lru_cache(None)
        def f(p: int, k: int) -> int:
            """位置 p, 剩余步数 k"""
            if abs(p - endPos) > k:
                return 0
            if k == 0:
                return 1
            return (f(p - 1, k - 1) + f(p + 1, k - 1)) % mod

        return f(startPos, k)

    # 组合数学
    def numberOfWays(self, startPos: int, endPos: int, k: int) -> int:
        """
        向终点走 x 步, 向反方向走 k - x 步
        -> + x - (k - x) = d
        -> x = (d + k) // 2
        -> comb(k, x), k 步挑 x 步
        """
        d = abs(startPos - endPos)
        if d > k or (k - d) & 1:
            return 0
        mod = 10**9 + 7
        return math.comb(k, (d + k) // 2) % mod


# 2401 - Longest Nice Subarray - MEDIUM
class Solution:
    # O(930n) / O(n), 30n + 均摊复杂度, 每次移动一次 l 指针, 移动一次 900, 移动 n 次
    # 9.3e7 -> 1e8, 4s, 快 TLE 了
    def longestNiceSubarray(self, nums: List[int]) -> int:
        """
        双指针, 对每一位求 1 的位置, 某一位 1 有 2 个, 就移动左边指针
        我们需要选出最长的区间，使得区间中每个二进制位最多出现一个 1
        1 <= nums[i] <= 1e9, 30位
        """
        l = 0
        ans = 1
        cnt = collections.defaultdict(int)
        z = nums[0]
        i = 0
        while z:  # O(30)
            if z & 1:
                cnt[i] += 1
            i += 1
            z //= 2
        for r in range(1, len(nums)):  # O(n)
            z = nums[r]
            i = 0
            while z:  # O(30)
                if z & 1:
                    cnt[i] += 1
                i += 1
                z //= 2
            while l < len(nums) and any(v >= 2 for v in cnt.values()):  # O(30)
                z = nums[l]
                i = 0
                while z:  # O(30)
                    if z & 1:
                        cnt[i] -= 1
                    i += 1
                    z //= 2
                l += 1
            ans = max(ans, r - l + 1)
        return ans

    # python 位运算好慢
    def longestNiceSubarray(self, nums: List[int]) -> int:
        ans = l = 0
        cnt = collections.defaultdict(int)
        for r in range(len(nums)):  # O(n)
            for k in range(30):
                cnt[k] += nums[r] >> k & 1
            while l < len(nums) and any(v >= 2 for v in cnt.values()):  # O(30)
                for k in range(30):
                    cnt[k] -= nums[l] >> k & 1
                l += 1
            ans = max(ans, r - l + 1)
        return ans

    # O(n * log(max(nums))) -> O(30n) / O(1)
    def longestNiceSubarray(self, nums: List[int]) -> int:
        """
        由于所有元素对按位与均为 0, 在优雅子数组中的从低到高的第 i 个位上, 至多有一个 1, 其余均为 0
        因此在本题数据范围下, 优雅子数组的长度不会超过 30
        """
        ans = 0
        for i, v in enumerate(nums):
            j = i - 1
            while j >= 0 and (v & nums[j]) == 0:
                v |= nums[j]
                j -= 1
            ans = max(ans, i - j)
        return ans

    # O(n) / O(1)
    def longestNiceSubarray(self, nums: List[int]) -> int:
        """由于优雅子数组的所有元素按位与均为 0, 可以理解成这些二进制数对应的集合没有交集, 所以可以用 xor 把它去掉"""
        ans = l = orr = 0
        for r, v in enumerate(nums):
            while orr & v:
                orr ^= nums[l]
                l += 1
            orr |= v
            ans = max(ans, r - l + 1)
        return ans


# 2402 - Meeting Rooms III - HARD
class Solution:
    # O(n + m(logm + logn)) / O(n)
    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        meetings.sort()
        cnt = [0] * n
        idle = list(range(n))
        using = []
        for s, e in meetings:
            while using and using[0][0] <= s:
                _, i = heapq.heappop(using)
                heapq.heappush(idle, i)
            if idle:
                i = heapq.heappop(idle)
                heapq.heappush(using, (e, i))
            else:
                end, i = heapq.heappop(using)
                heapq.heappush(using, (end + e - s, i))
            cnt[i] += 1
        mx = max(cnt)
        for i, v in enumerate(cnt):
            if v == mx:
                return i
        return -1

    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        meetings.sort()
        cnt = [0] * n
        idle = list(range(n))
        using = []
        for s, e in meetings:
            while using and using[0][0] <= s:
                heapq.heappush(idle, heapq.heappop(using)[1])
            if len(idle) == 0:
                end, i = heapq.heappop(using)
                e += end - s
            else:
                i = heapq.heappop(idle)
            cnt[i] += 1
            heapq.heappush(using, (e, i))
        ans = 0
        for i, c in enumerate(cnt):
            if c > cnt[ans]:
                ans = i
        return ans

    # 数据范围小, n <= 100 , 暴力
    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        cnt = [0] * n
        t = [0] * n
        for s, e in sorted(meetings):
            t = list(map(lambda x: max(x, s), t))
            choice = t.index(min(t))
            t[choice] += e - s
            cnt[choice] += 1
        return cnt.index(max(cnt))


# 2413 - Smallest Even Multiple - EASY
class Solution:
    def smallestEvenMultiple(self, n: int) -> int:
        return (n % 2 + 1) * n
        return n if n % 2 == 0 else 2 * n
        return math.lcm(2, n)


# 2414 - Length of the Longest Alphabetical Continuous Substring - MEDIUM
class Solution:
    def longestContinuousSubstring(self, s: str) -> int:
        pre = ord(s[0])
        ans = cur = 1
        for c in s[1:]:
            o = ord(c)
            if pre + 1 == o:
                cur += 1
            else:
                cur = 1
            pre = o
            ans = max(ans, cur)
        return ans

    def longestContinuousSubstring(self, s: str) -> int:
        ans = pre = 0
        for i in range(1, len(s)):
            if ord(s[i]) != ord(s[i - 1]) + 1:
                ans = max(ans, i - pre)
                pre = i
        return max(ans, len(s) - pre)

    def longestContinuousSubstring(self, s: str) -> int:
        ans = i = 0
        while i < len(s):
            j = i + 1
            while j < len(s) and ord(s[j - 1]) + 1 == ord(s[j]):
                j += 1
            ans = max(ans, j - i)
            i = j
        return ans


# 2415 - Reverse Odd Levels of Binary Tree - MEDIUM
class Solution:
    def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        q = [root]
        lv = 0
        while q:
            new = []
            for node in q:
                if node.left:
                    new += [node.left, node.right]
            if lv & 1:
                n = len(q)
                for i in range(n // 2):
                    q[i].val, q[n - i - 1].val = q[n - i - 1].val, q[i].val
            q = new
            lv += 1
        return root

    def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        q = [root]
        lv = 0
        while q[0].left:
            q = list(itertools.chain.from_iterable((x.left, x.right) for x in q))
            if lv == 0:
                n = len(q)
                for i in range(n // 2):
                    q[i].val, q[n - 1 - i].val = q[n - 1 - i].val, q[i].val
            lv ^= 1
        return root

    # 对称结构, 同时递归两个子树
    def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(n1: TreeNode, n2: TreeNode, odd: bool) -> None:
            if n1 is None:
                return
            if odd:
                n1.val, n2.val = n2.val, n1.val
            dfs(n1.left, n2.right, not odd)
            dfs(n1.right, n2.left, not odd)
            return

        dfs(root.left, root.right, True)
        return root


# 2416 - Sum of Prefix Scores of Strings - HARD
class Solution:
    def sumPrefixScores(self, words: List[str]) -> List[int]:
        d = {}
        for w in words:
            r = d
            for c in w:
                if c not in r:
                    r[c] = {}
                    r[(c, "#")] = 1
                else:
                    r[(c, "#")] += 1
                r = r[c]
        ans = []
        for w in words:
            r = d
            s = 0
            for c in w:
                if (c, "#") in r:
                    s += r[(c, "#")]
                r = r[c]
            ans.append(s)
        return ans

    def sumPrefixScores(self, words: List[str]) -> List[int]:
        d = {}
        for w in words:
            r = d
            for c in w:
                if c not in r:
                    r[c] = {}
                    r[c]["cnt"] = 1
                else:
                    r[c]["cnt"] += 1
                r = r[c]
        ans = []
        for w in words:
            r = d
            s = 0
            for c in w:
                s += r[c]["cnt"]
                r = r[c]
            ans.append(s)
        return ans


# use __slots__ if you are going to instantiate a lot (hundreds, thousands) of objects of the same class.
# __slots__ only exists as a memory optimization tool.
class Node:
    __slots__ = "son", "ids", "score"  # 访问属性更快, 省空间, 有点过优化

    def __init__(self):
        self.son = collections.defaultdict(Node)  # max(len()) = 26
        self.ids = []
        self.score = 0


class Solution:
    def sumPrefixScores(self, words: List[str]) -> List[int]:
        trie = Node()
        for i, word in enumerate(words):
            r = trie
            for c in word:
                r = r.son[c]
                r.score += 1
            r.ids.append(i)

        ans = [0] * len(words)

        def dfs(node: Node, summ: int) -> None:
            summ += node.score
            for i in node.ids:
                ans[i] = summ
            for child in node.son.values():
                if child:
                    dfs(child, summ)
            return

        dfs(trie, 0)
        return ans
