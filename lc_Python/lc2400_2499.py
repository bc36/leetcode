import bisect, collections, functools, heapq, itertools, math, operator, string
from typing import List, Optional, Tuple
import sortedcontainers


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


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
        我们需要选出最长的区间, 使得区间中每个二进制位最多出现一个 1
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


# 2404 - Most Frequent Even Element - EASY
class Solution:
    def mostFrequentEven(self, nums: List[int]) -> int:
        cnt = collections.Counter()
        ans = mx = 0
        for v in nums:
            if v & 1 == 0:
                cnt[v] += 1
                if cnt[v] == mx and v < ans:
                    mx = cnt[v]
                    ans = v
                if cnt[v] > mx:
                    mx = cnt[v]
                    ans = v
        return ans if len(cnt) else -1

    def mostFrequentEven(self, nums: List[int]) -> int:
        cnt = collections.Counter(v for v in nums if v % 2 == 0)
        if len(cnt) == 0:
            return -1
        mx = max(cnt.values())
        return min(k for k, v in cnt.items() if v == mx)


# 2405 - Optimal Partition of String - MEDIUM
class Solution:
    def partitionString(self, s: str) -> int:
        ans = 0
        vis = set()
        for c in s:
            if c in vis:
                vis.clear()
                ans += 1
            vis.add(c)
        return ans + 1

    def partitionString(self, s: str) -> int:
        ans = 1
        cur = ""
        for c in s:
            if c in cur:
                ans += 1
                cur = ""
            cur += c
        return ans


# 2406 - Divide Intervals Into Minimum Number of Groups - MEDIUM
class Solution:
    # 答案 / 划分与输入的顺序无关 -> 排序
    def minGroups(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        q = sortedcontainers.SortedList()
        for l, r in intervals:
            i = bisect.bisect_left(q, l)
            i -= 1
            if i == -1:
                q.add(r)
            else:
                q.remove(q[i])
                q.add(r)
        return len(q)

    def minGroups(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        q = []
        for l, r in intervals:
            if q and l > q[0]:
                heapq.heapreplace(q, r)
            else:
                heapq.heappush(q, r)
        return len(q)

    # 差分, 最大堆叠次数
    def minGroups(self, intervals: List[List[int]]) -> int:
        # 直接开数组慢, 900ms
        diff = [0] * 1000005
        for l, r in intervals:
            diff[l] += 1
            diff[r + 1] -= 1
        return max(itertools.accumulate(diff))

    def minGroups(self, intervals: List[List[int]]) -> int:
        # 500ms
        diff = collections.defaultdict(int)
        for l, r in intervals:
            diff[l] += 1
            diff[r + 1] -= 1
        arr = sorted((k, v) for k, v in diff.items())
        ans = cur = 0
        for _, v in arr:
            cur += v
            ans = max(ans, cur)
        return ans

    def minGroups(self, intervals: List[List[int]]) -> int:
        # 300ms
        diff = collections.defaultdict(int)
        for l, r in intervals:
            diff[l] += 1
            diff[r + 1] -= 1
        return max(itertools.accumulate(diff[k] for k in sorted(diff)))



# 2409 - Count Days Spent Together - EASY
class Solution:
    def countDaysTogether(
        self, arriveAlice: str, leaveAlice: str, arriveBob: str, leaveBob: str
    ) -> int:
        m = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        def f(s: str) -> int:
            x, y = s.split("-")
            x = int(x)
            y = int(y)
            return y + sum(m[: x - 1])

        aa = f(arriveAlice)
        la = f(leaveAlice)
        ab = f(arriveBob)
        lb = f(leaveBob)
        return max(0, min(la, lb) - max(aa, ab) + 1)

    def countDaysTogether(
        self, arriveAlice: str, leaveAlice: str, arriveBob: str, leaveBob: str
    ) -> int:
        m = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        presum = list(itertools.accumulate(m, initial=0))

        def calc(date: str) -> int:
            return presum[int(date[:2]) - 1] + int(date[3:])

        start = calc(min(leaveAlice, leaveBob))
        end = calc(max(arriveAlice, arriveBob))
        return max(start - end + 1, 0)


# 2410 - Maximum Matching of Players With Trainers - MEDIUM
class Solution:
    def matchPlayersAndTrainers(self, players: List[int], trainers: List[int]) -> int:
        players.sort()
        trainers.sort()
        ans = 0
        while players and trainers:
            if players[-1] <= trainers[-1]:
                players.pop()
                trainers.pop()
                ans += 1
            else:
                players.pop()
        return ans

    def matchPlayersAndTrainers(self, players: List[int], trainers: List[int]) -> int:
        players.sort()
        trainers.sort()
        ans = i = 0
        m = len(trainers)
        for p in players:
            while i < m and trainers[i] < p:
                i += 1
            if i < m:
                ans += 1
                i += 1
            else:
                break
        return ans


# 2411 - Smallest Subarrays With Maximum Bitwise OR - MEDIUM
class Solution:
    # O(nlogU) / O(1), U = max(nums) -> logU = 30
    def smallestSubarrays(self, nums: List[int]) -> List[int]:
        # 如果 nums[j] | nums[i] != nums[j], 说明 nums[j] 可以变大 -> 集合元素增多, 更新 nums[j] 和 ans[j]
        # 如果 nums[j] | nums[i] == nums[j], 说明 nums[i] 是左边所有 nums[0...j] 的子集, break
        # 分析, 为什么很容易 break:
        # 如果每个位置都不一样, 前 30 次循环之后, 后面每个都可以 break, O(30 + n)
        # 最坏结果, 前面的都一样, 最后 30 个才出现每个位置不同的情况, 每个 j 可以减到 0, 最多 O(30n)
        ans = [0] * len(nums)
        for i, v in enumerate(nums):
            ans[i] = 1
            for j in range(i - 1, -1, -1):
                if nums[j] | v == nums[j]:
                    break
                nums[j] |= v
                ans[j] = i - j + 1
        return ans

    def smallestSubarrays(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = [0] * n
        p = [0] * 30  # 记录该位为 1 的最近遍历到的数的索引
        for i in range(n - 1, -1, -1):
            s = bin(nums[i])[2:][::-1]
            for j, c in enumerate(s):
                if c == "1":
                    p[j] = i
            ans[i] = max(max(p) - i + 1, 1)
        return ans

    def smallestSubarrays(self, nums: List[int]) -> List[int]:
        # TODO, 421, 898, 1521
        # 该模板可以:
        # 求出所有子数组的按位或的结果, 以及值等于该结果的子数组的个数
        # 求按位或结果等于任意给定数字的子数组的最短长度 / 最长长度
        n = len(nums)
        ans = [0] * n
        ors = []  # 按位或的值 + 对应子数组的右端点的最小值
        for i in range(n - 1, -1, -1):
            ors.append([0, i])
            k = 0
            for p in ors:
                p[0] |= nums[i]
                if ors[k][0] == p[0]:
                    ors[k][1] = p[1]  # 合并相同值, 下标取最小的
                else:
                    k += 1
                    ors[k] = p
            del ors[k + 1 :]
            # 本题只用到了 ors[0], 如果题目改成任意给定数值, 可以在 ors 中查找
            ans[i] = ors[0][1] - i + 1
        return ans


# 2412 - Minimum Money Required Before Transactions - HARD
class Solution:
    def minimumMoney(self, transactions: List[List[int]]) -> int:
        """
        lose 为亏钱情况 cost - back 之和
        枚举交易, 分类讨论:
            1. cost <= back, 挣钱, 让其发生在亏钱之后, ans = lose + cost[i]
            2. cost > back, 亏钱, 让其发生在最后一笔亏钱时, 已经计算过了, 先把差值退回去, 再减 cost[i]
                ans = lose - (cost[i] - back[i]) + cost[i] = lose + back[i]
        """
        lose = mx = 0
        for c, b in transactions:
            lose += max(c - b, 0)
            mx = max(mx, min(c, b))
            # 拆开
            # if c > b:
            #     lose += c - b
            #     mx = b if b > mx else mx
            # else:
            #     mx = c if c > mx else mx
        return lose + mx

    def minimumMoney(self, transactions: List[List[int]]) -> int:
        """或者写成这样"""
        ans = lose = 0
        for c, b in transactions:
            if c > b:
                lose += c - b
        for c, b in transactions:
            if c > b:
                lose -= c - b
            ans = max(ans, lose + c)
            if c > b:
                lose += c - b
        return ans

    def minimumMoney(self, transactions: List[List[int]]) -> int:
        """
        哪个环节会导致交易失败? 只可能是需要 cost 的代价, 但当前金钱不足 cost 才会失败
        最差情况下, 进行交易 T 之前, 我们会首先完成其它所有负收益交易, 这样才会让我们的当前金钱变得最少
        初始至少需要 cost(T) - sum(负收益之和) 的金钱. 最终答案就是该式的最大值
        """
        ans = neg = 0
        for c, b in transactions:
            if c > b:
                neg += b - c
        for c, b in transactions:
            diff = b - c
            if diff < 0:
                ans = max(ans, c - (neg - diff))
            else:
                ans = max(ans, c - neg)
        return ans

    def minimumMoney(self, transactions: List[List[int]]) -> int:
        """
        summ 维护最大扣钱和, 然后枚举每一笔交易作为最少钱时购买的
        x > y: 拿的就是扣钱中的一个, 把减去的 y 补回来 / 反之继续拿新的, 不在原有和之中, 还得扣新的 x
        """
        ans = summ = 0
        for x, y in transactions:
            if x > y:
                summ += x - y
        for x, y in transactions:
            if x > y:
                ans = max(ans, y + summ)
            else:
                ans = max(ans, summ + x)
        return ans

    # 使用贪心排序的比较函数很难想
    #   1. 亏钱的排前面
    #   2. 亏钱的中, cashback 最大的排最后
    #   3. 赚钱的中, cost 最大的排最前
    # 不过问题的本质不是排序:
    #   其实只要保证亏钱部分 cashback 最大的放最后, 赚钱部分只把 cost 最大的那个放到最前就行了, 其他交易的顺序无关紧要

    # TODO 1665


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


# 2418 - Sort the People - EASY
class Solution:
    def sortPeople(self, names: List[str], heights: List[int]) -> List[str]:
        return [name for _, name in sorted(zip(heights, names), reverse=True)]


# 2419 - Longest Subarray With Maximum Bitwise AND - MEDIUM
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        mx = max(nums)
        ans = pre = 0
        for v in nums:
            if v == pre == mx:
                cur += 1
            elif v == mx:
                cur = 1
            else:
                cur = 0
            pre = v
            ans = max(ans, cur)
        return ans

    def longestSubarray(self, nums: List[int]) -> int:
        mx = max(nums)
        ans = cnt = 0
        for v in nums:
            if v == mx:
                cnt += 1
                ans = max(ans, cnt)
            else:
                cnt = 0
        return ans


# 2420 - Find All Good Indices - MEDIUM
class Solution:
    # f[i] 表示以第 i 个下标为结尾的非递增连续子数组最长是多少
    # g[i] 表示以第 i 个下标为开头的非递减连续子数组最长是多少
    def goodIndices(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        f = [1] * n
        g = [1] * n
        for i in range(1, n):
            if nums[i - 1] >= nums[i]:
                f[i] = f[i - 1] + 1
        for i in range(n - 1)[::-1]:
            if nums[i] <= nums[i + 1]:
                g[i] = g[i + 1] + 1
        ans = []
        for i in range(k, n - k):
            if f[i - 1] >= k and g[i + 1] >= k:
                ans.append(i)
        return ans


# 2421 - Number of Good Paths - HARD
class Solution:
    # 暴力, 从大到小考虑, 删除大节点, O(n^2)
    # 倒着考虑, 删除变合并 -> 并查集
    # 连通块对应最大值个数
    # O(nlogn) / O(n)
    def numberOfGoodPaths(self, vals: List[int], edges: List[List[int]]) -> int:
        n = len(vals)
        g = [[] for _ in range(n)]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)

        fa = list(range(n))
        # cnt[x] 表示在 x 所处连通块内, 节点值等于 vals[x] 的节点个数 -> 如果按照节点值从小到大合并, cnt[x] 也是连通块内的等于最大节点值的节点个数
        cnt = [1] * n

        def find(x: int) -> int:
            if fa[x] != x:
                fa[x] = find(fa[x])
            return fa[x]

        ans = n
        for v, x in sorted(zip(vals, range(n))):
            fx = find(x)
            for y in g[x]:
                fy = find(y)
                if fy == fx or vals[fy] > v:  # 后续再合并
                    continue  # 先只考虑最大节点值比 v 小的连通块
                if vals[fy] == v:  # 可以构成好路径
                    ans += cnt[fx] * cnt[fy]  # 乘法原理
                    cnt[fx] += cnt[fy]  # 统计连通块内节点值等于 v 的节点个数
                fa[fy] = fx  # 把小的节点值合并到大的节点值上
        return ans

    # 路径上的所有点权都小等于路径两端的点权 -> 先把点按点权从小到大排序, 然后依次加入树中, 这样加入过程中的所有路径都符合该条件
    # 考虑两个连通块合并时的情况, 当一条边合并两个连通块 x 和 y 时, x 里点权为 v 的节点都可以从这条边走到 y 里点权为 v 的节点
    # 因此两个连通块 x 和 y 合并的时候答案会增加 d[x] * d[y], 用并查集维护连通块的合并即可
    # O(nlogn) / O(n)
    def numberOfGoodPaths(self, vals: List[int], edges: List[List[int]]) -> int:
        n = len(vals)
        g = [[] for _ in range(n)]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)

        fa = list(range(n))

        def find(x: int) -> int:
            if fa[x] != x:
                fa[x] = find(fa[x])
            return fa[x]

        # d[x] 表示连通块 x 中点权等于 v 的点有几个
        d = collections.defaultdict(int)
        ans = j = 0
        arr = sorted(zip(vals, range(n)))
        vis = [False] * n
        for i in range(n):
            if arr[i][0] != arr[j][0]:  # 当前枚举的点权有变, 清空 map
                d.clear()
                j = i
            x = arr[i][1]
            vis[x] = True  # 将 x 加入树中
            # 一开始没有边和 x 连接, 它所处的连通块只有一个点(它本身)的点权等于 v
            d[x] = 1
            for y in g[x]:
                if vis[y]:
                    fx = find(x)
                    fy = find(y)
                    if fx == fy:
                        continue
                    ans += d[fx] * d[fy]
                    fa[fx] = fy
                    d[fy] += d[fx]
        return ans + n

    def numberOfGoodPaths(self, vals: List[int], edges: List[List[int]]) -> int:
        n = len(vals)
        g = [[] for _ in range(n)]
        for x, y in edges:
            # 保证后续合并时, 权值小的节点的 fa 指向权值大的节点
            if (vals[x], x) >= (vals[y], y):
                g[x].append(y)
            else:
                g[y].append(x)
        fa = list(range(n))
        cnt = [1] * n

        def find(x):
            if fa[x] != x:
                fa[x] = find(fa[x])
            return fa[x]

        ans = 0
        for v, x in sorted(zip(vals, range(n))):
            for y in g[x]:
                fx = find(x)
                fy = find(y)
                if vals[fy] == v:
                    ans += cnt[fx] * cnt[fy]
                    cnt[fy] += cnt[fx]
                else:
                    vals[fy] = v
                    cnt[fy] = cnt[fx]
                fa[fx] = fy
        return ans + n


# 2427 - Number of Common Factors - EASY
class Solution:
    # O(min(a, b)) / O(1)
    def commonFactors(self, a: int, b: int) -> int:
        return sum(a % i == b % i == 0 for i in range(1, min(a, b) + 1))

    # O(gcd(a, b)) / O(1)
    def commonFactors(self, a: int, b: int) -> int:
        g = math.gcd(a, b)
        return sum(a % i == b % i == 0 for i in range(1, g + 1))

    # O(sqrt(gcd(a, b))) / O(1), 枚举 a 和 b 的最大公因数 g 的因子
    def commonFactors(self, a: int, b: int) -> int:
        g = math.gcd(a, b)
        ans = 0
        i = 1
        while i * i <= g:
            if g % i == 0:
                ans += 1
                if i * i < g:
                    ans += 1
            i += 1
        return ans

    # O(sqrt(min(a, b))) / O(1)
    def commonFactors(self, a: int, b: int) -> int:
        g = math.gcd(a, b)
        # 1 + True = 2, 但是 1 + i * i < g 需要注意运算优先级
        return sum(1 + (i * i < g) for i in range(1, int(g**0.5) + 1) if g % i == 0)


# 2428 - Maximum Sum of an Hourglass - MEDIUM
class Solution:
    # O(mn) / O(1)
    def maxSum(self, grid: List[List[int]]) -> int:
        return max(
            grid[i][j]
            + grid[i - 1][j - 1]
            + grid[i - 1][j]
            + grid[i - 1][j + 1]
            + grid[i + 1][j - 1]
            + grid[i + 1][j]
            + grid[i + 1][j + 1]
            for i in range(1, len(grid) - 1)
            for j in range(1, len(grid[0]) - 1)
        )


# 2429 - Minimize XOR - MEDIUM
class Solution:
    def minimizeXor(self, num1: int, num2: int) -> int:
        k1 = bin(num1).count("1")
        k2 = bin(num2).count("1")
        if k1 > k2:
            d = k2
            x = 0
            for j in range(30, -1, -1):
                if num1 & 1 << j != 0:
                    x |= 1 << j
                    d -= 1
                if d == 0:
                    break
            return x
        elif k1 < k2:
            d = k2 - k1
            x = num1
            for j in range(0, 31):
                if num1 & 1 << j == 0:
                    x |= 1 << j
                    d -= 1
                if d == 0:
                    break
            return x
        else:
            return num1

    def minimizeXor(self, num1: int, num2: int) -> int:
        k1 = bin(num1).count("1")
        k2 = bin(num2).count("1")
        if k1 < k2:
            for _ in range(k2 - k1):  # 最低的 0 变成 1
                num1 |= num1 + 1
        elif k1 > k2:
            for _ in range(k1 - k2):  # 最低的 1 变成 0, 这里要求的是 x, 不是异或和
                num1 &= num1 - 1
        return num1

    def minimizeXor(self, num1: int, num2: int) -> int:
        k1 = bin(num1).count("1")
        k2 = bin(num2).count("1")
        while k1 < k2:
            num1 |= num1 + 1
            k2 -= 1
        while k1 > k2:
            num1 &= num1 - 1
            k2 += 1
        return num1

    # 贪心, 一开始按从最高位到最低位的顺序, 把 num1 所有的 1 都反转, 而 0 不变
    # 如果还有剩余反转次数, 按从最低位到最高位的顺序, 反转 num1 中未被反转的位(此时把 0 变成 1)
    def minimizeXor(self, num1: int, num2: int) -> int:
        a = [False] * 31  # num1 的二进制表示
        for i in range(30, -1, -1):
            a[i] = bool(num1 & 1 << i)
        used = [False] * 31
        ans = cnt = 0
        # cnt = bin(num2).count("1")
        # cnt = sum(num2 & (1 << i) > 0 for i in range(31))
        while num2:
            cnt += num2 & 1
            num2 >>= 1
        for i in range(30, -1, -1):
            if cnt == 0:
                break
            if a[i]:
                used[i] = True
                cnt -= 1
                ans |= 1 << i
        for i in range(31):
            if cnt == 0:
                break
            if not used[i]:
                cnt -= 1
                ans |= 1 << i
        return ans


# 2430 - Maximum Deletions on a String - HARD
class Solution:
    # 每次操作结束, 还剩一个完整字符(子)串 -> 子问题 -> dp
    # f[i]: 操作 s[i:] 需要的最大操作次数
    # ans = f[0]
    # f[i] = f[i + j] + 1 if s[i: i + j] == s[i + j: i + 2 * j]
    # f[i] = 1
    # 取 max
    # 问题转换为如何快速判断两个字串是否相同

    # lcp, 两个后缀的最长公共前缀
    # lcp[i][j] = s[i:] 和 s[j:] 的最长公共前缀
    # s[i: i + j] = = s[i + j: i + 2 * j] 等价于 lcp[i][i + j] >= j
    # lcp[i][j] = lcp[i + 1][j + 1] + 1 if s[i] == s[j] else 0

    # O(n^2) / O(n^2)
    def deleteString(self, s: str) -> int:
        n = len(s)
        if len(set(s)) == 1:
            return n
        # lcp[i][j] 表示 s[i:] 和 s[j:] 的最长公共前缀
        lcp = [[0] * (n + 1) for _ in range(n + 1)]

        for i in range(n - 1, -1, -1):
            for j in range(n - 1, i, -1):
                if s[i] == s[j]:
                    lcp[i][j] = lcp[i + 1][j + 1] + 1

        # 慢一点
        # for j in range(n - 1, -1, -1):
        #     for i in range(j - 1, -1, -1):
        #         if s[i] == s[j]:
        #             lcp[i][j] = lcp[i + 1][j + 1] + 1

        f = [0] * n
        for i in range(n - 1, -1, -1):
            for j in range(1, (n - i) // 2 + 1):  # i + 2 * j <= n
                if lcp[i][i + j] >= j:  # 说明 s[i : i + j] == s[i + j : i + 2 * j]
                    f[i] = max(f[i], f[i + j])
            f[i] += 1

        # for i in range(n - 1, -1, -1):
        #     f[i] = 1
        #     for j in range(1, (n - i) // 2 + 1):
        #         if lcp[i][i + j] >= j:
        #             f[i] = max(f[i], f[i + j] + 1)

        # for i in range(n - 1, -1, -1):
        #     f[i] = 1
        #     for j in range(i + 1, n):
        #         if lcp[i][j] >= j - i:
        #             f[i] = max(f[i], f[j] + 1)

        return f[0]

    def deleteString(self, s: str) -> int:
        n = len(s)
        if len(set(s)) == 1:
            return n
        lcp = [[0] * (n + 1) for _ in range(n + 1)]
        f = [1] * n
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    lcp[i][j] = lcp[i + 1][j + 1] + 1
                if lcp[i][j] >= j - i:
                    f[i] = max(f[i], f[j] + 1)
        return f[0]

    # 字符串哈希, 调用 getHash() py TLE, :(
    def deleteString(self, s: str) -> int:
        n = len(s)
        if len(set(s)) == 1:
            return n
        base = 131  # 哈希指数, 是一个经验值, 可以取 1331 等等
        mod = 998244353
        p = [0] * 4001
        h = [0] * 4001
        p[0] = 1
        for i in range(1, n + 1):
            p[i] = (p[i - 1] * base) % mod
            h[i] = (h[i - 1] * base + ord(s[i - 1])) % mod

        def getHash(l: int, r: int) -> int:
            return (h[r] - h[l - 1] * p[r - l + 1]) % mod

        f = [0] * (n + 1)
        for i in range(n, -1, -1):
            f[i] = 1
            for j in range(1, (n - i) // 2 + 1):  # j = span = length
                h1 = (h[i + j] - h[i] * p[j]) % mod
                h2 = (h[i + j * 2] - h[i + j] * p[j]) % mod
                if h1 == h2:
                    f[i] = max(f[i], f[i + j] + 1)

                # TLE
                # if getHash(i, i + j - 1) == getHash(i + j, i + j * 2 - 1):
                #     f[i] = max(f[i], f[i + j] + 1)

        return f[0]

    # O(n ** 3) / O(n), 切片比较快
    def deleteString(self, s: str) -> int:
        @functools.lru_cache(None)
        def dfs(s: str, i: int) -> int:
            if i == len(s):
                return 0
            t = span = 1
            while i + span * 2 <= len(s):
                if s[i : i + span] == s[i + span : i + span * 2]:
                    t = max(t, 1 + dfs(s, i + span))
                span += 1
            return t

        return dfs(s, 0)

    def deleteString(self, s: str) -> int:
        @functools.lru_cache(None)
        def dfs(p: int) -> int:
            t = 1
            for i in range(1, (len(s) - p) // 2 + 1):
                if s[p : p + i] == s[p + i : p + i * 2]:
                    t = max(t, dfs(p + i) + 1)
            return t

        return dfs(0)


# 2432 - The Employee That Worked on the Longest Task - EASY
class Solution:
    def hardestWorker(self, n: int, logs: List[List[int]]) -> int:
        p = mx = 0
        for i, t in logs:
            if t - p > mx:
                ans = i
                mx = t - p
            elif t - p == mx:
                ans = min(ans, i)
            p = t
        return ans


# 2433 - Find The Original Array of Prefix Xor - MEDIUM
class Solution:
    def findArray(self, pref: List[int]) -> List[int]:
        ans = [pref[0]]
        for i in range(1, len(pref)):
            ans.append(pref[i] ^ pref[i - 1])
        return ans

    def findArray(self, pref: List[int]) -> List[int]:
        return [pref[0]] + [x ^ y for x, y in pairwise(pref)]


# 2434 - Using a Robot to Print the Lexicographically Smallest String - MEDIUM
class Solution:
    # O(n + 26) / O(n + 26)
    def robotWithString(self, s: str) -> str:
        cnt = collections.Counter(s)
        st = []
        ans = []
        i = 97
        while cnt[chr(i)] == 0:
            i += 1
        for c in s:
            if cnt[chr(i)] != 0 and c != chr(i):
                st.append(c)
                cnt[c] -= 1
                continue
            cnt[chr(i)] -= 1
            ans.append(c)
            while cnt[chr(i)] == 0 and i <= ord("z"):
                i += 1
                while st and ord(st[-1]) <= i:
                    ans.append(st.pop())
        while st:
            ans.append(st.pop())
        return "".join(ans)

    def robotWithString(self, s: str) -> str:
        ans = []
        cnt = collections.Counter(s)
        i = 0
        st = []
        for c in s:
            cnt[c] -= 1
            while i < 25 and cnt[chr(i + 97)] == 0:
                i += 1
            st.append(c)
            while st and st[-1] <= chr(i + 97):
                ans.append(st.pop())
        return "".join(ans)


# 2435 - Paths in Matrix Whose Sum Is Divisible by K - HARD
class Solution:
    def numberOfPaths(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        mod = 10**9 + 7
        f = [[[0] * k for _ in range(n)] for _ in range(m)]
        f[0][0][grid[0][0] % k] = 1
        for i in range(m):
            for j in range(n):
                for h in range(k):
                    f[i][j][h] %= mod
                    if i + 1 < m:
                        f[i + 1][j][(h + grid[i + 1][j]) % k] += f[i][j][h]
                    if j + 1 < n:
                        f[i][j + 1][(h + grid[i][j + 1]) % k] += f[i][j][h]
        return f[m - 1][n - 1][0] % mod

    def numberOfPaths(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        mod = 10**9 + 7
        f = [[[0] * k for _ in range(n)] for _ in range(m)]
        x = 0
        for i in range(m):
            x = (x + grid[i][0]) % k
            f[i][0][x] = 1
        x = 0
        for j in range(n):
            x = (x + grid[0][j]) % k
            f[0][j][x] = 1
        for i in range(1, m):
            for j in range(1, n):
                for p in range(k):
                    x = (p - grid[i][j]) % k
                    f[i][j][p] = (f[i - 1][j][x] + f[i][j - 1][x]) % mod
        return f[-1][-1][0]

    def numberOfPaths(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        mod = 10**9 + 7
        f = [[[0] * k for _ in range(n + 1)] for _ in range(m + 1)]

        # f[0][1][0] = 1 / f[1][0][0], 为什么这样初始化:
        # 因为是从 f[1][1][] 开始算的, f[0][1][0] 和 f[1][0][0] 必须有一个是 1
        f[0][1][0] = 1

        for i in range(m):
            for j in range(n):
                for v in range(k):
                    f[i + 1][j + 1][(v + grid[i][j]) % k] = (
                        f[i + 1][j][v] + f[i][j + 1][v]
                    ) % mod

        return f[-1][-1][0]

    def numberOfPaths(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        mod = 10**9 + 7

        @functools.lru_cache(None)
        def dfs(i: int, j: int, v: int) -> int:
            if i >= m or j >= n:
                return 0
            v = (v + grid[i][j]) % k
            if i == m - 1 and j == n - 1 and v == 0:
                return 1

            return (dfs(i + 1, j, v) % mod + dfs(i, j + 1, v) % mod) % mod

        ans = dfs(0, 0, 0)
        dfs.cache_clear()
        return ans

    def numberOfPaths(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        mod = 10**9 + 7

        # 不需要 dfs.cache_clear()
        @functools.lru_cache(None)
        def dfs(i: int, j: int) -> List[int]:
            g = grid[i][j]
            ans = [0] * k
            if i == 0 == j:
                ans[g % k] = 1
                return ans
            if i:
                for v, r in enumerate(dfs(i - 1, j)):
                    v = (v + g) % k
                    ans[v] = (ans[v] + r) % mod
            if j:
                for v, r in enumerate(dfs(i, j - 1)):
                    v = (v + g) % k
                    ans[v] = (ans[v] + r) % mod
            return ans

        return dfs(m - 1, n - 1)[0]


# 2437 - Number of Valid Clock Times - EASY
class Solution:
    def countTime(self, time: str) -> int:
        l = []
        for i in range(24):
            for j in range(60):
                l.append("%02d:%02d" % (i, j))
                # l.append(f"{i:02d}:{j:02d}")
                # l.append("{:02d}:{:02d}".format(i, j))
        ans = 0
        for s in l:
            for i in range(5):
                if time[i] != "?" and time[i] != s[i]:
                    break
            else:
                ans += 1
        return ans

    def countTime(self, time: str) -> int:
        def count(time: str, limit: int) -> int:
            ans = 0
            for i in range(limit):
                if all(t == "?" or t == c for t, c in zip(time, f"{i:02d}")):
                    ans += 1
            return ans

        return count(time[:2], 24) * count(time[3:], 60)


# 2438 - Range Product Queries of Powers - MEDIUM
class Solution:
    # O(logn + q) / O(logn)
    def productQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        a = []
        while n:
            # a.append(n & -n)
            # n &= n - 1

            lb = n & -n
            a.append(lb)
            n ^= lb

        # a = []
        # for i in range(31):
        #     if n & 1 << i:
        #         a.append(2**i)

        p = [1]
        for v in a:
            p.append(p[-1] * v)
        mod = 10**9 + 7
        ans = []
        for l, r in queries:
            ans.append((p[r + 1] // p[l]) % mod)
        return ans


# 2439 - Minimize Maximum of Array - MEDIUM
class Solution:
    # 贪心 + 单调栈, 操作只会使 左边减小 右边增大, 栈内一定是单调不增的
    # O(n) / O(n)
    def minimizeArrayValue(self, nums: List[int]) -> int:
        st = []  # (val, count)
        for v in nums:
            if st:
                presum = pren = 0
                # 新加入的值比较大, 可以往左均摊
                while st and st[-1][0] <= (presum + v) // (pren + 1):
                    val, n = st.pop()
                    presum += val * n
                    pren += n
                # sum 总可以拆分成 a * n 或者 (a + 1) * x + a * (n - x) 的形式
                d, m = divmod(presum + v, pren + 1)
                if m != 0:
                    st.append((d + 1, m))
                    st.append((d, pren + 1 - m))
                else:
                    st.append((d, pren + 1))
            else:
                st.append((v, 1))
        # 返回栈内最大元素值
        return st[0][0]

    # 讨论如下:
    # 从 nums[0] 开始
    # 如果数组中只有 nums[0], 那么最大值为 nums[0]
    # nums[1], 如果 nums[1] <= nums[0], 最大值还是 nums[0], 否则可以平均这两个数, 平均后的最大值为平均值的向上取整
    # 再考虑 nums[2], 如果 nums[2] <= 之前计算出的最大值 -> 这三个数的平均值不超过前面算出的最大值, 那么最大值不变, 否则可以平均这三个数, 做法同上,
    # 以此类推直到最后一个数, 过程中的最大值为答案
    # O(n) / O(1)
    def minimizeArrayValue(self, nums: List[int]) -> int:
        # 不同考虑除 0 的情况了
        return max(
            (summ + i) // (i + 1) for i, summ in enumerate(itertools.accumulate(nums))
        )
        # return max(
        #     # (summ + i - 1) // i  # 向上取整
        #     (summ - 1) // i + 1  # 向上取整
        #     for i, summ in enumerate(itertools.accumulate(nums), start=1)
        # )

    # O(logU) / O(1), U = max(nums), 最小化最大值 -> 二分
    def minimizeArrayValue(self, nums: List[int]) -> int:
        def check(limit: int) -> bool:
            extra = 0
            for i in range(len(nums) - 1, 0, -1):
                extra = max(nums[i] + extra - limit, 0)
            return nums[0] + extra <= limit

        def check(x: int) -> bool:
            # 前方的较小数可以接受后方较大数多余的数字
            summ = 0
            for v in nums:
                if v - x > summ:
                    return False
                summ += x - v
            return True

        # l = 0
        # r = max(nums)
        # while l < r:
        #     m = (l + r) // 2
        #     if check(m):
        #         r = m
        #     else:
        #         l = m + 1
        # return l

        return bisect.bisect_left(range(max(nums)), True, key=check)


# 2440 - Create Components With Same Value - HARD
class Solution:
    # 价值是 sum(nums) 的因子
    # -> 1e6 有多少因子?
    # -> 240 (https://oeis.org/A066150), 或者利用 n**(1/3) 估算
    # -> 枚举因子, dfs
    # 统计子树大小 / 统计子树点权和
    # O(n * sqrt(sum(nums))) / O(n)
    def componentValue(self, nums: List[int], edges: List[List[int]]) -> int:
        g = [[] for _ in nums]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)

        def dfs(x: int, fa: int) -> int:
            s = nums[x]
            for y in g[x]:
                if y != fa:
                    r = dfs(y, x)
                    if r < 0:
                        return -1
                    s += r
            if s > target:
                return -1
            if s == target:
                return 0
            return s

        summ = sum(nums)
        # for i in range(summ, 0, -1): # 优化枚举起点
        for i in range(summ // max(nums), 0, -1):
            if summ % i == 0:
                target = summ // i
                if dfs(0, -1) == 0:
                    return i - 1
        return 0


# 2441 - Largest Positive Integer That Exists With Its Negative - EASY
class Solution:
    def findMaxK(self, nums: List[int]) -> int:
        s = set(nums)
        ans = -1
        for v in nums:
            if -v in s:
                ans = max(v, ans)
        return ans

    def findMaxK(self, nums: List[int]) -> int:
        s = set()
        ans = -1
        for v in nums:
            if -v in s:
                ans = max(ans, abs(v))
            s.add(v)
        return ans


# 2442 - Count Number of Distinct Integers After Reverse Operations - MEDIUM
class Solution:
    def countDistinctIntegers(self, nums: List[int]) -> int:
        s = set(nums)
        for v in nums:
            s.add(int(str(v)[::-1]))
        return len(s)

    def countDistinctIntegers(self, nums: List[int]) -> int:
        return len(set(nums) | set(int(str(v)[::-1]) for v in nums))

    def countDistinctIntegers(self, nums: List[int]) -> int:
        s = set()
        for v in nums:
            s.add(v)
            r = 0
            while v:
                r = r * 10 + v % 10
                v //= 10
            s.add(r)
        return len(s)


# 2443 - Sum of Number and Its Reverse - MEDIUM
class Solution:
    def sumOfNumberAndReverse(self, num: int) -> bool:
        for v in range(num, num // 2 - 1, -1):
            if v + int(str(v)[::-1]) == num:
                return True
        return False

    def sumOfNumberAndReverse(self, num: int) -> bool:
        return any(v + int(str(v)[::-1]) == num for v in range(num, num // 2 - 1, -1))

    def sumOfNumberAndReverse(self, num: int) -> bool:
        for v in range(num, num // 2 - 1, -1):
            r = 0
            x = v
            while x:
                r = r * 10 + x % 10
                x //= 10
            if v + r == num:
                return True
        return False


# 2444 - Count Subarrays With Fixed Bounds - HARD
class Solution:
    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
        ans = cmin = cmax = j = last = 0
        for i, v in enumerate(nums):
            if v < minK or v > maxK:
                j = last = i + 1
                cmin = cmax = 0
                continue
            if v == minK:
                cmin += 1
            if v == maxK:
                cmax += 1
            while j <= i:
                if nums[j] == minK:
                    cmin -= 1
                if nums[j] == maxK:
                    cmax -= 1
                if not cmin or not cmax:
                    if nums[j] == minK:
                        cmin += 1
                    if nums[j] == maxK:
                        cmax += 1
                    break
                j += 1
            if cmin and cmax:
                ans += j - last + 1
        return ans

    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
        ans = 0
        mii = mxi = pre = -1
        for i, v in enumerate(nums):
            if v < minK or v > maxK:
                mii = mxi = -1
                pre = i
            else:
                if v == minK:
                    mii = i
                if v == maxK:
                    mxi = i
                if mii != -1 and mxi != -1:
                    ans += min(mii, mxi) - pre
                # ans += max(0, min(mii, mxi) - pre)
        return ans

    # 枚举以 i 为右端点的定界子数组数量
    # 分别维护 maxK 和 minK 从 i 往左看, 第一次出现的位置 x, y
    # 表示左端点必须在 min⁡(x, y) 及其左侧, 否则子数组中会缺少 maxK 或 minK
    # 以 i 为右边界的子数组数量(如果存在) = min(x, y) - l
    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
        ans = 0
        mii = mxi = pre = -1
        for i, v in enumerate(nums):
            if not minK <= v <= maxK:
                pre = i
            if v == minK:
                mii = i
            if v == maxK:
                mxi = i
            ans += max(0, min(mii, mxi) - pre)
        return ans

    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
        ans = l = 0
        mii = mxi = -1
        for r, v in enumerate(nums):
            if v == minK:
                mii = r
            if v == maxK:
                mxi = r
            if v < minK or v > maxK:
                l = r + 1
            ans += max(0, min(mii, mxi) - l + 1)
        return ans

    # TODO
    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
        def ask(x: int, y: int) -> int:
            s = j = 0
            for v in nums:
                if x <= v <= y:
                    j += 1
                else:
                    j = 0
                s += j
            return s

        return (
            ask(minK, maxK)
            - ask(minK + 1, maxK)
            - ask(minK, maxK - 1)
            + ask(minK + 1, maxK - 1)
        )


# 2451 - Odd String Difference - EASY
class Solution:
    # O(mn) / O(m + n)
    def oddString(self, words: List[str]) -> str:
        n = len(words[0])
        d = collections.defaultdict(list)
        for w in words:
            diff = [0] * (n - 1)
            for j in range(n - 1):
                diff[j] = ord(w[j + 1]) - ord(w[j])
            d[tuple(diff)].append(w)
        x, y = d.values()
        return x[0] if len(x) == 1 else y[0]

    def oddString(self, words: List[str]) -> str:
        d = collections.defaultdict(list)
        for w in words:
            d[tuple(ord(x) - ord(y) for x, y in pairwise(w))].append(w)
        x, y = d.values()
        return x[0] if len(x) == 1 else y[0]


# 2452 - Words Within Two Edits of Dictionary - MEDIUM
class Solution:
    # O(qdn) / O(1), q = len(queries), d = len(dictionary), n = len(queries[i])
    def twoEditWords(self, queries: List[str], dictionary: List[str]) -> List[str]:
        ans = []
        for q in queries:
            for d in dictionary:
                diff = 0
                for x, y in zip(q, d):
                    if x != y:
                        diff += 1
                    if diff > 2:
                        break
                if diff <= 2:
                    ans.append(q)
                    break
        return ans

    def twoEditWords(self, queries: List[str], dictionary: List[str]) -> List[str]:
        ans = []
        for q in queries:
            for d in dictionary:
                if sum(x != y for x, y in zip(q, d)) <= 2:
                    ans.append(q)
                    break
        return ans

    # 每次对 queries[i] 做搜索不如把所有可能转移到 dictionary[i] 的情况找出来
    # 所以依次遍历可能转移到 dictionary[i] 的情况, 加入 set; 遍历 queries, 看是否在 set 中
    # 分析时间复杂度:
    # 最坏情况, dictionary[i] 有两个字符不同的备选
    # for w in dictionary:                                  O(d) 100
    #     for i in range(n):                                O(n) 100
    #         for a in string.ascii_lowercase:              O(26) 26
    #             new = w[:i] + a + w[i + 1:]               O(n) 100
    #             for j in range(i + 1, n):                 O(n) 100
    #                 for b in string.ascii_lowercase:      O(26) 26
    #                     neww = w[:j] + b + w[j + 1:]      O(n) 100
    #                     st.add(neww)                      total: 100**5 * 26**2 = 6.76e12
    # 即便不考虑切片复杂度, 去掉两个 100, 也会达到 6.76e8, 不一定能过


# 2453 - Destroy Sequential Targets - MEDIUM
class Solution:
    # O(n) / O(n)
    def destroyTargets(self, nums: List[int], space: int) -> int:
        cnt = collections.defaultdict(list)
        for v in nums:
            cnt[v % space].append(v)
        mx = ans = 0
        for vals in cnt.values():
            l = len(vals)
            v = min(vals)
            if l > mx or l == mx and v < ans:
                mx = l
                ans = v
        return ans

    def destroyTargets(self, nums: List[int], space: int) -> int:
        cnt = collections.defaultdict(int)
        for v in nums:
            cnt[v % space] += 1
        ans = nums[0]
        mx = cnt[nums[0] % space]
        for v in nums:
            t = cnt[v % space]
            if t > mx or t == mx and v < ans:
                mx = t
                ans = v
        return ans


# 2454 - Next Greater Element IV - HARD
class Solution:
    # s 单调递减栈, 如果 v 比 s 的栈顶大, 说明 v 是下个更大元素
    # 弹出 s 的栈顶, 加入 t, 如果 v 比 t 的栈顶大, 说明 v 是 t 栈顶的第二大元素
    # 注意:
    #     1. 先检查 t
    #     2. s 的栈顶移动到 t 时, 保持原始顺序 (直接 pop s 的栈顶移入 t 的话, t 就不是单调递减了)

    # O(n) / O(n)
    def secondGreaterElement(self, nums: List[int]) -> List[int]:
        s = []
        t = []
        ans = [-1] * len(nums)
        for i, v in enumerate(nums):
            while t and nums[t[-1]] < v:
                ans[t.pop()] = v
            dq = collections.deque()
            while s and nums[s[-1]] < v:
                dq.appendleft(s.pop())
            t.extend(dq)
            s.append(i)
        return ans

    def secondGreaterElement(self, nums: List[int]) -> List[int]:
        s = []
        t = []
        ans = [-1] * len(nums)
        for i, v in enumerate(nums):
            while t and nums[t[-1]] < v:
                ans[t.pop()] = v
            j = len(s) - 1
            while j >= 0 and nums[s[j]] < v:
                j -= 1
            t += s[j + 1 :]
            del s[j + 1 :]
            s.append(i)
        return ans

    # O(nlogK) / O(n)
    def secondGreaterElement(self, nums: List[int]) -> List[int]:
        s = []
        q = []
        ans = [-1] * len(nums)
        for i, v in enumerate(nums):
            while q and q[0][0] < v:
                ans[heapq.heappop(q)[1]] = v
            while s and nums[s[-1]] < v:
                x = s.pop()
                heapq.heappush(q, (nums[x], x))
            s.append(i)
        return ans

    # 名次树, 从上往下遍历, 找右边的第 k 个数 (解决第 k 大问题)
    def secondGreaterElement(self, nums: List[int]) -> List[int]:
        ans = [-1] * len(nums)
        k = 2  # 第 k 大
        s = sortedcontainers.SortedList()  # O(nlogn)
        # 先值排序, 保证名次, 再下标排序, 为了值重复时让其他数字先查到左边的值
        for _, i in sorted((-v, i) for i, v in enumerate(nums)):
            j = s.bisect_right(i) + k - 1  # 更大的元素 -> right / upper_bound
            if j < len(s):
                ans[i] = nums[s[j]]
            s.add(i)
        return ans


# 2455 - Average Value of Even Numbers That Are Divisible by Three - EASY
class Solution:
    def averageValue(self, nums: List[int]) -> int:
        t = s = 0
        for v in nums:
            if v % 6 == 0:
                t += 1
                s += v
        return s // t if t else 0


# 2456 - Most Popular Video Creator - MEDIUM
class Solution:
    def mostPopularCreator(
        self, creators: List[str], ids: List[str], views: List[int]
    ) -> List[List[str]]:
        d = dict()  # key: c, val: [sum, max, id]
        for c, i, v in zip(creators, ids, views):
            if c in d:
                summ, mx, j = d[c]
                summ += v
                if v > mx or v == mx and i < j:
                    mx = v
                    j = i
                d[c] = (summ, mx, j)
            else:
                d[c] = (v, v, i)
        ans = []
        mx = 0
        for c, (summ, _, i) in d.items():
            if summ > mx:
                mx = summ
                ans = [[c, i]]
            elif summ == mx:
                ans.append([c, i])
        return ans

    def mostPopularCreator(
        self, creators: List[str], ids: List[str], views: List[int]
    ) -> List[List[str]]:
        d = {}  # key: c, val: [sum, max, id]
        mx = 0
        for c, i, v in zip(creators, ids, views):
            if c in d:
                d[c][0] += v
                if v > d[c][1] or v == d[c][1] and i < d[c][2]:
                    d[c][1] = v
                    d[c][2] = i
            else:
                d[c] = [v, v, i]
            mx = max(mx, d[c][0])
        return [[c, i] for c, (summ, _, i) in d.items() if summ == mx]


# 2457 - Minimum Addition to Make Integer Beautiful - MEDIUM
class Solution:
    # 贪心, 从低位到高位, 把每一次逐次置 0 进位, 数位和才会变小, 注意进位后次高位会 +1

    # O(logn) / O(1)
    def makeIntegerBeautiful(self, n: int, target: int) -> int:
        arr = list(int(x) for x in str(n))[::-1]
        s = sum(arr)
        ans = 0
        i = 0
        while s > target:
            ans += (10 - arr[i]) * (10**i)
            if i + 1 < len(arr):
                arr[i + 1] += 1  # 进位, 下一位加 1
            else:
                arr.append(1)  # 或者没有下一位了, 补一个 1
            s = s - arr[i] + 1
            arr[i] = 0  # 进位, 该位变 0
            i += 1
        return ans

    # O((logn)**2) / O(1)
    def makeIntegerBeautiful(self, n: int, target: int) -> int:
        p = 1
        while True:
            x = y = n + (p - n % p) % p  # 进位后的数字, 再模一次是避免余数为 0 时发生进位
            s = 0
            while y:
                s += y % 10
                y //= 10
            if s <= target:
                return x - n
            p *= 10

    def makeIntegerBeautiful(self, n: int, target: int) -> int:
        n0 = n
        i = 0
        while sum(map(int, str(n))) > target:
            n = n // 10 + 1
            i += 1
        return n * (10**i) - n0


# 2458 - Height of Binary Tree After Subtree Removal Queries - HARD
class Solution:
    # 比赛做法:
    # 1. postorder traversal
    #    a. 求每个点对应的树高(层数), 从第 0 层开始
    #    b. 求每个节点下, 最大深度
    #    c. 求节点对应的父节点
    # 2. 对于每层节点, 因为同层节点上面高度一定, 所以根据节点下面深度排序
    # 3. a. 砍掉该节点, 该层还有其他节点, 则贪心地从排序后的同层结果中寻找, 从后往前遍历(有更大深度), 返回 高度 + 深度
    #    b. 该层只有一个节点, 则从上一层找 / 特殊情况: 上一层最大深度对应的子节点就是该节点 -> 所以需要知道父节点信息
    # O(nlogn + q) / O(n)
    def treeQueries(self, root: Optional[TreeNode], queries: List[int]) -> List[int]:
        level = collections.defaultdict(list)  # key: 树高, val: [(树深, 节点值) ... (树深, 节点值)]
        node2lv = dict()  # 节点对应树高
        parent = dict()  # 节点对应父节点

        def dfs(root: TreeNode, cur: int) -> int:
            if not root:
                return 0
            if root.left:
                parent[root.left.val] = root.val
            if root.right:
                parent[root.right.val] = root.val
            l = dfs(root.left, cur + 1)
            r = dfs(root.right, cur + 1)
            node2lv[root.val] = cur
            mx = max(l, r)
            level[cur].append((mx, root.val))
            return mx + 1

        dfs(root, 0)
        lv = []
        i = 0
        while i in level:
            lv.append(sorted(level[i]))
            i += 1
        ans = []
        for q in queries:
            l = node2lv[q]
            if len(lv[l]) == 1:
                i = len(lv[l - 1]) - 1
                arr = lv[l - 1]
                r = 0
                while i >= 0:
                    # 如果最大树深的节点正好是该节点的父节点, 则子树被砍掉后, 只有树高的贡献, 没有树深的贡献, 还得比较倒二的大小
                    if arr[i][1] == parent[q]:
                        r = l - 1
                        i -= 1
                        continue
                    else:
                        r = max(r, arr[i][0] + l - 1)
                        break
                    i -= 1
                ans.append(r)
            else:
                i = len(lv[l]) - 1
                arr = lv[l]
                while i >= 0:
                    if arr[i][1] != q:
                        ans.append(arr[i][0] + l)
                        break
                    i -= 1
        return ans

    # O(n + q) / O(n)
    def treeQueries(self, root: Optional[TreeNode], queries: List[int]) -> List[int]:
        height = collections.defaultdict(int)  # 每棵子树的高度, 叶子节点树高是 0

        def getHeight(root: TreeNode) -> int:
            if not root:
                return 0
            height[root] = 1 + max(getHeight(root.left), getHeight(root.right))
            return height[root]

        getHeight(root)
        ans = [0] * (len(height) + 1)  # 每个节点的答案

        # restH: 删除当前子树后剩余部分的树的高度
        def dfs(root: TreeNode, depth: int, restH: int) -> None:
            if not root:
                return
            depth += 1
            ans[root.val] = restH
            dfs(root.left, depth, max(restH, depth + height[root.right]))
            dfs(root.right, depth, max(restH, depth + height[root.left]))
            return

        dfs(root, -1, 0)
        return [ans[q] for q in queries]

    # 层序遍历, 将每层的节点最长路径(最大树高)的两个存储起来 (a, b), 查询的时候与同层的这两个值进行比较
    # O(n + q) / O(n)
    def treeQueries(self, root: Optional[TreeNode], queries: List[int]) -> List[int]:
        height = collections.defaultdict(int)  # 每棵子树的高度, 叶子节点树高是 0

        def getHeight(root: TreeNode) -> int:
            if not root:
                return 0
            height[root] = 1 + max(getHeight(root.left), getHeight(root.right))
            return height[root]

        getHeight(root)

        ans = [0] * (len(height) + 1)
        d = -1
        q = collections.deque([root])
        while q:
            # 求最大的两个树高, a >= b
            a = b = 0
            for o in q:
                if height[o] > a:
                    b = a
                    a = height[o]
                elif height[o] > b:
                    b = height[o]
            for _ in range(len(q)):
                o = q.popleft()
                if o.left:
                    q.append(o.left)
                if o.right:
                    q.append(o.right)
                ans[o.val] = d + (a if height[o] != a else b)
            d += 1
        return [ans[i] for i in queries]

    # DFS 序 + 前后缀
    # 树上时间戳, 进入和离开时间
    # 将树通过 DFS 序转成序列, 子树里的所有点是 DFS 序里的一个连续区间 / 同一棵子树节点所对应的 DFS 序是连续的一段区间
    # 问题转化为 -> 给定一个序列, 每次删除一个连续区间, 求序列里剩下的数的最大值, 显然删除一个连续区间后, 序列会剩下一个前缀以及一个后缀

    # 首先求 DFS 序, 并求出每个节点所对应的管辖区间(子树区间)
    # 同时记录 DFS 序为 i 的节点的深度 depth[i], (i 即是时间戳 time), 通过 depth 数组, 继续求出:
    # DFS 序 <= i 的节点中, 深度最大的节点的深度 goin[i]   (前缀 max 数组)
    # DFS 序 >= i 的节点中, 深度最大的节点的深度 goout[in] (后缀 max 数组)
    # O(n + q) / O(n)
    def treeQueries(self, root: Optional[TreeNode], queries: List[int]) -> List[int]:
        goin = collections.defaultdict(int)
        goout = collections.defaultdict(int)
        depth = collections.defaultdict(int)
        time = 0

        def dfs(root: TreeNode, d: int) -> None:
            if not root:
                return
            nonlocal time
            time += 1  # node 是第 clk 个被访问的点
            depth[time] = d  # 表示第 i 个被访问的点的深度
            goin[root.val] = time  # goin[i] 表示第 i 个点的子树对应的连续区间的左端点
            dfs(root.left, d + 1)
            dfs(root.right, d + 1)
            goout[root.val] = time  # goout[i] 表示第 i 个点的子树对应的连续区间的右端点
            return

        dfs(root, 0)  # root 深度为 0
        n = len(depth)
        f = [0] * (n + 2)  # f[i] 表示 max(depth[1], depth[2], ..., depth[i])
        g = [0] * (n + 2)  # g[i] 表示 max(depth[n], depth[n - 1], ..., depth[i])
        for i in range(1, n + 1):
            f[i] = max(f[i - 1], depth[i])
        for i in range(n, 0, -1):
            g[i] = max(g[i + 1], depth[i])
        # 树上询问转为区间询问处理
        return [max(f[goin[q] - 1], g[goout[q] + 1]) for q in queries]

    # TODO: https://leetcode.com/problems/height-of-binary-tree-after-subtree-removal-queries/discuss/2759353/C%2B%2BPython-Preoder-and-Postorder-DFS
    def treeQueries(self, root: Optional[TreeNode], queries: List[int]) -> List[int]:
        ans = collections.defaultdict(int)

        def dfs(root: TreeNode, h: int, maxh: int) -> int:
            if not root:
                return maxh
            ans[root.val] = max(ans[root.val], maxh)
            root.left, root.right = root.right, root.left
            return dfs(root.right, h + 1, dfs(root.left, h + 1, max(maxh, h)))

        dfs(root, 0, 0)
        dfs(root, 0, 0)
        return [ans[q] for q in queries]


# 2460 - Apply Operations to an Array - EASY
class Solution:
    # O(n) / O(n)
    def applyOperations(self, nums: List[int]) -> List[int]:
        for i in range(len(nums) - 1):
            if nums[i] == nums[i + 1]:
                nums[i] *= 2
                nums[i + 1] = 0
        return [v for v in nums if v] + [0] * sum(v == 0 for v in nums)

    # O(n) / O(1)
    def applyOperations(self, nums: List[int]) -> List[int]:
        j = 0
        for i in range(len(nums) - 1):
            if nums[i]:
                if nums[i] == nums[i + 1]:
                    nums[i] *= 2
                    nums[i + 1] = 0
                nums[j] = nums[i]  # 非零数字排在前面
                j += 1
        if nums[-1]:
            nums[j] = nums[-1]
            j += 1
        for i in range(j, len(nums)):
            nums[i] = 0
        return nums


# 2461 - Maximum Sum of Distinct Subarrays With Length K - MEDIUM
class Solution:
    # O(n) / O(k)
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        ans = l = summ = 0
        s = set()
        for r, v in enumerate(nums):
            while v in s or r - l == k:
                s.remove(nums[l])
                summ -= nums[l]
                l += 1
            s.add(v)
            summ += v
            if len(s) == k:
                ans = max(ans, summ)
        return ans

    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        ans = 0
        cnt = collections.Counter(nums[: k - 1])
        summ = sum(nums[: k - 1])
        for a, b in zip(nums[k - 1 :], nums):
            cnt[a] += 1
            summ += a
            if len(cnt) == k:
                ans = max(ans, summ)

            cnt[b] -= 1
            if cnt[b] == 0:
                del cnt[b]
            summ -= b
        return ans


# 2462 - Total Cost to Hire K Workers - MEDIUM
class Solution:
    def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
        if len(costs) < 2 * candidates:
            return sum(sorted(costs)[:k])
        ans = 0
        pre = costs[:candidates]
        heapq.heapify(pre)
        suf = costs[-candidates:]
        heapq.heapify(suf)
        l = candidates
        r = len(costs) - 1 - candidates
        while k and l <= r:
            if pre[0] <= suf[0]:
                ans += heapq.heapreplace(pre, costs[l])
                l += 1
            else:
                ans += heapq.heapreplace(suf, costs[r])
                r -= 1
            k -= 1
        return sum(sorted(pre + suf)[:k]) + ans

    def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
        ans = 0
        if candidates * 2 < len(costs):
            pre = costs[:candidates]
            heapq.heapify(pre)
            suf = costs[-candidates:]
            heapq.heapify(suf)
            l = candidates
            r = len(costs) - 1 - candidates
            while k and l <= r:
                if pre[0] <= suf[0]:
                    ans += heapq.heapreplace(pre, costs[l])
                    l += 1
                else:
                    ans += heapq.heapreplace(suf, costs[r])
                    r -= 1
                k -= 1
            costs = pre + suf
        costs.sort()
        return ans + sum(costs[:k])


# 2463 - Minimum Total Distance Traveled - HARD
class Solution:
    # 时间复杂度 = O(状态个数) * O(单个状态的转移次数) * O(计算转移来源是谁)
    #          = O(nm) * O(m) * O(1)
    # O(n * m * m) / O(nm)
    def minimumTotalDistance(self, robot: List[int], factory: List[List[int]]) -> int:
        factory.sort()
        robot.sort()
        n = len(factory)
        m = len(robot)
        # [j, n - j] 个工厂修理 [i, m - 1] 个机器人
        @functools.lru_cache(None)
        def dfs(i: int, j: int) -> int:
            if j == m:
                return 0
            if i == n - 1:
                if m - j > factory[i][1]:
                    return math.inf
                return sum(abs(x - factory[i][0]) for x in robot[j:])
            r = dfs(i + 1, j)
            s = 0
            k = 1
            while k <= factory[i][1] and j + k - 1 < m:
                s += abs(robot[j + k - 1] - factory[i][0])
                r = min(r, s + dfs(i + 1, j + k))
                k += 1
            return r

        return dfs(0, 0)

    def minimumTotalDistance(self, robot: List[int], factory: List[List[int]]) -> int:
        robot.sort()
        factory.sort()

        @functools.lru_cache(None)
        def dfs(i: int, j: int) -> int:
            if i == len(robot):
                return 0
            if j == len(factory):
                return math.inf
            r = 0
            ans = dfs(i, j + 1)
            for k in range(factory[j][1]):
                if i + k == len(robot):
                    break
                r += abs(robot[i + k] - factory[j][0])
                ans = min(ans, r + dfs(i + k + 1, j + 1))
            return ans

        ans = dfs(0, 0)
        dfs.cache_clear()
        return ans

    def minimumTotalDistance(self, robot: List[int], factory: List[List[int]]) -> int:
        robot = sorted(robot)
        factory = sorted(factory)

        @functools.lru_cache(None)
        def dfs(i: int, j: int, k: int) -> int:
            if i == len(robot):
                return 0
            if j == len(factory):
                return math.inf
            if k == factory[j][1]:
                return dfs(i, j + 1, 0)
            r = abs(robot[i] - factory[j][0]) + dfs(i + 1, j, k + 1)
            r = min(r, dfs(i, j + 1, 0))
            return r

        return dfs(0, 0, 0)

    def minimumTotalDistance(self, robot: List[int], factory: List[List[int]]) -> int:
        robot.sort()
        factory.sort()
        # to fix robot[i] and its following roberts with factory[j] already fix k robert
        @functools.lru_cache(None)
        def dfs(i: int, j: int, k: int) -> int:
            if i == len(robot):
                return 0
            if j == len(factory):
                return math.inf
            res1 = dfs(i, j + 1, 0)
            res2 = (
                dfs(i + 1, j, k + 1) + abs(robot[i] - factory[j][0])
                if factory[j][1] > k
                else math.inf
            )
            return min(res1, res2)

        return dfs(0, 0, 0)

    def minimumTotalDistance(self, robot: List[int], factory: List[List[int]]) -> int:
        factory.sort()
        robot.sort()
        m = len(robot)
        # 前 i 个工厂, 修理前 j 个机器人
        # f[i][j] = f[i - 1][j]
        #         = f[i - 1][j - k] + cost(i, j, k)
        #         的最小值
        # cost(i, j, k) 表示第 i 个工厂, 修理从 j 往前的 k 个机器人, 移动距离之和,
        # 枚举 k, 取 min(), 0 <= k <= min(j, limit[i])
        f = [0] + [math.inf] * m
        for p, limit in factory:
            for j in range(m, 0, -1):
                cost = 0
                for k in range(1, min(j, limit) + 1):
                    cost += abs(robot[j - k] - p)
                    f[j] = min(f[j], f[j - k] + cost)
        return f[m]

    def minimumTotalDistance(self, robot: List[int], factory: List[List[int]]) -> int:
        robot.sort()
        factory.sort()
        n = len(robot)
        m = len(factory)
        # f[i][j] 表示已经送走了前 i 个机器人, 且第 i 个机器人送去工厂 j 的最小总距离
        # g[i][j] 是 f[i][j] 的前缀 min, 辅助计算 f[i][j], g[i][j] = min(f[i][k] for k in range(j + 1))
        f = [[math.inf] * (m + 1) for _ in range(n + 1)]
        g = [[math.inf] * (m + 1) for _ in range(n + 1)]
        f[0][0] = 0
        for j in range(m + 1):
            g[0][j] = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d = 0
                k = i - 1
                while 0 <= k and i - k <= factory[j - 1][1]:
                    d += abs(robot[k] - factory[j - 1][0])
                    f[i][j] = min(f[i][j], g[k][j - 1] + d)
                    k -= 1
            for j in range(1, m + 1):
                g[i][j] = min(g[i][j - 1], f[i][j])
        return g[n][m]

    def minimumTotalDistance(self, robot: List[int], factory: List[List[int]]) -> int:
        robot.sort()
        factory.sort()
        n = len(robot)
        m = len(factory)
        f = [math.inf] * (n + 1)
        f[n] = 0
        for j in range(m - 1, -1, -1):
            for i in range(n):
                cur = 0
                for k in range(1, min(factory[j][1], n - i) + 1):
                    cur += abs(robot[i + k - 1] - factory[j][0])
                    f[i] = min(f[i], f[i + k] + cur)
        return f[0]


# 2465 - Number of Distinct Averages - EASY
class Solution:
    def distinctAverages(self, nums: List[int]) -> int:
        nums.sort()
        # no need to divide by 2
        return len(set(nums[i] + nums[-i - 1] for i in range(len(nums) // 2)))


# 2466 - Count Ways To Build Good Strings - MEDIUM
class Solution:
    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        @functools.lru_cache
        def f(n: int):
            return (
                0 if n < 0 else 1 if n == 0 else (f(n - zero) + f(n - one)) % 1000000007
            )

        return sum(f(v) for v in range(low, high + 1)) % 1000000007

    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        mod = 10**9 + 7
        f = [0] * (high + 1)
        f[0] = 1
        for i in range(high + 1 - min(zero, one)):
            if i + zero <= high:
                f[i + zero] = (f[i] + f[i + zero]) % mod
            if i + one <= high:
                f[i + one] = (f[i] + f[i + one]) % mod
        return sum(f[low:]) % mod

    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        mod = 10**9 + 7
        f = [1] + [0] * high
        ans = 0
        for i in range(1, high + 1):
            if i >= one:
                f[i] = (f[i] + f[i - one]) % mod
            if i >= zero:
                f[i] = (f[i] + f[i - zero]) % mod
            if i >= low:
                ans = (ans + f[i]) % mod
        return ans


# 2469 - Convert the Temperature - EASY
class Solution:
    def convertTemperature(self, celsius: float) -> List[float]:
        return [celsius + 273.15, celsius * 1.8 + 32]


# 2490 - Circular Sentence - EASY
class Solution:
    def isCircularSentence(self, sentence: str) -> bool:
        if sentence[-1] != sentence[0]:
            return False
        s = sentence.split()
        for i in range(len(s) - 1):
            if s[i][-1] != s[i + 1][0]:
                return False
        return True

    def isCircularSentence(self, sentence: str) -> bool:
        return sentence[0] == sentence[-1] and all(
            sentence[i - 1] == sentence[i + 1]
            for i, c in enumerate(sentence)
            if c == " "
        )


# 2491 - Divide Players Into Teams of Equal Skill - MEDIUM
class Solution:
    # O(nlogn) / O(1)
    def dividePlayers(self, skill: List[int]) -> int:
        skill.sort()
        ans = 0
        s = skill[0] + skill[-1]
        for i in range(len(skill) // 2):
            x = skill[i]
            y = skill[-1 - i]
            if x + y != s:
                return -1
            ans += x * y
        return ans

    # O(n) / O(n)
    def dividePlayers(self, skill: List[int]) -> int:
        n = len(skill)
        summ = sum(skill)
        if summ % (n // 2) != 0:
            return -1
        t = summ // (n // 2)
        ans = 0
        cnt = collections.Counter(skill)
        for k, v in cnt.items():
            if v != cnt[t - k]:
                return -1
            ans += v * k * (t - k)
        return ans // 2


# 2492 - Minimum Score of a Path Between Two Cities - MEDIUM
class Solution:
    # O(m + n) / O(m + n)
    def minScore(self, n: int, roads: List[List[int]]) -> int:
        g = [[] for _ in range(n)]
        for x, y, d in roads:
            g[x - 1].append((y - 1, d))
            g[y - 1].append((x - 1, d))
        ans = math.inf
        vis = [False] * n

        def dfs(x: int) -> None:
            nonlocal ans
            vis[x] = True
            for y, d in g[x]:
                ans = min(ans, d)
                if not vis[y]:
                    dfs(y)
            return

        dfs(0)
        return ans

    def minScore(self, n: int, roads: List[List[int]]) -> int:
        g = collections.defaultdict(list)
        for x, y, _ in roads:
            g[x].append(y)
            g[y].append(x)
        vis = set()
        q = [1]
        while q:
            new = []
            for x in q:
                for y in g[x]:
                    if y not in vis:
                        new.append(y)
                        vis.add(y)
            q = new
        ans = math.inf
        for x, y, w in roads:
            if x in vis and y in vis:
                ans = min(ans, w)
        return ans

    def minScore(self, n: int, roads: List[List[int]]) -> int:
        p = list(range(n + 1))
        w = [math.inf] * (n + 1)  # 集合对应最小元素

        def find(x: int) -> int:
            """path compression"""
            if p[x] != x:
                p[x] = find(p[x])
            return p[x]

        for x, y, v in roads:
            a = find(x)
            b = find(y)
            w[a] = min(w[a], w[b], v)
            p[b] = a
        return w[find(1)]


# 2493 - Divide Nodes Into the Maximum Number of Groups - HARD
class Solution:
    # | y - x | = 1
    # 数据范围比较小, 枚举每个点作为起点, BFS, 求出最大编号(最大深度)
    # 每个连通块的最大深度相加, 即为答案
    # O(mn) / O(m + n), m = len(edges)
    def magnificentSets(self, n: int, edges: List[List[int]]) -> int:
        g = [[] for _ in range(n)]
        for x, y in edges:
            g[x - 1].append(y - 1)
            g[y - 1].append(x - 1)
        time = [0] * n
        clock = 0

        def bfs(start: int) -> int:
            depth = 0
            nonlocal clock
            clock += 1
            time[start] = clock
            q = [start]
            while q:
                tmp = q
                q = []
                for x in tmp:
                    for y in g[x]:
                        if time[y] != clock:
                            time[y] = clock
                            q.append(y)
                depth += 1
            return depth

        color = [0] * n

        def is_bipartite(x: int, c: int) -> bool:
            nodes.append(x)
            color[x] = c
            for y in g[x]:
                if color[y] == c or color[y] == 0 and not is_bipartite(y, -c):
                    return False
            return True

        ans = 0
        for i, c in enumerate(color):
            if c:
                continue
            nodes = []
            if not is_bipartite(i, 1):
                return -1  # 不是二分图(有奇环)
            ans += max(bfs(x) for x in nodes)  # 枚举连通块的每个点, 作为起点 BFS, 求最大深度
        return ans

    def magnificentSets(self, n: int, edges: List[List[int]]) -> int:
        class UnionFind:
            def __init__(self, n: int) -> None:
                self.p = [i for i in range(n)]

            def find(self, x: int) -> int:
                """path compression"""
                if self.p[x] != x:
                    self.p[x] = self.find(self.p[x])
                return self.p[x]

            def union(self, x: int, y: int) -> None:
                """x's root -> y"""
                px = self.find(x)
                py = self.find(y)
                if px == py:
                    return
                self.p[px] = py
                return

        g = collections.defaultdict(list)
        uf = UnionFind(n + 1)
        for x, y in edges:
            uf.union(x, y)
            g[x].append(y)
            g[y].append(x)

        # 二分图判断
        note = [-1] * (n + 1)
        for i in range(1, n + 1):
            if note[i] == -1:
                note[i] = 0
                q = collections.deque([i])
                while q:
                    x = q.popleft()
                    for y in g[x]:
                        if note[y] == -1:
                            note[y] = 1 - note[x]
                            q.append(y)
                        elif note[y] == note[x]:
                            return -1  # 不满足二分图条件

        largest_diameter = collections.defaultdict(int)
        for i in range(1, n + 1):
            q = collections.deque([i])
            vis = {i}
            cnt = 0
            while q:
                cnt += 1
                for _ in range(len(q)):
                    x = q.popleft()
                    for y in g[x]:
                        if y not in vis:
                            vis.add(y)
                            q.append(y)
            largest_diameter[uf.find(i)] = max(largest_diameter[uf.find(i)], cnt)
        return sum(largest_diameter.values())


# 2496 - Maximum Value of a String in an Array - EASY
class Solution:
    def maximumValue(self, strs: List[str]) -> int:
        ans = 0
        for w in strs:
            try:
                ans = max(ans, int(w))
            except:
                ans = max(ans, len(w))
        return ans

    def maximumValue(self, strs: List[str]) -> int:
        ans = 0
        for w in strs:
            if any(c.isalpha() for c in w):
                ans = max(ans, len(w))
            else:
                ans = max(ans, int(w))
        return ans


# 2497 - Maximum Star Sum of a Graph - MEDIUM
class Solution:
    def maxStarSum(self, vals: List[int], edges: List[List[int]], k: int) -> int:
        g = [[] for _ in range(len(vals))]
        # 如果直接连边, 不考虑值的情况, 后续就会变得复杂
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)
        ans = -math.inf
        for x, arr in enumerate(g):
            arr.sort(key=lambda x: vals[x])
            t = vals[x]
            t += sum(vals[y] for y in arr[-k:] if vals[y] > 0)
            ans = max(ans, t)
        return ans

    def maxStarSum(self, vals: List[int], edges: List[List[int]], k: int) -> int:
        g = [[] for _ in range(len(vals))]
        for x, y in edges:
            if vals[y] > 0:
                g[x].append(vals[y])
            if vals[x] > 0:
                g[y].append(vals[x])
        return max(sum(heapq.nlargest(k, g[i])) + v for i, v in enumerate(vals))
        return max(v + sum(heapq.nlargest(k, arr)) for v, arr in zip(vals, g))


# 2498 - Frog Jump II - MEDIUM
class Solution:
    # 二分
    def maxJump(self, stones: List[int]) -> int:
        def check(m: int) -> bool:
            can = [False] * (n + 1)
            can[0] = True
            for i in range(1, n + 1):
                if stones[i] <= m:
                    can[i] = True
                else:
                    break
            for i in range(1, n + 1):
                for j in range(i - 1, 0, -1):
                    if can[i]:
                        break
                    if stones[i - 1] - stones[j - 1] > m:
                        break
                    if i - j >= 2:
                        can[i] = can[i] or can[j]
                if not can[i]:
                    break
            return can[-1]

        def check(m: int) -> bool:
            used = set()
            j = 0
            for i in range(1, n):
                if stones[i] - stones[j] > m:
                    if i - 1 == j:
                        return False
                    used.add(i - 1)
                    j = i - 1
            j = 0
            for i in range(1, n):
                if i in used:
                    continue
                if stones[i] - stones[j] > m:
                    return False
                j = i
            return True

        l = 0
        r = stones[-1]
        n = len(stones)
        while l < r:
            m = (l + r) // 2
            if check(m):
                r = m
            else:
                l = m + 1
        return l

    # 1. 转换成两个青蛙不相交跳到终点
    # 2. 如果有空的石头, 则跳到空石头不会使答案更差 -> 所有石头会被使用
    # 3. 全部使用的话, 交替跳是最好的
    # 对于 A B C D ... 中, A B 是两个起点, C D 是两个落点, A -> C, B -> D 比 A -> D, B -> C 更优
    def maxJump(self, stones: List[int]) -> int:
        ans = stones[1] - stones[0]
        for i in range(2, len(stones)):
            ans = max(ans, stones[i] - stones[i - 2])
        return ans


# 2499 - Minimum Total Cost to Make Arrays Unequal - HARD
class Solution:
    # 统计 same = nums1[i] == nums2[i] 个数, 以及 same 的众数 mode, same 是必要交换的
    # 1. 众数 mode 没有超过 same 的一半
    #   a. same 是偶数, 两两交换
    #   b. same 是奇数, mode 最大是 same // 2, 所以 same 至少会有三类, 多出来的数可以和 nums[0] 交换
    #      比如 11223 或者 31122, 众数交换之后, 多出来的和 0 位置的数字换一下
    # 2. mode 超过 same 的一半
    #    需要外部数字, 满足 x != y and x != mode and y != mode 的数字来消化多余的众数
    #    未消化完则 return -1

    # same 是奇数, 和 nums[0] 交换比较难想
    # O(n) / O(n)
    def minimumTotalCost(self, nums1: List[int], nums2: List[int]) -> int:
        ans = same = modeCnt = mode = 0
        d = collections.defaultdict(int)
        for i, (x, y) in enumerate(zip(nums1, nums2)):
            if x == y:
                ans += i
                same += 1
                d[x] += 1
                if d[x] > modeCnt:
                    modeCnt = d[x]
                    mode = x
        for i, (x, y) in enumerate(zip(nums1, nums2)):
            if modeCnt * 2 > same and x != y and x != mode and y != mode:
                ans += i
                same += 1
        return ans if modeCnt * 2 <= same else -1
