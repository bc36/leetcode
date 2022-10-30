import bisect, collections, functools, math, itertools, heapq, string, operator
from typing import List, Optional
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
                    ors[k][1] = p[1]  # 合并相同值，下标取最小的
                else:
                    k += 1
                    ors[k] = p
            del ors[k + 1 :]
            # 本题只用到了 ors[0]，如果题目改成任意给定数值，可以在 ors 中查找
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
