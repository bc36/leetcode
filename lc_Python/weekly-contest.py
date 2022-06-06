from typing import List
import collections, functools, itertools, heapq, math


########################
# 274 / 3道 / 2022.1.1 #
########################
# 5969
# https://leetcode-cn.com/problems/destroying-asteroids/
# https://leetcode-cn.com/contest/weekly-contest-274/problems/destroying-asteroids/
class Solution:
    def asteroidsDestroyed(self, mass: int, asteroids: List[int]) -> bool:
        asteroids.sort()
        for asteroid in asteroids:
            if mass < asteroid:
                return False
            mass += asteroid
        return True


# 5970
# https://leetcode-cn.com/problems/maximum-employees-to-be-invited-to-a-meeting/
# https://leetcode-cn.com/contest/weekly-contest-274/problems/maximum-employees-to-be-invited-to-a-meeting/
class Solution:
    def maximumInvitations(self, f: List[int]) -> int:
        # functional graph
        # either table consists of one cycle
        # or multiple (two chains + cycle of two)
        # bidirectional edges
        edges = collections.defaultdict(list)
        n = len(f)
        for i in range(n):
            edges[f[i]].append(i)
            edges[i].append(f[i])
        seen = [False] * n
        self.l = 0

        def dfs(v, p, d):
            self.l = max(self.l, d)
            seen[v] = True
            for i in edges[v]:
                if i != p and not seen[i]:
                    dfs(i, v, d + 1)

        total = 0
        best = 0
        for i in range(n):
            if seen[i]:
                continue
            if f[f[i]] == i:  # cycle of two
                self.l = 0
                dfs(i, f[i], 0)
                cur = self.l
                self.l = 0
                dfs(f[i], i, 0)
                cur += self.l
                total += cur + 2
        for i in range(n):
            if not seen[i]:
                dfs(i, i, 0)
                l = [i]
                j = i
                cur = 0
                s = set()
                s.add(i)
                while True:
                    j = f[j]
                    if j in s:
                        while True:
                            cur += 1
                            if l.pop() == j:
                                break
                        best = max(best, cur)
                        break
                    s.add(j)
                    l.append(j)
        return max(total, best)

    def maximumInvitations(self, favorite: List[int]) -> int:
        # there must be one or more cycle in this graph
        # each connected region in the graph will lead to 1 cycle
        # Either
        #  all cycles of size 2 fulfilled, each node can have 1 edge
        #  one cycle fulfilled, nothing else
        visited = [False] * len(favorite)
        depth = [None] * len(favorite)
        represent = [None] * len(favorite)
        cycles = []

        def dfs(n, d=0):
            if visited[n]:
                if represent[n] is None:
                    # found cycle
                    represent[n] = n
                    cycles.append((n, d - depth[n]))
                    return n
                else:
                    # revisit
                    return represent[n]
            visited[n] = True
            depth[n] = d
            ans = dfs(favorite[n], d + 1)
            represent[n] = ans
            return ans

        for i in range(len(favorite)):
            dfs(i)
        ans = 0
        for i, j in cycles:
            ans = max(ans, j)
        rev = []
        for i in favorite:
            rev.append([])
        for index, i in enumerate(favorite):
            rev[i].append(index)

        def michi(x, nashi):
            answer = 0
            for i in rev[x]:
                if i == nashi:
                    continue
                answer = max(answer, michi(i, nashi) + 1)
            return answer

        ans2 = 0
        for i0, j in cycles:
            if j != 2:
                continue
            i1 = favorite[i0]
            assert i0 == favorite[i1]
            ans2 += 2 + michi(i0, i1) + michi(i1, i0)
        return max(ans, ans2)

    def maximumInvitations(self, favorite: List[int]) -> int:
        n = len(favorite)
        f = [0] * n
        e = [set() for i in range(n)]
        num = n + 1
        for i in range(n):
            e[favorite[i]].add(i)

        def findquan(x, num):
            if f[x] == num:
                return
            f[x] = num
            findquan(favorite[x], num)

        def find(x, y):
            if f[x] == y:
                nonlocal num
                findquan(x, num)
                num += 1
                return
            if f[x] != 0:
                return

            f[x] = y
            if f[favorite[x]] == 0 or f[favorite[x]] == y:
                find(favorite[x], y)

        def finddeep(x):
            d = 0
            nonlocal n
            for i in e[x]:
                if f[i] <= n:
                    d = max(d, finddeep(i))
            return d + 1

        for i in range(n):
            if f[i] == 0:
                find(i, i + 1)
        ans = 0
        ans2 = 0
        done = set()
        for i in range(n):
            if f[i] > n and f[i] not in done:
                done.add(f[i])
                p = i
                num = 1
                while favorite[p] != i:
                    num += 1
                    p = favorite[p]
                if num > 2:
                    ans = max(ans, num)
                if num == 2:
                    ans2 += finddeep(p) + finddeep(i)
        return max(ans, ans2)

    def maximumInvitations(self, A: List[int]) -> int:
        res = 0
        pairs = []
        d = collections.defaultdict(list)
        for i, a in enumerate(A):
            if A[a] == i:
                pairs.append(i)
            else:
                d[a].append(i)
        # print(pairs, d)
        @functools.lru_cache(None)
        def cal(x):
            if d[x] == []:
                return 1
            mx = 0
            for c in d[x]:
                mx = max(mx, cal(c))
            return mx + 1

        for a in pairs:
            # print(a, cal(a))
            res += cal(a)
        # print(res)

        def check(i):
            dd = {}
            cur = i
            while True:
                dd[cur] = len(dd)
                cur = A[cur]
                if cur in d:
                    for k in dd:
                        d[k] = -1
                    break
                if cur in dd:
                    length = len(dd) - dd[cur]
                    for k, v in dd.items():
                        if v < len(dd) - length:
                            d[k] = -1
                        else:
                            d[k] = length
                    break

        d = {}
        for i, a in enumerate(A):
            if i in d:
                continue
            else:
                check(i)
        return max(res, max(d.values()))

    def maximumInvitations(self, f: List[int]) -> int:
        def dfs(cur):
            res = 0
            mark[cur] = True
            for nxt in e[cur]:
                if nxt != f[cur]:
                    res = max(res, dfs(nxt))
            res += 1
            return res

        def dfs2(cur, root):
            nonlocal res
            nonlocal counter
            nxt = f[cur]
            if rank[nxt] >= rank[root]:
                res = max(res, depth[cur] + 1 - depth[nxt])
            elif rank[nxt] < 0:
                rank[nxt] = counter
                counter += 1
                depth[nxt] = depth[cur] + 1
                dfs2(nxt, root)

        n = len(f)
        e = [set() for _ in range(n)]
        for i in range(n):
            e[f[i]].add(i)
        q = []
        for i in range(n):
            if f[f[i]] == i:
                q.append(i)
        mark = [False] * n
        res = 0
        for root in q:
            res += dfs(root)
        depth = [-1] * n
        rank = [-1] * n
        counter = 0
        for i in range(n):
            if not mark[i]:
                depth[i] = 0
                rank[i] = counter
                counter += 1
                dfs2(i, i)
        return res


########################
# 275 / 1道 / 2022.1.8 #
########################
# https://leetcode-cn.com/problems/minimum-swaps-to-group-all-1s-together-ii/
# https://leetcode.com/problems/minimum-swaps-to-group-all-1s-together-ii/
# 5977. 最少交换次数来组合所有的 1 II
class Solution:
    # 滑动窗口, '1'的总数的连续段滑动, 找包含最多'1'的段
    def minSwaps(self, nums: List[int]) -> int:
        ones, n = nums.count(1), len(nums)
        x, onesInWindow = 0, 0
        for i in range(n * 2):
            if i >= ones and nums[i % n - ones]:
                x -= 1
            if nums[i % n] == 1:
                x += 1
            onesInWindow = max(x, onesInWindow)
        return ones - onesInWindow

    # 找包含最少'0'的段
    def minSwaps(self, nums: List[int]) -> int:
        a = sum(nums)  # '1'的个数
        t = sum(nums[:a])  # 窗口大小
        r = a - t
        for i in range(len(nums)):
            t -= nums[i]
            i = (i + a) % len(nums)
            t += nums[i]
            r = min(r, a - t)
        return r

    # 前缀和
    def minSwaps(self, nums: List[int]) -> int:
        cnt = nums.count(1)
        n = len(nums)
        t = [0]
        res = 0
        for i in range(n):
            t.append(t[-1] + nums[i])
        for i in range(cnt, n + 1):
            d = t[i] - t[i - cnt]
            res = max(res, d)
        for i in range(cnt):
            d = t[i] + t[n] - t[n - (cnt - i)]
            res = max(res, d)
        return cnt - res


# https://leetcode-cn.com/problems/count-words-obtained-after-adding-a-letter/
# https://leetcode.com/problems/count-words-obtained-after-adding-a-letter/
# 5978. 统计追加字母可以获得的单词数
class Solution:
    # 暴力, 字符排序后加入set, 不要直接append到list, 数量太大超时
    def wordCount(self, sw: List[str], tw: List[str]) -> int:
        dic = collections.defaultdict(set)
        for i in range(len(sw)):
            dic[len(sw[i])].add("".join(sorted(sw[i])))
        ans = 0
        for i in range(len(tw)):
            s = "".join(sorted(tw[i]))
            for j in range(len(s)):
                if s[:j] + s[j + 1 :] in dic[len(s) - 1]:
                    ans += 1
                    break
        return ans

    # frozenset
    def wordCount(self, startWords: List[str], targetWords: List[str]) -> int:
        ss = set(frozenset(w) for w in startWords)
        res = 0
        for w in targetWords:
            s = set(w)
            for c in w:
                if s.difference(c) in ss:
                    res += 1
                    break
        return res

    # 用二进制第i位数表示第i个小写字母, 每个单词算出来一个固定的数字
    def wordCount(self, startWords: List[str], targetWords: List[str]) -> int:
        s = set()

        def word2int(s):
            n = 0
            for ch in s:
                n |= 1 << (ord(ch) - ord("a"))
            return n

        for i in startWords:
            s.add(word2int(i))
        ans = 0
        for w in targetWords:
            n = word2int(w)
            for ch in w:
                if n ^ word2int(ch) in s:  # XOR 异或去掉这个字符
                    ans += 1
                    break
        return ans


# https://leetcode-cn.com/problems/earliest-possible-day-of-full-bloom/
# https://leetcode.com/problems/earliest-possible-day-of-full-bloom/
# 5979. 全部开花的最早一天
# 长的慢的先种, 播种的时间不能避免, 直接加
class Solution:
    def earliestFullBloom(self, plantTime: List[int], growTime: List[int]) -> int:
        data = list(zip(plantTime, growTime))
        data.sort(key=lambda x: -x[1])  # sort by grow time in descending order
        # data.sort(key=lambda x: x[1], reverse=True)
        ans = 0
        allPlantTime = 0
        for plant, grow in data:
            allPlantTime += plant
            ans = max(ans, allPlantTime + grow)
        return ans

    def earliestFullBloom(self, plantTime: List[int], growTime: List[int]) -> int:
        heap = []
        for i in range(0, len(plantTime)):
            heapq.heappush(heap, (-growTime[i], plantTime[i]))
        ans = 0
        cur_time = 0
        while heap:
            g, p = heapq.heappop(heap)
            cur_time += p
            ans = max(ans, cur_time - g)
        return ans


########################
# 276 / 2道 / 2022.1.6 #
########################
# https://leetcode-cn.com/problems/solving-questions-with-brainpower/
# https://leetcode.com/problems/solving-questions-with-brainpower/
# 5982. 解决智力问题
# 10^5 数据规模 O(n^2)明显不行, 因为dp[i]的值取决于dp[i]数组后面值，所以要倒推
class Solution:
    def mostPoints(self, questions: List[List[int]]) -> int:
        @functools.lru_cache(None)
        def solve(t=0):
            if t >= len(questions):
                return 0
            points, brainpower = questions[t]
            return max(points + solve(t + brainpower + 1), solve(t + 1))

        return solve()

    def mostPoints(self, q: List[List[int]]) -> int:
        n = len(q)
        r = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            r[i] = q[i][0]
            if i + q[i][1] + 1 < n:
                r[i] += r[i + q[i][1] + 1]
            r[i] = max(r[i], r[i + 1])
        return r[0]

    def mostPoints(self, questions: List[List[int]]) -> int:
        a, m = [0] * len(questions), 0
        for i in range(len(questions) - 1, -1, -1):
            a[i] = max(
                m,
                questions[i][0]
                + (
                    0
                    if i + questions[i][1] + 1 >= len(questions)
                    else a[i + questions[i][1] + 1]
                ),
            )
            m = a[i]
        return a[0]

    def mostPoints(self, questions: List[List[int]]) -> int:
        # 状态: 每个位置能得到的最大分数
        # 假设前面的都跳过，那么最后一个问题一定能解决，最高分初始化为最后一题的得分
        cnt = [questions[-1][0]] * len(questions)
        # 从倒数第二项往前推
        for i in range(len(questions) - 2, -1, -1):
            # 如果当前下标+冷静期>边界，那么选它的话后面的都不能选，最高分为questions[i][0]; 不选它的话最高分为cnt[i+1]
            if i + questions[i][1] + 1 > len(questions) - 1:
                cnt[i] = max(questions[i][0], cnt[i + 1])
            # 否则就是选了它, 后面还有可选的
            else:
                cnt[i] = max(questions[i][0] + cnt[i + questions[i][1] + 1], cnt[i + 1])
        return cnt[0]

    def mostPoints(self, questions: List[List[int]]) -> int:
        n = len(questions)
        f = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            q = questions[i]
            j = i + q[1] + 1
            f[i] = max(f[i + 1], q[0] + (f[j] if j < n else 0))
        return f[0]

    def mostPoints(self, q: List[List[int]]) -> int:
        n = len(q)
        dp = [0] * n
        dp[-1] = q[-1][0]
        for i in range(n - 2, -1, -1):
            pre = 0
            if i + q[i][1] + 1 < n:
                pre = dp[i + 1 + q[i][1]]
            dp[i] = max(dp[i + 1], q[i][0] + pre)
        return dp[0]

    def mostPoints(self, q: List[List[int]]) -> int:
        n = len(q)
        dp = [i for i, _ in q]
        for i in range(n - 2, -1, -1):
            if i + q[i][1] + 1 < n:
                dp[i] = max(dp[i + 1], dp[i + 1 + q[i][1]] + q[i][0])
            else:
                dp[i] = max(dp[i + 1], dp[i])
        return dp[0]


# https://leetcode-cn.com/problems/maximum-running-time-of-n-computers/
# https://leetcode.com/problems/maximum-running-time-of-n-computers/
# 5983. 同时运行 N 台电脑的最长时间
class Solution:
    def maxRunTime(self, n: int, batteries: List[int]) -> int:
        batteries = sorted(batteries, reverse=True)
        sumt = sum(batteries)
        for t in batteries:
            if t > sumt // n:
                n -= 1
                sumt -= t
            else:
                return sumt // n

    def maxRunTime(self, n: int, batteries: List[int]) -> int:
        def check(amt):
            return sum(min(amt, v) for v in batteries) >= n * amt

        lo, hi = 0, sum(batteries) // n + 1
        while lo + 1 < hi:
            x = (lo + hi) >> 1
            if check(x):
                lo = x
            else:
                hi = x
        return lo

    def maxRunTime(self, n: int, batteries: List[int]) -> int:
        l, r = 1, 100000000000005
        while l < r:
            t, mid = 0, (l + r) // 2
            for x in batteries:
                t += min(x, mid)
            if t // n >= mid:
                l = mid + 1
            else:
                r = mid
        return l - 1


#########################
# 277 / 3道 / 2022.1.22 #
#########################

# https://leetcode.com/problems/maximum-good-people-based-on-statements/
# 5992. 基于陈述统计最多好人数
# 二进制枚举的方法就是通过二进制数mask的每一位表示第i个人是好人还是坏人, 所以这个mask表示的是所有人的状态
# 我们只判断好人的陈词, 并不判断坏人的陈述, 因为坏人无论是说真还是说假都无法提高好人的个数, 对答案没有贡献

#########################
# 278 / 2道 / 2022.1.29 #
#########################


# https://leetcode-cn.com/problems/find-substring-with-given-hash-value/
# 5994. 查找给定哈希值的子串
# 正向滑窗, 需要除以power再求模. 除法不满足取余的恒等性(本题的power和modulo也不一定满足互质)
# 所以需要倒序, 乘法满足取余恒等. 除法取余: 逆元
# 所有要做减法求模的行为要小心结果为负数，我们减法之前先加上 mod，保证结果为正
class Solution:
    def subStrHash(
        self, s: str, power: int, modulo: int, k: int, hashValue: int
    ) -> str:
        ans = m = 0
        p, n = 1, len(s)
        for i in range(n - 1, n - k - 1, -1):
            m = (m * power + ord(s[i]) - 96) % modulo
            p = p * power % modulo
        if m == hashValue:
            ans = n - k
        for i in range(i - 1, -1, -1):
            m = (m * power + ord(s[i]) - 96 - (ord(s[i + k]) - 96) * p) % modulo
            if m == hashValue:
                ans = i
        return s[ans : ans + k]


# https://leetcode-cn.com/problems/groups-of-strings/
# 5995. 字符串分组
class Solution:
    def groupStrings(self, words):
        parent = [i for i in range(len(words))]

        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            parent[find(j)] = find(i)
            return

        words.sort(key=lambda word: len(word))
        for i in range(len(words)):
            words[i] = "".join(sorted(words[i]))
        hashtable = {words[i]: i for i in range(len(words))}
        visited = {}
        for i, word in enumerate(words):
            if len(word) == 1:
                union(i, 0)
                continue
            if hashtable[word] != i:
                union(i, hashtable[word])
                continue
            for j in range(len(word)):
                cur = word[:j] + word[j + 1 :]
                if cur in hashtable:
                    union(i, hashtable[cur])
                if cur in visited:
                    union(i, visited[cur])
                else:
                    visited[cur] = i

        parent = [find(i) for i in range(len(words))]
        count = collections.Counter(parent)
        return [len(count), max(count.values())]


class UnionFind:
    def __init__(self):
        self.fa = None
        self.size = None

    def init_set(self, n: int) -> None:
        self.fa = list(range(n))
        self.size = [1] * n

    def find_set(self, x: int) -> int:
        if x != self.fa[x]:
            self.fa[x] = self.find_set(self.fa[x])
        return self.fa[x]

    def union_sets(self, x: int, y: int) -> bool:
        x, y = self.find_set(x), self.find_set(y)
        if x == y:
            return False
        if self.size[x] < self.size[y]:
            x, y = y, x
        self.fa[y] = x
        self.size[x] += self.size[y]
        return True

    def count(self):
        counter = collections.Counter(map(self.find_set, self.fa))
        return [len(counter), max(counter.values())]


def get_mask(s: str) -> int:
    mask = 0
    for c in s:
        mask |= 1 << (ord(c) - 97)
    return mask


class Solution:
    def groupStrings(self, words: List[str]) -> List[int]:
        n = len(words)
        uf = UnionFind()
        uf.init_set(n)

        r_mask = dict()
        for group, word in enumerate(words):
            mask = get_mask(word)
            if mask in r_mask:
                uf.union_sets(group, r_mask[mask])
            else:
                r_mask[mask] = group

        d_mask = dict()
        for mask, group in r_mask.items():
            for i in range(26):
                if mask >> i & 1:
                    t = mask & ~(1 << i)
                    if t in r_mask:
                        uf.union_sets(group, r_mask[t])
                    if t in d_mask:
                        uf.union_sets(group, d_mask[t])
                    else:
                        d_mask[t] = group

        return uf.count()


########################
# 279 / 2道 / 2022.2.5 #
########################


# https://leetcode-cn.com/problems/design-bitset/
# 6002. 设计位集
class Bitset:
    def __init__(self, size: int):
        self.arr = [0] * size
        self.ones = 0
        self.reverse = 0  # flag

    def fix(self, idx: int) -> None:
        if self.reverse ^ self.arr[idx] == 0:
            self.arr[idx] ^= 1
            self.ones += 1

    def unfix(self, idx: int) -> None:
        if self.reverse ^ self.arr[idx] == 1:
            self.arr[idx] ^= 1
            self.ones -= 1

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


# https://leetcode-cn.com/problems/minimum-time-to-remove-all-cars-containing-illegal-goods/
# 6003. 移除所有载有违禁货物车厢所需的最少时间
class Solution:
    def minimumTime(self, s: str) -> int:
        n = ans = len(s)
        pre = 0
        for idx, char in enumerate(s):
            if char == "1":
                pre = min(pre + 2, idx + 1)
            ans = min(ans, pre + n - idx - 1)
        return ans


#########################
# 280 / 3道 / 2022.2.12 #
#########################
# https://leetcode-cn.com/contest/weekly-contest-280/


# https://leetcode-cn.com/problems/maximum-and-sum-of-array/
# 6007. 数组的最大与和
# 状态压缩 状压 dp
class Solution:
    def maximumANDSum(self, nums: List[int], numSlots: int) -> int:
        # backtracking
        # try to enumerate all possible
        n = len(nums)

        @functools.lru_cache(None)
        def dfs(i, mask=0):
            if i == n:
                return 0
            res = 0
            for slot in range(numSlots):
                if not (mask >> (slot * 2) & 1):
                    res = max(
                        res, (nums[i] & (slot + 1)) + dfs(i + 1, mask | (1 << slot * 2))
                    )
                elif not (mask >> (slot * 2) & 2):
                    res = max(
                        res,
                        (nums[i] & (slot + 1))
                        + dfs(i + 1, mask | (1 << (slot * 2 + 1))),
                    )
            return res

        return dfs(0)

    def maximumANDSum(self, nums: List[int], numSlots: int) -> int:
        @functools.lru_cache(None)
        def fn(k, m):
            """Return max AND sum."""
            if k == len(nums):
                return 0
            ans = 0
            for i in range(numSlots):
                if m & 1 << 2 * i == 0 or m & 1 << 2 * i + 1 == 0:
                    if m & 1 << 2 * i == 0:
                        mm = m ^ 1 << 2 * i
                    else:
                        mm = m ^ 1 << 2 * i + 1
                    ans = max(ans, (nums[k] & i + 1) + fn(k + 1, mm))
            return ans

        return fn(0, 0)

    # O(m * (3 ** m)), m = numSlots
    def maximumANDSum(self, nums: List[int], numSlots: int) -> int:
        @functools.lru_cache(None)
        def dp(i, mask):
            res = 0
            if i == len(nums):
                return 0
            for slot in range(1, numSlots + 1):
                b = 3 ** (slot - 1)
                if mask // b % 3 > 0:
                    res = max(res, (nums[i] & slot) + dp(i + 1, mask - b))
            return res

        return dp(0, 3**numSlots - 1)

    # O(nm * (3 ** m)), m = numSlots
    def maximumANDSum(self, nums: List[int], numSlots: int) -> int:
        @functools.lru_cache(None)
        def f(i, mask):
            if i < 0:
                return 0
            t, w, res = mask, 1, 0
            for k in range(1, numSlots + 1):
                if t % 3:
                    res = max(res, f(i - 1, mask - w) + (k & nums[i]))
                t, w = t // 3, w * 3
            return res

        return f(len(nums) - 1, 3**numSlots - 1)

    # O(m * (3 ** m)), m = numSlots
    def maximumANDSum(self, nums: List[int], numSlots: int) -> int:
        @functools.lru_cache(None)
        def f(mask):
            t, cnt = mask, 0
            for k in range(1, numSlots + 1):
                cnt += 2 - (t % 3)
                t //= 3
            i = len(nums) - 1 - cnt
            if i < 0:
                return 0
            t, w, res = mask, 1, 0
            for k in range(1, numSlots + 1):
                if t % 3:
                    res = max(res, f(mask - w) + (k & nums[i]))
                t, w = t // 3, w * 3
            return res

        return f(3**numSlots - 1)

    def maximumANDSum(self, nums: List[int], numSlots: int) -> int:
        f = [0] * (1 << (numSlots * 2))
        for i, fi in enumerate(f):
            c = i.bit_count()
            if c >= len(nums):
                continue
            for j in range(numSlots * 2):
                if (i & (1 << j)) == 0:  # 枚举空篮子 j
                    s = i | (1 << j)
                    f[s] = max(f[s], fi + ((j // 2 + 1) & nums[c]))
        return max(f)


###################
# 281 / 2022.2.19 #
###################
# https://leetcode-cn.com/contest/weekly-contest-281/


# https://leetcode-cn.com/problems/count-array-pairs-divisible-by-k/
# 6015. Count Array Pairs Divisible by K
# 每个数都可能贡献一些因子 / gcd / 最大公因子
class Solution:
    def countPairs(self, nums: List[int], k: int) -> int:
        divisors = []
        d = 1
        while d * d <= k:  # 预处理 k 的所有因子
            if k % d == 0:
                divisors.append(d)
                if d * d < k:
                    divisors.append(k // d)
            d += 1
        ans = 0
        cnt = collections.defaultdict(int)
        for n in nums:
            ans += cnt[k // math.gcd(n, k)]
            for d in divisors:
                if n % d == 0:
                    cnt[d] += 1
        return ans

    # O(n * log100000 + k * k), k is the number of divisors of k
    def countPairs(self, nums: List[int], k: int) -> int:
        cnt = collections.Counter(math.gcd(n, k) for n in nums)
        ans = 0
        for a in cnt:
            for b in cnt:
                if a <= b and a * b % k == 0:
                    if a < b:
                        ans += cnt[a] * cnt[b]
                    else:
                        ans += cnt[a] * (cnt[a] - 1) // 2
        return ans

    def countPairs(self, nums: List[int], k: int) -> int:
        cnt, ans = collections.Counter(), 0
        for n in nums:
            g = math.gcd(n, k)
            for c in cnt:
                if (c * g) % k == 0:
                    ans += cnt[c]
            cnt[g] += 1
        return ans

    def countPairs(self, nums: List[int], k: int) -> int:
        mx = k
        for x in nums:
            mx = max(mx, x)
        cnt = [0] * (mx + 1)
        for x in nums:
            cnt[x] += 1
        # n * (1 + 1/2 + 1/3 + 1/4 ...) -> n * logn
        for i in range(1, mx + 1):
            for j in range(2 * i, mx + 1, i):
                cnt[i] += cnt[j]
        ans = 0
        for x in nums:
            ans += cnt[k // math.gcd(k, x)]
        for x in nums:
            if (x * x) % k == 0:
                ans -= 1
        return ans // 2  # i < j


###################
# 282 / 2022.2.26 #
###################
# https://leetcode-cn.com/contest/weekly-contest-282/

# https://leetcode-cn.com/problems/minimum-time-to-finish-the-race/

##################
# 283 / 2022.3.5 #
##################
# https://leetcode-cn.com/contest/weekly-contest-283/

# https://leetcode-cn.com/contest/weekly-contest-283/problems/append-k-integers-with-minimal-sum/
# 贪心 or check 空隙等差数列

# https://leetcode-cn.com/contest/weekly-contest-283/problems/replace-non-coprime-numbers-in-array/
# gcd, Greatest Common Divisor
# lcm, Least Common Multiple

###################
# 284 / 2022.3.12 #
###################
# https://leetcode-cn.com/contest/weekly-contest-284/

# https://leetcode-cn.com/contest/weekly-contest-284/problems/maximize-the-topmost-element-after-k-moves/
# 分类讨论

# https://leetcode-cn.com/contest/weekly-contest-284/problems/minimum-weighted-subgraph-with-the-required-paths/
# dijkstra
# 图论经典套路: 枚举中间点
# 记 d1[i] 为从 src1 出发到达点 i 的最短路，d2[i] 为从 src2 出发到达点 i 的最短路
# d3[i] 为从点 i 出发到达 dest 的最短路(可以将原图中所有边反向，然后从 dest 出发跑 dijkstra 得到)
# 枚举中间点 i，答案就是 min(d1[i] + d2[i] + d3[i])

###################
# 285 / 2022.3.19 #
###################
# https://leetcode-cn.com/contest/weekly-contest-285/

# dp, 贪心, 二进制枚举,
# https://leetcode-cn.com/contest/weekly-contest-285/problems/maximum-points-in-an-archery-competition/

# 线段树, segment tree
# https://leetcode-cn.com/contest/weekly-contest-285/problems/longest-substring-of-one-repeating-character/

###################
# 286 / 2022.3.26 #
###################
# https://leetcode-cn.com/contest/weekly-contest-286/

# 数学, 构造
# 左半部分，第q个回文数在前导0允许的情况下是
# 第10 ^ (intLength // 2 + intLength % 2 - 1) + q - 1个,
# 恰巧左半部分每一个数都会对应构造出一个回文数来
# L = 7, the first one: [1000][001], the 376th one: [1375][731]
# https://leetcode-cn.com/contest/weekly-contest-286/problems/find-palindrome-with-fixed-length/

# 分组背包, 前缀和, dp
# https://leetcode-cn.com/contest/weekly-contest-286/problems/maximum-value-of-k-coins-from-piles/

##################
# 288 / 2022.4.9 #
##################
# https://leetcode-cn.com/contest/weekly-contest-288/

# https://leetcode-cn.com/contest/weekly-contest-288/problems/maximum-total-beauty-of-the-gardens/
# 贪心, 枚举后缀

# https://leetcode-cn.com/contest/weekly-contest-289/

# 1048 / https://leetcode-cn.com/contest/weekly-contest-289/problems/maximum-trailing-zeros-in-a-cornered-path/
# 前缀和

# 739 / https://leetcode-cn.com/contest/weekly-contest-289/problems/longest-path-with-different-adjacent-characters/
# 树形DP, 树的直径 lc1245, lc687

# https://leetcode-cn.com/contest/weekly-contest-290/

# 1104 / https://leetcode-cn.com/contest/weekly-contest-290/problems/count-number-of-rectangles-containing-each-point/
# 树状数组, 二维偏序问题, 有序容器 SortedList, 二分排序

# https://leetcode-cn.com/contest/weekly-contest-291/

# 1078 / https://leetcode-cn.com/contest/weekly-contest-291/problems/total-appeal-of-a-string/
# 字符串贡献问题, 动态规划

# https://leetcode-cn.com/contest/weekly-contest-292/

# 2057 / https://leetcode-cn.com/contest/weekly-contest-292/problems/count-number-of-texts/
# dfs, dp, python大数运算取模TLE / MLE

# 928 / https://leetcode-cn.com/contest/weekly-contest-292/problems/check-if-there-is-a-valid-parentheses-string-path/
# dfs, dp, bfs


# https://leetcode-cn.com/contest/weekly-contest-293/
# 542 / https://leetcode.cn/contest/weekly-contest-293/problems/count-integers-in-intervals/

# https://leetcode-cn.com/contest/weekly-contest-294/
# 146 / https://leetcode.cn/contest/weekly-contest-294/problems/sum-of-total-strength-of-wizards/


# https://leetcode-cn.com/contest/weekly-contest-295/
# 194 / https://leetcode.cn/contest/weekly-contest-295/problems/steps-to-make-array-non-decreasing/
# 单调栈

# 739 / https://leetcode.cn/contest/weekly-contest-295/problems/minimum-obstacle-removal-to-reach-corner/
# 0-1 BFS / Dijkstra

"""
AK, 313 / 676, 2022-06-04
https://leetcode-cn.com/contest/weekly-contest-296/
WA: T1(1)
FT: 0:42:44
T4: 1532 pass
"""
