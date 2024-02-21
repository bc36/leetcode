import bisect, collections, copy, functools, heapq, itertools, math, operator, string
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


# 901 - Online Stock Span - MEDIUM
class StockSpanner:
    # O(n) / O(n), 往前数最多有连续多少日的是小于等于今日的, 同 lc 496,
    def __init__(self):
        self.st = []
        self.i = -1

    def next(self, price: int) -> int:
        self.i += 1
        r = -1
        while self.st and price >= self.st[-1][0]:
            _, r = self.st.pop()
        if r != -1:
            self.st.append((price, r))
            return self.i - r + 1
        else:
            self.st.append((price, self.i))
            return 1


class StockSpanner:
    def __init__(self):
        self.st = []
        self.i = -1

    def next(self, price: int) -> int:
        self.i += 1
        r = -1
        while self.st and price >= self.st[-1][0]:
            _, r = self.st.pop()
        pre = r if r != -1 else self.i
        self.st.append((price, pre))
        return self.i - pre + 1


class StockSpanner:
    def __init__(self):
        self.st = [(math.inf, -1)]
        self.i = -1

    def next(self, price: int) -> int:
        self.i += 1
        while price >= self.st[-1][0]:
            self.st.pop()
        self.st.append((price, self.i))
        return self.i - self.st[-2][1]


# 902 - Numbers At Most N Given Digit Set - HARD
class Solution:
    def atMostNGivenDigitSet(self, digits: List[str], n: int) -> int:
        s = str(n)

        @functools.cache
        def f(i: int, isLimit: bool, isNum: bool) -> int:
            """数位dp
            f(i, isLimit, isNum) 表示构造从左往右第 i 位及其之后数位的合法方案数
            isLimit 表示当前是否受到了 n 的约束. 若为真, 则第 i 位填入的数字至多为 s[i], 否则至多为 9
            isNum 表示 i 前面的数位是否填了数字. 若为假, 则当前位可以跳过（不填数字）, 或者要填入的数字至少为 1; 若为真, 则必须填数字, 且要填入的数字从 0 开始. 这样我们可以控制构造出的是一位数/两位数/三位数等等
            """
            if i == len(s):
                return int(isNum)  # 如果填了数字, 则为 1 种合法方案
            res = 0
            if not isNum:  # 前面不填数字, 那么可以跳过当前数位, 也不填数字
                # isLimit 改为 False, 因为没有填数字, 位数都比 n 要短, 自然不会受到 n 的约束
                # isNum 仍然为 False, 因为没有填任何数字
                res = f(i + 1, False, False)
            up = s[i] if isLimit else "9"  # 根据是否受到约束, 决定可以填的数字的上限
            # 注意: 对于一般的题目而言, 如果此时 isNum 为 False, 则必须从 1 开始枚举, 由于本题 digits 没有 0, 所以无需处理这种情况
            for d in digits:  # 枚举要填入的数字 d
                if d > up:
                    break  # d 超过上限, 由于 digits 是有序的, 后面的 d 都会超过上限, 故退出循环
                # is_limit: 如果当前受到 n 的约束, 且填的数字等于上限, 那么后面仍然会受到 n 的约束
                # isNum 为 True, 因为填了数字
                res += f(i + 1, isLimit and d == up, True)
            return res

        return f(0, True, False)


# 904 - Fruit Into Baskets - MEDIUM
class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        ans = l = 0
        d = {}
        for r, v in enumerate(fruits):
            d[v] = d.get(v, 0) + 1
            while len(d) > 2:
                d[fruits[l]] -= 1
                if d[fruits[l]] == 0:
                    del d[fruits[l]]
                l += 1
            ans = max(ans, r - l + 1)
            r += 1
        return ans


# 905 - Sort Array By Parity - EASY
class Solution:
    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        j = len(nums) - 1
        for i in range(j):
            while j >= 0 and nums[j] & 1:
                j -= 1
            if i >= j:
                break
            if nums[i] & 1:
                nums[i], nums[j] = nums[j], nums[i]
                j -= 1
        return nums

    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        return [n for n in nums if not n & 1] + [n for n in nums if n & 1]

    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        return sorted(nums, key=lambda x: x & 1)


# 907 - Sum of Subarray Minimums - MEDIUM
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        n = len(arr)
        st = []
        left = [-1] * n
        for i in range(n):
            while st and arr[st[-1]] >= arr[i]:  # 注意边界
                st.pop()
            if st:
                left[i] = st[-1]
            st.append(i)
        st = []
        right = [n] * n
        for i in range(n - 1, -1, -1):
            while st and arr[st[-1]] > arr[i]:  # 有重复元素, 避免重复统计
                st.pop()
            if st:
                right[i] = st[-1]
            st.append(i)
        ans = 0
        for i, (v, l, r) in enumerate(zip(arr, left, right)):
            ans = (ans + v * (i - l) * (r - i)) % (10**9 + 7)
        return ans

    def sumSubarrayMins(self, arr: List[int]) -> int:
        n = len(arr)
        st = []
        left = [0] * n
        for i in range(n):
            while st and arr[st[-1]] >= arr[i]:
                st.pop()
            left[i] = i - (st[-1] if st else -1)
            st.append(i)
        st = []
        right = [0] * n
        for i in range(n)[::-1]:
            while st and arr[st[-1]] > arr[i]:
                st.pop()
            right[i] = (st[-1] if st else n) - i
            st.append(i)
        ans = 0
        for l, r, v in zip(left, right, arr):
            ans = (ans + l * r * v) % (10**9 + 7)
        return ans


# 908 - Smallest Range I - EASY
class Solution:
    def smallestRangeI(self, nums: List[int], k: int) -> int:
        mi = min(nums)
        mx = max(nums)
        if mx - mi <= 2 * k:
            return 0
        return mx - mi - 2 * k if mx - mi > 2 * k else 0

    def smallestRangeI(self, nums: List[int], k: int) -> int:
        return max(0, max(nums) - min(nums) - 2 * k)


# 910 - Smallest Range II - MEDIUM
class Solution:
    def smallestRangeII(self, nums: List[int], k: int) -> int:
        nums.sort()
        li = nums[0]
        rx = nums[-1]
        ans = rx - li
        for i in range(len(nums) - 1):
            lx = nums[i]
            ri = nums[i + 1]
            ans = min(ans, max(rx - k, lx + k) - min(li + k, ri - k))
        return ans


# 911 - Online Election - MEDIUM
class TopVotedCandidate:
    def __init__(self, persons: List[int], times: List[int]):
        cnt = collections.defaultdict(int)
        cur = -1
        cnt[cur] = -1
        self.lead = []
        for p in persons:
            cnt[p] += 1
            if cnt[p] >= cnt[cur]:
                cur = p
            self.lead.append(cur)
        self.times = times

    def q(self, t: int) -> int:
        return self.lead[bisect.bisect_right(self.times, t) - 1]


class TopVotedCandidate:
    def __init__(self, persons: List[int], times: List[int]):
        self.winner = []
        d = {}
        cmax = 0
        for p in persons:
            if p not in d:
                d[p] = 1
            else:
                d[p] += 1
            cur = d[p]
            if cur >= cmax:
                self.winner.append(p)
                cmax = cur
            else:
                self.winner.append(self.winner[-1])
        self.time = times

    def q(self, t: int) -> int:
        q = bisect.bisect_right(self.time, t)
        return self.winner[q - 1]


# 913 - Cat and Mouse — HARD
class Solution:
    # 思路:
    # 将该问题视为状态方程, 状态公式[mouse,cat,turn]代表老鼠的位置, 猫的位置, 下一步轮到谁走
    # 猫胜利的状态为[i,i,1]或[i,i,2]（i!=0）, 1代表老鼠走, 2代表猫走
    # 老鼠胜利的状态为[0,i,1]或[0,i,2]
    # 用0代表未知状态, 1代表老鼠赢, 2代表猫赢
    # 由最终的胜利状态, 回推
    # 假如当前父节点轮次是1（父节点轮次是2同样的道理）
    # 父节点=1 if 子节点是1
    # 或者
    # 父节点=2 if 所有子节点是2
    def catMouseGame(self, graph: List[List[int]]) -> int:
        n = len(graph)
        degrees = [[[0] * 2 for _ in range(n)] for _ in range(n)]  # (m,c,t)
        for i in range(n):
            for j in range(n):
                if j == 0:
                    continue
                degrees[i][j][0] += len(graph[i])
                degrees[i][j][1] += len(graph[j]) - (0 in graph[j])

        dp = [[[0] * 2 for _ in range(n)] for _ in range(n)]  # (m,c,t)
        queue = collections.deque()
        for i in range(1, n):
            states = [(i, i, 0), (i, i, 1), (0, i, 0), (0, i, 1)]
            results = [2, 2, 1, 1]
            for (m, c, t), rv in zip(states, results):
                dp[m][c][t] = rv
            queue.extend(states)

        while queue:
            m, c, t = queue.popleft()
            rv = dp[m][c][t]
            if t == 0:  # mouse
                for pre in graph[c]:
                    if pre == 0 or dp[m][pre][1] != 0:
                        continue
                    if rv == 2:
                        dp[m][pre][1] = 2
                        queue.append((m, pre, 1))
                    else:
                        degrees[m][pre][1] -= 1
                        if degrees[m][pre][1] == 0:
                            dp[m][pre][1] = 1
                            queue.append((m, pre, 1))
            else:
                for pre in graph[m]:
                    if dp[pre][c][0] != 0:
                        continue
                    if rv == 1:
                        dp[pre][c][0] = 1
                        queue.append((pre, c, 0))
                    else:
                        degrees[pre][c][0] -= 1
                        if degrees[pre][c][0] == 0:
                            dp[pre][c][0] = 2
                            queue.append((pre, c, 0))

        return dp[1][2][0]

    def catMouseGame(self, graph: List[List[int]]) -> int:
        n = len(graph)
        # search(step,cat,mouse) 表示步数=step, 猫到达位置cat, 鼠到达位置mouse的情况下最终的胜负情况

        @functools.lru_cache(None)
        def search(mouse, cat, step):
            # mouse到达洞最多需要n步(初始step=1) 说明mouse走n步还没达洞口 且cat也没抓住mouse
            if step == 2 * (n):
                return 0
            # cat抓住mouse
            if cat == mouse:
                return 2
            # mouse入洞
            if mouse == 0:
                return 1
            # 奇数步: mouse走
            if step % 2 == 0:
                # 对mouse最优的策略: 先看是否能mouse赢 再看是否能平 如果都不行则cat赢
                drawFlag = False
                for nei in graph[mouse]:
                    ans = search(nei, cat, step + 1)
                    if ans == 1:
                        return 1
                    elif ans == 0:
                        drawFlag = True
                if drawFlag:
                    return 0
                return 2
            else:
                # 对cat最优的策略: 先看是否能cat赢 再看是否能平 如果都不行则mouse赢
                drawFlag = False
                for nei in graph[cat]:
                    if nei == 0:
                        continue
                    ans = search(mouse, nei, step + 1)
                    if ans == 2:
                        return 2
                    elif ans == 0:
                        drawFlag = True
                if drawFlag:
                    return 0
                return 1

        return search(1, 2, 0)


# 915 - Partition Array into Disjoint Intervals - MEDIUM
class Solution:
    # O(3n) / O(2n)
    def partitionDisjoint(self, nums: List[int]) -> int:
        n = len(nums)
        mx = [0] * n
        mx[0] = nums[0]
        for i in range(1, len(nums)):
            mx[i] = max(mx[i - 1], nums[i])
        mi = [1e6] * n
        mi[-1] = nums[-1]
        for i in range(len(nums) - 2, -1, -1):
            mi[i] = min(mi[i + 1], nums[i])
        for i in range(n - 1):
            if mx[i] <= mi[i + 1]:
                return i + 1
        return -1

    # O(2n) / O(n)
    def partitionDisjoint(self, nums: List[int]) -> int:
        n = len(nums)
        mi = [1e6] * n
        mi[-1] = nums[-1]
        for i in range(len(nums) - 2, -1, -1):
            mi[i] = min(mi[i + 1], nums[i])
        mx = nums[0]
        for i in range(n - 1):
            if mx <= mi[i + 1]:
                return i + 1
            mx = max(mx, nums[i])
        return -1

    # O(n) / O(1)
    def partitionDisjoint(self, nums: List[int]) -> int:
        leftMx = mx = nums[0]
        p = 0
        for i in range(1, len(nums) - 1):
            mx = max(mx, nums[i])
            if nums[i] < leftMx:
                leftMx = mx
                p = i
        return p + 1


# 917 - Reverse Only Letters - EASY
class Solution:
    def reverseOnlyLetters(self, s: str) -> str:
        a, z, A, Z = ord("a"), ord("z"), ord("A"), ord("Z")
        arr = []
        for i in range(len(s)):
            if a <= ord(s[i]) <= z or A <= ord(s[i]) <= Z:
                arr.append(s[i])
        ans = ""
        for i in range(len(s)):
            if a <= ord(s[i]) <= z or A <= ord(s[i]) <= Z:
                ans += arr.pop()
            else:
                ans += s[i]
        return ans

    def reverseOnlyLetters(self, s: str) -> str:
        s = list(s)
        l, r = 0, len(s) - 1
        while l < r:
            while l < r and not s[l].isalpha():
                l += 1
            while l < r and not s[r].isalpha():
                r -= 1

            if l < r:
                s[l], s[r] = s[r], s[l]
            l += 1
            r -= 1
        return "".join(s)


# 918 - Maximum Sum Circular Subarray - MEDIUM
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        dpx = dpn = 0  # mx = mi = 0 -> wrong
        mx = mi = nums[0]  # help solve for all elements being negative
        for i in range(len(nums)):
            dpx = nums[i] + max(dpx, 0)
            mx = max(mx, dpx)
            dpn = nums[i] + min(dpn, 0)
            mi = min(mi, dpn)
        return max(sum(nums) - mi, mx) if mx > 0 else mx

    # reducing the number of times 'max()' and 'min()' are used will reduce the runtime
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        dpx = dpn = 0
        mx = mi = nums[0]
        for n in nums:
            dpx = n + dpx if dpx > 0 else n
            if dpx > mx:
                mx = dpx
            dpn = n + dpn if dpn < 0 else n
            if dpn < mi:
                mi = dpn
        return max(sum(nums) - mi, mx) if mx > 0 else mx


# 919 - Complete Binary Tree Inserter - MEDIUM
class CBTInserter:
    def __init__(self, root: TreeNode):
        self.root = root
        self.candidate = collections.deque()
        dq = collections.deque([root])
        while dq:
            n = dq.popleft()
            if n.left:
                dq.append(n.left)
            if n.right:
                dq.append(n.right)
            if not n.left or not n.right:
                self.candidate.append(n)

    def insert(self, val: int) -> int:
        c = self.candidate
        child = TreeNode(val)
        ans = c[0].val
        if not c[0].left:
            c[0].left = child
        else:
            c[0].right = child
            c.popleft()
        c.append(child)
        return ans

    def get_root(self) -> TreeNode:
        return self.root


class CBTInserter:
    def __init__(self, root: TreeNode):
        self.root = root
        self.cnt = 0
        dq = collections.deque([root])
        while dq:
            self.cnt += 1
            n = dq.popleft()
            if n.left:
                dq.append(n.left)
            if n.right:
                dq.append(n.right)

    # TODO
    def insert(self, val: int) -> int:
        self.cnt += 1
        child = TreeNode(val)
        root = self.root
        highbit = self.cnt.bit_length() - 1
        for i in range(highbit - 1, 0, -1):
            if self.cnt & (1 << i):
                root = root.right
            else:
                root = root.left
        if self.cnt & 1:
            root.right = child
        else:
            root.left = child
        return root.val

    def get_root(self) -> TreeNode:
        return self.root


# 921 - Minimum Add to Make Parentheses Valid - MEDIUM
class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        while "()" in s:
            s = s.replace("()", "")
        return len(s)

    def minAddToMakeValid(self, s: str) -> int:
        ans = l = 0
        for c in s:
            if c == "(":
                l += 1
            else:
                if l <= 0:
                    ans += -l + 1
                    l = 0
                else:
                    l -= 1
        ans += l
        return ans

    def minAddToMakeValid(self, s: str) -> int:
        ans = l = 0
        for c in s:
            if c == "(":
                l += 1
            elif l > 0:
                l -= 1
            else:
                ans += 1
        return ans + l


# 922 - Sort Array By Parity II - EASY
class Solution:
    def sortArrayByParityII(self, nums: List[int]) -> List[int]:
        nums.sort(key=lambda x: x & 1)
        nums[::2], nums[1::2] = nums[: len(nums) // 2], nums[len(nums) // 2 :]
        return nums

    def sortArrayByParityII(self, nums: List[int]) -> List[int]:
        o = [v for v in nums if v & 1]
        e = [v for v in nums if not v & 1]
        return [v for x in zip(e, o) for v in x]


# 925 - Long Pressed Name - EASY
class Solution:
    def isLongPressedName(self, name: str, typed: str) -> bool:
        i = 0
        for j in range(len(typed)):
            if i < len(name) and name[i] == typed[j]:
                i += 1
            elif j > 0 and typed[j - 1] == typed[j]:
                continue
            else:
                return False
        return i == len(name)


# 926 - Flip String to Monotone Increasing - MEDIUM
class Solution:
    # O(n) / O(1)
    def minFlipsMonoIncr(self, s: str) -> int:
        dp0 = dp1 = 0
        for c in s:
            dp00, dp11 = dp0, min(dp0, dp1)
            if c == "1":
                dp00 += 1
            else:
                dp11 += 1
            dp0, dp1 = dp00, dp11
        return min(dp0, dp1)

    # O(n) / O(n)
    def minFlipsMonoIncr(self, s: str) -> int:
        pre = [0]
        # flip 1 in the left and 0 in the right
        for x in s:
            pre.append(pre[-1] + int(x))
        # pre[j]: 1s in the left
        # pre[-1] - pre[j]: 1s in the right
        # len(s) - j - (pre[-1] - pre[j]): 0s in the right
        return min(pre[j] + len(s) - j - (pre[-1] - pre[j]) for j in range(len(pre)))


# 927 - Three Equal Parts - HARD
class Solution:
    # 1. 每一部分的 1 的数量是一定的
    # 2. -> 右边部分的大小可以确定
    # 3. -> 调整左边部分的大小 (对确定个数的 1 之后, 在右边补零)
    # 4. -> 确定中间部分的大小 (最后剩余部分)
    # O(n) / O(1)
    def threeEqualParts(self, arr: List[int]) -> List[int]:
        summ = sum(arr)
        if summ % 3:
            return [-1, -1]
        if not summ:
            return [0, 2]
        n = len(arr)
        t = summ // 3
        x = y = z = one = l = 0
        r = n - 1
        while r > -1 and one < t:
            if arr[r]:
                one += 1
                z |= 1 << (n - 1 - r)
            r -= 1
        one = 0
        while l < r and one < t:
            x = (x << 1) + arr[l]
            one += arr[l]
            l += 1
        while x < z and l < r and arr[l] == 0:
            x <<= 1
            l += 1
        for i in range(l, r + 1):
            y = (y << 1) + arr[i]
            if y >= z:
                r = i
                break
        if not x == y == z:
            return [-1, -1]
        return [l - 1, r + 1]

    # 判断三组 1 从第一个 1 开始是否构成同样的分布
    def threeEqualParts(self, arr: List[int]) -> List[int]:
        def find(x: int) -> int:
            s = 0
            for i, v in enumerate(arr):
                s += v
                if s == x:
                    return i
            return -1

        n = len(arr)
        t, mod = divmod(sum(arr), 3)
        if mod:
            return [-1, -1]
        if t == 0:
            return [0, 2]
        i = find(1)
        j = find(t + 1)
        k = find(t * 2 + 1)
        while k < n and arr[i] == arr[j] == arr[k]:
            i += 1
            j += 1
            k += 1
        return [i - 1, j] if k == n else [-1, -1]

    def threeEqualParts(self, arr: List[int]) -> List[int]:
        summ = sum(arr)
        if summ % 3:
            return [-1, -1]
        if not summ:
            return [0, 2]
        t = summ // 3
        cnt = 0
        i = j = k = -1
        for idx, v in enumerate(arr):
            if v:
                cnt += 1
            if i == -1 and cnt:
                i = idx
            if j == -1 and cnt > t:
                j = idx
            if cnt > t * 2:
                k = idx
                break
        l = len(arr) - k
        if arr[i : i + l] == arr[j : j + l] == arr[k:]:
            return [i + l - 1, j + l]
        return [-1, -1]


# 929 - Unique Email Addresses - EASY
class Solution:
    def numUniqueEmails(self, emails: List[str]) -> int:
        s = set()
        for e in emails:
            if "+" in e:
                plus = e.index("+")
                at = e.index("@")
            else:
                at = plus = e.index("@")
            a = "".join(e[:plus].split("."))
            b = e[at:]
            s.add(a + b)
        return len(s)

    def numUniqueEmails(self, emails: List[str]) -> int:
        s = set()
        for e in emails:
            a = e.split("@")[0]
            b = e.split("@")[1]
            f = a.split("+")[0].replace(".", "")
            email = f + "@" + b
            s.add(email)
        return len(s)


# 931 - Minimum Falling Path Sum - MEDIUM
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        n = len(matrix)
        f = [[math.inf] * n for _ in range(n)]
        for j in range(n):
            f[0][j] = matrix[0][j]
        for i in range(1, n):
            f[i][0] = min(f[i - 1][0], f[i - 1][1]) + matrix[i][0]
            for j in range(1, n - 1):
                f[i][j] = (
                    min(f[i - 1][j - 1], f[i - 1][j], f[i - 1][j + 1]) + matrix[i][j]
                )
            f[i][n - 1] = min(f[i - 1][n - 2], f[i - 1][n - 1]) + matrix[i][n - 1]
        return min(f[n - 1])

    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        n = len(matrix)
        f = [[math.inf] * n for _ in range(n)]
        for j in range(n):
            f[0][j] = matrix[0][j]
        for i in range(1, n):
            for j in range(n):
                f[i][j] = f[i - 1][j] + matrix[i][j]
                if j > 0:
                    f[i][j] = min(f[i][j], f[i - 1][j - 1] + matrix[i][j])
                if j < n - 1:
                    f[i][j] = min(f[i][j], f[i - 1][j + 1] + matrix[i][j])
        return min(f[n - 1])

    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        n = len(matrix)
        f = [[float("inf")] + matrix[i] + [float("inf")] for i in range(n)]
        for i in range(1, n):
            for j in range(1, n + 1):
                f[i][j] = f[i][j] + min(f[i - 1][j - 1], f[i - 1][j], f[i - 1][j + 1])
        return min(f[-1])


# 933 - Number of Recent Calls - EASY
class RecentCounter:
    def __init__(self):
        self.dq = collections.deque()

    def ping(self, t: int) -> int:
        self.dq.append(t)
        while self.dq[0] < t - 3000:
            self.dq.popleft()
        return len(self.dq)


# 934 - Shortest Bridge - MEDIUM
class Solution:
    # O(mn) / O(mn)
    def shortestBridge(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        dq = collections.deque()

        # def dfs(i: int, j: int) -> None:
        #     grid[i][j] = 2
        #     for x, y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
        #         if 0 <= x < m and 0 <= y < n and grid[x][y] == 1:
        #             dfs(x, y)
        #             dq.append((x, y))
        #     return

        def dfs(r: int, c: int) -> None:
            if 0 <= r < m and 0 <= c < n:
                if grid[r][c] != 1:
                    return
                dq.append((r, c))
                grid[r][c] = 2
                dfs(r + 1, c)
                dfs(r - 1, c)
                dfs(r, c - 1)
                dfs(r, c + 1)
            return

        f = False
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    dq.append((i, j))
                    dfs(i, j)
                    f = True
                    break
            if f:
                break
        ans = 0
        while dq:
            for _ in range(len(dq)):
                i, j = dq.popleft()
                for x, y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= x < m and 0 <= y < n:
                        if grid[x][y] == 1:
                            return ans
                        if grid[x][y] == 0:
                            grid[x][y] = 2
                            dq.append((x, y))
            ans += 1
        return ans

    def shortestBridge(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        island = []
        ok = False
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v == 1:
                    q = [(i, j)]
                    island.append((i, j))
                    grid[i][j] = 2
                    while q:
                        x, y = q.pop()
                        for nx, ny in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
                            if 0 <= nx < m and 0 <= ny < n:
                                if grid[nx][ny] == 1:
                                    grid[nx][ny] = -1  # sink the first island
                                    island.append((nx, ny))
                                    q.append((nx, ny))
                    ok = True
                    break
            if ok:
                break
        q = island
        ans = 0
        while q:
            nxt = []
            for x, y in q:
                for nx, ny in (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1):
                    if 0 <= nx < m and 0 <= ny < n:
                        if grid[nx][ny] == 1:
                            return ans
                        if grid[nx][ny] == 0:
                            grid[nx][ny] = -1
                            nxt.append((nx, ny))
            q = nxt
            ans += 1
        return -1


# 935 - Knight Dialer - MEDIUM
class Solution:
    # 0         -> 4 6
    # 1 3 7 9   -> 2 8 / 4 6
    # 2 8       -> 1 3 7 9
    # 4 6       -> 0 / 1 3 7 9
    def knightDialer(self, n: int) -> int:
        if n == 1:
            return 10
        n1379, n46, n28, n0 = 4, 2, 2, 1
        mod = 10**9 + 7
        for _ in range(n - 1):
            n1379, n46, n28, n0 = 2 * (n46 + n28), n1379 + n0 * 2, n1379, n46
        return (n1379 + n46 + n28 + n0) % mod

    def knightDialer(self, n: int) -> int:
        x1 = x2 = x3 = x4 = x5 = x6 = x7 = x8 = x9 = x0 = 1
        for _ in range(n - 1):
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x0 = (
                x6 + x8,
                x7 + x9,
                x4 + x8,
                x3 + x9 + x0,
                0,
                x1 + x7 + x0,
                x2 + x6,
                x1 + x3,
                x2 + x4,
                x4 + x6,
            )
        return (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x0) % (10**9 + 7)


# 937 - Reorder Data in Log Files - EASY
class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        a = []
        b = []
        for log in logs:
            if log[-1].isalpha():
                a.append(log)
            else:
                b.append(log)
        a.sort(key=lambda x: (x[x.index(" ") + 1 :], x[: x.index(" ") + 1]))
        return a + b

    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        def comparator(log: str) -> tuple:
            identity, res = log.split(" ", 1)
            if res[0].isalpha():
                return (0, res, identity)
            else:
                return (1, "")

        return sorted(logs, key=comparator)

    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        def comparator(log: str) -> tuple:
            identity, res = log.split(" ", 1)
            return (0, res, identity) if res[0].isalpha() else (1,)

        return sorted(logs, key=comparator)


# 938 - Range Sum of BST - EASY
class Solution:
    # preorder
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        def dfs(root: TreeNode, ans: List[int]):
            if not root:
                return
            # process
            if root.val >= low and root.val <= high:
                ans.append(root.val)
            # left node and right node
            if root.left:
                dfs(root.left, ans)
            if root.right:
                dfs(root.right, ans)
            return

        ans = []
        dfs(root, ans)
        return sum(ans)

    # preorder
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        def dfs(root: TreeNode):
            if not root:
                return 0
            # process
            val = 0
            if root.val >= low and root.val <= high:
                val = root.val
            # left node and right node
            return val + dfs(root.left) + dfs(root.right)

        return dfs(root)

    # search the whole tree (see next solution to speed up)
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        if not root:
            return 0
        val = root.val if root.val >= low and root.val <= high else 0
        return (
            val
            + self.rangeSumBST(root.left, low, high)
            + self.rangeSumBST(root.right, low, high)
        )

    # since its a 'binary search tree' which means that left.val < root.val < right.val
    # so we can speed up by jump some unqualified node (the value greater than high or smalller than low)
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        if not root:
            return 0
        if root.val > high:
            return self.rangeSumBST(root.left, low, high)
        if root.val < low:
            return self.rangeSumBST(root.right, low, high)
        return (
            root.val
            + self.rangeSumBST(root.left, low, high)
            + self.rangeSumBST(root.right, low, high)
        )

    # bfs
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        dq = collections.deque()
        ans = 0
        if root:
            dq.append(root)
        while dq:
            # process a layer of nodes
            for _ in range(len(dq)):
                # get one node to process from left side -> FIFO
                node = dq.popleft()
                # add the qualified value
                if node.val >= low and node.val <= high:
                    ans += node.val
                # add new children node to dq
                # guarantee the node is not 'None'
                # if there is no 'if' judgement, 'None' will append to the 'dq',
                # and in the next level loop, when node poped,
                # we will get 'None.val', and get Exception
                if node.left:
                    dq.append(node.left)
                if node.right:
                    dq.append(node.right)
        return ans


# 940 - Distinct Subsequences II - HARD
class Solution:
    def distinctSubseqII(self, s: str) -> int:
        # 定义 f[i][j] 表示前 i 个字符中, 以字符 j 结尾的不同子序列的个数
        # f[i][s[i]] = 1 + sum(f[i-1][j] for j in range(26))

        # f[i] 表示以 s[i] 为最后一个字符的子序列的数目
        mod = 10**9 + 7
        f = [0] * 26
        for c in s:
            p = ord(c) - ord("a")
            for i in range(26):
                if p != i:
                    f[p] += f[i]
            f[p] = (1 + f[p]) % mod
        return sum(f) % mod

    def distinctSubseqII(self, s: str) -> int:
        mod = 10**9 + 7
        total = 0
        f = [0] * 26
        for c in s:
            i = ord(c) - ord("a")
            others = total - f[i]
            f[i] = total + 1
            total = (f[i] + others) % mod
        return total

    def distinctSubseqII(self, s: str) -> int:
        mod = 10**9 + 7
        total = 0
        f = [0] * 26
        for c in s:
            i = ord(c) - ord("a")
            pre = f[i]
            f[i] = total + 1
            total = (total - pre + f[i]) % mod
        return total


# 941 - Valid Mountain Array - EASY
class Solution:
    def validMountainArray(self, arr: List[int]) -> bool:
        i = 0
        while i + 1 < len(arr) and arr[i] < arr[i + 1]:
            i += 1
        if i == 0 or i == len(arr) - 1:
            return False
        while i + 1 < len(arr) and arr[i] > arr[i + 1]:
            i += 1
        return i == len(arr) - 1

    def validMountainArray(self, arr: List[int]) -> bool:
        if len(arr) < 3:
            return False
        i, j = 0, len(arr) - 1
        while i < len(arr) - 1 and arr[i] < arr[i + 1]:
            i += 1
        while j >= 0 and arr[j] < arr[j - 1]:
            j -= 1
        return i == j and i != 0 and j != len(arr) - 1


# 942 - DI String Match - EASY
class Solution:
    def diStringMatch(self, s: str) -> List[int]:
        i = 0
        j = len(s)
        ans = []
        for c in s:
            if c == "I":
                ans.append(i)
                i += 1
            else:
                ans.append(j)
                j -= 1
        ans.append(i)
        return ans


# 944 - Delete Columns to Make Sorted - EASY
class Solution:
    def minDeletionSize(self, m: List[str]) -> int:
        return sum(any(a > b for a, b in zip(col, col[1:])) for col in zip(*m))


# 945 - Minimum Increment to Make Array Unique - MEDIUM
class Solution:
    def minIncrementForUnique(self, nums: List[int]) -> int:
        ans = t = 0
        p = -1
        for v in sorted(nums):
            if p == v:
                t += 1
            else:
                t = max(t - (v - 1 - p), 0)  # pre 和 v 之间的可以留出多少空隙
            ans += t  # 除左端点之外, 值相同连续区间需要统一往右挪一步, 区间长度 t
            p = v
        return ans

    def minIncrementForUnique(self, nums: List[int]) -> int:
        nums.sort()
        nxt = nums[0]
        ans = 0
        for v in nums:
            if v < nxt:
                ans += nxt - v
                nxt += 1
            else:
                nxt = v + 1
        return ans


# 946 - Validate Stack Sequences - MEDIUM
class Solution:
    def validateStackSequences(self, ps: List[int], pp: List[int]) -> bool:
        st = []
        i = 0
        for v in ps:
            while st and st[-1] == pp[i]:
                st.pop()
                i += 1
            st.append(v)
        while st and st[-1] == pp[i]:
            st.pop()
            i += 1
        return not st

    def validateStackSequences(self, ps: List[int], pp: List[int]) -> bool:
        st = []
        i = 0
        for v in ps:
            st.append(v)
            while st and i < len(pp) and st[-1] == pp[i]:
                st.pop()
                i += 1
        return st == []

    def validateStackSequences(self, ps: List[int], pp: List[int]) -> bool:
        st = []
        i = 0
        for v in pp:
            while (not st or st[-1] != v) and i < len(ps):
                st.append(ps[i])
                i += 1
            if st[-1] != v:
                return False
            else:
                st.pop()
        return True

    def validateStackSequences(self, ps: List[int], pp: List[int]) -> bool:
        i = j = 0
        for v in ps:
            ps[i] = v
            while i >= 0 and ps[i] == pp[j]:
                i -= 1
                j += 1
            i += 1
        return i == 0


# 952 - Largest Component Size by Common Factor - HARD

"""
for some problems that input can be used for other test cases,
put the cache outside the class Solution,
each instance can reuse cache and speed up
"""


@functools.lru_cache(None)
def get_prime_factor(n: int) -> set:
    if n == 1:
        return set()
    ans = set()
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            n //= i
            ans.add(i)
            ans = ans.union(get_prime_factor(n))
            return ans
    ans.add(n)
    return ans


class Solution:
    def largestComponentSize(self, nums) -> int:
        class UnionFind:
            def __init__(self, n: int) -> None:
                self.p = [i for i in range(n)]
                self.sz = [1] * n

            def find(self, x: int) -> int:
                if self.p[x] != x:
                    self.p[x] = self.find(self.p[x])
                return self.p[x]

            def union(self, x: int, y: int) -> None:
                px = self.find(x)
                py = self.find(y)
                if px == py:
                    return
                self.p[px] = py
                self.sz[py] += self.sz[px]
                return

        conn = collections.defaultdict(list)
        for i, n in enumerate(nums):
            for fac in get_prime_factor(n):
                conn[fac].append(i)
        uf = UnionFind(len(nums))
        for group in conn.values():
            for i1, i2 in pairwise(group):
                uf.union(i1, i2)
        return max(uf.sz)


@functools.lru_cache(None)
def get_factors(n: int) -> collections.defaultdict(int):
    if n == 1:
        return collections.defaultdict(int)
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            tmp = copy.deepcopy(get_factors(n // i))
            tmp[i] += 1
            return tmp
    tmp = collections.defaultdict(int)
    tmp[n] = 1
    return tmp


class Solution:
    def largestComponentSize(self, nums: List[int]) -> int:
        def find(x: int) -> int:
            if p[x] != x:
                p[x] = find(p[x])
            return p[x]

        p = [i for i in range(max(nums) + 1)]
        cnt = collections.Counter()
        for i in range(len(nums)):
            for n in get_factors(nums[i]).keys():
                if n != 1:
                    p[find(nums[i])] = find(n)
        for i in range(len(nums)):
            cnt[find(nums[i])] += 1
        return cnt.most_common(1)[0][1]


# 953 - Verifying an Alien Dictionary - EASY
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        trans = str.maketrans(order, "abcdefghijklmnopqrstuvwxyz")
        nw = [w.translate(trans) for w in words]
        for i in range(len(words) - 1):
            if nw[i] > nw[i + 1]:
                return False
        return True

    def isAlienSorted(self, words: List[str], order: str) -> bool:
        m = {c: i for i, c in enumerate(order)}
        words = [[m[c] for c in w] for w in words]
        return all(w1 <= w2 for w1, w2 in zip(words, words[1:]))

    def isAlienSorted(self, words: List[str], order: str) -> bool:
        return words == sorted(words, key=lambda w: map(order.index, w))

    # compare each character in word[i] and word[i+1]
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        order_map = {val: index for index, val in enumerate(order)}
        # check the next word letter one by one
        for i in range(len(words) - 1):
            for j in range(len(words[i])):
                # find a mismatch letter between words[i] and words[i + 1],
                if j >= len(words[i + 1]):  # ("apple", "app")
                    return False
                if words[i][j] != words[i + 1][j]:
                    if order_map[words[i][j]] > order_map[words[i + 1][j]]:
                        return False
                    break
        return True

    def isAlienSorted(self, words: List[str], order: str) -> bool:
        m = {c: i for i, c in enumerate(order)}
        for i, w in enumerate(words[:-1]):
            j = 0
            while j < len(w):
                if j == len(words[i + 1]):
                    return False
                a = m[w[j]] - m[words[i + 1][j]]
                if a > 0:
                    return False
                elif a < 0:
                    break
                j += 1
        return True


# 954 - Array of Doubled Pairs - MEDIUM
class Solution:
    # O(nlogn) / O(n)
    def canReorderDoubled(self, arr: List[int]) -> bool:
        cnt = collections.Counter(arr)
        for x in sorted(cnt, key=abs):
            if cnt[x] > cnt[2 * x]:
                return False
            cnt[2 * x] -= cnt[x]
        return True


# 958 - Check Completeness of a Binary Tree - MEDIUM
class Solution:
    # 树的 size 是否等于最后一个节点的 index
    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        arr = [1]
        q = [(root, 1)]
        size = 1
        while q:
            nxt = []
            for r, i in q:
                if r.left:
                    nxt.append((r.left, i * 2))
                    size += 1
                    arr.append(i * 2)
                if r.right:
                    nxt.append((r.right, i * 2 + 1))
                    size += 1
                    arr.append(i * 2 + 1)
            q = nxt
        return arr[-1] == size

    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        arr = [(root, 1)]
        i = 0
        while i < len(arr):
            r, v = arr[i]
            i += 1
            if r:
                arr.append((r.left, 2 * v))
                arr.append((r.right, 2 * v + 1))
        return arr[-1][1] == len(arr)

    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        dq = collections.deque([root])
        # 在出现一次 empty 后, 后面的必须都是 empty
        # 注意: 当前层空节点靠右, 但下一层还有节点, 也是非完全的
        hasEmpty = False
        while dq:
            for _ in range(len(dq)):
                r = dq.popleft()
                if r == None:
                    hasEmpty = True
                else:
                    if hasEmpty:
                        return False
                    dq.append(r.left)
                    dq.append(r.right)
        return True


# 965 - Univalued Binary Tree - EASY
class Solution:
    def isUnivalTree(self, root: TreeNode) -> bool:
        def dfs(root: TreeNode, pre: int):
            if not root:
                return True
            if root.val != pre:
                return False
            return dfs(root.left, root.val) and dfs(root.right, root.val)

        return dfs(root, root.val)


# 967 - Numbers With Same Consecutive Differences - MEDIUM
class Solution:
    # O(2 ** n) / O(2 ** n)
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        def dfs(n: int, num: int) -> None:
            if n == 0:
                ans.append(num)
                return
            for x in set([num % 10 + k, num % 10 - k]):
                if 0 <= x < 10:
                    dfs(n - 1, num * 10 + x)
            return

        ans = []
        for i in range(1, 10):
            dfs(n - 1, i)
        return ans

    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        q = list(range(1, 10))
        for _ in range(n - 1):
            new = set()
            for v in q:
                if v % 10 + k < 10:
                    new.add(v * 10 + v % 10 + k)
                if 0 <= v % 10 - k < 10:
                    new.add(v * 10 + v % 10 - k)
            q = list(new)
        return q

    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        cur = range(1, 10)
        for i in range(n - 1):
            cur = {
                x * 10 + y for x in cur for y in [x % 10 + k, x % 10 - k] if 0 <= y <= 9
            }
        return list(cur)


# 969 - Pancake Sorting - MEDIUM
class Solution:
    def pancakeSort(self, arr: List[int]) -> List[int]:
        i = len(arr)
        ans = []
        while i > -1:
            # every two moves make one number to its correct position
            for j in range(i):
                if arr[j] == i:
                    # moves the current largest number to the first position.
                    ans.append(j + 1)
                    arr[:] = arr[: j + 1][::-1] + arr[j + 1 :]
                    # reverse the first i elements
                    # so that the current largest number is moved to its correct position.
                    ans.append(i)
                    arr[:] = arr[:i][::-1] + arr[i:]
                    break
            i -= 1
        return ans

    def pancakeSort(self, arr: List[int]) -> List[int]:
        ans = []
        n = len(arr)
        while n:
            idx = arr.index(n)
            ans.append(idx + 1)
            arr = arr[: idx + 1][::-1] + arr[idx + 1 :]
            ans.append(n)
            arr = arr[:n][::-1] + arr[n:]
            n -= 1
        return ans


# 970 - Powerful Integers - MEDIUM
class Solution:
    def powerfulIntegers(self, x: int, y: int, bound: int) -> List[int]:
        if bound < 2:
            return []
        if x > y:
            return self.powerfulIntegers(y, x, bound)
        if x == 1:
            if y == 1:
                return [2]
            ans = []
            jy = 1
            while 1 + jy <= bound:
                ans.append(1 + jy)
                jy *= y
            return ans
        s = set()
        ix = 1
        while ix <= bound:
            jy = 1
            while ix + jy <= bound:
                s.add(ix + jy)
                jy *= y
            ix *= x
        return list(s)

    def powerfulIntegers(self, x: int, y: int, bound: int) -> List[int]:
        ans = set()
        ix = 1
        while ix <= bound:
            jy = 1
            while ix + jy <= bound:
                ans.add(ix + jy)
                jy *= y
                if y == 1:
                    break
            if x == 1:
                break
            ix *= x
        return list(ans)

    def powerfulIntegers(self, x: int, y: int, bound: int) -> List[int]:
        res = set()
        if bound == 2:
            return [2]
        # 2 ** 20 ~= 10 ^ 6
        for i in range(21):
            for j in range(21):
                tmp = x**i + y**j
                if tmp <= bound:
                    res.add(tmp)
                else:
                    break

        return list(res)


# 973 - K Closest Points to Origin - MEDIUM
class Solution:
    # Pay attention that if the points are at the same distance,
    # different coordinates should be returned.
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        # [info[0]: square, info[1]: position index]
        info = [[x[0] * x[0] + x[1] * x[1], i] for i, x in enumerate(points)]
        # key: square, value: position index
        distance = {}
        for i in info:
            if i[0] not in distance:
                distance[i[0]] = [i[1]]
            else:
                distance[i[0]].append(i[1])
        order = list(distance.keys())
        order.sort()
        ans, i = [], 0
        while len(ans) < k:
            if distance[order[i]]:
                ans.append(points[distance[order[i]].pop()])
            else:
                i += 1
        return ans

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        points.sort(key=lambda x: (x[0] ** 2 + x[1] ** 2))
        return points[:k]

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        a = sorted([(x * x + y * y, i) for i, (x, y) in enumerate(points)])
        return [points[i] for _, i in a[:k]]

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap = []
        for x, y in points:
            dist = -(x * x + y * y)
            if len(heap) == k:
                heapq.heappushpop(heap, (dist, x, y))
            else:
                heapq.heappush(heap, (dist, x, y))
        return [(x, y) for (_, x, y) in heap]

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        q = [(-(x**2) - y**2, i) for i, (x, y) in enumerate(points[:k])]
        heapq.heapify(q)
        for i in range(k, len(points)):
            x, y = points[i]
            dist = -(x**2) - y**2
            heapq.heappushpop(q, (dist, i))
        ans = [points[i] for (_, i) in q]
        return ans


# 976 - Largest Perimeter Triangle - EASY
class Solution:
    def largestPerimeter(self, a: List[int]) -> int:
        a.sort()
        for i in range(len(a) - 1, 1, -1):
            if a[i - 1] + a[i - 2] > a[i]:
                return sum(a[i - 2 : i + 1])
        return 0


# 977 - Squares of a Sorted Array - EASY
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        return sorted([num**2 for num in nums])

    def sortedSquares(self, nums: List[int]) -> List[int]:
        left, right = 0, len(nums) - 1
        ans = [0] * len(nums)
        while left <= right:
            al, ar = abs(nums[left]), abs(nums[right])
            if al > ar:
                ans[right - left] = al**2
                left += 1
            else:
                ans[right - left] = ar**2
                right -= 1
        return ans


# 979 - Distribute Coins in Binary Tree - MEDIUM
class Solution:
    def distributeCoins(self, root: Optional[TreeNode]) -> int:
        def dfs(node: Optional[TreeNode]) -> int:
            """
            1. 多了问父节点要, 少了给父节点
            2. 给和拿其实没区别, 路径必定经过, 所以需要考虑有多少点会经过这条边
            3. 每个节点的值 = 其子节点个数 + 1"""
            if node is None:
                return 0
            d = dfs(node.left) + dfs(node.right) + node.val - 1
            ans[0] += abs(d)
            return d

        ans = [0]
        dfs(root)
        return ans[0]


# 980 - Unique Paths III - HARD
class Solution:
    # O(3 ^ (mn)) / O(mn)
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        def dfs(x: int, y: int, left: int) -> int:
            """dfs(x, y, left) 表示从 (x, y) 出发, 还剩下 left 个无障碍方格(不含终点)需要访问时的不同路径个数"""
            if x < 0 or x >= m or y < 0 or y >= n or grid[x][y] < 0:
                return 0
            if grid[x][y] == 2:
                return left == 0
            grid[x][y] = -1
            ans = (
                dfs(x - 1, y, left - 1)
                + dfs(x, y - 1, left - 1)
                + dfs(x + 1, y, left - 1)
                + dfs(x, y + 1, left - 1)
            )
            grid[x][y] = 0
            return ans

        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v == 1:
                    return dfs(i, j, sum(row.count(0) for row in grid) + 1)

    # O(mn * 2 ^ (mn)) / O(mn * 2 ^ (mn))
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        @functools.lru_cache(None)
        def dfs(x: int, y: int, vis: int) -> int:
            """如果访问了 (x, y), 就把 vis 从低到高第 nx + y 个比特位标记成 1"""
            if x < 0 or x >= m or y < 0 or y >= n or vis >> (x * n + y) & 1:
                return 0
            vis |= 1 << (x * n + y)
            if grid[x][y] == 2:
                return vis == (1 << m * n) - 1  # 全集(所有格子的坐标集合)
            return (
                dfs(x - 1, y, vis)
                + dfs(x, y - 1, vis)
                + dfs(x + 1, y, vis)
                + dfs(x, y + 1, vis)
            )

        vis = 0
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v < 0:
                    vis |= 1 << (i * n + j)
                elif v == 1:
                    sx, sy = i, j
        return dfs(sx, sy, vis)


# 981 - Time Based Key-Value Store - MEDIUM
class TimeMap:
    def __init__(self):
        self.d = collections.defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.d[key].append((timestamp, value))
        return

    def get(self, key: str, timestamp: int) -> str:
        # p = bisect.bisect_right(self.d[key], (timestamp, "z" * 100))
        # p = bisect.bisect_right(self.d[key], (timestamp, chr(ord("z") + 1)))
        p = bisect.bisect_right(self.d[key], (timestamp, chr(127)))
        return "" if p == 0 else self.d[key][p - 1][1]


# 982 - Triples with Bitwise AND Equal To Zero - HARD
class Solution:
    # O(n^2) / O(U), U = max(nums), 2400ms
    def countTriplets(self, nums: List[int]) -> int:
        cnt = collections.Counter(x & y for x in nums for y in nums)
        return sum(v for x, v in cnt.items() for y in nums if x & y == 0)

    # 集合 A 和集合 B 没有交集 -> B 是 A 的补集的子集
    # O(n^2 + nU) / O(U), U = max(nums), 390ms
    def countTriplets(self, nums: List[int]) -> int:
        cnt = [0] * (1 << 16)
        for x in nums:
            for y in nums:
                cnt[x & y] += 1
        ans = 0
        for v in nums:
            v ^= 0xFFFF
            x = v
            while True:  # 枚举 v 的子集(包括空集)
                ans += cnt[x]
                x = (x - 1) & v
                if x == v:  # 当 x = 0 时 -> -1 的二进制全为 1 -> 等式成立
                    break
        return ans

    # 预处理每个 nums[k] 的补集的子集的出现次数 cnt, 360ms
    def countTriplets(self, nums: List[int]) -> int:
        cnt = [0] * (1 << 16)
        cnt[0] = len(nums)  # 直接统计空集
        for v in nums:
            v ^= 0xFFFF
            x = v
            while x:  # 枚举 v 的非空子集
                cnt[x] += 1
                x = (x - 1) & v
        return sum(cnt[x & y] for x in nums for y in nums)

    # 仔细计算 cnt 的实际大小 u, 相应的全集就是 u - 1, 260ms
    def countTriplets(self, nums: List[int]) -> int:
        u = 1
        for v in nums:
            while u <= v:
                u <<= 1
        cnt = [0] * u
        cnt[0] = len(nums)  # 直接统计空集
        for v in nums:
            v ^= u - 1
            x = v
            while x:  # 枚举 v 的非空子集
                cnt[x] += 1
                x = (x - 1) & v
        return sum(cnt[x & y] for x in nums for y in nums)


# 985 - Sum of Even Numbers After Queries - MEDIUM
class Solution:
    def sumEvenAfterQueries(
        self, nums: List[int], queries: List[List[int]]
    ) -> List[int]:
        summ = sum(v for v in nums if not v & 1)
        ans = []
        for v, i in queries:
            x = nums[i] + v
            if nums[i] % 2 == v % 2:
                if v & 1:
                    summ += x
                else:
                    summ += v
            else:
                if nums[i] % 2 == 0:
                    summ -= nums[i]
            nums[i] = x
            ans.append(summ)
        return ans

    def sumEvenAfterQueries(
        self, nums: List[int], queries: List[List[int]]
    ) -> List[int]:
        summ = sum(v for v in nums if v % 2 == 0)
        ans = []
        for v, i in queries:
            if nums[i] % 2 == 0:
                summ -= nums[i]
            nums[i] += v
            if nums[i] % 2 == 0:
                summ += nums[i]
            ans.append(summ)
        return ans


# 986 - Interval List Intersections - MEDIUM
class Solution:
    def intervalIntersection(
        self, first: List[List[int]], second: List[List[int]]
    ) -> List[List[int]]:
        ans, i, j = [], 0, 0
        while i < len(first) and j < len(second):
            lo = max(first[i][0], second[j][0])
            hi = min(first[i][1], second[j][1])
            if lo <= hi:
                ans.append([lo, hi])
            if first[i][1] < second[j][1]:
                i += 1
            else:
                j += 1
            # or
            # if first[i][1] == hi:
            #     i += 1
            # else:
            #     j += 1
        return ans


# 987 - Vertical Order Traversal of a Binary Tree - HARD
class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        dq = collections.deque([(root, 0)])
        ans = collections.defaultdict(list)
        while dq:
            tmp = collections.defaultdict(list)
            for _ in range(len(dq)):
                n, p = dq.popleft()
                tmp[p].append(n.val)
                if n.left:
                    dq.append((n.left, p - 1))
                if n.right:
                    dq.append((n.right, p + 1))
            for k, v in tmp.items():
                ans[k].extend(sorted(v))
        return [v for _, v in sorted(ans.items())]
        return [ans[k] for k in sorted(ans)]  # sorting directly will return sorted keys


# 989 - Add to Array-Form of Integer - EASY
class Solution:
    def addToArrayForm(self, num: List[int], k: int) -> List[int]:
        num.reverse()
        i = carry = 0
        n = len(num)
        while k or carry:
            tmp = carry
            if k:
                tmp += k % 10
                k //= 10
            if i < n:
                tmp += num[i]
                num[i] = tmp % 10
            else:
                num.append(tmp % 10)
            carry = tmp // 10
            i += 1
        return list(reversed(num))

    def addToArrayForm(self, num: List[int], k: int) -> List[int]:
        i = len(num) - 1
        carry = 0
        while k or carry:
            if i == -1:
                i = 0
                num = [0] + num
            v = k % 10 + num[i] + carry
            # num[i] = v % 10
            # carry = v // 10
            carry, num[i] = divmod(v, 10)
            k //= 10
            i -= 1
        return num


# 990 - Satisfiability of Equality Equations - MEDIUM
class Solution:
    # O(n + ClogC) / O(C), C = 26
    def equationsPossible(self, equations: List[str]) -> bool:
        p = list(range(26))

        def find(x: int) -> int:
            if p[x] != x:
                p[x] = find(p[x])
            return p[x]

        check = []
        for a, b, _, c in equations:
            ia, ic = ord(a) - 97, ord(c) - 97
            fa, fc = find(ia), find(ic)
            if b == "=":
                p[fc] = fa
            else:
                check.append((ia, ic))

        return not any(find(ia) == find(ic) for ia, ic in check)


# 992 - Subarrays with K Different Integers - HARD
class Solution:
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        def atMostK(nums, k):
            ans = left = right = distinct = 0
            cnt = collections.Counter()
            while right < len(nums):
                if cnt[nums[right]] == 0:
                    distinct += 1
                cnt[nums[right]] += 1
                while distinct > k:
                    cnt[nums[left]] -= 1
                    if cnt[nums[left]] == 0:
                        distinct -= 1
                    left += 1
                ans += right - left + 1
                right += 1
            return ans

        return atMostK(nums, k) - atMostK(nums, k - 1)

    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        ret = 0
        prev_good = 0
        counter = dict()
        left, right = 0, 0
        # keep moving right
        for right in range(len(nums)):
            counter[nums[right]] = counter.setdefault(nums[right], 0) + 1
            # now we have k distinct
            if len(counter.keys()) == k:
                # the first time we meet k distinct
                if prev_good == 0:
                    prev_good = 1
                # we can move left to find the shortest good to get new good
                while counter[nums[left]] > 1:
                    counter[nums[left]] -= 1
                    left += 1
                    prev_good += 1
            # now we have more than k distinct
            elif len(counter.keys()) > k:
                # we remove the first of previous shortest good and appending the right
                # to get a new good
                prev_good = 1
                counter.pop(nums[left])
                left += 1
                # we can move left to reach the shortest good to get new good
                while counter[nums[left]] > 1:
                    counter[nums[left]] -= 1
                    left += 1
                    prev_good += 1
            ret += prev_good
        return ret

    # 对于任意一个右端点, 能够与其对应的左端点们必然相邻
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        freq = {}
        ans = start = l = 0
        for v in nums:
            freq[v] = freq.get(v, 0) + 1
            if len(freq) == k + 1:
                del freq[nums[l]]  # 只剩一个了, 直接删除
                l += 1
                start = l
            if len(freq) == k:
                while freq[nums[l]] > 1:
                    freq[nums[l]] -= 1
                    l += 1
                ans += l - start + 1
        return ans


# 993 - Cousins in Binary Tree - EASY
class Solution:
    # O(n) / O(n)
    def isCousins(self, root: Optional[TreeNode], x: int, y: int) -> bool:
        q = [(root, None)]
        while q:
            new = []
            m = n = None
            for a, fa in q:
                if a.val == x:
                    m = (a, fa)
                elif a.val == y:
                    n = (a, fa)
                if a.left:
                    new.append((a.left, a))
                if a.right:
                    new.append((a.right, a))
            if m and n:
                return m[1] != n[1]
            elif (m and not n) or (n and not m):
                return False
            q = new
        return False

    def isCousins(self, root: Optional[TreeNode], x: int, y: int) -> bool:
        info = {}

        def dfs(root: TreeNode, fa: TreeNode, d: int) -> None:
            if root:
                info[root.val] = (fa, d)
                dfs(root.left, root, d + 1)
                dfs(root.right, root, d + 1)
            return

        dfs(root, None, 0)
        return info[x][0] != info[y][0] and info[x][1] == info[y][1]

    # O(n) / O(1)
    def isCousins(self, root: Optional[TreeNode], x: int, y: int) -> bool:
        def dfs(
            root: TreeNode, fa: TreeNode, depth: int, t: int
        ) -> Tuple[TreeNode, int, bool]:
            if not root:
                return None, None, False
            if root.val == t:
                return fa, depth, True
            res = dfs(root.left, root, depth + 1, t)
            if res[2]:
                return res
            return dfs(root.right, root, depth + 1, t)

        xx = dfs(root, None, 0, x)
        yy = dfs(root, None, 0, y)
        return xx[0] != yy[0] and xx[1] == yy[1]


# 994 - Rotting Oranges - MEDIUM
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        starts, m, n, fresh = [], len(grid), len(grid[0]), 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    starts.append((i, j))
                elif grid[i][j] == 1:
                    fresh += 1
        ans, dq = 0, collections.deque(starts)
        while dq:
            for _ in range(len(dq)):
                x, y = dq.popleft()
                for i, j in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                    if 0 <= i < m and 0 <= j < n and grid[i][j] == 1:
                        grid[i][j] = 2
                        fresh -= 1
                        dq.append((i, j))
            if not dq:
                break
            ans += 1
        return ans if fresh == 0 else -1


# 997 - Find the Town Judge - EASY
class Solution:
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        o = [0] * n
        i = [0] * n
        for a, b in trust:
            o[a - 1] += 1
            i[b - 1] += 1
        for j in range(n):
            if i[j] == n - 1 and o[j] == 0:
                return j + 1
        return -1


# 998 - Maximum Binary Tree II - MEDIUM
class Solution:
    def insertIntoMaxTree(self, root: TreeNode, val: int) -> TreeNode:
        if not root or root.val < val:
            return TreeNode(val, root)
        root.right = self.insertIntoMaxTree(root.right, val)
        return root
