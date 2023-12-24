import bisect, collections, functools, heapq, itertools, math, operator, random, string
from typing import List, Optional, Tuple
import sortedcontainers


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 300 - Longest Increasing Subsequence - MEDIUM
class Solution:
    # O(n ^ 2) / O(n)
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1] * len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    # O(nlogn) / O(n)
    def lengthOfLIS(self, nums: List[int]) -> int:
        tail = []
        for num in nums:
            idx = bisect.bisect_left(tail, num)
            if idx == len(tail):
                tail.append(num)
            else:
                tail[idx] = num
        # keep the 'tail' ordered
        # replace smaller element
        # 'tail' may not be the exact LIS, but has the same length
        # the element after idx is useless at each insert
        # 'tail' will maintain the maximum length of LIS
        return len(tail)

    def lengthOfLIS(self, nums: List[int]) -> int:
        tail = []
        for n in nums:
            l, r = 0, len(tail)
            while l < r:
                mid = l + r >> 1
                if tail[mid] < n:
                    l = mid + 1
                else:
                    r = mid
            if l == len(tail):
                tail.append(n)
            else:
                tail[l] = n
        return len(tail)


# 301 - Remove Invalid Parentheses - HARD
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        l = r = 0
        for c in s:
            if c == "(":
                l += 1
            elif c == ")":
                if l:
                    l -= 1
                else:
                    r += 1
        ans = []

        # cl cr: left or right count
        # dl dr: left or right remain
        @functools.lru_cache(None)
        def dfs(idx, cl, cr, dl, dr, path):
            if idx == len(s):
                if not dl and not dr:
                    ans.append(path)
                return
            if cr > cl or dl < 0 or dr < 0:
                return
            ch = s[idx]
            if ch == "(":
                dfs(idx + 1, cl, cr, dl - 1, dr, path)
            elif ch == ")":
                dfs(idx + 1, cl, cr, dl, dr - 1, path)
            dfs(idx + 1, cl + (ch == "("), cr + (ch == ")"), dl, dr, path + ch)

        dfs(0, 0, 0, l, r, "")
        return ans


# 303 - Range Sum Query - Immutable - EASY
# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(left,right)
class NumArray:
    def __init__(self, nums: List[int]):
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        self.nums = nums

    def sumRange(self, left: int, right: int) -> int:
        return self.nums[right] - (self.nums[left - 1] if left > 0 else 0)


class NumArray:
    def __init__(self, nums: List[int]):
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        self.nums = [0] + nums

    def sumRange(self, left: int, right: int) -> int:
        return self.nums[right + 1] - self.nums[left]


# 304 - Range Sum Query 2D - Immutable - MEDIUM
# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)class NumMatrix:
class NumMatrix:
    # Add zero row and column!!
    # remove preprocess and additional 'if' in 'sumRegion'
    def __init__(self, matrix: List[List[int]]):
        m, n = len(matrix), len(matrix[0])
        self.presum = [[0] * (n + 1) for _ in range(m + 1)]
        self.presum[0][0] = matrix[0][0]
        # for i in range(1, m):
        #     self.presum[i][0] = matrix[i][0] + self.presum[i - 1][0]
        # for j in range(1, n):
        #     self.presum[0][j] = matrix[0][j] + self.presum[0][j - 1]
        for i in range(m):
            for j in range(n):
                self.presum[i + 1][j + 1] = (
                    matrix[i][j]
                    + self.presum[i][j + 1]
                    + self.presum[i + 1][j]
                    - self.presum[i][j]
                )
        return

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return (
            self.presum[row2 + 1][col2 + 1]
            - self.presum[row2 + 1][col1]
            - self.presum[row1][col2 + 1]
            + self.presum[row1][col1]
        )


class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        m, n = len(matrix), len(matrix[0])
        for i in range(m):
            for j in range(1, n):
                matrix[i][j] += matrix[i][j - 1]
        for i in range(1, m):
            for j in range(n):
                matrix[i][j] += matrix[i - 1][j]
        self.matrix = matrix

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return (
            self.matrix[row2][col2]
            - (self.matrix[row1 - 1][col2] if row1 > 0 else 0)
            - (self.matrix[row2][col1 - 1] if col1 > 0 else 0)
            + (self.matrix[row1 - 1][col1 - 1] if row1 > 0 and col1 > 0 else 0)
        )


# 306 - Additive Number - MEDIUM
class Solution:
    def isAdditiveNumber(self, num: str) -> bool:
        for i in range(1, len(num)):
            for j in range(i + 1, len(num)):
                first, second, remaining = num[:i], num[i:j], num[j:]
                if (first.startswith("0") and first != "0") or (
                    second.startswith("0") and second != "0"
                ):
                    continue
                while remaining:
                    third = str(int(first) + int(second))
                    if not remaining.startswith(third):
                        break
                    first = second
                    second = third
                    remaining = remaining[len(third) :]
                if not remaining:
                    return True
        return False

    def isAdditiveNumber(self, num: str) -> bool:
        def check(i, j):
            a = num[: i + 1]
            b = num[i + 1 : j + 1]
            if (a.startswith("0") and a != "0") or (b.startswith("0") and b != "0"):
                return False
            c = str(int(a) + int(b))
            temp = a + b + c
            while len(temp) <= len(num):
                if num == temp:
                    return True
                b, c = c, str(int(b) + int(c))
                temp += c
            return False

        for j in range(1, len(num) - 1):
            for i in range(j):
                if check(i, j):
                    return True
        return False


# 307 - Range Sum Query - Mutable - MEDIUM
# segment tree
# O(n + logn) / O(n), constructor: O(n), update / sunRange: O(logn)
class SegmentTree:
    """根节点下标 0"""

    def __init__(self, nums: List[int]):
        n = len(nums)
        self.t = [0] * (n * 4)
        self.build(nums, 0, 0, n - 1)

    def build(self, nums: List[int], o: int, l: int, r: int):
        if l == r:
            self.t[o] = nums[l]
            return
        m = l + r >> 1
        self.build(nums, o * 2 + 1, l, m)
        self.build(nums, o * 2 + 2, m + 1, r)
        self.t[o] = self.t[o * 2 + 1] + self.t[o * 2 + 2]
        return

    def update(self, o: int, l: int, r: int, idx: int, val: int):
        """将 idx 下标位置更新为 val, self.update(0, 0, n - 1, idx, val)"""
        if l == r:
            self.t[o] = val
            return
        m = l + r >> 1
        if idx <= m:
            self.update(o * 2 + 1, l, m, idx, val)
        else:
            self.update(o * 2 + 2, m + 1, r, idx, val)
        self.t[o] = self.t[o * 2 + 1] + self.t[o * 2 + 2]
        return

    def query(self, o: int, l: int, r: int, L: int, R: int) -> int:
        """返回 [L, R] 闭区间内元素和, self.query(0, 0, self.n - 1, L, R)"""
        if L <= l and r <= R:
            return self.t[o]
        m = l + r >> 1
        res = 0
        if L <= m:
            res += self.query(o * 2 + 1, l, m, L, R)
        if R > m:
            res += self.query(o * 2 + 2, m + 1, r, L, R)
        return res


class NumArray:
    def __init__(self, nums: List[int]):
        self.n = len(nums)
        self.st = SegmentTree(nums)

    def update(self, idx: int, val: int) -> None:
        self.st.update(0, 0, self.n - 1, idx, val)

    def sumRange(self, left: int, right: int) -> int:
        return self.st.query(0, 0, self.n - 1, left, right)


class SegmentTree:
    """基本款"""

    def __init__(self, nums: List[int]):
        """根节点下标 1, 管辖范围 1 - n"""
        n = len(nums)
        self.t = [0] * (n * 4)
        self.build(nums, 1, 1, n)

    def update(self, o: int, l: int, r: int, idx: int, val: int) -> None:
        """给 idx 下标位置 += val, self.update(1, 1, n, idx, val)"""
        if l == r:
            self.t[o] = val
            return
        m = l + r >> 1
        if idx <= m:
            self.update(o << 1, l, m, idx, val)
        else:
            self.update(o << 1 | 1, m + 1, r, idx, val)
        self.t[o] = self.t[o << 1] + self.t[o << 1 | 1]  # push up
        return

    def query(self, o: int, l: int, r: int, L: int, R: int) -> int:
        """返回 [L, R] 闭区间内元素和, self.query(1, 1, n, L, R)"""
        if L <= l and r <= R:
            return self.t[o]
        res = 0
        m = l + r >> 1
        if L <= m:
            res += self.query(o << 1, l, m, L, R)
        if R > m:
            res += self.query(o << 1 | 1, m + 1, r, L, R)
        return res

    def build(self, nums: List[int], o: int, l: int, r: int) -> None:
        if l == r:
            self.t[o] = nums[l - 1]
            return
        m = l + r >> 1
        self.build(nums, o << 1, l, m)
        self.build(nums, o << 1 | 1, m + 1, r)
        self.t[o] = self.t[o << 1] + self.t[o << 1 | 1]
        return


class NumArray:
    def __init__(self, nums: List[int]):
        self.n = len(nums)
        self.st = SegmentTree(nums)

    def update(self, idx: int, val: int) -> None:
        self.st.update(1, 1, self.n, idx + 1, val)

    def sumRange(self, left: int, right: int) -> int:
        return self.st.query(1, 1, self.n, left + 1, right + 1)


# iterative
class NumArray:
    def __init__(self, nums: List[int]):
        self.n = len(nums)
        self.t = [0] * self.n + nums
        for i in range(self.n - 1, 0, -1):
            # self.t[i] = self.t[i << 1] + self.t[i << 1 | 1]
            self.t[i] = self.t[i * 2] + self.t[i * 2 + 1]

    def update(self, i: int, val: int) -> None:
        i = self.n + i
        self.t[i] = val
        while i > 1:
            # self.t[i >> 1] = self.t[i] + self.t[i ^ 1]
            self.t[i // 2] = self.t[i] + self.t[(i - 1) if (i % 2) else (i + 1)]
            i //= 2

    def sumRange(self, left: int, right: int) -> int:
        left = self.n + left
        right = self.n + right
        ans = 0
        while left <= right:
            if left % 2:  # if left & 1
                ans += self.t[left]
                left += 1
            left //= 2  # left >>= 1
            if not (right % 2):  # if not (right & 1)
                ans += self.t[right]
                right -= 1
            right //= 2  # right >>= 1
        return ans


# Fenwick tree
# O(nlogn + logn) / O(n), constructor: O(nlogn), add / query: O(logn)
class BIT:
    def __init__(self, n: int):
        self.tree = [0] * n

    def add(self, i: int, d: int = 1) -> None:
        while i < len(self.tree):
            self.tree[i] += d
            i += i & -i
        return

    def query(self, i: int) -> int:
        res = 0
        while i > 0:
            res += self.tree[i]
            i &= i - 1
        return res

    def rsum(self, l: int, r: int) -> int:
        return self.query(r) - self.query(l - 1)


class NumArray:
    def __init__(self, nums: List[int]):
        self.bit = BIT(len(nums) + 1)
        for i, x in enumerate(nums):
            self.bit.add(i + 1, x)

    def update(self, index: int, val: int) -> None:
        ori = self.bit.rsum(index + 1, index + 1)
        self.bit.add(index + 1, val - ori)
        return

    def sumRange(self, left: int, right: int) -> int:
        return self.bit.rsum(left + 1, right + 1)


class NumArray:
    def __init__(self, nums: List[int]):
        self.bit = BIT(len(nums) + 1)
        for i, x in enumerate(nums):
            self.bit.add(i + 1, x)
        self.nums = [0] + nums

    def update(self, index: int, val: int) -> None:
        self.bit.add(index + 1, val - self.nums[index + 1])
        self.nums[index + 1] = val
        return

    def sumRange(self, left: int, right: int) -> int:
        return self.bit.rsum(left + 1, right + 1)


# TLE
class NumArray:
    def __init__(self, nums):
        self.update = nums.__setitem__
        self.sumRange = lambda i, j: sum(nums[i : j + 1])


class NumArray:
    def __init__(self, nums: List[int]):
        self.nums = nums
        self.sum = sum(self.nums)

    def update(self, index: int, val: int) -> None:
        self.sum = self.sum + val - self.nums[index]
        self.nums[index] = val

    def sumRange(self, left: int, right: int) -> int:
        return self.sum - sum(self.nums[:left]) - sum(self.nums[right + 1 :])


# 309 - Best Time to Buy and Sell Stock with Cooldown - MEDIUM
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        n = len(prices)
        f0, f1, f2 = -prices[0], 0, 0
        for i in range(1, n):
            f0, f1, f2 = max(f0, f2 - prices[i]), f0 + prices[i], max(f1, f2)
            # newf0 = max(f0, f2 - prices[i])
            # newf1 = f0 + prices[i]
            # newf2 = max(f1, f2)
            # f0, f1, f2 = newf0, newf1, newf2
        return max(f1, f2)


# 310 - Minimum Height Trees - MEDIUM
class Solution:
    # O(n) / O(n)
    # Topological Sorting, find the middle nodes in the longest path of a graph
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if not edges:
            return [0]
        g = collections.defaultdict(list)
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)
        leaves = []
        degree = []
        for i in range(n):
            if len(g[i]) == 1:
                leaves.append(i)
            degree.append(len(g[i]))
        while n > 2:
            new = []
            for l in leaves:
                for node in g[l]:
                    degree[node] -= 1
                    if degree[node] == 1:
                        new.append(node)
            n -= len(leaves)
            leaves = new
        return leaves


# 312 - Burst Balloons - HARD
class Solution:
    # not pass
    @functools.lru_cache(None)
    def maxCoins(self, nums: List[int]) -> int:
        def backtrack(nums: List[int], cur):
            if len(nums) == 0:
                self.ans = max(self.ans, cur)
                return
            for i in range(len(nums)):
                left = nums[i - 1] if i - 1 >= 0 else 1
                right = nums[i + 1] if i + 1 < len(nums) else 1
                cur += left * nums[i] * right
                backtrack(nums[: i - 1] + nums[i + 1 :], cur)
            return

        nums = [n for n in nums if n]
        self.ans = -math.inf
        backtrack(nums[1:-1], 0)
        return self.ans

    def maxCoins(self, A: List[int]) -> int:
        # a test case that all elements are '100'
        if len(A) > 1 and len(set(A)) == 1:
            return (A[0] ** 3) * (len(A) - 2) + A[0] ** 2 + A[0]
        A, n = [1] + A + [1], len(A) + 2
        dp = [[0] * n for _ in range(n)]
        # why bottom to up: must solve subquestion first
        for i in range(n - 2, -1, -1):
            for j in range(i + 2, n):
                dp[i][j] = max(
                    A[i] * A[k] * A[j] + dp[i][k] + dp[k][j] for k in range(i + 1, j)
                )
        return dp[0][n - 1]

    def maxCoins(self, nums: List[int]) -> int:
        if len(nums) > 1 and len(set(nums)) == 1:  # speed up
            return (nums[0] ** 3) * (len(nums) - 2) + nums[0] ** 2 + nums[0]
        nums = [1] + nums + [1]
        # or: nums = [1] + [n for n in nums if n] + [1]
        dp = [[0] * len(nums) for _ in range(len(nums))]
        for i in range(len(nums) - 1, -1, -1):
            for j in range(i + 2, len(nums)):
                for k in range(i + 1, j):
                    dp[i][j] = max(
                        dp[i][j], dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j]
                    )
        return dp[0][-1]

    def maxCoins(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        elif len(set(nums)) == 1:
            n = nums[0]
            return n**3 * (len(nums) - 2) + n * n + n
        nums = [1] + [n for n in nums if n] + [1]
        N = len(nums)

        @functools.lru_cache(None)
        def helper(lo, hi):
            if lo > hi:
                return 0
            res = -math.inf
            for i in range(lo, hi + 1):
                gain = nums[i] * nums[lo - 1] * nums[hi + 1]
                res = max(res, gain + helper(lo, i - 1) + helper(i + 1, hi))
            return res

        return helper(1, N - 2)


# 314 - Binary Tree Vertical Order Traversal - MEDIUM
class Solution:
    # dfs
    def verticalOrder(self, root: TreeNode) -> List[List[int]]:
        # Use a dict to store our answers, keys will be column idxs.
        ans = collections.defaultdict(list)

        def dfs(root: TreeNode, row: int, col: int) -> None:
            if not root:
                return
            # Append root vals to column in our dict.
            ans[col].append((row, root.val))
            dfs(root.left, row + 1, col - 1)
            dfs(root.right, row + 1, col + 1)
            return

        dfs(root, 0, 0)
        # Sort our dict by keys (column vals)
        ans = dict(sorted(ans.items()))
        ret = []
        # Loop through our sorted dict appending vals sorted by height (top down order).
        for _, v in ans.items():
            ret.append([x[1] for x in sorted(v, key=lambda x: x[0])])
        return ret

    # bfs
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        nodes = collections.defaultdict(list)
        dq = collections.deque([(root, 0)])
        while dq:
            node, pos = dq.popleft()
            if node:
                nodes[pos].append(node.val)
                dq.append((node.left, pos - 1))
                dq.append((node.right, pos + 1))
        # sorted by keys of defaultdict
        return [nodes[i] for i in sorted(nodes)]


# 316 - Remove Duplicate Letters - MEDIUM
class Solution:
    # O(n * 26 * 26) / O(26 * 2)
    def removeDuplicateLetters(self, s: str) -> str:
        cnt = collections.Counter(s)
        st = []
        for c in s:
            if c not in st:  # maximum length of st = 26
                while st and st[-1] > c and cnt[st[-1]] > 0:
                    st.pop()
                st.append(c)
            cnt[c] -= 1
        return "".join(st)

    # O(n * 26) / O(26 * 3)
    def removeDuplicateLetters(self, s: str) -> str:
        pos = {c: i for i, c in enumerate(s)}  # last occurence
        st = []
        vis = set()
        for i, c in enumerate(s):
            if c not in vis:
                while st and c < st[-1] and i < pos[st[-1]]:
                    vis.remove(st.pop())
                st.append(c)
                vis.add(c)
        return "".join(st)


# 318 - Maximum Product of Word Lengths - MEDIUM
class Solution:
    def maxProduct(self, words: List[str]) -> int:
        s = [set(x) for x in words]
        maxL = 0
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if len(s[i].intersection(s[j])) == 0:
                    maxL = max(maxL, len(words[i]) * len(words[j]))
        return maxL


# 319 - Bulb Switcher - MEDIUM
class Solution:
    def bulbSwitch(self, n: int) -> int:
        ans, i = 0, 1
        while i * i <= n:
            i += 1
            ans += 1
        return ans

    def bulbSwitch(self, n: int) -> int:
        return int(math.sqrt(n))


# 322 - Coin Change - MEDIUM
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [0] + [float("inf")] * amount
        for i in range(1, amount + 1):
            dp[i] = min(dp[i - c] if i - c >= 0 else float("inf") for c in coins) + 1
        return dp[-1] if dp[-1] != float("inf") else -1

    def coinChange(self, coins, amount):
        dp = [0] + [float("inf")] * amount
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[-1] if dp[-1] != float("inf") else -1

    def coinChange(self, coins: List[int], amount: int) -> int:
        ans = 0
        dq = collections.deque([amount])
        vis = set()
        while dq:
            for _ in range(len(dq)):
                val = dq.popleft()
                if val == 0:
                    return ans
                for coin in coins:
                    if val >= coin and val - coin not in vis:
                        vis.add(val - coin)
                        dq.append(val - coin)
            ans += 1
        return -1

    def coinChange(self, coins: List[int], amount: int) -> int:
        @functools.lru_cache(None)
        def dp(amount: int) -> int:
            if amount == 0:
                return 0
            ans = math.inf
            for coin in coins:
                if amount >= coin:
                    ans = min(ans, dp(amount - coin) + 1)
            return ans

        ans = dp(amount)
        return ans if ans != math.inf else -1

    def coinChange(self, coins: List[int], amount: int) -> int:
        @functools.lru_cache(None)
        def dp(amount: int) -> int:
            if amount == 0:
                return 0
            if amount < 0:
                return float("inf")
            return min(dp(amount - coin) + 1 for coin in coins)

        return dp(amount) if dp(amount) != float("inf") else -1


# 324 - Wiggle Sort II - MEDIUM
class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        arr = sorted(nums)
        left = arr[: (len(nums) + 1) // 2]
        right = arr[(len(nums) + 1) // 2 :]
        i = 0
        while i < len(nums):
            if left:
                nums[i] = left.pop()
                i += 1
            if right:
                nums[i] = right.pop()
                i += 1
        return

    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        nums.sort()
        n = len(nums)
        nums[::2], nums[1::2] = (
            nums[: (n + 1) // 2][::-1],
            nums[(n + 1) // 2 :][::-1],
        )
        return


# 328 - Odd Even Linked List - MEDIUM
class Solution:
    # O(n) / O(n)
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        num = 1
        odd, even = ListNode(-1), ListNode(-1)
        cpodd, cpeven = odd, even
        while head:
            if num & 1:
                odd.next = ListNode(head.val)
                odd = odd.next
            else:
                even.next = ListNode(head.val)
                even = even.next
            head = head.next
            num += 1
        odd.next = cpeven.next
        return cpodd.next

    # O(1) / O(n)
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        odd, even = head, head.next
        evenHead = even
        while even and even.next:
            odd.next = odd.next.next
            even.next = even.next.next
            odd = odd.next
            even = even.next
        odd.next = evenHead
        return head


# 329 - Longest Increasing Path in a Matrix - HARD
class Solution:
    # O(mn) / O(mn)
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        @functools.lru_cache(None)
        def dfs(i: int, j: int) -> int:
            t = 1
            for x, y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                # if 0 <= x < m and 0 <= y < n and matrix[x][y] > matrix[i][j]:  # works too, decreasing order
                if 0 <= x < m and 0 <= y < n and matrix[x][y] < matrix[i][j]:
                    t = max(t, dfs(x, y) + 1)
            return t

        m = len(matrix)
        n = len(matrix[0])
        return max(dfs(i, j) for i in range(m) for j in range(n))

    # O(mn * logmn) / O(mn)
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m = len(matrix)
        n = len(matrix[0])
        g = [[1 for _ in range(n)] for _ in range(m)]
        pair = []
        for i in range(m):
            for j in range(n):
                pair.append([matrix[i][j], i, j])
        pair.sort()
        for i in range(m * n):
            v, x, y = pair[i]
            for nx, ny in [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]:
                if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] > v:
                    g[nx][ny] = max(g[nx][ny], g[x][y] + 1)
        return max(max(r) for r in g)

    # O(mn) / O(mn), topological sort
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m = len(matrix)
        n = len(matrix[0])
        f = [[1 for _ in range(n)] for _ in range(m)]
        ind = [[0 for _ in range(n)] for _ in range(m)]  # in degree
        dq = collections.deque()
        for i in range(m):
            for j in range(n):
                for x, y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= x < m and 0 <= y < n and matrix[x][y] < matrix[i][j]:
                        ind[i][j] += 1
                if ind[i][j] == 0:
                    dq.append((i, j))
        while dq:
            for _ in range(len(dq)):
                i, j = dq.popleft()
                for x, y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= x < m and 0 <= y < n and matrix[x][y] > matrix[i][j]:
                        f[x][y] = max(f[x][y], f[i][j] + 1)
                        ind[x][y] -= 1
                        if ind[x][y] == 0:
                            dq.append((x, y))
        return max(max(r) for r in f)

    # optimize 'dp'
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m = len(matrix)
        n = len(matrix[0])
        ind = [[0 for _ in range(n)] for _ in range(m)]  # in degree
        dq = collections.deque()
        for i in range(m):
            for j in range(n):
                for x, y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= x < m and 0 <= y < n and matrix[x][y] < matrix[i][j]:
                        ind[i][j] += 1
                if ind[i][j] == 0:
                    dq.append((i, j))
        ans = 0
        while dq:
            ans += 1
            for _ in range(len(dq)):
                i, j = dq.popleft()
                for x, y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= x < m and 0 <= y < n and matrix[x][y] > matrix[i][j]:
                        ind[x][y] -= 1
                        if ind[x][y] == 0:
                            dq.append((x, y))
        return ans


# 334 - Increasing Triplet Subsequence - MEDIUM
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        if len(nums) < 3:
            return False
        first, second = float("inf"), float("inf")
        for n in nums:
            if n <= first:
                first = n
            elif first < n <= second:
                second = n
            elif n > second:
                return True
        return False


# 337 - House Robber III - MEDIUM
# recursive
class Solution:
    memory = {}

    def rob(self, root: TreeNode) -> int:
        if root is None:
            return 0
        if root.left is None and root.right is None:
            return root.val
        if self.memory.get(root) is not None:
            return self.memory[root]
        # rob root
        val1 = root.val
        if root.left:
            val1 += self.rob(root.left.left) + self.rob(root.left.right)
        if root.right:
            val1 += self.rob(root.right.left) + self.rob(root.right.right)
        # not rob root
        val2 = self.rob(root.left) + self.rob(root.right)
        self.memory[root] = max(val1, val2)
        return max(val1, val2)


# dp
class Solution:
    def rob(self, root: TreeNode) -> int:
        result = self.rob_tree(root)
        return max(result[0], result[1])

    def rob_tree(self, node: TreeNode) -> int:
        if node is None:
            return (0, 0)  # (rob this node，not rob this node)
        left = self.rob_tree(node.left)
        right = self.rob_tree(node.right)
        val1 = node.val + left[1] + right[1]  # rob node
        val2 = max(left[0], left[1]) + max(right[0], right[1])  # not rob this node
        return (val1, val2)


# 338 - Counting Bits - EASY
class Solution:
    # x is even, bits[x] = bits[x//2]
    # x is odd, bits[x] = bits[(x+1)//2] + 1
    def countBits(self, n: int) -> List[int]:
        bits = [0]
        for i in range(1, n + 1):
            bits.append(bits[i >> 1] + (i & 1))
        return bits

    def countBits(self, n: int) -> List[int]:
        bits = [0]
        for i in range(1, n + 1):
            bits.append(bits[i & (i - 1)] + 1)
        return bits

    def countBits(self, n: int) -> List[int]:
        bits = [0]
        highBit = 0
        for i in range(1, n + 1):
            if i & (i - 1) == 0:  # n > 0 && n & (n - 1) == 0
                highBit = i  # is the power of 2
            bits.append(bits[i - highBit] + 1)
        return bits


# 339 - Nested List Weight Sum - MEDIUM
class NestedInteger:
    def __init__(self, value=None):
        """
        If value is not specified, initializes an empty list.
        Otherwise initializes a single integer equal to value.
        """

    def isInteger(self):
        """
        @return True if this NestedInteger holds a single integer, rather than a nested list.
        :rtype bool
        """

    def add(self, elem):
        """
        Set this NestedInteger to hold a nested list and adds a nested integer elem to it.
        :rtype void
        """

    def setInteger(self, value):
        """
        Set this NestedInteger to hold a single integer equal to value.
        :rtype void
        """

    def getInteger(self):
        """
        @return the single integer that this NestedInteger holds, if it holds a single integer
        Return None if this NestedInteger holds a nested list
        :rtype int
        """

    def getList(self):
        """
        @return the nested list that this NestedInteger holds, if it holds a nested list
        Return None if this NestedInteger holds a single integer
        :rtype List[NestedInteger]
        """


class Solution:
    # dfs
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        self.ans = 0

        def dfs(nestedList: List[NestedInteger], depth: int):
            if not nestedList:
                return
            for i in nestedList:
                if i.isInteger():
                    self.ans += i.getInteger() * depth
                else:
                    dfs(i.getList(), depth + 1)

        dfs(nestedList, 1)
        return self.ans

    # bfs
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        ans, depth = 0, 1
        stack = collections.deque([nestedList])
        while stack:
            for _ in range(len(stack)):
                n = stack.popleft()
                for i in n:
                    if i.isInteger():
                        ans += i.getInteger() * depth
                    else:
                        stack.append(i.getList())
            depth += 1
        return ans

    """
    flatten trick about a list of lists
    >>> sum([[1, 2], [2, 4]], [])
    [1, 2, 2, 4]
    """

    def depthSum(self, nestedList):
        depth, ret = 1, 0
        while nestedList:
            ret += depth * sum([x.getInteger() for x in nestedList if x.isInteger()])
            nestedList = sum([x.getList() for x in nestedList if not x.isInteger()], [])
            depth += 1
        return ret


# 343 - Integer Break - MEDIUM
class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0] * (n + 1)
        for i in range(2, n + 1):
            for j in range(i):
                dp[i] = max(dp[i], j * (i - j), j * dp[i - j])
        return dp[n]

    def integerBreak(self, n: int) -> int:
        if n <= 3:
            return n - 1
        d, mod = n // 3, n % 3
        if mod == 0:
            return 3**d
        if mod == 1:
            return 3 ** (d - 1) * 4
        return 3**d * 2


# 344 - Reverse String - EASY
class Solution:
    def reverseString(self, s: List[str]) -> None:
        for i in range(len(s) // 2):
            s[i], s[-i - 1] = s[-i - 1], s[i]
        # s[:] = s[::-1]
        # s.reverse()
        return


# 345 - Reverse Vowels of a String - EASY
class Solution:
    def reverseVowels(self, s: str) -> str:
        a = []
        b = []
        for i, c in enumerate(s):
            if c in "aeiouAEIOU":
                a.append(c)
            else:
                b.append((c, i))
        j = 0
        ans = ""
        for i in range(len(s)):
            if j < len(b) and i == b[j][1]:
                ans += b[j][0]
                j += 1
            else:
                ans += a.pop()
        return ans

    def reverseVowels(self, s: str) -> str:
        n = len(s)
        s = list(s)
        i = 0
        j = n - 1
        while i < j:
            while i < n and s[i] not in "aeiouAEIOU":
                i += 1
            while j > 0 and s[j] not in "aeiouAEIOU":
                j -= 1
            if i < j:
                s[i], s[j] = s[j], s[i]
                i += 1
                j -= 1
        return "".join(s)


# 346


# 347 - Top K Frequent Elements - MEDIUM
class Solution:
    # O(nlogn) / O(n)
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        cnt = collections.Counter(nums)
        return [i[0] for i in sorted(cnt.items(), key=lambda x: x[1])[-k:]]
        # return [i[0] for i in sorted(cnt.items(), key=lambda x: x[1], reverse=True)[:k]]

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        cnt = collections.Counter(nums)
        times = sorted(cnt.items(), key=lambda k: k[1])
        ans = []
        while k != 0 and len(times) > 0:
            ans.append(times.pop()[0])
            k -= 1
        return ans

    # O(n + Klogn)/ O(n)
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        ans = []
        cnt = collections.Counter(nums)
        max_heap = [(-val, key) for key, val in cnt.items()]
        heapq.heapify(max_heap)  # heapify costs 'n'
        for _ in range(k):
            ans.append(heapq.heappop(max_heap)[1])  # heappop costs 'Klogn'
        return ans

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        cnt = collections.Counter(nums)
        return heapq.nlargest(k, cnt.keys(), key=cnt.get)


# 349 - Intersection of Two Arrays - EASY
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        return list(set(nums1) & set(nums2))
        return set(nums1).intersection(set(nums2))


# 350 - Intersection of Two Arrays II - EASY
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        nums2.sort()
        p1 = p2 = 0
        ans = []
        while p1 < len(nums1) and p2 < len(nums2):
            if nums1[p1] > nums2[p2]:
                p2 += 1
            elif nums1[p1] < nums2[p2]:
                p1 += 1
            else:
                ans.append(nums1[p1])
                p1 += 1
                p2 += 1
        return ans

    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        cnt1, cnt2 = collections.Counter(nums1), collections.Counter(nums2)
        s1, s2 = set(nums1), set(nums2)
        ans = []
        for n in s1.intersection(s2):
            ans += [n] * min(cnt1[n], cnt2[n])
        return ans

    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        cnt = collections.Counter(nums1)
        ans = []
        for num in nums2:
            if cnt[num] > 0:
                ans += (num,)
                cnt[num] -= 1
        return ans


# 357 - Count Numbers with Unique Digits - MEDIUM
class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        if n == 0:
            return 1
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 10
        for i in range(2, n + 1):
            dp[i] = (dp[i - 1] - dp[i - 2]) * (10 - (i - 1)) + dp[i - 1]
        return dp[-1]

    def countNumbersWithUniqueDigits(self, n: int) -> int:
        if n == 0:
            return 1
        if n == 1:
            return 10
        ans = 10
        cur = 9
        for i in range(n - 1):
            cur *= 9 - i
            ans += cur
        return ans


# 367 - Valid Perfect Square - EASY
# binary search
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        left, right = 0, num
        while left <= right:
            mid = (left + right) // 2
            if mid * mid > num:
                right = mid - 1
            elif mid * mid < num:
                left = mid + 1
            else:
                return True
        return False


# math: sum of odd -> 1+3+5+7+... = n^2
#       (n+1)^2 - n^2 = 2n+1
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        odd = 1
        while num > 0:
            num -= odd
            odd += 2
        if num == 0:
            return True
        return False


# 368 - Largest Divisible Subset - MEDIUM
# dynamic programming
# dp[i]: considering the first i numbers,
#        have the largest divisible subset ending with index i
# since we have to give the final solution,
# we need extra 'g[]' to record where does each state transfer from
#
# For the problem of finding the number of solutions,
# it is the most common means to use an extra array
# to record where the state is transferred from.
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        nums.sort()
        n = len(nums)
        dp, g = [0] * n, [0] * n
        for i in range(n):
            # including number itself, so length start with 1
            length, prev_idx = 1, i
            for j in range(i):
                if nums[i] % nums[j] == 0:
                    # update the max length and where it come from
                    if dp[j] + 1 > length:
                        length = dp[j] + 1
                        prev_idx = j
            # record final 'length' and 'come from'
            dp[i] = length
            g[i] = prev_idx
        max_len = idx = -1
        for i in range(n):
            if dp[i] > max_len:
                max_len = dp[i]
                idx = i
        ans = []
        while len(ans) < max_len:
            ans.append(nums[idx])
            idx = g[idx]
        ans.reverse()
        return ans


# greedy
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        nums.sort()
        f = [[x] for x in nums]  # answer at nums[i]
        for i in range(len(nums)):
            for j in range(i):
                if nums[i] % nums[j] == 0 and len(f[i]) < len(f[j]) + 1:
                    f[i] = f[j] + [nums[i]]
        return max(f, key=len)


# 372 Super Pow
class Solution:
    def superPow(self, a: int, b: List[int]) -> int:
        return pow(a, int("".join(map(str, b))), 1337)


class Solution:
    def superPow(self, a: int, b: List[int]) -> int:
        ans = 1
        for digit in b:
            ans = pow(ans, 10, 1337) * pow(a, digit, 1337) % 1337
        return ans


class Solution:
    def superPow(self, a: int, b: List[int]) -> int:
        ans = 1
        for digit in reversed(b):
            ans = ans * pow(a, digit, 1337) % 1337
            a = pow(a, 10, 1337)
        return ans


# 373 - Find K Pairs with Smallest Sums - MEDIUM
class Solution:
    def kSmallestPairs(
        self, nums1: List[int], nums2: List[int], k: int
    ) -> List[List[int]]:
        def push(i: int, j: int):
            if i < len(nums1) and j < len(nums2):
                heapq.heappush(queue, [nums1[i] + nums2[j], i, j])
            return

        queue, ans = [], []
        push(0, 0)
        while queue and len(ans) < k:
            _, i, j = heapq.heappop(queue)
            ans.append([nums1[i], nums2[j]])
            push(i, j + 1)
            if j == 0:
                push(i + 1, 0)
        return ans

    def kSmallestPairs(
        self, nums1: List[int], nums2: List[int], k: int
    ) -> List[List[int]]:
        ans = []
        queue = [(nums1[i] + nums2[0], i, 0) for i in range(min(k, len(nums1)))]
        while queue and len(ans) < k:
            _, i, j = heapq.heappop(queue)
            ans.append([nums1[i], nums2[j]])
            if j + 1 < len(nums2):
                heapq.heappush(queue, (nums1[i] + nums2[j + 1], i, j + 1))
        return ans


# 374 - Guess Number Higher or Lower - EASY
# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num: int) -> int:


class Solution:
    def guessNumber(self, n: int) -> int:
        left, right, mid = 1, n, 1
        while guess(mid) != 0:
            if guess(mid) > 0:
                left = mid + 1
            else:
                right = mid - 1
            mid = (right + left) // 2
        return mid


def guess(self, n: int) -> int:
    pick = 1  # specify internally
    if n > pick:
        return 1
    elif n < pick:
        return -1
    else:
        return 0


class Solution:
    def guessNumber(self, n: int) -> int:
        left, right = 1, n
        while left < right:
            mid = (left + right) // 2
            if guess(mid) <= 0:
                right = mid  # in [left, mid]
            else:
                left = mid + 1  # in [mid+1, right]

        # at this time left == right
        return left


# 375 - Guess Number Higher or Lower II - MEDIUM
# dp[i][j] means that whatever the number we pick in in [i, j], the minimum money we use to win the game
# dp[1][1] means we have 1 number 1 -> dp[1][1] = 1
# dp[1][2] means we have 2 numbers 1, 2 -> dp[1][2] = 1
# dp[2][3] means we have 2 numbers 2, 3 -> dp[2][3] = 2
# dp[1][3] means we have 3 numbers 1, 2, 3
#   -> dp[2][3] = min(max(0,1+dp[2][3]), max(0,2+dp[1][1],2+dp[3][3]), max(0,3+dp[1][2]))
#                       guess 1                   guess 2                     guess 3
# we can use the downside and leftside value to calcutate dp[i][j]
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        # # intialize
        dp = [[0] * (n + 1) for _ in range(n + 1)]  # dp[n+1][n+1]
        for i in range(n + 1):
            dp[i][i] = 0
        # start with the second column
        for j in range(2, n + 1):
            # from bottom to top
            i = j - 1
            while i >= 1:
                # calculate every split point
                for k in range(i + 1, j):
                    dp[i][j] = min(k + max(dp[i][k - 1], dp[k + 1][j]), dp[i][j])
                # calculate both sides
                dp[i][j] = min(dp[i][j], i + dp[i + 1][j], j + dp[i][j - 1])
                dp[i][j] = min(dp[i][j], j + dp[i][j - 1])
                i -= 1

        return dp[1][n]

    def getMoneyAmount(self, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n - 1, 0, -1):
            for j in range(i + 1, n + 1):
                dp[i][j] = min(k + max(dp[i][k - 1], dp[k + 1][j]) for k in range(i, j))
        return dp[1][n]


# 376 - Wiggle Subsequence - MEDIUM
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        pre, cur, ans = 0, 0, 1
        for i in range(len(nums) - 1):
            cur = nums[i + 1] - nums[i]
            if cur * pre <= 0 and cur != 0:
                ans += 1
                pre = cur
        return ans


# 377 - Combination Sum IV - MEDIUM
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [1] + [0] * target
        for i in range(1, target + 1):
            for num in nums:
                if num <= i:
                    dp[i] += dp[i - num]
        return dp[-1]


# 380 - Insert Delete GetRandom O(1) - MEDIUM
class RandomizedSet:
    def __init__(self):
        self.arr = []
        self.v2i = {}

    def insert(self, val: int) -> bool:
        if val in self.v2i:
            return False
        self.v2i[val] = len(self.arr)
        self.arr.append(val)
        return True

    # replace the element to be deleted with the last element, then pop the last one
    def remove(self, val: int) -> bool:
        if val not in self.v2i:
            return False
        i = self.v2i[val]
        self.v2i[self.arr[-1]] = i
        self.arr[i] = self.arr[-1]
        self.arr.pop()
        del self.v2i[val]
        return True

    def getRandom(self) -> int:
        return random.choice(self.arr)


# 382 - Linked List Random Node - MEDIUM
class Solution:
    def __init__(self, head: Optional[ListNode]):
        self.node = []
        while head:
            self.node.append(head.val)
            head = head.next
        return

    def getRandom(self) -> int:
        return random.choice(self.node)


# reservoir sampling


# 383 - Ransom Note - EASY
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        return not collections.Counter(ransomNote) - collections.Counter(magazine)


# 384 - Shuffle an Array - MEDIUM
# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.reset()
# param_2 = obj.shuffle()
class Solution:
    def __init__(self, nums: List[int]):
        self.nums = nums[:]
        self.cp = nums[:]

    def reset(self) -> List[int]:
        self.nums[:] = self.cp[:]
        return self.nums

    def shuffle(self) -> List[int]:
        random.shuffle(self.nums)
        return self.nums

    # Fisher-Yates Algorithm
    # the same as built-in function: 'random.shuffle'
    def shuffle(self) -> List[int]:
        n = len(self.nums)
        for i in range(n):
            idx = random.randrange(i, n)
            self.nums[i], self.nums[idx] = self.nums[idx], self.nums[i]
        return self.nums


# 385 - Mini Parser - MEDIUM
class Solution:
    def deserialize(self, s: str) -> NestedInteger:
        def dfs(elem):
            if type(elem) == int:
                return NestedInteger(elem)
            li = NestedInteger()
            for i in elem:
                li.add(dfs(i))
            return li

        return dfs(eval(s))

    def deserialize(self, s: str) -> NestedInteger:
        return NestedInteger(eval(s))

    def deserialize(self, s: str) -> NestedInteger:
        stack = []
        num = ""
        last = None
        for c in s:
            if c.isdigit() or c == "-":
                num += c
            elif c == "," and num:
                stack[-1].add(NestedInteger(int(num)))
                num = ""
            elif c == "[":
                elem = NestedInteger()
                if stack:
                    stack[-1].add(elem)
                stack.append(elem)
            elif c == "]":
                if num:
                    stack[-1].add(NestedInteger(int(num)))
                    num = ""
                last = stack.pop()
        return last if last else NestedInteger(int(num))


# 386 - Lexicographical Numbers - MEDIUM
class Solution:
    def lexicalOrder(self, n: int) -> List[int]:
        ans = []
        num = 1
        for _ in range(n):
            ans.append(num)
            if num * 10 <= n:
                num *= 10
            else:
                while num % 10 == 9 or num >= n:
                    num //= 10
                num += 1
        return ans

    def lexicalOrder(self, n: int) -> List[int]:
        ans = []
        cur = 1
        for _ in range(n):
            ans.append(cur)
            if cur * 10 <= n:
                cur *= 10
            else:
                if cur >= n:
                    cur //= 10
                cur += 1
                while cur % 10 == 0:
                    cur //= 10
        return ans

    def lexicalOrder(self, n: int) -> List[int]:
        ans = []
        a = 1
        while len(ans) < n:
            while a <= n:
                ans.append(a)
                a *= 10
            while a % 10 == 9 or a >= n:
                a //= 10
            a += 1
        return ans

    def lexicalOrder(self, n):
        def dfs(k):
            if k <= n:
                ans.append(k)
                t = 10 * k
                if t <= n:
                    for i in range(10):
                        dfs(t + i)
            return

        ans = []
        for i in range(1, 10):
            dfs(i)
        return ans

    def lexicalOrder(self, n: int) -> List[int]:
        return sorted(range(1, n + 1), key=str)


# 387 - First Unique Character in a String - EASY
class Solution:
    # fastest
    def firstUniqChar(self, s: str) -> int:
        candi = [chr(i) for i in range(97, 123)]
        ans = float("inf")
        for ch in candi:
            if ch in s and s.find(ch) == s.rfind(ch):
                if ans > s.find(ch):
                    ans = s.find(ch)
        return ans if ans != float("inf") else -1

    # dict is ordered after Python3.6
    def firstUniqChar(self, s: str) -> int:
        frequency = collections.Counter(s)
        for i, ch in enumerate(s):
            if frequency[ch] == 1:
                return i
        return -1

    def firstUniqChar(self, s: str) -> int:
        dic = {}
        for i, ch in enumerate(s):
            if (not dic.get(ch)) and (ch not in s[i + 1 :]):
                return i
            dic[ch] = True
        return -1


# 388 - Longest Absolute File Path - MEDIUM
class Solution:
    def lengthLongestPath(self, s: str) -> int:
        ans = 0
        m = {-1: 0}
        for p in s.split("\n"):
            depth = p.count("\t")
            m[depth] = m[depth - 1] + len(p) - depth
            if p.count("."):
                ans = max(ans, m[depth] + depth)  # depth = '/'
        return ans

    # why this version not work?
    def lengthLongestPath(self, s: str) -> int:
        ans = 0
        m = {-1: 0}
        for p in s.split("\n"):
            depth = p.count("\t")
            m[depth] = m[depth - 1] + len(p)
            # if it does not have '.', the m[depth] will be wrong
            if p.count("."):
                ans = max(ans, m[depth])
        return ans


# 389 - Find the Difference - EASY
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        xor = 0
        for ch in s + t:
            xor ^= ord(ch)
        return chr(xor)
        return list((collections.Counter(t) - collections.Counter(s)))[0]
        return chr(sum([ord(x) for x in t]) - sum([ord(x) for x in s]))


# 390 - Elimination Game - MEDIUM
class Solution(object):
    def lastRemaining(self, n):
        def helper(n: int, isLeft: bool) -> int:
            if n == 1:
                return 1
            # if started from left side the odd elements will be removed, the only remaining ones will the the even i.e.
            # [1 2 3 4 5 6 7 8 9] => [2 4 6 8] => 2*[1 2 3 4]
            if isLeft:
                return 2 * helper(n // 2, False)
            # same as left side the odd elements will be removed
            elif n % 2 == 1:
                return 2 * helper(n // 2, True)
            # even elements will be removed and the only left ones will be [1 2 3 4 5 6] => [1 3 5] => 2*[1 2 3] - 1
            else:
                return 2 * helper(n // 2, True) - 1

        return helper(n, True)

    def lastRemaining(self, n: int) -> int:
        startLeft, ans, step = True, 1, 1
        while n > 1:
            if startLeft or n % 2 == 1:
                ans += step
            startLeft = not startLeft
            step *= 2
            n //= 2
        return ans


# 392 - Is Subsequence - EASY
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        i = j = 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1
        return i == len(s)

    def isSubsequence(self, s: str, t: str) -> bool:
        if not s:
            return True
        j = 0
        for i in range(len(t)):
            if t[i] == s[j]:
                j += 1
            if j == len(s):
                return True
        return False


# 393 - UTF-8 Validation - MEDIUM
class Solution:
    def validUtf8(self, data: List[int]) -> bool:
        k = 0
        for n in data:
            if k == 0:
                if n & 0b10000000 == 0b00000000:
                    k = 0
                elif n & 0b11100000 == 0b11000000:
                    k = 1
                elif n & 0b11110000 == 0b11100000:
                    k = 2
                elif n & 0b11111000 == 0b11110000:
                    k = 3
                else:
                    return False
            else:
                if n & 0b11000000 != 0b10000000:
                    return False
                k -= 1
        return k == 0


# 394 - Decode String - MEDIUM
class Solution:
    def decodeString(self, s: str) -> str:
        st = []
        d = 0
        ans = ""
        for c in s:
            if c == "[":
                st.append(ans)
                st.append(d)
                ans = ""
                d = 0
            elif c == "]":
                pre_num = st.pop()
                pre_string = st.pop()
                ans = pre_string + pre_num * ans
            elif c.isdigit():
                d = d * 10 + int(c)
            else:
                ans += c
        return ans


# 396 - Rotate Function - MEDIUM
class Solution:
    # O(n) / O(1)
    def maxRotateFunction(self, nums: List[int]) -> int:
        n = len(nums)
        s = sum(nums)
        dp = [0] * n
        dp[0] = sum(i * v for i, v in enumerate(nums))
        for i in range(1, n):
            dp[i] = dp[i - 1] + s - nums[-i] * n
        return max(dp)

    def maxRotateFunction(self, nums: List[int]) -> int:
        mx = cur = sum(i * v for i, v in enumerate(nums))
        s = sum(nums)
        n = len(nums)
        for i in range(1, n):
            cur += s - n * nums[-i]
            # mx = max(mx, cur)  # slow
            mx = cur if cur > mx else mx  # fast
        return mx

    def maxRotateFunction(self, nums: List[int]) -> int:
        ans = cur = sum(idx * num for idx, num in enumerate(nums))
        s = sum(nums)
        n = len(nums)
        while nums:
            cur += s - nums.pop() * n
            ans = cur if cur > ans else ans
        return ans


# 397 - Integer Replacement - MEDIUM
# memo
class Solution:
    def __init__(self):
        self.cache = collections.defaultdict(int)

    def integerReplacement(self, n: int) -> int:
        if n == 1:
            return 0
        if n in self.cache:
            return self.cache.get(n)
        if n % 2 == 0:
            self.cache[n] = 1 + self.integerReplacement(n // 2)
        else:
            self.cache[n] = 2 + min(
                self.integerReplacement(n // 2), self.integerReplacement(n // 2 + 1)
            )
        return self.cache[n]


class Solution:
    @functools.lru_cache(None)
    def integerReplacement(self, n: int) -> int:
        if n == 1:
            return 0
        if n % 2 == 0:
            return 1 + self.integerReplacement(n // 2)
        return 2 + min(
            self.integerReplacement(n // 2), self.integerReplacement(n // 2 + 1)
        )


# bfs
class Solution:
    def integerReplacement(self, n: int) -> int:
        dq = collections.deque([n])
        ans = 0
        while dq:
            n = len(dq)
            for _ in range(n):
                number = dq.popleft()
                if number == 1:
                    return ans
                if number % 2 == 0:
                    dq.append(number // 2)
                else:
                    dq.append(number + 1)
                    dq.append(number - 1)
            ans += 1
        return ans


# 398 - Random Pick Index - MEDIUM
class Solution:
    # init: O(n), pick: O(1) / O(n)
    def __init__(self, nums: List[int]):
        self.d = collections.defaultdict(list)
        for i, v in enumerate(nums):
            self.d[v].append(i)

    def pick(self, target: int) -> int:
        return random.choice(self.d[target])


class Solution:
    # init: O(1), pick: O(n) / O(1)
    # Reservoir Sampling: we cannot load all the data at once
    def __init__(self, nums: List[int]):
        self.nums = nums

    def pick(self, target: int) -> int:
        ans = cnt = 0
        for i, n in enumerate(self.nums):
            if n == target:
                cnt += 1
                if random.randrange(cnt) == 0:
                    ans = i
                # if random.randint(1, cnt) == cnt:
                #     ans = i
                # if random.randint(1, cnt) == 1:
                #     ans = i
        return ans


class Solution:
    def __init__(self, nums: List[int]):
        self.nums = nums

    def pick(self, target: int) -> int:
        ans = []
        for i, n in enumerate(self.nums):
            if n == target:
                ans.append(i)
        return ans[random.randrange(len(ans))]
