from typing import List, Optional
import collections, math, functools


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 301 - Remove Invalid Parentheses - HARD
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        return


# 314 - Binary Tree Vertical Order Traversal - MEDIUM
# dfs
class Solution:
    def verticalOrder(self, root: TreeNode) -> List[List[int]]:
        # Use a dict to store our answers, keys will be column idxs.
        ans = collections.defaultdict(list)

        def dfs(node, row, col) -> None:
            if not node:
                return
            # Append node vals to column in our dict.
            ans[col].append((row, node.val))
            # Traverse l and r.
            dfs(node.left, row + 1, col - 1)
            dfs(node.right, row + 1, col + 1)
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
class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        nodes = collections.defaultdict(list)
        queue = collections.deque([(root, 0)])
        while queue:
            node, pos = queue.popleft()
            if node:
                nodes[pos].append(node.val)
                queue.append((node.left, pos - 1))
                queue.append((node.right, pos + 1))
        # sorted the keys of defaultdict
        return [nodes[i] for i in sorted(nodes)]


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


class Solution:
    def bulbSwitch(self, n: int) -> int:
        return int(math.sqrt(n))


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


# dfs
class Solution:
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
class Solution:
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


'''
flatten trick about a list of lists
>>> sum([[1, 2], [2, 4]], [])
[1, 2, 2, 4]
'''


class Solution(object):
    def depthSum(self, nestedList):
        depth, ret = 1, 0
        while nestedList:
            ret += depth * sum(
                [x.getInteger() for x in nestedList if x.isInteger()])
            nestedList = sum(
                [x.getList() for x in nestedList if not x.isInteger()], [])
            depth += 1
        return ret


# 347 - Top K Frequent Elements - MEDIUM
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        cnt = collections.Counter(nums)
        # sorted by value and get the key
        return [i[0] for i in sorted(cnt.items(), key=lambda x: x[1])[-k:]]
        # return [i[0] for i in sorted(cnt.items(), key=lambda x: x[1], reverse=True)[:k]]


class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        cnt = collections.Counter(nums)
        # convert to 'tuple' to sort, because 'dict' is unordered
        times = sorted(cnt.items(), key=lambda k: k[1])
        ans = []
        while k != 0 and len(times) > 0:
            ans.append(times.pop()[0])
            k -= 1
        return ans


# 349 - Intersection of Two Arrays - EASY
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        return set(nums1).intersection(set(nums2))


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
                    dp[i][j] = min(k + max(dp[i][k - 1], dp[k + 1][j]),
                                   dp[i][j])
                # calculate both sides
                dp[i][j] = min(dp[i][j], i + dp[i + 1][j], j + dp[i][j - 1])
                dp[i][j] = min(dp[i][j], j + dp[i][j - 1])
                i -= 1

        return dp[1][n]


class Solution:
    def getMoneyAmount(self, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n - 1, 0, -1):
            for j in range(i + 1, n + 1):
                dp[i][j] = min(k + max(dp[i][k - 1], dp[k + 1][j])
                               for k in range(i, j))
        return dp[1][n]


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
            self.cache[n] = 2 + min(self.integerReplacement(n // 2),
                                    self.integerReplacement(n // 2 + 1))
        return self.cache[n]


class Solution:
    @functools.lru_cache(None)
    def integerReplacement(self, n: int) -> int:
        if n == 1:
            return 0
        if n % 2 == 0:
            return 1 + self.integerReplacement(n // 2)
        return 2 + min(self.integerReplacement(n // 2),
                       self.integerReplacement(n // 2 + 1))


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
