from typing import List
import collections, math


# 301 - Remove Invalid Parentheses - HARD
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        return


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