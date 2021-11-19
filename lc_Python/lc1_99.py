import bisect, collections, functools, random, operator, math
from posix import X_OK
from typing import AnyStr, Iterable, List
'''
Function usually used

bit operation
&   bitwise AND
|   bitwise OR
^   bitwise XOR
~   bitwise NOT Inverts all the bits (~x = -x-1)
<<  left shift
>>  right shift
'''

# For viewing definitions

# LIBRARY FUNCTION
bisect.bisect_left()
collections.Counter(dict)
collections.deque(Iterable)
random.randint()
functools.reduce()
operator.xor()
# CLASS METHOD
dict.setdefault()


# 3 - Longest Substring Without Repeating Characters - MEDIUM
# sliding window + hashmap
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        left, right, ans = 0, 0, 0
        dic = {}
        while right < len(s):
            if s[right] in dic and dic[s[right]] >= left:
                left = dic[s[right]] + 1
            dic[s[right]] = right
            right += 1
            ans = max(ans, right - left)
        return ans


# ord(), chr() / byte -> position
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        exist = [0 for _ in range(256)]
        left, right, ans = 0, 0, 0
        while right < len(s):
            if exist[ord(s[right]) - 97] == 0:
                exist[ord(s[right]) - 97] += 1
                right += 1
            else:
                exist[ord(s[right]) - 97] -= 1
                left += 1

            ans = max(ans, right - left)
        return ans


# 20 - Valid Parentheses - EASY
# stack
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for ch in s:
            if len(stack) == 0 and (ch == ")" or ch == "]" or ch == "}"):
                return False
            elif ch == "(" or ch == "[" or ch == "{":
                stack.append(ch)
            elif ch == ")":
                if stack.pop() != "(":
                    return False
            elif ch == "]":
                if stack.pop() != "[":
                    return False
            elif ch == "}":
                if stack.pop() != "{":
                    return False
        return len(stack) == 0


# 31 - Next Permutation - MEDUIM
# find the first number that is greater than the adjecent number on the right
# then swap this number with the smallest number among the numbers larger than it on the right.
# then sort the numbers to the right side of this number in ascending order
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # greater save the number value and position
        greater = [[nums[-1], -1]]
        for i in range(len(nums) - 2, -1, -1):
            # find 'first number'
            if nums[i] < nums[i + 1]:
                # find the swap position
                greater.sort()
                for pair in greater:
                    if nums[i] < pair[0]:
                        # swap
                        nums[i], nums[pair[1]] = nums[pair[1]], nums[i]
                        # make the rest number ascending order
                        rightSide = nums[i + 1:]
                        rightSide.sort()
                        nums[i + 1:] = rightSide
                        return

            # update 'greater'
            greater.append([nums[i], i])

        # did not find such number
        nums.reverse()
        return


# better
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        # i == -1 means that the whole list is descending order
        if i >= 0:
            j = len(nums) - 1
            # find the smaller number to be swapped
            while j >= 0 and nums[i] >= nums[j]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        # swap to make the right list ascending order
        left, right = i + 1, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1


# 43 - Multiply Strings - MEDIUM
class Solution:
    def multiply(self, num1, num2):
        res = [0] * (len(num1) + len(num2))
        for i, e1 in enumerate(reversed(num1)):
            for j, e2 in enumerate(reversed(num2)):
                res[i + j] += int(e1) * int(e2)
                res[i + j + 1] += res[i + j] // 10
                res[i + j] %= 10
        # reverse, prepare to output
        while len(res) > 1 and res[-1] == 0:
            res.pop()
        return ''.join(map(str, res[::-1]))


# 46 - Permutations - MEDIUM
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []

        def backtrack(nums, tmp):
            if not nums:
                res.append(tmp)
                return
            for i in range(len(nums)):
                backtrack(nums[:i] + nums[i + 1:], tmp + [nums[i]])

        backtrack(nums, [])
        return res


# 50 - Pow(x, n) - MEDIUM
'''
operators '>>', '&' are just used for 'int' and not used for 'float', '%' can be.
e.g.: 
>>> 5.00 >> 1
TypeError: unsupported operand type(s) for >>: 'float' and 'int'
>>> 5.00 & 1
TypeError: unsupported operand type(s) for &: 'float' and 'int
>>> 5.00 % 2
1.0
'''


# iterative
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n < 0:
            n = -n
            x = 1 / x
        ans = 1
        while n != 0:
            if n & 1:
                ans *= x
            x *= x
            n >>= 1  # equal to n //= 2
        return ans


# recursive
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if not n:
            return 1
        if n < 0:
            return 1 / self.myPow(x, -n)
        if n % 2:
            return x * self.myPow(x, n - 1)
        return self.myPow(x * x, n / 2)


# 56 - Merge Intervals - MEDIUM
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        ans = []
        intervals.sort()
        i = 0
        while i < len(intervals):
            # no need to use 'left', since intervals has been sorted
            right = intervals[i][1]
            j = i + 1
            while j < len(intervals) and right >= intervals[j][0]:
                right = max(intervals[j][1], right)
                j += 1
            ans.append([intervals[i][0], right])
            i = j

        return ans


# two pointers
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        res = []
        intervals = sorted(intervals, key=lambda x: x[0])
        left, right = intervals[0][0], intervals[0][1]
        for i in range(1, len(intervals)):
            if intervals[i][0] <= right:
                right = max(right, intervals[i][1])
            else:
                res.append([left, right])
                left = intervals[i][0]
                right = intervals[i][1]
        res.append([left, right])
        return res


# 62 - Unique Paths - MEDIUM
# dp[i][j] peresent the maximum value of paths that can reach this point
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
        dp = [[0 for _ in range(n)] for j in range(m)]
        # initialize
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]


class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1] * n] + [[1] + [0] * (n - 1)] * (m - 1)
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]


# combination
# To make the machine get to the corner, the number of steps to the right and the number of steps to the left are fixed
# m - 1 down && n - 1 right -> m + n - 1 times movement
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        return math.comb(m + n - 2, n - 1)


# 71 - Simplify Path - MEDIUM
class Solution:
    def simplifyPath(self, path: str) -> str:
        # places = [p for p in path.split("/") if p != "." and p != ""]
        sp = path.split("/")
        stack = []
        for i in sp:
            if i == "" or i == ".":
                continue
            elif i == "..":
                if len(stack) > 0:
                    stack.pop()
            else:
                stack.append(i)
        return "/" + "/".join(stack)


class Solution:
    def simplifyPath(self, path: str) -> str:
        stack = []
        for p in path.split("/"):
            if stack and p == "..":
                stack.pop()
            elif p not in "..":
                stack.append(p)
        return "/" + "/".join(stack)


# 76 - Minimum Window Substring - HARD
#
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        needdic = collections.Counter(t)
        need = len(t)
        left, right = 0, float("inf")  # record answer
        i = 0
        for j, ch in enumerate(s):
            # ch in needdic
            if needdic[ch] > 0:
                need -= 1
            needdic[ch] -= 1
            if need == 0:
                # move left point
                while i < j and needdic[s[i]] < 0:
                    needdic[s[i]] += 1
                    i += 1
                # update new answer
                if j - i < right - left:
                    left, right = i, j
                i += 1
                needdic[s[i]] += 1
                need += 1
        return "" if right > len(s) else s[left:right + 1]


# 88 - Merge Sorted Array - EASY
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int],
              n: int) -> None:
        i, j, insertPos = m - 1, n - 1, m + n - 1
        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[insertPos] = nums1[i]
                i -= 1
            else:
                nums1[insertPos] = nums2[j]
                j -= 1
            insertPos -= 1
        while j >= 0:
            nums1[insertPos] = nums2[j]
            j -= 1
            insertPos -= 1
        return


class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int],
              n: int) -> None:
        nums1[m:] = nums2
        nums1.sort()
        return


# 96 - Unique Binary Search Trees - MEDIUM
