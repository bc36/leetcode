import bisect, collections, functools, random, operator, math
from posix import X_OK
from typing import AnyStr, Iterable, List, Optional
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


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 1 - Two Sum - EASY
# [3, 3] 6: nums.index(3) will return 0, not 1
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for i in range(len(nums)):
            if nums[i] in dic:
                return [i, dic[nums[i]]]
            dic[target - nums[i]] = i
        return None


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        right = len(nums) - 1
        while nums:
            num = nums.pop()
            if target - num in nums:
                return [nums.index(target - num), right]
            right -= 1
        return None


# 2 - Add Two Numbers - MEDIUM
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode],
                      l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(-1)
        head = dummy
        carry = 0
        while l1 or l2 or carry:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0
            head.next = ListNode((v1 + v2 + carry) % 10)
            head = head.next
            carry = (v1 + v2 + carry) // 10
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return dummy.next


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


# 12 - Integer to Roman - MEDIUM
class Solution:
    def intToRoman(self, num: int) -> str:
        pairs = (("M", 1000), ("CM", 900), ("D", 500), ("CD", 400), ("C", 100),
                 ("XC", 90), ("L", 50), ("XL", 40), ("X", 10), ("IX", 9),
                 ("V", 5), ("IV", 4), ("I", 1))
        ret = ""
        for ch, val in pairs:
            ret += (num // val) * ch
            num %= val
        return ret


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
        # nums are in descending order
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        # i == -1 means that the whole list is descending order
        if i >= 0:
            j = len(nums) - 1
            # find the smaller number to be swapped
            # find the last "ascending" position
            while j >= 0 and nums[i] >= nums[j]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        # swap to make the right list ascending order
        left, right = i + 1, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
        return


# 37 - Sudoku Solver - HARD
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        def check(x: int, y: int, n: str) -> bool:
            for i in range(9):
                if board[x][i] == n or board[i][y] == n:
                    return False
            row = x // 3 * 3
            col = y // 3 * 3
            for i in range(row, row + 3):
                for j in range(col, col + 3):
                    if board[i][j] == n:
                        return False
            return True

        def backtrack(cur: int) -> bool:
            if cur == 81:
                return True
            x, y = cur // 9, cur % 9
            if board[x][y] != ".":
                return backtrack(cur + 1)
            for i in range(1, 10):
                if check(x, y, str(i)):
                    board[x][y] = str(i)
                    # backtrack until 'cur' == 81
                    if backtrack(cur + 1):
                        return True
                    board[x][y] = "."
            return False

        backtrack(0)

        # def backtrack(board: List[List[str]]) -> bool:
        #     for i in range(9):
        #         for j in range(9):
        #             if board[i][j] != ".":
        #                 continue
        #             for k in range(1, 10):
        #                 if check(i, j, str(k)):
        #                     board[i][j] = str(k)
        #                     if backtrack(board):
        #                         return True
        #                     board[i][j] = "."
        #             return False
        #     return True

        # backtrack(board)

        return


# 42 - Trapping Rain Water - HARD
class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        left = [0] * n
        right = [0] * n
        maxL, maxR = 0, 0
        for i in range(n):
            if height[i] > maxL:
                maxL = height[i]
            left[i] = maxL
            if height[n - 1 - i] > maxR:
                maxR = height[n - 1 - i]
            right[n - 1 - i] = maxR
        ans = 0
        for i in range(n):
            ans += min(left[i], right[i]) - height[i]
        return ans


# 43 - Multiply Strings - MEDIUM
class Solution:
    def multiply(self, num1, num2):
        ret = [0] * (len(num1) + len(num2))
        for i, e1 in enumerate(reversed(num1)):
            for j, e2 in enumerate(reversed(num2)):
                ret[i + j] += int(e1) * int(e2)
                ret[i + j + 1] += ret[i + j] // 10
                ret[i + j] %= 10
        # reverse, prepare to output
        while len(ret) > 1 and ret[-1] == 0:
            ret.pop()
        return ''.join(map(str, ret[::-1]))


# 46 - Permutations - MEDIUM
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        ret = []

        def backtrack(nums, tmp):
            if not nums:
                ret.append(tmp)
                return
            for i in range(len(nums)):
                backtrack(nums[:i] + nums[i + 1:], tmp + [nums[i]])

        backtrack(nums, [])
        return ret


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


# 53 - Maximum Subarray - EASY
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [nums[0]] + [0] * (len(nums) - 1)
        for i in range(1, len(nums)):
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
        return max(dp)


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        pre, ans = 0, nums[0]
        for i in range(len(nums)):
            pre = max(pre + nums[i], nums[i])
            if pre > ans:
                ans = pre
        return ans


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
        ret = []
        intervals = sorted(intervals, key=lambda x: x[0])
        left, right = intervals[0][0], intervals[0][1]
        for i in range(1, len(intervals)):
            if intervals[i][0] <= right:
                right = max(right, intervals[i][1])
            else:
                ret.append([left, right])
                left = intervals[i][0]
                right = intervals[i][1]
        ret.append([left, right])
        return ret


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


# 83 - Remove Duplicates from Sorted List - EASY
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        '''
        slower
        dummy = ListNode(-1,head)
        faster
        dummy = ListNode(-1)
        dummy.next = head
        '''
        dummy = ListNode(-1)
        dummy.next = head
        while head.next:
            if head.val == head.next.val:
                head.next = head.next.next
            else:
                head = head.next
        return dummy.next


class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        ans = ListNode(-101)
        dummy = ans
        while head:
            if head.val != dummy.val:
                dummy.next = ListNode(head.val)
                dummy = dummy.next
            head = head.next
        return ans.next


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
