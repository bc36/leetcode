import bisect, collections, functools, random, operator, math, itertools, re, os, heapq, queue
from typing import List, Optional
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


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 1 - Two Sum - EASY
# [3, 3] 6: nums.index(3) will return 0, not 1
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for i in range(len(nums)):
            if target - nums[i] in dic:
                return [dic[target - nums[i]], i]
            dic[nums[i]] = i
        return None

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

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        carry = 0
        dummy = head = ListNode(-1)
        while l1 or l2 or carry:
            a = l1.val if l1 else 0
            b = l2.val if l2 else 0
            if l1: l1 = l1.next
            if l2: l2 = l2.next
            head.next = ListNode((a + b + carry) % 10)
            carry = (a + b + carry) // 10
            head = head.next
        return dummy.next


# 3 - Longest Substring Without Repeating Characters - MEDIUM
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        res = 0
        longest_substr = ''
        for ch in s:
            if ch not in longest_substr:
                longest_substr += ch
                if len(longest_substr) > res:
                    res += 1
            else:
                i = longest_substr.find(ch)
                longest_substr = longest_substr[i + 1:] + ch
        return res

    # sliding window + hashmap
    def lengthOfLongestSubstring(self, s: str) -> int:
        slow, fast, ans = 0, 0, 0
        dic = {}
        while fast < len(s):
            if s[fast] in dic and dic[s[fast]] >= slow:
                slow = dic[s[fast]] + 1
            dic[s[fast]] = fast
            fast += 1
            ans = max(ans, fast - slow)
        return ans

    # set
    def lengthOfLongestSubstring(self, s: str) -> int:
        st, ans, slow, fast = set(), 0, 0, 0
        while fast < len(s):
            if s[fast] not in st:
                st.add(s[fast])
                fast += 1
            else:
                st.remove(s[slow])
                slow += 1
            ans = max(ans, len(st))
        return ans

    # ord(), chr() / byte -> position
    def lengthOfLongestSubstring(self, s: str) -> int:
        exist = [0 for _ in range(256)]
        slow, fast, ans = 0, 0, 0
        while fast < len(s):
            if exist[ord(s[fast])] == 0:
                exist[ord(s[fast])] += 1
                fast += 1
            else:
                exist[ord(s[slow])] -= 1
                slow += 1
            ans = max(ans, fast - slow)
        return ans


# 5 - Longest Palindromic Substring - MEDIUM
class Solution:
    # Longest Common Substring, not subsequence(1143), O(n^2)
    def longestPalindrome(self, s: str) -> str:
        return

    # Expand Around Center, O(n^2)
    def longestPalindrome(self, s: str) -> str:
        def helper(s, l, r):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            return s[l + 1:r]

        ans = ""
        for i in range(len(s)):
            # odd case, like "aba"
            tmp = helper(s, i, i)
            if len(tmp) > len(ans):
                ans = tmp
            # even case, like "abba"
            tmp = helper(s, i, i + 1)
            if len(tmp) > len(ans):
                ans = tmp
        return ans

    # dp, O(n^2)
    def longestPalindrome(self, s: str) -> str:
        dp = [[False] * len(s) for _ in range(len(s))]
        for i in range(len(s)):
            dp[i][i] = True
        ans = s[0]
        for j in range(len(s)):
            for i in range(j):
                if s[i] == s[j] and (dp[i + 1][j - 1] or j == i + 1):
                    dp[i][j] = True
                    if j - i + 1 > len(ans):
                        ans = s[i:j + 1]
        return ans

    # Manacher's Algorithm, O(n)


# 8 - String to Integer (atoi) - MEDIUM
class Solution:
    def myAtoi(self, s: str) -> int:
        return max(min(int(*re.findall('^[\+\-]?\d+', s.lstrip())), 2**31 - 1),
                   -2**31)


# 11 - Container With Most Water - MEDIUM
class Solution:
    def maxArea(self, height: List[int]) -> int:
        ans, i, j, ans = 0, 0, len(height) - 1
        while i < j:
            if height[i] < height[j]:
                ans = max(ans, height[i] * (j - i))
                i += 1
            else:
                ans = max(ans, height[j] * (j - i))
                j -= 1
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


# 15 - 3Sum - MEDIUM
class Solution:
    # narrow down 'left' and 'right' for each 'i'
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if not nums or len(nums) < 3:
            return []
        nums.sort()
        ans = []
        for i in range(len(nums) - 2):
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            left, right = i + 1, len(nums) - 1
            while left < right:
                if nums[i] + nums[left] + nums[right] == 0:
                    ans.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
                elif nums[i] + nums[left] + nums[right] < 0:
                    left += 1
                else:
                    right -= 1
        return ans


# 17 - Letter Combinations of a Phone Number - MEDIUM
class Solution:
    dic = {
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz'
    }

    def letterCombinations(self, digits: str) -> List[str]:
        a = []
        for i in range(len(digits)):
            a.append(digits[i])
        if len(a) == 4:
            return [
                ''.join(i) for i in (itertools.product(self.dic[
                    a[0]], self.dic[a[1]], self.dic[a[2]], self.dic[a[3]]))
            ]
        if len(a) == 3:
            return [
                ''.join(i) for i in (itertools.product(self.dic[
                    a[0]], self.dic[a[1]], self.dic[a[2]]))
            ]
        if len(a) == 2:
            return [
                ''.join(i)
                for i in (itertools.product(self.dic[a[0]], self.dic[a[1]]))
            ]
        if len(a) == 1:
            return [''.join(i) for i in (itertools.product(self.dic[a[0]]))]
        return []

    def letterCombinations(self, digits: str) -> List[str]:
        ans = [''] if digits else []
        for d in digits:
            cur = []
            for ch in self.dic[d]:
                for i in ans:
                    cur.append(i + ch)
            ans = cur
        return ans

    def letterCombinations(self, digits: str) -> List[str]:
        def dfs(i: int, cur: str):
            if i == len(digits):
                ans.append(cur)
                return
            s = self.dic[digits[i]]
            for ch in s:
                dfs(i + 1, cur + ch)
            return

        if not digits: return []
        ans = []
        dfs(0, '')
        return ans


# 18 - 4Sum - MEDIUM
class Solution:
    # > 1300ms
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        def twoSum(arr, target):
            if len(arr) < 2:
                return []
            i, j = 0, len(arr) - 1
            ans = []
            while i < j:
                if arr[i] + arr[j] > target:
                    j -= 1
                elif arr[i] + arr[j] < target:
                    i += 1
                else:
                    ans.append([arr[i], arr[j]])
                    i += 1
                    j -= 1
            return ans

        nums.sort()
        ans = set()
        if len(nums) < 4:
            return
        for i in range(len(nums) - 3):
            for j in range(i + 1, len(nums) - 2):
                s = twoSum(nums[j + 1:], target - nums[i] - nums[j])
                for k in range(len(s)):
                    ans.add((nums[i], nums[j], s[k][0], s[k][1]))
        return ans

    # < 200ms
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        def twoSum(nums, t):
            res = []
            s = set()
            for x in nums:
                if not res or res[-1][1] != x:
                    if t - x in s:
                        res.append([t - x, x])
                s.add(x)
            return res

        def kSum(nums, t, k):
            res = []
            if not nums:
                return res
            # speed up
            average = t // k
            if nums[0] > average or nums[-1] < average:
                return res
            if k == 2:
                return twoSum(nums, t)
            for i in range(len(nums)):
                if i == 0 or nums[i - 1] != nums[i]:
                    for subset in kSum(nums[i + 1:], t - nums[i], k - 1):
                        res.append([nums[i]] + subset)
            return res

        nums.sort()
        return kSum(nums, target, 4)


# 19 - Remove Nth Node From End of List - MEDIUM
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode],
                         n: int) -> Optional[ListNode]:
        self.n = n
        dummy = ListNode(-1, head)

        def helper(head: Optional[ListNode]):
            if head:
                helper(head.next)
            else:
                return
            if self.n == 0:
                if head.next:
                    head.next = head.next.next
                else:
                    head.next = None
            self.n -= 1

        helper(dummy)
        return dummy.next

    def removeNthFromEnd(self, head, n):
        fast = slow = head
        for _ in range(n):
            fast = fast.next
        if not fast:
            return head.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return head


# 20 - Valid Parentheses - EASY
class Solution:
    # stack
    def isValid(self, s: str) -> bool:
        dic = {'(': ')', '{': '}', '[': ']'}
        stack = []
        for i in s:
            if i in dic:
                stack.append(i)
            elif len(stack) == 0 or dic[stack.pop()] != i:
                return False
        return len(stack) == 0

    def isValid(self, s: str) -> bool:
        preLen = len(s)
        while True:
            s = s.replace("()", "").replace("[]", "").replace("{}", "")
            if preLen == len(s):
                break
            preLen = len(s)
        return len(s) == 0

    def isValid(self, s: str) -> bool:
        stack = []
        for ch in s:
            if ch == '(':
                stack.append(')')
            elif ch == '[':
                stack.append(']')
            elif ch == '{':
                stack.append('}')
            else:
                if not stack or ch != stack[-1]:
                    return False
                stack.pop()
        return len(stack) == 0


# 21. Merge Two Sorted Lists - EASY
class Solution:
    # iterative
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = head = ListNode(-1)
        while l1 and l2:
            if l1.val >= l2.val:
                head.next = l2
                l2 = l2.next
            else:
                head.next = l1
                l1 = l1.next
            head = head.next
        head.next = l1 or l2
        return dummy.next

    # recursive
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1 or not l2:
            return l1 or l2
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2


# 22 - Generate Parentheses - MEDIUM
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def dfs(left: int, right: int, cur: str):
            if left == right == n:
                ans.append(cur)
                return
            if left < n:
                dfs(left + 1, right, cur + '(')
            if left > right:
                dfs(left, right + 1, cur + ')')
            return

        ans = []
        dfs(0, 0, '')
        return ans

    def generateParenthesis(self, n: int) -> List[str]:
        ans, s = [], [("", 0, 0)]
        while s:
            cur, l, r = s.pop()
            if l - r < 0 or l > n or r > n:
                continue
            if l == r == n:
                ans.append(cur)
            s.append((cur + "(", l + 1, r))
            s.append((cur + ")", l, r + 1))
        return ans


# 23 - Merge k Sorted Lists - HARD
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        hp = []
        dummy = head = ListNode(-1)
        for i in range(len(lists)):
            if lists[i]:
                # heap cannot compare 'ListNode'
                # need 'i' just to avoid comparing 'ListNode'
                heapq.heappush(hp, (lists[i].val, i, lists[i]))
        while hp:
            n, i, cur = heapq.heappop(hp)
            head.next = ListNode(n)
            head = head.next
            if cur.next:
                cur = cur.next
                heapq.heappush(hp, (cur.val, i, cur))
        return dummy.next

    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        dummy = head = ListNode(-1)
        q = []
        for i in range(len(lists)):
            if lists[i]:
                q.append((lists[i].val, i))
        heapq.heapify(q)
        while q:
            val, i = heapq.heappop(q)
            head.next = ListNode(val)
            head = head.next
            lists[i] = lists[i].next
            if lists[i]:
                heapq.heappush(q, (lists[i].val, i))
        return dummy.next

    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        def merge2Lists(l1, l2):
            if not l1: return l2
            if not l2: return l1
            if l1.val < l2.val:
                l1.next = merge2Lists(l1.next, l2)
                return l1
            else:
                l2.next = merge2Lists(l1, l2.next)
                return l2

        def merge(lists, left, right):
            if left == right:
                return lists[left]
            mid = left + (right - left) // 2
            l1 = merge(lists, left, mid)
            l2 = merge(lists, mid + 1, right)
            return merge2Lists(l1, l2)

        if not lists:
            return
        n = len(lists)
        return merge(lists, 0, n - 1)


# 24 - Swap Nodes in Pairs - MEDIUM
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy = ListNode(-1, head)
        pre = dummy
        while head and head.next:
            slow, fast = head, head.next
            pre.next = fast
            slow.next = fast.next
            fast.next = slow
            head = slow.next
            pre = slow
        return dummy.next

    def swapPairs(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        newHead = head.next
        head.next = self.swapPairs(newHead.next)
        newHead.next = head
        return newHead


# 31 - Next Permutation - MEDIUM
class Solution:
    # find the first number that is greater than the adjecent number on the right
    # then swap this number with the smallest number among the numbers larger than it on the right.
    # then sort the numbers to the right side of this number in ascending order
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


# 33 - Search in Rotated Sorted Array - MEDIUM
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            if nums[0] <= nums[mid]:  # left half in order
                if nums[0] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:  # right half in order
                if nums[mid] <= target <= nums[-1]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1


# 34 - Find First and Last Position of Element in Sorted Array - MEDIUM
class Solution:
    def searchRange(self, nums, target):
        def search(n: int) -> int:
            lo, hi = 0, len(nums)
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] >= n:
                    hi = mid
                else:
                    lo = mid + 1
            return lo

        lo = search(target)
        return [lo, search(target + 1) - 1] if target in nums[lo:lo +
                                                              1] else [-1, -1]


# 35 - Search Insert Position - EASY
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (right + left) >> 1
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left

    def searchInsert(self, nums: List[int], target: int) -> int:
        return bisect.bisect_left(nums, target)


# 36 - Valid Sudoku - MEDIUM
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        seen = []
        for i, row in enumerate(board):
            for j, c in enumerate(row):
                if c != '.':
                    seen += [(c, j), (i, c), (i // 3, j // 3, c)]
        return len(seen) == len(set(seen))

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        r = [[False] * 9 for _ in range(9)]
        c = [[False] * 9 for _ in range(9)]
        sub = [[False] * 9 for _ in range(9)]
        for i, row in enumerate(board):
            for j, ch in enumerate(row):
                if ch != '.':
                    ch = int(ch) - 1
                    if r[i][ch] or c[j][ch] or sub[i // 3 * 3 + j // 3][ch]:
                        return False
                    r[i][ch] = c[j][ch] = sub[i // 3 * 3 + j // 3][ch] = True
        return True


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
        # Another sulotion:
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


# 39 - Combination Sum - MEDIUM
class Solution:
    def combinationSum(self, candidates: List[int],
                       target: int) -> List[List[int]]:
        def backtrack(begin: int, path: List[int], target: int):
            if target == 0:
                ans.append(path)
                return
            for i in range(begin, len(candidates)):
                if target < candidates[i]:
                    break
                backtrack(i, path + [candidates[i]], target - candidates[i])
            return

        ans = []
        candidates.sort()
        backtrack(0, [], target)
        return ans


# 40 - Combination Sum II - MEDIUM
class Solution:
    def combinationSum2(self, candidates: List[int],
                        target: int) -> List[List[int]]:
        def backtrack(begin: int, path: List[int], target: int):
            if target == 0:
                ans.append(path)
                return
            for i in range(begin, len(candidates)):
                if candidates[i] > target: break
                if i > begin and candidates[i - 1] == candidates[i]: continue
                backtrack(i + 1, path + [candidates[i]],
                          target - candidates[i])
            return

        ans = []
        candidates.sort()
        backtrack(0, [], target)
        return ans


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

    def trap(self, height: List[int]) -> int:
        left = [0] * len(height)
        right = [0] * len(height)
        ans = prel = prer = 0
        for i in range(len(height)):
            if height[i] > prel:
                left[i] = prel = height[i]
            else:
                left[i] = prel
            if height[~i] > prer:
                right[~i] = prer = height[~i]
            else:
                right[~i] = prer
        for i in range(len(height)):
            if height[i] < left[i] and height[i] < right[i]:
                ans += min(left[i], right[i]) - height[i]
        return ans

    def trap(self, height: List[int]) -> int:
        w = [0] * len(height)  # w: water level
        left = right = 0
        for i in range(len(height)):
            left = max(left, height[i])
            w[i] = left  # over-fill it to left max height
        for i in range(len(height) - 1, -1, -1):
            right = max(right, height[i])
            w[i] = min(w[i], right) - height[i]  # drain to the right height
        return sum(w)

    # monotonic stack
    def trap(self, height: List[int]) -> int:
        ans, stack = 0, []
        for i in range(len(height)):
            while stack and height[i] > height[stack[-1]]:
                w = stack.pop()  # water level
                if not stack:  # do not have wall at left
                    break
                left = stack[-1]
                curWidth = i - left - 1
                curHeight = min(height[left], height[i]) - height[w]
                ans += curWidth * curHeight
            stack.append(i)
        return ans

    # two pointers
    def trap(self, height: List[int]) -> int:
        ans = pl = lmax = rmax = 0
        pr = len(height) - 1
        while pl <= pr:
            if lmax < rmax:
                ans += max(0, lmax - height[pl])
                lmax = max(lmax, height[pl])
                pl += 1
            else:
                ans += max(0, rmax - height[pr])
                rmax = max(rmax, height[pr])
                pr -= 1
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


# 45 - Jump Game II - MEDIUM
class Solution:
    # slow, O(n^2)
    def jump(self, nums: List[int]) -> int:
        if len(nums) == 1: return 0
        n = len(nums)
        dp = [0] + [float('inf')] * (n - 1)  # dp[i]: minimum step to 'i'
        for i in range(n - 1):
            if i + nums[i] >= n - 1:
                return dp[i] + 1
            for step in range(1, nums[i] + 1):
                dp[i + step] = min(dp[i] + 1, dp[i + step])
        return dp[-1]

    # greedy, O(n), find the next reachable area
    def jump(self, nums: List[int]) -> int:
        if len(nums) == 1: return 0
        l = r = times = 0
        while r < len(nums) - 1:
            times += 1
            nxt = max(i + nums[i] for i in range(l, r + 1))
            l, r = r + 1, nxt
        return times

    # greedy, O(n), when reach the boundry of reachable area, 'step++'
    def jump(self, nums: List[int]) -> int:
        cur = ans = nxt = 0
        for i in range(len(nums) - 1):
            nxt = max(nums[i] + i, nxt)
            if i == cur:
                cur = nxt
                ans += 1
        return ans


# 46 - Permutations - MEDIUM
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(res: List[int], path: List[int]):
            if not res:
                ret.append(path)
                return
            for i in range(len(res)):
                backtrack(res[:i] + res[i + 1:], path + [res[i]])
            return

        ret = []
        backtrack(nums, [])
        return ret

    def permute(self, nums: List[int]) -> List[List[int]]:
        # [(2, 3), (3, 2)]
        # return list(itertools.permutations(nums, len(nums)))
        # [[2, 3], [3, 2]]
        return list(map(list, itertools.permutations(nums, len(nums))))


# 47 - Permutations II - MEDIUM
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def backtrack(path: List[int], check):
            if len(path) == len(nums):
                ret.append(path)
                return
            for i in range(len(nums)):
                if check[i] == 1:
                    continue
                if i > 0 and nums[i] == nums[i - 1] and check[i - 1] == 0:
                    continue
                check[i] = 1
                backtrack(path + [nums[i]], check)
                check[i] = 0
            return

        ret, check = [], [0] * len(nums)
        nums.sort()
        backtrack([], check)
        return ret


# 48 - Rotate Image - MEDIUM
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        m = len(matrix)
        # horizontal
        for i in range(m):
            for j in range(m // 2):
                matrix[i][j], matrix[i][m - 1 - j] = matrix[i][m - 1 -
                                                               j], matrix[i][j]
        # diagonal
        for i in range(m - 1):
            for j in range(m - 1 - i):
                matrix[i][j], matrix[m - 1 - j][m - 1 -
                                                i] = matrix[m - 1 -
                                                            j][m - 1 -
                                                               i], matrix[i][j]
        return

    def rotate(self, matrix: List[List[int]]) -> None:
        m = len(matrix)
        for i in range(m // 2):
            for j in range(m):
                matrix[i][j], matrix[m - 1 - i][j] = matrix[m - 1 -
                                                            i][j], matrix[i][j]
        for i in range(m):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        return


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


class Solution:
    # iterative
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

    def maxSubArray(self, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            nums[i] = nums[i] + max(nums[i - 1], 0)
        return max(nums)

    def maxSubArray(self, nums: List[int]) -> int:
        pre, ans = 0, nums[0]
        for i in range(len(nums)):
            pre = max(pre + nums[i], nums[i])
            if pre > ans:
                ans = pre
        return ans


# 55 - Jump Game - MEDIUM
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        can_reach = 0
        for i in range(len(nums)):
            if i > can_reach:
                return False
            can_reach = max(can_reach, i + nums[i])
        return True

    def canJump(self, nums: List[int]) -> bool:
        rightmost, n = 0, len(nums)
        for i in range(n):
            if i <= rightmost:
                rightmost = max(rightmost, i + nums[i])
                if rightmost >= n - 1:
                    return True
        return False


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

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        ans = []
        intervals = sorted(intervals, key=lambda x: x[0])
        left, right = intervals[0]
        for i in range(1, len(intervals)):
            if intervals[i][0] <= right:
                right = max(right, intervals[i][1])
            else:
                ans.append([left, right])
                left, right = intervals[i]
        ans.append([left, right])
        return ans


# 59 - Spiral Matrix II - MEDIUM
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        matrix = [[0] * n for _ in range(n)]
        row = col = dirIdx = 0
        for i in range(n * n):
            matrix[row][col] = i + 1
            dx, dy = dirs[dirIdx]
            r, c = row + dx, col + dy
            if r < 0 or r >= n or c < 0 or c >= n or matrix[r][c] > 0:
                dirIdx = (dirIdx + 1) % 4  # rotate
                dx, dy = dirs[dirIdx]
            row, col = row + dx, col + dy
        return matrix

    def generateMatrix(self, n: int) -> List[List[int]]:
        ans = [[0] * n for _ in range(n)]
        left, right, top, down, num = 0, n - 1, 0, n - 1, 1
        while left <= right and top <= down:
            for i in range(left, right + 1):
                ans[top][i] = num
                num += 1
            top += 1
            for i in range(top, down + 1):
                ans[i][right] = num
                num += 1
            right -= 1
            for i in range(right, left - 1, -1):
                ans[down][i] = num
                num += 1
            down -= 1
            for i in range(down, top - 1, -1):
                ans[i][left] = num
                num += 1
            left += 1
        return ans


# 62 - Unique Paths - MEDIUM
class Solution:
    # dp[i][j] peresent the maximum value of paths that can reach this point
    def uniquePaths(self, m: int, n: int) -> int:
        # dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
        dp = [[0 for _ in range(n)] for _ in range(m)]
        # initialize
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]

    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1] * n] + [[1] + [0] * (n - 1)] * (m - 1)
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]

    # combination
    # To make the machine get to the corner, the number of steps to the right and the number of steps to the left are fixed
    # m - 1 down && n - 1 right -> m + n - 1 times movement
    def uniquePaths(self, m: int, n: int) -> int:
        return math.comb(m + n - 2, n - 1)


# 63 - Unique Paths II - MEDIUM
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if obstacleGrid[0][0] == 1: return 0
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            if obstacleGrid[i][0] == 1: break
            dp[i][0] = 1
        for j in range(n):
            if obstacleGrid[0][j] == 1: break
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] != 1:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        firstObs = n
        for i in range(n):
            if obstacleGrid[0][i] == 1:
                firstObs = i
                break
        dp = [1] * firstObs + [0] * (n - firstObs)
        for i in range(1, m):
            if obstacleGrid[i][0] == 1:
                dp[0] = 0
            for j in range(1, n):
                if obstacleGrid[i][j] != 1:
                    dp[j] += dp[j - 1]
                else:
                    dp[j] = 0
        return dp[-1]


# 64 - Minimum Path Sum - MEDIUM
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = grid[0][0]
        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        for j in range(1, n):
            dp[0][j] = dp[0][j - 1] + grid[0][j]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        return dp[-1][-1]


# 67 - Add Binary - EASY
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        ans, num = '', (int(a, 2) + int(b, 2))
        while num:
            if num & 1: ans += '1'
            else: ans += '0'
            num >>= 1
        return ans[::-1] if ans else '0'

    def addBinary(self, a: str, b: str) -> str:
        x, y = int(a, 2), int(b, 2)
        return bin(x + y)[2:]


# 70 - Climbing Stairs - EASY
class Solution:
    # dp
    def climbStairs(self, n: int) -> int:
        if n < 3:
            return n
        dp = [1, 2] + [0] * (n - 2)
        for i in range(2, n):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[-1]

    # dp optimized
    def climbStairs(self, n: int) -> int:
        if n < 3:
            return n
        one, two, ans = 1, 2, 0
        for _ in range(2, n):
            ans = one + two
            one, two = two, ans
        return ans

    # memo 1
    def __init__(self):
        self.memo = {}

    def climbStairs(self, n: int) -> int:
        if n < 3:
            return n
        if n in self.memo:
            return self.memo[n]
        self.memo[n] = self.climbStairs(n - 1) + self.climbStairs(n - 2)
        return self.memo[n]

    # memo 2
    memo = {1: 1, 2: 2}

    def climbStairs(self, n: int) -> int:
        if n < 3:
            return self.memo[n]
        if n in self.memo:
            return self.memo[n]
        self.memo[n] = self.climbStairs(n - 1) + self.climbStairs(n - 2)
        return self.memo[n]

    @functools.lru_cache(None)
    def climbStairs(self, n: int) -> int:
        if n < 3:
            return n
        return self.climbStairs(n - 1) + self.climbStairs(n - 2)


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

    def simplifyPath(self, path: str) -> str:
        stack = []
        for p in path.split("/"):
            if stack and p == "..":
                stack.pop()
            elif p not in "..":
                stack.append(p)
        return "/" + "/".join(stack)

    def simplifyPath(self, path: str) -> str:
        return os.path.realpath(path)


# 73 - Set Matrix Zeroes - MEDIUM
class Solution:
    # O(2 * m * n), O(m + n)
    def setZeroes(self, matrix: List[List[int]]) -> None:
        r, c = set(), set()
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    r.add(i)
                    c.add(j)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i in r or j in c:
                    matrix[i][j] = 0
        return

    # O(2 * m * n), space optimized, O(2), two flags
    def setZeroes(self, matrix: List[List[int]]) -> None:
        m, n = len(matrix), len(matrix[0])
        col0 = any(matrix[i][0] == 0 for i in range(m))
        row0 = any(matrix[0][j] == 0 for j in range(n))
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        if col0:
            for i in range(m):
                matrix[i][0] = 0
        if row0:
            for j in range(n):
                matrix[0][j] = 0
        return

    # O(2 * m * n), space optimized, O(1), one flags
    def setZeroes(self, matrix: List[List[int]]) -> None:
        m, n = len(matrix), len(matrix[0])
        col0 = False
        for i in range(m):
            if matrix[i][0] == 0:
                col0 = True
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0
        for i in range(m - 1, -1, -1):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
            if col0:
                matrix[i][0] = 0
        return


# 74 - Search a 2D Matrix - MEDIUM
class Solution:
    # zigzag search
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        j, i = len(matrix[0]) - 1, 0
        while i < len(matrix):
            if matrix[i][j] >= target:
                while j >= 0:
                    if matrix[i][j] == target:
                        return True
                    j -= 1
            i += 1
        return False

    # binary search
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        n = len(matrix[0])
        lo, hi = 0, len(matrix) * n
        while lo < hi:
            mid = (lo + hi) // 2
            x = matrix[mid // n][mid % n]
            if x < target:
                lo = mid + 1
            elif x > target:
                hi = mid
            else:
                return True
        return False


# 75 - Sort Colors - MEDIUM
class Solution:
    # O(2n) / O(n)
    def sortColors(self, nums: List[int]) -> None:
        cnt = collections.Counter(nums)
        i = 0
        for k in sorted(cnt.keys()):
            while cnt[k] != 0:
                nums[i] = k
                i += 1
                cnt[k] -= 1
        return

    # O(n) / O(1)
    def sortColors(self, nums: List[int]) -> None:
        p = 0
        for i in range(len(nums)):
            if nums[i] == 0:
                nums[p], nums[i] = nums[i], nums[p]
                p += 1
        for i in range(p, len(nums)):
            if nums[i] == 1:
                nums[p], nums[i] = nums[i], nums[p]
                p += 1
        return

    # O(n) / O(1)
    def sortColors(self, nums: List[int]) -> None:
        p0 = p1 = 0
        for i in range(len(nums)):
            if nums[i] == 1:
                nums[i], nums[p1] = nums[p1], nums[i]
                p1 += 1
            elif nums[i] == 0:
                nums[i], nums[p0] = nums[p0], nums[i]
                if p0 < p1:
                    nums[i], nums[p1] = nums[p1], nums[i]
                p0 += 1
                p1 += 1
        return


# 76 - Minimum Window Substring - HARD
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


# 77 - Combinations - MEDIUM
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        return list(itertools.combinations(range(1, n + 1), k))

    # pretty slow: > 500ms
    def combine(self, n: int, k: int) -> List[List[int]]:
        def backtrack(res: List[int], path: List[int], k: int):
            if not k:
                ans.append(path)
            # for i in range(len(res) - (k - len(path)) + 1):
            for i in range(len(res)):
                # optimize: there are not enough numbers remaining: > 90ms
                # if len(res) - i < k:
                #     return
                backtrack(res[i + 1:], path + [res[i]], k - 1)
            return

        ans = []
        backtrack(list(range(1, n + 1)), [], k)
        return ans

    def combine(self, n: int, k: int) -> List[List[int]]:
        def backtrack(n: int, k: int, startIndex: int):
            if len(path) == k:
                ans.append(path[:])
                return
            for i in range(startIndex, n - (k - len(path)) + 2):
                path.append(i)
                backtrack(n, k, i + 1)
                path.pop()
            return

        ans, path = [], []
        backtrack(n, k, 1)
        return ans


# 78 - Subsets - MEDIUM
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        q = [[]]
        for i in range(len(nums)):
            for j in range(len(q)):
                q.append(q[j] + [nums[i]])
        return q

    def subsets(self, nums: List[int]) -> List[List[int]]:
        def helper(i, tmp):
            ans.append(tmp)
            for j in range(i, len(nums)):
                helper(j + 1, tmp + [nums[j]])
            return

        ans = []
        helper(0, [])
        return ans


# 79 - Word Search - MEDIUM
class Solution:
    # slow, > 6s
    def exist(self, board: List[List[str]], word: str) -> bool:
        def search(x, y, i):
            if i == len(word) - 1: return True
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < m and 0 <= ny < n and board[nx][ny] == word[i +
                                                                         1]:
                    board[x][y] = '#'
                    if search(nx, ny, i + 1):
                        return True
                    board[x][y] = word[i]
            return False

        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:
                    if search(i, j, 0):
                        return True
        return False

    # fast, < 100ms
    # TODO do not know how it works
    def exist(self, board, word):
        def backtracking(r, c, step=0):
            if step == len(word): return True
            # Q1: 'word[~step]'
            if 0 <= r < m and 0 <= c < n and board[r][c] == word[~step] and (
                    r, c) not in visited:
                visited.add((r, c))
                HashMap[(r, c,
                         step)] += 1  # Q2: how it works to speed up, why?
                for nr, nc in (r, c + 1), (r, c - 1), (r - 1, c), (r + 1, c):
                    if HashMap[(nr, nc, step + 1)] < n:
                        if backtracking(nr, nc, step + 1):
                            return True
                visited.remove((r, c))
                return False

        m, n = len(board), len(board[0])
        visited, HashMap = set(), collections.defaultdict(int)
        for i, j in itertools.product(range(m), range(n)):
            if backtracking(i, j):
                return True
        return False


# 80 - Remove Duplicates from Sorted Array II - MEDIUM
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        cur = 2
        for i in range(2, len(nums)):
            if nums[i] != nums[cur - 2]:
                nums[cur] = nums[i]
                cur += 1
        return cur

    def removeDuplicates(self, nums: List[int]) -> int:
        def solve(k):  # Repeat up to k
            cur = 0
            for n in nums:
                if cur < k or nums[cur - k] != n:
                    nums[cur] = n
                    cur += 1
            return cur

        return solve(2)


# 82 - Remove Duplicates from Sorted List II - MEDIUM
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        dummy = ListNode(-101, head)
        pre, cur = dummy, head
        while cur:
            while cur.next and cur.val == cur.next.val:
                cur = cur.next
            if pre.next == cur:
                pre = pre.next  # no duplicate nodes between pre and cur
            else:
                pre.next = cur.next  # have duplicate nodes, don't move pre
            cur = cur.next
        return dummy.next

    def deleteDuplicates(self, head: ListNode) -> ListNode:
        dummy = ListNode(-101, head)
        cur = dummy
        while cur.next and cur.next.next:
            if cur.next.val == cur.next.next.val:
                x = cur.next.val
                while cur.next and cur.next.val == x:
                    cur.next = cur.next.next
            else:
                cur = cur.next
        return dummy.next


# 83 - Remove Duplicates from Sorted List - EASY
class Solution:
    # O(n) / O(1)
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        dummy = ListNode(-1, head)
        while head.next:
            if head.val == head.next.val:
                head.next = head.next.next
            else:
                head = head.next
        return dummy.next

    # O(n) / O(1)
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        dummy = pre = ListNode(-101, head)
        while head:
            if pre.val == head.val:
                pre.next = head.next
            else:
                pre = pre.next
            head = head.next
        return dummy.next

    # O(n) / O(n)
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = pre = ListNode(-101)
        while head:
            if head.val != pre.val:
                pre.next = ListNode(head.val)
                pre = pre.next
            head = head.next
        return dummy.next


# 84 - Largest Rectangle in Histogram - HARD
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = [-1]
        heights.append(0)
        ans = 0
        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i - stack[-1] - 1
                ans = max(ans, h * w)
            stack.append(i)
        return ans


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

    def merge(self, nums1: List[int], m: int, nums2: List[int],
              n: int) -> None:
        nums1[m:] = nums2
        nums1.sort()
        return


# 90 - Subsets II - MEDIUM
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums, path, ret):
            ret.append(path)
            for i in range(len(nums)):
                if i != 0 and nums[i] == nums[i - 1]: continue
                dfs(nums[i + 1:], path + [nums[i]], ret)
            return

        nums.sort()
        ret = []
        dfs(nums, [], ret)
        return ret

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums, index, path, ret):
            ret.append(path)
            for i in range(index, len(nums)):
                if i > index and nums[i] == nums[i - 1]: continue
                dfs(nums, i + 1, path + [nums[i]], ret)
            return

        nums.sort()
        ret = []
        dfs(nums, 0, [], ret)
        return ret

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        ret, cur = [[]], []
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i - 1]:
                cur = [item + [nums[i]] for item in cur]
            else:
                cur = [item + [nums[i]] for item in ret]
            ret += cur
        return ret


# 91 - Decode Ways - MEDIUM
class Solution:
    def numDecodings(self, s: str) -> int:
        if s[0] == '0': return 0
        dp = [1] + [0] * len(s)
        for i in range(1, len(s) + 1):
            if s[i - 1] != '0':
                dp[i] += dp[i - 1]
            if i > 1 and s[i - 2] != '0' and int(s[i - 2:i]) < 27:
                dp[i] += dp[i - 2]
        return dp[-1]

    @functools.lru_cache(None)
    def numDecodings(self, s: str) -> int:
        if len(s) == 1:
            return int(s[0] != '0')
        if len(s) == 0:
            return 1
        one = two = 0
        if s[-1] != '0':
            one += self.numDecodings(s[:-1])
        if s[-2] != '0' and int(s[-2:]) < 27:
            two += self.numDecodings(s[:-2])
        return one + two


# 94 - Binary Tree Inorder Traversal - EASY
class Solution:
    # recursively
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        return self.inorderTraversal(
            root.left) + [root.val] + self.inorderTraversal(root.right)

    # iteratively
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack, ans = [], []
        while stack or root:
            if root:
                stack.append(root)
                root = root.left
            else:
                cur = stack.pop()
                ans.append(cur.val)
                root = cur.right
        return ans

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        stack, ans = [], []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            cur = stack.pop()
            ans.append(cur.val)
            root = cur.right
        return ans


# 96 - Unique Binary Search Trees - MEDIUM
class Solution(object):
    def numTrees(self, n):
        dp = [0] * (n + 1)
        dp[0], dp[1] = 1, 1
        for i in range(2, n + 1):
            for j in range(1, i + 1):
                dp[i] += dp[j - 1] * dp[i - j]
        return dp[n]


# 98 - Validate Binary Search Tree - MEDIUM
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        def valid(root: TreeNode, lower=float('-inf'), upper=float('inf')):
            if not root:
                return True
            if root.val <= lower or root.val >= upper:
                return False
            return valid(root.left, lower, root.val) and valid(
                root.right, root.val, upper)

        return valid(root)

    def isValidBST(self, root: TreeNode) -> bool:
        def isValid(root, minv, maxv):
            if not root:
                return True
            if minv and root.val <= minv.val: return False
            if maxv and root.val >= maxv.val: return False
            return isValid(root.left, minv, root) and isValid(
                root.right, root, maxv)

        return isValid(root, None, None)

    def isValidBST(self, root: TreeNode) -> bool:
        pre, stack = float('-inf'), []
        while root or stack:
            if root:
                stack.append(root)
                root = root.left
            else:
                cur = stack.pop()
                if pre >= cur.val:
                    return False
                pre = cur.val
                root = cur.right
        return True