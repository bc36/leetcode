from operator import le
from typing import Iterable, List
import collections, itertools


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 1200 - Minimum Absolute Difference - EASY
class Solution:
    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        diff = float('inf')
        arr.sort()
        for i in range(len(arr) - 1):
            diff = min(arr[i + 1] - arr[i], diff)
        ans, left, right = [], 0, 1
        while right < len(arr):
            if arr[right] - arr[left] > diff:
                left += 1
            elif arr[right] - arr[left] < diff:
                right += 1
            else:
                ans.append([arr[left], arr[right]])
                left += 1
                right += 1
        return ans

    def minimumAbsDifference(self, a: List[int]) -> List[List[int]]:
        a.sort()
        diff = min(a[i] - a[i - 1] for i in range(1, len(a)))
        return [[a[i - 1], a[i]] for i in range(1, len(a))
                if a[i] - a[i - 1] == diff]


# 1217 - Minimum Cost to Move Chips to The Same Position - EASY
class Solution:
    def minCostToMoveChips(self, position: List[int]) -> int:
        odd, even = 0, 0
        for chip in position:
            if chip & 1:
                even += 1
            else:
                odd += 1
        return min(odd, even)


class Solution:
    def minCostToMoveChips(self, position: List[int]) -> int:
        cost = [0, 0]
        for chip in position:
            cost[chip & 1] += 1
        return min(cost)


# 1218 - Longest Arithmetic Subsequence of Given Difference - MEDIUM
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        dp = collections.defaultdict(int)
        for i in arr:
            dp[i] = dp[i - difference] + 1
        return max(dp.values())


# 1220 - Count Vowels Permutation - HARD
class Solution:
    def countVowelPermutation(self, n: int) -> int:
        # only: ae, ea, ei, ia, ie, io, iu, oi, ou, ua
        a, e, i, o, u = 1, 1, 1, 1, 1
        for _ in range(n - 1):
            a, e, i, o, u = e + u + i, a + i, o + e, i, i + o
        return (a + e + i + o + u) % (10**9 + 7)


# 1249 - Minimum Remove to Make Valid Parentheses - MEDIUM
class Solution:
    # left must less than right
    # make the invalid parentheses as special character "*"
    # remove the extra "(" and "*"
    def minRemoveToMakeValid(self, s: str) -> str:
        s = list(s)
        left = 0
        for i in range(len(s)):
            if s[i] == ")":
                if left <= 0:
                    s[i] = "*"
                else:
                    left -= 1
            if s[i] == "(":
                left += 1
        # remove "*"
        for i in range(len(s)):
            if s[i] == "*":
                s[i] = ""
        # remove extra "("
        i = len(s) - 1
        while left > 0 and i >= 0:
            if s[i] == "(":
                s[i] = ""
                left -= 1
            i -= 1
        return "".join(s)

    def minRemoveToMakeValid(self, s: str) -> str:
        left = []
        s = list(s)
        for i in range(len(s)):
            # record the index of each "("
            if s[i] == '(':
                left.append(i)
            elif s[i] == ')':
                # no "(" to be paired, remove invalid ")"
                if not left:
                    s[i] = ''
                # has extra "(" to be paired, remove one "("
                # pop will remove the rightmost "("
                else:
                    left.pop()
        for i in left:
            s[i] = ''
        return ''.join(s)

    def minRemoveToMakeValid(self, s: str) -> str:
        arr = list(s)
        stack = []
        for i, ch in enumerate(s):
            if ch == '(':
                stack.append(i)
            elif ch == ')':
                if stack:
                    stack.pop()
                else:
                    arr[i] = ''  # remove extra ')'
        while stack:
            arr[stack.pop()] = ''  # remove extra '('
        return ''.join(arr)


# 1286 - Iterator for Combination - MEDIUM
class CombinationIterator:
    def __init__(self, characters: str, combinationLength: int):
        self.dq = collections.deque(
            itertools.combinations(characters, combinationLength))

    def next(self) -> str:
        return ''.join(self.dq.popleft())

    def hasNext(self) -> bool:
        return len(self.dq) > 0


# 1290 - Convert Binary Number in a Linked List to Integer - EASY
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        s = ""
        while head:
            s += str(head.val)
            head = head.next
        return int(s, 2)

    def getDecimalValue(self, head: ListNode) -> int:
        ans = head.val
        while head := head.next:
            ans = (ans << 1) + head.val
        return ans


# 1291 - Sequential Digits - MEDIUM
class Solution:
    def sequentialDigits(self, low: int, high: int) -> List[int]:
        def dfs(now: int):
            if now > high or now % 10 == 9:
                return
            new = now * 10 + now % 10 + 1
            if low <= new <= high:
                ans.append(new)
            dfs(new)
            return

        ans = []
        for i in range(1, 10):
            dfs(i)
        return sorted(ans)

    def sequentialDigits(self, low: int, high: int) -> List[int]:
        ans = []
        dq = collections.deque(range(1, 10))
        while dq:
            cur = dq.popleft()
            if low <= cur <= high:
                ans.append(cur)
            mod = cur % 10
            if mod < 9:
                dq.append(cur * 10 + mod + 1)
        return ans

    def sequentialDigits(self, low: int, high: int) -> List[int]:
        ans = []
        for i in range(1, 9):
            num = nxt = i
            while num <= high and nxt < 10:
                if num >= low:
                    ans.append(num)
                nxt += 1
                num = num * 10 + nxt
        return sorted(ans)