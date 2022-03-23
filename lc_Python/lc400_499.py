import collections, bisect, functools, math, heapq
from typing import List


# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 400 - Nth Digit - MEDIUM
class Solution:
    def findNthDigit(self, n: int) -> int:
        k = 1
        while k * (10**k) < n:
            n += 10**k
            k += 1
        return int(str(n // k)[n % k])

    def findNthDigit(self, n):
        n -= 1
        for digits in range(1, 11):
            first = 10**(digits - 1)
            if n < 9 * first * digits:
                return int(str(first + n / digits)[n % digits])
            n -= 9 * first * digits


# 402 - Remove K Digits - MEDIUM
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []
        for d in num:
            while stack and k and stack[-1] > d:
                stack.pop()
                k -= 1
            stack.append(d)
        if k > 0:
            stack = stack[:-k]
        return ''.join(stack).lstrip('0') or "0"


# 408 - Valid Word Abbreviation - EASY
class Solution:
    def validWordAbbreviation(self, word: str, abbr: str) -> bool:
        i = j = 0
        while i < len(word) and j < len(abbr):
            if abbr[j].isalpha():
                if word[i] != abbr[j]:
                    return False
                i += 1
                j += 1
            else:
                if abbr[j] == "0":
                    return False
                tmp = ""
                while j < len(abbr) and abbr[j].isdigit():
                    tmp = tmp + abbr[j]
                    j += 1
                i += int(tmp)
        return i == len(word) and j == len(abbr)


# 409 - Longest Palindrome - EASY
class Solution:
    def longestPalindrome(self, s: str) -> int:
        cnt = collections.Counter(s)
        ans, odd = 0, False
        for k in cnt:
            if cnt[k] & 1:
                odd = True
            ans += cnt[k] // 2 * 2
        return ans + 1 if odd else ans

    def longestPalindrome(self, s: str) -> int:
        arr = [0] * 128
        for ch in s:
            arr[ord(ch) - ord('a')] += 1
        odd = 0
        for n in arr:
            odd += n & 1
        return len(s) - odd + 1 if odd else len(s)


# 413 - Arithmetic Slices - MEDIUM
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        dp = [0] * len(nums)
        for i in range(1, len(nums) - 1):
            if nums[i - 1] + nums[i + 1] == nums[i] * 2:
                dp[i] = dp[i - 1] + 1
        return sum(dp)

    # (1,2,3)->1 (1,2,3,4)->3 (1,2,3,4,5)->6 (1,2,3,4,5,6)->10
    # add a number to an Arithmetic Slices, each increment is added by 1
    # so there are two ways of understanding:
    # 1. the rule of equal variance series / 2. space optimized dp
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        if len(nums) < 3:
            return 0
        ans = add = 0
        for i in range(2, len(nums)):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                add += 1
                ans += add
            else:
                add = 0
        return ans


# 415 - Add Strings - EASY
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        ans = ""
        i, j, carry = len(num1) - 1, len(num2) - 1, 0
        while i >= 0 or j >= 0 or carry != 0:
            n1 = int(num1[i]) if i >= 0 else 0
            n2 = int(num2[j]) if j >= 0 else 0
            tmp = n1 + n2 + carry
            carry = tmp // 10
            ans = str(tmp % 10) + ans
            i, j = i - 1, j - 1
        return ans


# 419 - Battleships in a Board - MEDIUM
class Solution:
    def countBattleships(self, board):
        total = 0
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 'X':
                    flag = 1
                    if j > 0 and board[i][j - 1] == 'X': flag = 0
                    if i > 0 and board[i - 1][j] == 'X': flag = 0
                    total += flag
        return total


# 421 - Maximum XOR of Two Numbers in an Array - MEDIUM
class Solution:
    # 1000ms
    def findMaximumXOR(self, nums: List[int]) -> int:
        ans = 0
        for i in reversed(range(32)):
            prefixes = set([x >> i for x in nums])
            ans <<= 1
            candidate = ans + 1
            for p in prefixes:
                if candidate ^ p in prefixes:
                    ans = candidate
                    break
        return ans

    # 300ms
    def findMaximumXOR(self, nums: List[int]) -> int:
        ans = 0
        max_len = max(nums).bit_length()
        if not max_len:
            return 0
        k = 1 << (max_len - 1)
        while k:
            seen = set()
            ans ^= k
            for num in nums:
                seen.add(cur := (ans & num))
                if cur ^ ans in seen:
                    break
            else:
                ans ^= k
            k >>= 1
        return ans


# 423 - Reconstruct Original Digits from English - MEDIUM
class Solution:
    def originalDigits(self, s: str) -> str:
        n0 = s.count("z")
        n2 = s.count("w")
        n4 = s.count("u")
        n6 = s.count("x")
        n8 = s.count("g")
        n1 = s.count("o") - n0 - n2 - n4
        n3 = s.count("t") - n2 - n8
        n5 = s.count("f") - n4
        n7 = s.count("s") - n6
        n9 = s.count("i") - n5 - n6 - n8

        ns = (n0, n1, n2, n3, n4, n5, n6, n7, n8, n9)
        return "".join((str(i) * n for i, n in enumerate(ns)))


# 426 - Convert Binary Search Tree to Sorted Doubly Linked List - MEDIUM
# inorder, bfs
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if not root:
            return None
        dummy = Node(-1)
        pre = dummy
        stack, node = [], root
        while stack or node:
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            # node.left, prev.right, prev = prev, node, node
            node.left = pre
            pre.right = node
            pre = node
            node = node.right
        dummy.right.left, pre.right = pre, dummy.right
        return dummy.right


# 432 - All O`one Data Structure - HARD
# TODO


# 435 - Non-overlapping Intervals - MEDIUM
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: x[1])
        ans = 0
        mxr = -math.inf
        for l, r in intervals:
            if mxr <= l:
                mxr = r
            # elif mxr <= r:
            else:
                ans += 1
        return ans

    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals = sorted(intervals, key=lambda x: x[1])
        mxr = intervals[0][1]
        can = 1
        for i in range(1, len(intervals)):
            if intervals[i][0] >= mxr:
                can += 1
                mxr = intervals[i][1]
        return len(intervals) - can


# 438 - Find All Anagrams in a String - MEDIUM
class Solution:
    # sliding window + list
    def findAnagrams(self, s: str, p: str) -> List[int]:
        if len(s) < len(p):
            return []
        ans, ss, pp = [], [0] * 26, [0] * 26
        for i in range(len(p)):
            ss[ord(s[i]) - ord('a')] += 1
            pp[ord(p[i]) - ord('a')] += 1
        if ss == pp:
            ans.append(0)
        k = len(p)
        for i in range(len(p), len(s)):
            ss[ord(s[i]) - ord('a')] += 1
            ss[ord(s[i - k]) - ord('a')] -= 1
            if ss == pp:
                ans.append(i - k + 1)
        return ans

    # sliding window + two pointers
    def findAnagrams(self, s: str, p: str) -> List[int]:
        ans = []
        if len(s) < len(p):
            return ans
        p_cnt = [0] * 26
        s_cnt = [0] * 26
        for i in range(len(p)):
            p_cnt[ord(p[i]) - ord('a')] += 1

        left = 0
        for right in range(len(s)):
            cur_right = ord(s[right]) - ord('a')
            s_cnt[cur_right] += 1
            while s_cnt[cur_right] > p_cnt[cur_right]:
                # move left pointer to satisfy 's_cnt[cur_right] == p_cnt[cur_right]'
                cur_left = ord(s[left]) - ord('a')
                s_cnt[cur_left] -= 1
                left += 1
            if right - left + 1 == len(p):
                ans.append(left)
        return ans


# 440 - K-th Smallest in Lexicographical Order - HARD
class Solution:
    # O(logn * logn) / O(1)
    def findKthNumber(self, n: int, k: int) -> int:
        def getCnt(first: int) -> int:
            cnt = 0
            last = first
            while first <= n:
                cnt += min(last, n) - first + 1
                first *= 10
                last = last * 10 + 9
            return cnt

        cnt = prefix = 1
        while cnt < k:
            add = getCnt(prefix)
            if cnt + add > k:
                prefix *= 10
                cnt += 1
            else:
                prefix += 1
                cnt += add
        return prefix

    def findKthNumber(self, n: int, k: int) -> int:
        def cal_steps(n1, n2):
            step = 0
            while n1 <= n:
                step += min(n2, n + 1) - n1
                n1 *= 10
                n2 *= 10
            return step

        cur = 1
        k -= 1
        while k:
            steps = cal_steps(cur, cur + 1)
            if steps <= k:
                k -= steps
                cur += 1
            else:
                k -= 1
                cur *= 10
        return cur


# 441 - Arranging Coins - EASY
class Solution:
    def arrangeCoins(self, n: int) -> int:
        for i in range(n):
            i += 1
            n -= i
            if n == 0:
                return i
            if n < 0:
                return i - 1


# 443 - String Compression - MEDIUM
class Solution:
    def compress(self, chars: List[str]) -> int:
        i = j = 0  # read / write
        while i < len(chars):
            chars[j] = chars[i]
            j += 1
            count = 0
            while i < len(chars) and chars[i] == chars[j - 1]:
                count += 1
                i += 1
            if count > 1:
                for ch in str(count):
                    chars[j] = ch
                    j += 1
        return j

    def compress(self, chars: List[str]) -> int:
        l = r = 0
        while r < len(chars):
            count = 1
            while r + 1 < len(chars) and chars[r] == chars[r + 1]:
                count += 1
                r += 1
            chars[l] = chars[r]
            if count > 1:
                s = str(count)
                chars[l + 1:l + 1 + len(s)] = s
                l += len(s)
            l += 1
            r += 1
        return l


# 448 - Find All Numbers Disappeared in an Array - EASY
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        n = [0] * len(nums)
        for i in range(len(nums)):
            n[nums[i] - 1] = 1
        ans = []
        for i in range(len(n)):
            if n[i] == 0:
                ans.append(i + 1)
        return ans

    # marker the scaned number as negative
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        for i in range(len(nums)):
            index = abs(nums[i]) - 1
            nums[index] = -abs(nums[index])
        return [i + 1 for i in range(len(nums)) if nums[i] > 0]


# 452 - Minimum Number of Arrows to Burst Balloons - MEDIUM
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points = sorted(points, key=lambda x: x[1])
        ans, end = 0, float('-inf')
        for p in points:
            if p[0] > end:
                ans += 1
                end = p[1]
        return ans


# 454 - 4Sum II - MEDIUM
class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int],
                     nums3: List[int], nums4: List[int]) -> int:
        dic = collections.defaultdict(int)
        for n1 in nums1:
            for n2 in nums2:
                dic[n1 + n2] += 1
        ans = 0
        for n3 in nums3:
            for n4 in nums4:
                ans += dic[-n3 - n4]
        return ans

    def fourSumCount(self, nums1: List[int], nums2: List[int],
                     nums3: List[int], nums4: List[int]) -> int:
        ab = collections.Counter(a + b for a in nums1 for b in nums2)
        return sum(ab[-c - d] for c in nums3 for d in nums4)


# 461 - Hamming Distance - EASY
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        ans = 0
        while x != 0 and y != 0:
            if x & 1 != y & 1:
                ans += 1
            x >>= 1
            y >>= 1
        while x != 0:
            if x & 1:
                ans += 1
            x >>= 1
        while y != 0:
            if y & 1:
                ans += 1
            y >>= 1
        return ans


# 475 - Heaters - MEDIUM
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        heaters = heaters + [float('-inf'), float('inf')]
        houses.sort()
        heaters.sort()
        ans, i = 0, 0
        for h in houses:
            while h > heaters[i + 1]:
                i += 1
            dis = min(h - heaters[i], heaters[i + 1] - h)
            ans = max(ans, dis)
        return ans


# 476 - Number Complement - EASY
class Solution:
    def findComplement(self, num: int) -> int:
        mask = num
        mask |= mask >> 1
        mask |= mask >> 2
        mask |= mask >> 4
        mask |= mask >> 8
        mask |= mask >> 16
        return num ^ mask

    def findComplement(self, num: int) -> int:
        a = 1  # sum is -1
        while True:
            if num >= a:
                a <<= 1
            else:
                return a - num - 1


# 495 - Teemo Attacking - EASY
class Solution:
    def findPoisonedDuration(self, timeSeries: List[int],
                             duration: int) -> int:
        ans = 0
        for i in range(1, len(timeSeries)):
            ans += min(duration, timeSeries[i] - timeSeries[i - 1])
        return ans + duration

    # reduce the number of function calls can speed up the operation
    def findPoisonedDuration(self, timeSeries: List[int],
                             duration: int) -> int:
        ans = 0
        lastTime = timeSeries[0]
        for i in timeSeries[1:]:
            if i - lastTime > duration:
                ans += duration
            else:
                ans += i - lastTime
            lastTime = i
        return ans + duration


# 496 - Next Greater Element I - EASY
class Solution:
    # brutal-force solution
    def nextGreaterElement(self, nums1: List[int],
                           nums2: List[int]) -> List[int]:
        m, n = len(nums1), len(nums2)
        ret = [0] * m
        for i in range(m):
            j = nums2.index(nums1[i])
            k = j + 1
            while k < n and nums2[k] < nums2[j]:
                k += 1
            ret[i] = nums2[k] if k < n else -1
        return ret

    # stack
    def nextGreaterElement(self, nums1: List[int],
                           nums2: List[int]) -> List[int]:
        stack = []
        dic = {}  # save the next greater element
        for i in range(len(nums2) - 1, -1, -1):
            while stack and nums2[i] > stack[-1]:
                stack.pop()
            dic[nums2[i]] = -1 if len(stack) == 0 else stack[-1]
            stack.append(nums2[i])
        return [dic[n1] for n1 in nums1]

    def nextGreaterElement(self, nums1: List[int],
                           nums2: List[int]) -> List[int]:
        stack = []
        dic = {}  # save the next greater element
        for num in nums2[::-1]:
            while stack and num > stack[-1]:
                stack.pop()
            if stack:
                dic[num] = stack[-1]
            stack.append(num)
        return [dic.get(num, -1) for num in nums1]
        # stack, dic = [], {}
        # for n in nums2:
        #     while (len(stack) and stack[-1] < n):
        #         dic[stack.pop()] = n
        #     stack.append(n)
        # for i in range(len(nums1)):
        #     nums1[i] = dic.get(nums1[i], -1)
        # return nums1
