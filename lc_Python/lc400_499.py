from typing import List


# 441 - Arranging Coins - EASY
# math
class Solution:
    def arrangeCoins(self, n: int) -> int:
        for i in range(n):
            i += 1
            n -= i
            if n == 0:
                return i
            if n < 0:
                return i - 1


# 495 - Teemo Attacking - EASY
class Solution:
    def findPoisonedDuration(self, timeSeries: List[int],
                             duration: int) -> int:
        ans = 0
        for i in range(1, len(timeSeries)):
            ans += min(duration, timeSeries[i] - timeSeries[i - 1])
        return ans + duration


# reduce the number of function calls can speed up the operation
class Solution:
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
# brutal-force solution


class Solution:
    def nextGreaterElement(self, nums1: List[int],
                           nums2: List[int]) -> List[int]:
        m, n = len(nums1), len(nums2)
        res = [0] * m
        for i in range(m):
            j = nums2.index(nums1[i])
            k = j + 1
            while k < n and nums2[k] < nums2[j]:
                k += 1
            res[i] = nums2[k] if k < n else -1
        return res


# stack
class Solution:
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


class Solution:
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
        '''
        stack, dic = [], {}
        for n in nums2:
            while (len(stack) and stack[-1] < n):
                dic[stack.pop()] = n
            stack.append(n)
        for i in range(len(nums1)):
            nums1[i] = dic.get(nums1[i], -1)
        return nums1
        '''
