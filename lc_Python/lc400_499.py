import collections
from typing import List


# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        for i in range(len(nums)):
            index = abs(nums[i]) - 1
            nums[index] = -abs(nums[index])
        return [i + 1 for i in range(len(nums)) if nums[i] > 0]


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
        # stack, dic = [], {}
        # for n in nums2:
        #     while (len(stack) and stack[-1] < n):
        #         dic[stack.pop()] = n
        #     stack.append(n)
        # for i in range(len(nums1)):
        #     nums1[i] = dic.get(nums1[i], -1)
        # return nums1
