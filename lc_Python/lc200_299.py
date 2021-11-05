from operator import le
from typing import List
import collections, lc100_199


# 235 - Lowest Common Ancestor of a Binary Search Tree - EASY
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# recursive solution
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        return root


# Non-recursive solution
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        while (root.val - p.val) * (root.val - q.val) > 0:
            if root.val > p.val and root.val > q.val:
                root = root.left
            else:
                root = root.right
        return root


# 236 - Lowest Common Ancestor of a Binary Tree - MEDIUM
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        if not root or p == root or q == root:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if not left:
            return right
        if not right:
            return left
        return root


# 260 - Single Number III - MEDUIM
# Hash / O(n) + O(n)
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        cnt = collections.Counter(nums)
        ans = [num for num, times in cnt.items() if times == 1]
        return ans


# "lsb" is the last 1 of its binary representation, means that two numbers are different in that bit
# split nums[] into two lists, one with that bit as 0 and the other with that bit as 1.
# separately perform XOR operation, find the number that appears once in each list.
# O(n) + O(1)
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        xorSum = 0
        for i in nums:
            xorSum ^= i
        lsb = xorSum & -xorSum
        # mask = 1
        # while(xorSum & mask == 0):
        #     mask = mask << 1
        ans1, ans2 = 0, 0
        for i in nums:
            if i & lsb > 0:
                ans1 ^= i
            else:
                ans2 ^= i
        return [ans1, ans2]


# 268 - Missing Number - EASY
# sort
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        nums.sort()
        for i in range(len(nums)):
            if nums[i] != i:
                return i
        return len(nums)


# XOR
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        ans = len(nums)
        for i in range(len(nums)):
            ans = ans ^ i ^ nums[i]
        return ans


# math
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        # (0 + n) * (n + 1) // 2
        n = len(nums)
        total = (0 + n) * (n + 1) // 2
        sum = 0
        for v in nums:
            sum += v
        return total - sum