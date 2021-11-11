from operator import le
from typing import List
import collections, lc100_199


# 206 - Reverse Linked List - EASY
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre = None
        while head:
            tmp = head.next
            head.next = pre
            pre = head
            head = tmp
        return pre


# 215 - Kth Largest Element in an Array - MEDIUM
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums.sort()
        return nums[-k]


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
# need to know the status of left and right subtrees
# then we can proceed to the next step, so we use postorder traversal
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        if not root or p.val == root.val or q.val == root.val:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if not left:
            return right
        if not right:
            return left
        return root


# three cases:
# 1. root == p || root == q
# 2. p, q are subtree in two sides (p in left, q in right and vice versa)
# 3. p, q on the same side of subtree, recursive


class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        if root.val == p.val: return root
        if root.val == q.val: return root
        leftNode = self.lowestCommonAncestor(root.left, p, q)
        rightNode = self.lowestCommonAncestor(root.right, p, q)
        if leftNode and rightNode: return root
        return leftNode if leftNode else rightNode


# iterative solution


class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        stack = [root]
        parent = {root: None}
        # find p's and q's parents
        while p not in parent or q not in parent:
            node = stack.pop()
            if node.left:
                parent[node.left] = node
                stack.append(node.left)
            if node.right:
                parent[node.right] = node
                stack.append(node.right)
        ancestors = set()
        # Backtracking p's all ancestors, until to the root
        while p:
            ancestors.add(p)
            p = parent[p]
            print(p)
        print(ancestors)
        # find q's ancestor, if not, until to the root
        while q not in ancestors:
            q = parent[q]
        return q


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


# 299 - Bulls and Cows - MEDIUM
class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        bull = 0
        numS, numG = [0] * 10, [0] * 10
        for k, _ in enumerate(secret):
            if secret[k] == guess[k]:
                bull += 1
            else:
                numS[int(secret[k])] += 1
                numG[int(guess[k])] += 1
        cow = sum([min(numS[k], numG[k]) for k, _ in enumerate(numS)])

        # base = set(guess)
        # for i in base:
        #     if i in secret:
        #         cow += min(guess.count(i),secret.count(i))
        # cow = cow-bull

        # str(bull) + "A" + str(cow) + "B"
        # "{}A{}B".format(bull, cow)
        return f'{bull}A{cow}B'