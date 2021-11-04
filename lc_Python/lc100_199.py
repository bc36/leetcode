from typing import List
import collections, functools

# 121 - Best Time to Buy and Sell Stock - EASY
# Dynamic Programming
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        hisLowPrice, ans = prices[0], 0
        for price in prices:
            ans = max(ans, price - hisLowPrice)
            hisLowPrice = min(hisLowPrice, price)
        return ans

# 129 - Sum Root to Leaf Numbers - MEDIUM
# dfs
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        def dfs(root: TreeNode, pre: int) -> int:
            if not root:
                return 0
            cur = pre * 10 + root.val
            if not root.left and not root.right:
                return cur
            return dfs(root.left, cur) + dfs(root.right, cur)
        return dfs(root, 0)
# bfs
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        total = 0
        nodes = collections.deque([root])
        # (vals) can be optimized spatially. before each node put into deque, change the value of node
        vals = collections.deque([root.val])
        while nodes:
            node = nodes.popleft()
            val = vals.popleft()
            if not node.left and not node.right:
                total += val
            else:
                if node.left:
                    nodes.append(node.left) 
                    vals.append(node.left.val + val * 10)   
                if node.right:
                    nodes.append(node.right) 
                    vals.append(node.right.val + val * 10)  

        return total



# 136 - Single Number - EASY
# XOR operation
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ans = 0
        for i in nums:
            ans ^= i
        return ans

# lambda arguments: expression
# reduce(func, seq)
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        # return functools.reduce(operator.xor, nums)
        return functools.reduce(lambda x, y: x ^ y, nums)

# 137 - Single Number II - MEDIUM
# sort, jump 3 element
# use HashMap also works
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        cnt = collections.Counter(nums)
        ans = [num for num, times in cnt.items() if times == 1]
        return ans[0]
# 没看懂
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        b1,b2 = 0,0 # 出现一次的位，和两次的位
        for n in nums:
            # 既不在出现一次的b1，也不在出现两次的b2里面，我们就记录下来，出现了一次，再次出现则会抵消
            b1 = (b1 ^ n) & ~ b2
            # 既不在出现两次的b2里面，也不再出现一次的b1里面(不止一次了)，记录出现两次，第三次则会抵消 
            b2 = (b2 ^ n) & ~ b1
        return b1