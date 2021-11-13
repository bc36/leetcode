from typing import List
import collections


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


'''
Binary tree - 二叉树
'''


# 二叉树层序遍历 / BFS
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        results = []
        if not root:
            return results
        que = collections.deque([root])
        while que:
            size = len(que)
            thisLevel = []
            # for _ in list(nodeStack): list冻结deque, 否则在更改deque的同时迭代报错如下:
            # RuntimeError: deque mutated during iteration
            for _ in range(size):
                # 倒序弹出
                cur = que.popleft()
                thisLevel.append(cur.val)
                if cur.left:
                    que.append(cur.left)
                if cur.right:
                    que.append(cur.right)
            results.append(thisLevel)

        return results


# 二叉树递归遍历 / DFS
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []

        def dfs(root: TreeNode, depth: int):
            if not root:
                return
            if len(res) == depth:
                # start the current depth
                res.append([])
            # fulfil the current depth, do some operation
            res[depth].append(root.val)
            # process child nodes for the next depth
            if root.left:
                dfs(root.left, depth + 1)
            if root.right:
                dfs(root.right, depth + 1)

        dfs(root, 0)
        return res


# 单调栈(单调递增或单调递减) 解决 下一个更大元素 等问题
class Solution:
    def nextGreaterElement(self, nums: List[int]) -> List[int]:
        ans = [0] * len()
        stack = []
        for i in range(len(nums) - 1, -1, -1):
            while stack and stack[-1] < nums[i]:
                stack.pop()
            ans[i] = -1 if len(stack) == 0 else stack[-1]
            # 下标入栈泛化性更好
            stack.append(nums[i])

        return ans


# 链表 recursive lc-203
