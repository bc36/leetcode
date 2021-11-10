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
            for _ in range(size):
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

        def dfs(root, depth):
            if not root:
                return []
            if len(res) == depth:
                # start the current depth
                res.append([])
            # fulfil the current depth
            res[depth].append(root.val)
            # process child nodes for the next depth
            if root.left:
                dfs(root.left, depth + 1)
            if root.right:
                dfs(root.right, depth + 1)

        dfs(root, 0)
        return res