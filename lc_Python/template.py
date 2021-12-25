from typing import List
import collections, itertools, functools


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


"""
Binary tree - 二叉树
"""


# 二叉树层序遍历 / BFS
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        results = []
        if not root:
            return results
        dq = collections.deque([root])
        while dq:
            size = len(dq)
            thisLevel = []
            # for _ in list(nodeStack): list冻结deque, 否则在更改deque的同时迭代报错如下:
            # RuntimeError: deque mutated during iteration
            for _ in range(size):
                # 倒序弹出
                cur = dq.popleft()
                thisLevel.append(cur.val)
                if cur.left:
                    dq.append(cur.left)
                if cur.right:
                    dq.append(cur.right)
            results.append(thisLevel)

        return results


# 二叉树递归遍历 / DFS
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        ret = []

        def dfs(root: TreeNode, depth: int):
            if not root:
                return
            if len(ret) == depth:
                # start the current depth
                ret.append([])
            # fulfil the current depth, do some operation
            ret[depth].append(root.val)
            # process child nodes for the next depth
            if root.left:
                dfs(root.left, depth + 1)
            if root.right:
                dfs(root.right, depth + 1)

        dfs(root, 0)
        return ret


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

# 图
'''
邻接表(adjacency list): 出度 / 逆邻接表(inverse adjacency list): 入度
BFS 维护一个入度为0的队列
    拓扑排序 是专门应用于有向图的算法
    需要两个辅助数据结构: 邻接表(set()*n, defaultdict(list)), 入度表(list, defaultdict(int))
DFS 检查有无cycle 
    why逆邻接表: 若求路径时 使用邻接表, 则第一个元素在递归过程中会在栈底, 会逆序输出拓扑排序
    单纯 detect cycle 哪个表无所谓
'''


# 并查集 Union-Find / 可检测有无cycle
class UnionFind:
    def __init__(self):
        """
        记录每个节点的父节点
        """
        self.father = {}

    def find(self, x):
        """
        查找根节点
        路径压缩
        """
        root = x

        while self.father[root] != None:
            root = self.father[root]

        # 路径压缩
        while x != root:
            original_father = self.father[x]
            self.father[x] = root
            x = original_father

        return root

    def union(self, x, y, val):
        """
        合并两个节点
        """
        root_x, root_y = self.find(x), self.find(y)

        if root_x != root_y:
            self.father[root_x] = root_y

    def is_connected(self, x, y):
        """
        判断两节点是否相连
        """
        return self.find(x) == self.find(y)

    def add(self, x):
        """
        添加新节点
        """
        if x not in self.father:
            self.father[x] = None

# 堆
# 所有的k，都有 heap[k] <= heap[2*k+1] 和 heap[k] <= heap[2*k+2]
# 最小的元素总是在根结点：heap[0]