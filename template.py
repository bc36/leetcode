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
# 最小的元素总是在根结点：heap[0] => 最小堆
# 保存负数变最大堆

# 前缀树
# Trie 208

# BFS最短路径问题（BFS，DFS的思考）
# 例题 1091, 994
# 典型的BFS最短路径问题，用DFS也可以求解，但是容易超时。

# 在二维矩阵中搜索，什么时候用BFS，什么时候用DFS？
# 1.如果只是要找到某一个结果是否存在，那么DFS会更高效。因为DFS会首先把一种可能的情况尝试到底，才会回溯去尝试下一种情况，只要找到一种情况，就可以返回了。但是BFS必须所有可能的情况同时尝试，在找到一种满足条件的结果的同时，也尝试了很多不必要的路径；
# 2.如果是要找所有可能结果中最短的，那么BFS回更高效。因为DFS是一种一种的尝试，在把所有可能情况尝试完之前，无法确定哪个是最短，所以DFS必须把所有情况都找一遍，才能确定最终答案（DFS的优化就是剪枝，不剪枝很容易超时）。而BFS从一开始就是尝试所有情况，所以只要找到第一个达到的那个点，那就是最短的路径，可以直接返回了，其他情况都可以省略了，所以这种情况下，BFS更高效。

# BFS解法中的visited为什么可以全局使用？
# BFS是在尝试所有的可能路径，哪个最快到达终点，哪个就是最短。那么每一条路径走过的路不同，visited（也就是这条路径上走过的点）也应该不同，那么为什么visited可以全局使用呢？
# 因为我们要找的是最短路径，那么如果在此之前某个点已经在visited中，也就是说有其他路径在小于或等于当前步数的情况下，到达过这个点，证明到达这个点的最短路径已经被找到。那么显然这个点没必要再尝试了，因为即便去尝试了，最终的结果也不会是最短路径了，所以直接放弃这个点即可。
