import collections, heapq
from typing import List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 921 - Minimum Add to Make Parentheses Valid - MEDIUM
class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        while "()" in s:
            s = s.replace("()", "")
        return len(s)


# stack
class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        left = right = 0
        for ch in s:
            if ch == "(":
                # an extra opening parenthesis
                left += 1
            else:
                left -= 1
            if left == -1:
                # need a closing parenthesis to pair an extra opening parenthesis
                right += 1
                left += 1
        return left + right


# 938 - Range Sum of BST - EASY
# dfs
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        ans = []

        # preorder
        def dfs(root: TreeNode, ans: List[int]):
            if not root:
                return
            # process
            if root.val >= low and root.val <= high:
                ans.append(root.val)
            # left node and right node
            if root.left:
                dfs(root.left, ans)
            if root.right:
                dfs(root.right, ans)
            return

        dfs(root, ans)
        return sum(ans)


class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        # preorder
        def dfs(root: TreeNode):
            if not root:
                return 0
            # process
            val = 0
            if root.val >= low and root.val <= high:
                val = root.val
            # left node and right node
            return val + dfs(root.left) + dfs(root.right)

        return dfs(root)


# search the whole tree (see next solution to speed up)
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        if not root:
            return 0
        val = root.val if root.val >= low and root.val <= high else 0
        return val + self.rangeSumBST(root.left, low, high) + self.rangeSumBST(
            root.right, low, high)


# since its a 'binary search tree' which means that left.val < root.val < right.val
# so we can speed up by jump some unqualified node (the value greater than high or smalller than low)
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        if not root:
            return 0
        if root.val > high:
            return self.rangeSumBST(root.left, low, high)
        if root.val < low:
            return self.rangeSumBST(root.right, low, high)
        return root.val + self.rangeSumBST(
            root.left, low, high) + self.rangeSumBST(root.right, low, high)


# bfs
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        dq = collections.deque()
        ans = 0
        if root:
            dq.append(root)
        while dq:
            # process a layer of nodes
            for _ in range(len(dq)):
                # get one node to process from left side -> FIFO
                node = dq.popleft()
                # add the qualified value
                if node.val >= low and node.val <= high:
                    ans += node.val
                # add new children node to dq
                # guarantee the node is not 'None'
                # if there is no 'if' judgement, 'None' will append to the 'dq',
                # and in the next level loop, when node poped,
                # we will get 'None.val', and get Exception
                if node.left:
                    dq.append(node.left)
                if node.right:
                    dq.append(node.right)
        return ans


# 953 - Verifying an Alien Dictionary - EASY
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        trans = str.maketrans(order, 'abcdefghijklmnopqrstuvwxyz')
        nw = [w.translate(trans) for w in words]
        for i in range(len(words) - 1):
            if nw[i] > nw[i + 1]:
                return False
        return True


class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        m = {c: i for i, c in enumerate(order)}
        words = [[m[c] for c in w] for w in words]
        return all(w1 <= w2 for w1, w2 in zip(words, words[1:]))


class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        return words == sorted(words, key=lambda w: map(order.index, w))


# compare each character in word[i] and word[i+1]
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        order_map = {val: index for index, val in enumerate(order)}
        for i in range(len(words) - 1):
            for j in range(len(words[i])):
                # If we do not find a mismatch letter between words[i] and words[i + 1],
                # we need to examine the case when words are like ("apple", "app").
                if j >= len(words[i + 1]):
                    return False
                if words[i][j] != words[i + 1][j]:
                    if order_map[words[i][j]] > order_map[words[i + 1][j]]:
                        return False
                    # if we find the first different character and they are sorted,
                    # then there's no need to check remaining letters
                    break
        return True


# 973 - K Closest Points to Origin - MEDIUM
# Pay attention that if the points are at the same distance,
# different coordinates should be returned.
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        info = [[x[0] * x[0] + x[1] * x[1], i] for i, x in enumerate(points)]
        # key: square, value: position index
        distance = {}
        for i in info:
            if i[0] not in distance:
                distance[i[0]] = [i[1]]
            else:
                distance[i[0]].append(i[1])
        order = list(distance.keys())
        order.sort()
        ans = []
        i = 0
        while len(ans) < k:
            if distance[order[i]]:
                ans.append(points[distance[order[i]].pop()])
            else:
                i += 1
        return ans


class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap = []
        for (x, y) in points:
            dist = -(x * x + y * y)
            if len(heap) == k:
                heapq.heappushpop(heap, (dist, x, y))
            else:
                heapq.heappush(heap, (dist, x, y))
        return [(x, y) for (_, x, y) in heap]


class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        points.sort(key=lambda x: (x[0]**2 + x[1]**2))
        return points[:k]


class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        q = [(-x**2 - y**2, i) for i, (x, y) in enumerate(points[:k])]
        heapq.heapify(q)

        n = len(points)
        for i in range(k, n):
            x, y = points[i]
            dist = -x**2 - y**2
            heapq.heappushpop(q, (dist, i))
        ans = [points[identity] for (_, identity) in q]
        return ans