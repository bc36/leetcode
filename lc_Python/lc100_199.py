import bisect, collections, copy, functools, heapq, itertools, math, operator, random, string
from typing import Callable, List, Literal, Optional, Tuple
import sortedcontainers


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 101 - Symmetric Tree - EASY
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        def dfs(left: TreeNode, right: TreeNode) -> bool:
            if not (left or right):
                return True
            if not (left and right):
                return False
            if left.val != right.val:
                return False
            return dfs(left.left, right.right) and dfs(left.right, right.left)

        return dfs(root.left, root.right)

    def isSymmetric(self, root: TreeNode) -> bool:
        def isSym(p: TreeNode, q: TreeNode) -> bool:
            if p and q and p.val == q.val:
                return isSym(p.left, q.right) and isSym(p.right, q.left)
            return p == q

        return not root or isSym(root.left, root.right)

    def isSymmetric(self, root: TreeNode) -> bool:
        if not root or not (root.left or root.right):
            return True
        dq = collections.deque([root.left, root.right])
        while dq:
            left = dq.popleft()
            right = dq.popleft()
            if not (left or right):
                continue
            if not (left and right):
                return False
            if left.val != right.val:
                return False
            dq.append(left.left)
            dq.append(right.right)
            dq.append(left.right)
            dq.append(right.left)
        return True


# 102 - Binary Tree Level Order Traversal - MEDIUM
class Solution:
    # bfs: breadth-first search
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        ans = []
        dq = collections.deque()
        if root:
            dq.append(root)
        while dq:
            level = []
            for _ in list(dq):
                node = dq.popleft()
                if node.left:
                    dq.append(node.left)
                if node.right:
                    dq.append(node.right)
                level.append(node.val)
            ans.append(level)
        return ans

    # dfs: depth-first search
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        ans = []

        def dfs(root: TreeNode, level: int):
            if not root:
                return
            if level == len(ans):
                ans.append([])
            ans[level].append(root.val)
            dfs(root.left, level + 1)
            dfs(root.right, level + 1)
            return

        dfs(root, 0)
        return ans


# 103 - Binary Tree Zigzag Level Order Traversal - MEDIUM
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        ans = []
        dq = collections.deque([root])
        f = True
        while dq:
            lv = collections.deque([])
            for _ in range(len(dq)):
                n = dq.popleft()
                if f:
                    lv.append(n.val)
                else:
                    lv.appendleft(n.val)
                if n.left:
                    dq.append(n.left)
                if n.right:
                    dq.append(n.right)
            f = not f
            ans.append(list(lv))
        return ans

    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        ans = []
        q = [root]
        f = 0
        while q:
            lv = []
            nxt = []
            for n in q:
                lv.append(n.val)
                if n.left:
                    nxt.append(n.left)
                if n.right:
                    nxt.append(n.right)
            lv = lv[::-1] if f & 1 else lv
            ans.append(lv)
            f += 1
            q = nxt
        return ans

    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        ans = []
        q = [root]
        l2r = 1
        while q:
            nxt = []
            v = [x.val for x in q]
            if l2r:
                ans.append(v)
            else:
                ans.append(v[::-1])
            l2r = 1 - l2r
            for x in q:
                if x.left:
                    nxt.append(x.left)
                if x.right:
                    nxt.append(x.right)
            q = nxt
        return ans


# 104 - Maximum Depth of Binary Tree - EASY
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        def dfs(r, s):
            if not r:
                self.ans = max(self.ans, s)
                return
            dfs(r.left, s + 1)
            dfs(r.right, s + 1)
            return

        self.ans = 0
        dfs(root, 0)
        return self.ans

    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        dq = collections.deque([root])
        step = 0
        while dq:
            for _ in range(len(dq)):
                r = dq.popleft()
                if r.left:
                    dq.append(r.left)
                if r.right:
                    dq.append(r.right)
            step += 1
        return step


# 105 - Construct Binary Tree from Preorder and Inorder Traversal - MEDIUM
class Solution:
    # slow, need to find index every time and new four sublists
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder:
            return None
        root = TreeNode(preorder[0])
        idx = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1 : idx + 1], inorder[:idx])
        root.right = self.buildTree(preorder[idx + 1 :], inorder[idx + 1 :])
        return root

    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        dic = {val: idx for idx, val in enumerate(inorder)}

        def helper(p: int, ileft: int, iright: int) -> TreeNode:
            if ileft > iright:
                return None
            root = TreeNode(preorder[p])
            idx = dic[preorder[p]]
            root.left = helper(p + 1, ileft, idx - 1)
            root.right = helper(p + idx - ileft + 1, idx + 1, iright)
            return root

        return helper(0, 0, len(inorder) - 1)


# 108 - Convert Sorted Array to Binary Search Tree - EASY
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        mid = len(nums) // 2
        left = nums[:mid]
        right = nums[mid + 1 :]
        root = TreeNode(nums[mid])
        if left:
            root.left = self.sortedArrayToBST(left)
        if right:
            root.right = self.sortedArrayToBST(right)
        return root

    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        def helper(left, right):
            if left > right:
                return None
            mid = (left + right) // 2
            root = TreeNode(nums[mid])
            root.left = helper(left, mid - 1)
            root.right = helper(mid + 1, right)
            return root

        return helper(0, len(nums) - 1)


# 112 - Path Sum - EASY
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        stack = []
        if root:
            stack.append((root, targetSum))
        while stack:
            n, t = stack.pop()
            if t - n.val == 0 and not (n.left or n.right):
                return True
            if n.left:
                stack.append((n.left, t - n.val))
            if n.right:
                stack.append((n.right, t - n.val))
        return False

    def hasPathSum(self, root: Optional[TreeNode], t: int) -> bool:
        if not root:
            return False
        if not (root.right or root.left):
            return root.val == t
        return self.hasPathSum(root.left, t - root.val) or self.hasPathSum(
            root.right, t - root.val
        )


# 113 - Path Sum II - MEDIUM
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        ans = list()
        path = list()

        def dfs(root: TreeNode, t: int):
            if not root:
                return
            path.append(root.val)
            t -= root.val
            if not root.left and not root.right and t == 0:
                ans.append(path[:])
            dfs(root.left, t)
            dfs(root.right, t)
            path.pop()
            return

        dfs(root, targetSum)
        return ans

    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        ans = []
        if not root:
            return []

        def dfs(r: TreeNode, path: List[int], t: int):
            if not r.left and not r.right:
                if t + r.val == targetSum:
                    path.append(r.val)
                    ans.append(path)
                return
            if r.left:
                dfs(r.left, path + [r.val], t + r.val)
            if r.right:
                dfs(r.right, path + [r.val], t + r.val)
            return

        dfs(root, [], 0)
        return ans

    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        stack = []
        ans = []
        if root:
            stack.append((root, targetSum, []))
        while stack:
            n, t, p = stack.pop()
            p.append(n.val)
            if t - n.val == 0 and not (n.left or n.right):
                n1 = p[:]
                ans.append(n1)
            if n.left:
                n2 = p[:]
                stack.append((n.left, t - n.val, n2))
            if n.right:
                n3 = p[:]
                stack.append((n.right, t - n.val, n3))
        return ans


# 116 - Populating Next Right Pointers in Each Node - MEDIUM
class Solution:
    # O(n) / O(n)
    def connect(self, root: "Optional[Node]") -> "Optional[Node]":
        if not root:
            return None
        dq = collections.deque([root])
        while dq:
            sz = len(dq)
            for i in range(sz):
                n = dq.popleft()
                if i < sz - 1:
                    n.next = dq[0]
                # if i == sz - 1:
                #     n.next = None
                # else:
                #     n.next = dq[0]
                if n.left:
                    dq.append(n.left)
                if n.right:
                    dq.append(n.right)
        return root


# 117 - Populating Next Right Pointers in Each Node II - MEDIUM
class Solution:
    def connect(self, root: "Node") -> "Node":
        if not root:
            return
        cur = root
        while cur:
            dummy = Node(None)
            pre = dummy
            while cur:
                if cur.left:
                    pre.next = cur.left
                    pre = pre.next
                if cur.right:
                    pre.next = cur.right
                    pre = pre.next
                cur = cur.next
            cur = dummy.next
        return root


# 118 - Pascal's Triangle - EASY
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        ans = [[1]]
        for _ in range(numRows - 1):
            new = [1]
            for j in range(len(ans[-1]) - 1):
                new.append(ans[-1][j] + ans[-1][j + 1])
            new += [1]
            ans.append(new)
        return ans

    # fastest
    def generate(self, numRows: int) -> List[List[int]]:
        pascal = [[1] * (i + 1) for i in range(numRows)]
        for i in range(numRows):
            for j in range(1, i):
                pascal[i][j] = pascal[i - 1][j - 1] + pascal[i - 1][j]
        return pascal

    def generate(self, numRows: int) -> List[List[int]]:
        #    1 3 3 1 0
        # +  0 1 3 3 1
        # =  1 4 6 4 1
        res = [[1]]
        for _ in range(1, numRows):
            res += [map(lambda x, y: x + y, res[-1] + [0], [0] + res[-1])]
        return res[:numRows]


# 119 - Pascal's Triangle II - EASY
class Solution(object):
    def getRow(self, rowIndex: int) -> List[int]:
        row = [1]
        for _ in range(rowIndex):
            row = [x + y for x, y in zip([0] + row, row + [0])]
        return row

    def getRow(self, rowIndex: int) -> List[int]:
        row = [1]
        for _ in range(rowIndex):
            row = [1] + [row[j] + row[j + 1] for j in range(len(row) - 1)] + [1]
        return row


# 120 - Triangle - MEDIUM
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        dp = [[0] * len(x) for x in triangle]
        dp[0][0] = triangle[0][0]
        for i in range(1, len(triangle)):
            dp[i][0] = dp[i - 1][0] + triangle[i][0]
            for j in range(1, i):
                dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1]) + triangle[i][j]
            dp[i][i] = dp[i - 1][-1] + triangle[i][i]
        return min(dp[-1])


# 121 - Best Time to Buy and Sell Stock - EASY
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        mi = prices[0]
        for p in prices:
            ans = max(ans, p - mi)
            mi = min(p, mi)
        return ans

    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        buy = math.inf
        for p in prices:
            buy = min(buy, p)
            ans = max(ans, p - buy)
        return ans


# 124 - Binary Tree Maximum Path Sum - HARD
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        def getMax(root: TreeNode) -> int:
            # nonlocal ans
            if not root:
                return 0
            l = max(0, getMax(root.left))
            r = max(0, getMax(root.right))
            self.ans = max(self.ans, root.val + l + r)
            return root.val + max(l, r)

        self.ans = float("-inf")
        getMax(root)
        return self.ans


# 125 - Valid Palindrome - EASY
class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = s.lower()
        s = "".join(
            [
                ch
                for ch in s
                if (97 <= ord(ch) and ord(ch) <= 122)
                or (48 <= ord(ch) and ord(ch) <= 57)
            ]
        )
        # s.isalnum(): alphabet or numeric
        # s = "".join(ch.lower() for ch in s if ch.isalnum())
        return s == s[::-1]


# 127 - Word Ladder - HARD
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0
        s = set(wordList)
        dq = collections.deque([beginWord])
        step = 1
        while dq:
            n = len(dq)
            for _ in range(n):
                cur = dq.popleft()
                for i in range(len(cur)):
                    for ch in "qwertyuiopasdfghjklzxcvbnm":
                        new = cur[:i] + ch + cur[i + 1 :]
                        if new == endWord:
                            return step + 1
                        if new in s:
                            dq.append(new)
                            s.remove(new)
            step += 1
        return 0

    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0
        mapper = collections.defaultdict(list)
        for word in wordList:
            for i in range(len(word)):
                key = word[:i] + "*" + word[i + 1 :]
                mapper[key].append(word)
        dq = collections.deque([(beginWord, 1)])
        seen = set(beginWord)
        while dq:
            cur, step = dq.popleft()
            for i in range(len(cur)):
                key = cur[:i] + "*" + cur[i + 1 :]
                for word in mapper[key]:
                    if word == endWord:
                        return step + 1
                    if word not in seen:
                        seen.add(word)
                        dq.append((word, step + 1))
        return 0

    # bi-directional bfs
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordSet = set(wordList)
        startSet, endSet = set([beginWord]), set([endWord])
        if endWord not in wordSet:
            return 0
        steps = 1
        while startSet and endSet:
            if len(startSet) > len(endSet):
                startSet, endSet = endSet, startSet
            steps += 1
            nextSet = set()
            for word in startSet:
                for i in range(len(word)):
                    for ch in "abcdefghijklmnopqrstuvwxyz":
                        temp = word[:i] + ch + word[i + 1 :]
                        if temp in endSet:
                            return steps
                        if temp in wordSet:
                            nextSet.add(temp)
                            wordSet.remove(temp)
            startSet = nextSet
        return 0


# 128 - Longest Consecutive Sequence - MEDIUM
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        nums = set(nums)
        longest = 0
        for num in nums:
            if num - 1 not in nums:
                curNum = num
                curLen = 1
                while curNum + 1 in nums:
                    curNum += 1
                    curLen += 1
                """
                'curLen' can be optimized
                nextOne = num + 1
                while nextOne in nums:
                    nextOne += 1
                longest = max(longest, nextOne - num)
                """
                longest = max(longest, curLen)
        return longest

    def longestConsecutive(self, nums: List[int]) -> int:
        nums, maxlen = set(nums), 0
        while nums:
            num = nums.pop()
            l, r = num - 1, num + 1
            while l in nums:
                nums.remove(l)
                l -= 1
            while r in nums:
                nums.remove(r)
                r += 1
            l += 1
            r -= 1
            maxlen = max(maxlen, r - l + 1)
        return maxlen


# 129 - Sum Root to Leaf Numbers - MEDIUM
class Solution:
    # dfs
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


# 130 - Surrounded Regions - MEDIUM
class Solution:
    # search from edge
    # dfs
    def solve(self, board: List[List[str]]) -> None:
        row, col = len(board), len(board[0])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def dfs(i: int, j: int):
            if 0 <= i < row and 0 <= j < col and board[i][j] == "O":
                board[i][j] = "*"
                for x, y in directions:
                    dfs(i + x, j + y)
            return

        for i in range(row):
            dfs(i, 0)
            dfs(i, col - 1)
        for j in range(col):
            dfs(0, j)
            dfs(row - 1, j)
        for i in range(row):
            for j in range(col):
                board[i][j] = "X" if board[i][j] != "*" else "O"
        return

    # bfs
    def solve(self, board: List[List[str]]) -> None:
        dq = collections.deque([])
        row, col = len(board), len(board[0])
        for r in range(row):
            for c in range(col):
                if (r in [0, row - 1] or c in [0, col - 1]) and board[r][c] == "O":
                    dq.append((r, c))
        while dq:
            r, c = dq.popleft()
            if 0 <= r < row and 0 <= c < col and board[r][c] == "O":
                board[r][c] = "*"
                dq.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])

        for i in range(row):
            for j in range(col):
                board[i][j] = "X" if board[i][j] != "*" else "O"
        return


# 131 - Palindrome Partitioning - MEDIUM
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def dfs(path, s):
            if len(s) == 0:
                ans.append(list(path))
                return
            for i in range(1, len(s) + 1):
                if s[:i] == s[:i][::-1]:
                    dfs(path + [s[:i]], s[i:])
            return

        ans = []
        dfs([], s)
        return ans


# 133 - Clone Graph - MEDIUM
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = [neighbors if neighbors is not None else []]


class Solution:
    def cloneGraph(self, node: "Node") -> "Node":
        def dfs(node: "Node") -> "Node":
            if not node:
                return
            if node.val in seen:
                return seen[node.val]
            clone = Node(node.val, [])
            seen[node.val] = clone
            for n in node.neighbors:
                clone.neighbors.append(dfs(n))
            return clone

        seen = {}
        return dfs(node)

    def cloneGraph(self, node: "Node") -> "Node":
        if not node:
            return
        clone = Node(node.val, [])
        seen = {node: clone}
        queue = collections.deque([node])
        while queue:
            cur = queue.popleft()
            for n in cur.neighbors:
                if n not in seen:
                    seen[n] = Node(n.val, [])
                    queue.append(n)
                seen[cur].neighbors.append(seen[n])
        return clone


# 134 - Gas Station - MEDIUM
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1
        surplus = start = 0
        for i in range(len(gas)):
            surplus += gas[i] - cost[i]
            # if surplus < 0 again, it means the answer is at the rest of stations
            # in other words, the answer is the station next to the last station with -surplus
            if surplus < 0:
                start = i + 1
                surplus = 0
        return start

    # the answer is the next station of the station with 'minSurplus'
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        total = idx = 0
        surplus = float("inf")
        for i in range(len(gas)):
            total += gas[i] - cost[i]
            if total < surplus:
                idx = i
                surplus = total
        return (idx + 1) % len(gas) if total >= 0 else -1

    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1
        ans = pre = g = 0
        for i in range(len(gas)):
            g += gas[i] - cost[i]
            if g < pre:
                pre = g
                # ans = (i + 1) % len(gas)
                # whether the first place is the answer will be judged at index 0, rather than at the last index
                ans = i + 1
        return ans


# 136 - Single Number - EASY
class Solution:
    # XOR operation
    def singleNumber(self, nums: List[int]) -> int:
        ans = 0
        for i in nums:
            ans ^= i
        return ans

    # lambda arguments: expression
    # reduce(func, seq)
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


class Solution:  # TODO
    def singleNumber(self, nums: List[int]) -> int:
        b1, b2 = 0, 0  # 出现一次的位，和两次的位
        for n in nums:
            # 既不在出现一次的b1，也不在出现两次的b2里面，我们就记录下来，出现了一次，再次出现则会抵消
            b1 = (b1 ^ n) & ~b2
            # 既不在出现两次的b2里面，也不再出现一次的b1里面(不止一次了)，记录出现两次，第三次则会抵消
            b2 = (b2 ^ n) & ~b1
        return b1


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)


# 138 - Copy List with Random Pointer - MEDIUM
class Node:
    def __init__(self, x: int, next: "Node" = None, random: "Node" = None):
        self.val = int(x)
        self.next = next
        self.random = random


class Solution:
    def copyRandomList(self, head: "Node") -> "Node":
        return copy.deepcopy(head)


class Solution:
    def copyRandomList(self, head: "Node") -> "Node":
        if not head:
            return None
        dic = {}
        headCP = head
        # save node value
        while head:
            valHead = Node(head.val)
            dic[head] = valHead
            head = head.next
        head = headCP
        tmp = dic[head]
        ans = tmp
        # process random pointer
        while head:
            if head.next:
                tmp.next = dic[head.next]
            if head.random:
                tmp.random = dic[head.random]
            head = head.next
            tmp = tmp.next
        return ans


# 139 - Word Break - MEDIUM
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        f = [True] + [False] * len(s)
        mx = max(len(x) for x in wordDict)
        wd = set(wordDict)
        for i in range(len(s)):
            for j in range(i + 1, min(i + 1 + mx, len(s) + 1)):
                if f[i] and s[i:j] in wd:
                    f[j] = True
        return f[-1]

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        f = [True] + [False] * len(s)
        for i in range(1, len(s) + 1):
            for w in wordDict:
                if i >= len(w):
                    f[i] = f[i] or (f[i - len(w)] and w == s[i - len(w) : i])
        return f[-1]

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        f = [True] + [False] * len(s)
        for i in range(len(s) + 1):
            for w in wordDict:
                if i + len(w) < len(s) + 1 and s[i : i + len(w)] == w:
                    f[i + len(w)] = f[i] or f[i + len(w)]
        return f[-1]

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        f = [True] + [False] * len(s)
        for i in range(1, len(s) + 1):
            for w in wordDict:
                if i + len(w) - 1 > len(s):
                    continue
                if f[i - 1] and s[i - 1 : i - 1 + len(w)] == w:
                    f[i - 1 + len(w)] = True
        return f[-1]


# 141 - Linked List Cycle - EASY
class Solution:
    # O(n) / O(n)
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        seen = set()
        while head:
            if head in seen:
                return True
            seen.add(head)
            head = head.next
        return False

    # O(n) / O(1)
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

    # O(n) / O(1)
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        try:
            slow = head
            fast = head.next
            while slow is not fast:
                slow = slow.next
                fast = fast.next.next
            return True
        except:
            return False

    # O(n) / O(0), change the 'val' in linked list, may not allowed
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        while head:
            if head.val == "#":
                return True
            head.val = "#"
            head = head.next
        return False


# 142 - Linked List Cycle II - MEDIUM
class Solution:
    # O(n), O(n)
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        s = set()
        while head:
            if head in s:
                return head
            s.add(head)
            head = head.next
        return None

    # 1. f = 2s
    # 2. f = s + n * cycle
    # 3. from head to entrance: a + n * cycle
    # s = n * cycle, so let another point from head move 'a' step with 'slow'
    # 'slow' move 'a + 1 * cycle' to entrance, 'point' move 'a + 0 * cycle' to entrance
    # O(n), O(1)
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        try:
            fast = head.next
            slow = head
            while fast is not slow:
                fast = fast.next.next
                slow = slow.next
        except:
            return None
        # since fast starts at head.next, we need to move slow one step forward
        slow = slow.next
        while head is not slow:
            head = head.next
            slow = slow.next
        return head

    # O(n), O(1)
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                p = head
                while slow != p:
                    slow = slow.next
                    p = p.next
                return slow
        return None


# 143 - Reorder List - MEDIUM
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        pre = None
        while slow:
            """
            # pre, slow, slow.next = slow, slow.next, pre # wrong
            # pre, slow.next, slow = slow, pre, slow.next # right
            # slow, slow.next, pre = slow.next, pre, slow # wrong
            # slow.next, slow, pre = pre, slow.next, slow # right
            # slow, pre, slow.next = slow.next, slow, pre # wrong
            """
            slow.next, pre, slow = pre, slow, slow.next  # right
        while pre.next:
            head.next, head = pre, head.next
            pre.next, pre = head, pre.next
        return


# 144 - Binary Tree Preorder Traversal - EASY
class Solution:
    # recursively
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        return (
            [root.val]
            + self.preorderTraversal(root.left)
            + self.preorderTraversal(root.right)
        )

    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def preorder(root):
            if not root:
                return
            ans.append(root.val)
            preorder(root.left)
            preorder(root.right)
            return

        ans = []
        preorder(root)
        return ans

    # iteratively
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack = [root]
        ans = []
        while stack:
            n = stack.pop()
            if n:
                ans.append(n.val)
                stack.append(n.right)  # LIFO
                stack.append(n.left)
        return ans


# 145 - Binary Tree Postorder Traversal - EASY
class Solution:
    # recursively
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        return (
            self.postorderTraversal(root.left)
            + self.postorderTraversal(root.right)
            + [root.val]
        )

    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def postorder(root):
            if not root:
                return
            postorder(root.left)
            postorder(root.right)
            ans.append(root.val)
            return

        ans = []
        postorder(root)
        return ans

    # iteratively
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        stack, ans = [root], []
        while root:
            n = stack.pop()
            if n:
                ans.append(root.val)
                stack.append(root.left)  # LIFO
                stack.append(root.right)
        return ans[::-1]


# 146 - LRU Cache - MEDIUM
# Least Recently Used (LRU) cache.
class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.dic = {}
        self.seq = collections.deque()

    def get(self, key: int) -> int:
        value = self.dic.get(key, -1)
        if value != -1:
            self.seq.remove(key)  # O(n)
            self.seq.append(key)
        return value

    def put(self, key: int, value: int) -> None:
        # have the same key
        if key in self.dic:
            self.dic[key] = value
            self.seq.remove(key)  # O(n)
            self.seq.append(key)
            return
        # whether cache reach to the capacity
        if len(self.dic) == self.cap:
            delete = self.seq.popleft()
            # self.dic.pop(delete)
            del self.dic[delete]
        # insert
        self.dic[key] = value
        self.seq.append(key)
        return


# OrderedDict
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        # del self.cache[key]
        # # del is faster, pop() or popitem() used to get the return value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return


# Doubly linked list + Hashmap
class ListNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.pre: ListNode = None
        self.next: ListNode = None


class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.d = {}
        self.head = ListNode()
        self.tail = ListNode()
        # head <-> tail
        self.head.next = self.tail
        self.tail.pre = self.head

    def remove(self, node: ListNode) -> None:
        #      hashmap[key]                               hashmap[key]
        #           |                                          |
        #           V              -->                         V
        # pre <-> node <-> next         pre <-> next   ...   node
        node.pre.next = node.next
        node.next.pre = node.pre
        return

    def add(self, node: ListNode) -> None:
        #                 hashmap[key]                 hashmap[key]
        #                      |                            |
        #                      V        -->                 V
        # pre <-> tail  ...  node                pre <-> node <-> tail
        node.pre = self.tail.pre
        node.next = self.tail
        self.tail.pre.next = node
        self.tail.pre = node
        return

    def move_to_end(self, key: int) -> None:
        node = self.d[key]
        self.remove(node)
        self.add(node)
        return

    def get(self, key: int) -> int:
        if key not in self.d:
            return -1
        self.move_to_end(key)
        node = self.d[key]
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.d:
            self.d[key].value = value
            self.move_to_end(key)
        else:
            if len(self.d) == self.cap:
                self.d.pop(self.head.next.key)
                self.remove(self.head.next)
            node = ListNode(key, value)
            self.d[key] = node
            self.add(node)
        return


class Node:
    def __init__(self, key=-1, val=-1, pre=None, next=None) -> None:
        self.key = key  # 删除非法的 head.next 时, 需要删除 self.d[key], 所以需要 node 需要保存 key
        self.val = val
        self.pre: Node = pre
        self.next: Node = next


class DoubleEndedLinkedList:
    def __init__(self) -> None:
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.pre = self.head

    def remove(self, node: Node) -> None:
        node.pre.next = node.next
        node.next.pre = node.pre
        return

    def add(self, node: Node) -> None:
        node.pre = self.tail.pre
        node.next = self.tail
        self.tail.pre.next = node
        self.tail.pre = node
        return

    def moveToEnd(self, node: Node) -> None:
        self.remove(node)
        self.add(node)
        return


class LRUCache:
    def __init__(self, capacity: int):
        self.l = DoubleEndedLinkedList()
        self.d = {}
        self.c = capacity

    def get(self, key: int) -> int:
        if key not in self.d:
            return -1
        self.l.moveToEnd(self.d[key])
        return self.d[key].val

    def put(self, key: int, value: int) -> None:
        if key in self.d:
            self.d[key].val = value
            self.l.moveToEnd(self.d[key])
        else:
            if len(self.d) == self.c:
                del self.d[self.l.head.next.key]
                self.l.remove(self.l.head.next)
            new = Node(key=key, val=value)
            self.d[key] = new
            self.l.add(new)
        return


# 147 - Insertion Sort List - MEDIUM
class Solution:
    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(-1, head)
        while head and head.next:
            if head.val <= head.next.val:
                head = head.next
                continue
            pre = dummy
            while pre.next.val < head.next.val:
                pre = pre.next
            cur = head.next
            head.next = cur.next
            cur.next = pre.next
            pre.next = cur
            if cur.val > head.val:
                pre.next = head
                cur.next = head.next
                head.next = cur
        return dummy.next


# 148 - Sort List - MEDIUM
class Solution:
    # O(nlogn) / O(n)
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        arr = []
        while head:
            arr.append(head.val)
            head = head.next
        arr.sort()
        head = dummy = ListNode(-1)
        for n in arr:
            head.next = ListNode(n)
            head = head.next
        return dummy.next

    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        arr = []
        cur = head
        while cur:
            arr.append(cur.val)
            cur = cur.next
        cur = head
        for num in sorted(arr):
            cur.val = num
            cur = cur.next
        return head

    # O(nlogn) / O(logn)
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        slow = head
        fast = head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        start = slow.next
        slow.next = None  # break the list
        return self.mergeTwoLists(self.sortList(head), self.sortList(start))

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = head = ListNode(-1)
        while l1 and l2:
            if l1.val >= l2.val:
                head.next = l2
                l2 = l2.next
            else:
                head.next = l1
                l1 = l1.next
            head = head.next
        head.next = l1 or l2
        return dummy.next


# 149 - Max Points on a Line - HARD
class Solution:
    # O(n ^ 2)
    def maxPoints(self, points: List[List[int]]) -> int:
        def getSlope(i, j):
            a = points[i]
            b = points[j]
            if a[0] == b[0]:
                return float("inf")
            else:
                return (b[1] - a[1]) / (b[0] - a[0])

        ans = 1
        for i in range(len(points) - 1):
            dic = collections.defaultdict(int)
            for j in range(i + 1, len(points)):
                slope = getSlope(i, j)
                dic[slope] += 1
            ans = max(ans, max(dic.values()) + 1)
        return ans

    def maxPoints(self, points: List[List[int]]) -> int:
        ans = 1
        for i in range(len(points) - 1):
            cnt = collections.Counter()
            for j in range(i + 1, len(points)):
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                cnt[dy / dx if dx else math.inf] += 1
            ans = max(ans, max(cnt.values()) + 1)
        return ans

    # O(n ^ 2 * logm), gcd: logm
    # python3, math.gcd always return positive number
    def maxPoints(self, points: List[List[int]]) -> int:
        if len(points) < 3:
            return len(points)

        def gcd(a: int, b: int) -> int:
            while b:
                a, b = b, a % b
            return a

        ans = 0
        for i in range(len(points) - 1):
            d = collections.defaultdict(int)
            for j in range(i + 1, len(points)):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                g = gcd(dy, dx)
                d[(dy // g, dx // g)] += 1
            ans = max(ans, max(d.values()) + 1)  # plus 'i' self
        return ans


# 151 - Reverse Words in a String - MEDIUM
class Solution:
    def reverseWords(self, s: str) -> str:
        t = s.split()
        return " ".join(t[::-1])

    def reverseWords(self, s: str) -> str:
        return " ".join(reversed(s.split()))


# 152 - Maximum Product Subarray - MEDIUM
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        maxF = minF = ans = nums[0]
        for i in range(1, len(nums)):
            maxF, minF = max(maxF * nums[i], nums[i], minF * nums[i]), min(
                maxF * nums[i], nums[i], minF * nums[i]
            )
            # mx, mn = maxF, minF
            # maxF = max(mx * nums[i], nums[i], mn * nums[i])
            # minF = min(mx * nums[i], nums[i], mn * nums[i])
            ans = max(maxF, ans)
        return ans

    # TODO https://leetcode.com/problems/maximum-product-subarray/discuss/183483/JavaC%2B%2BPython-it-can-be-more-simple
    def maxProduct(self, nums: List[int]) -> int:
        revnums = nums[::-1]
        for i in range(1, len(nums)):
            nums[i] *= nums[i - 1] or 1
            revnums[i] *= revnums[i - 1] or 1
        return max(nums + revnums)

    # Kadane
    def maxProduct(self, nums: List[int]) -> int:
        prefix, suffix, ans = 0, 0, float("-inf")
        for i in range(len(nums)):
            prefix = (prefix or 1) * nums[i]
            suffix = (suffix or 1) * nums[~i]
            ans = max(ans, prefix, suffix)
        return ans


# 153 - Find Minimum in Rotated Sorted Array - MEDIUM
class Solution:
    def findMin(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        while l < r:
            mid = (l + r) // 2
            if nums[mid] > nums[r]:
                l = mid + 1
            else:
                r = mid
        return nums[l]


# 155 - Min Stack - MEDIUM
class MinStack:
    # O(1) / O(2n)
    def __init__(self):
        self.st = []
        self.mi = [math.inf]

    def push(self, val: int) -> None:
        self.st.append(val)
        self.mi.append(min(val, self.mi[-1]))
        return

    def pop(self) -> None:
        self.st.pop()
        self.mi.pop()
        return

    def top(self) -> int:
        return self.st[-1]

    def getMin(self) -> int:
        return self.mi[-1]


class MinStack:
    # O(1) / O(2n)
    def __init__(self):
        self.s = [(0, math.inf)]

    def push(self, val: int) -> None:
        self.s.append((val, min(self.s[-1][1], val)))
        return

    def pop(self) -> None:
        self.s.pop()
        return

    def top(self) -> int:
        return self.s[-1][0]

    def getMin(self) -> int:
        return self.s[-1][1]


class MinStack:
    # O(1) / O(n), save diff in stack
    def __init__(self):
        self.st = []
        self.mi = 0

    def push(self, val: int) -> None:
        if not self.st:
            self.st.append(0)
            self.mi = val
        else:
            diff = val - self.mi
            self.st.append(diff)
            if diff < 0:
                self.mi = val
        return

    def pop(self) -> None:
        diff = self.st.pop()
        if diff < 0:
            self.mi -= diff  # 更新最小值
        return

    def top(self) -> int:
        if self.st[-1] < 0:
            return self.mi  # 栈顶就是最小的
        return self.st[-1] + self.mi  # 栈顶不是最小的, 复原

    def getMin(self) -> int:
        return self.mi


# 160 - Intersection of Two Linked Lists - EASY
class Solution:
    # O(m + n) / O(m)
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        s = set()
        while headA:
            s.add(headA)
            headA = headA.next
        while headB:
            if headB in s:
                return headB
            headB = headB.next
        return None

    # O(m + n) / O(1)
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        a = headA
        b = headB
        aex = False  # flag: whether a has been exchanged
        bex = False
        while a and b:
            if a == b:
                return a
            else:
                a = a.next
                b = b.next
            if not a and not aex:
                a = headB
                aex = True
            if not b and not bex:
                b = headA
                bex = True
        return None

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        a = headA
        b = headB
        while a != b:
            a = a.next if a else headB
            b = b.next if b else headA
        return a


# 162 - Find Peak Element - MEDIUM
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (right + left) // 2
            if nums[mid] > nums[mid + 1]:
                right = mid
            else:
                left = mid + 1
        return left

    # climbing to the greater side
    def findPeakElement(self, nums: List[int]) -> int:
        idx = random.randint(0, len(nums) - 1)

        # helper function: help to handle boundary situations
        def getValue(i: int) -> int:
            if i == -1 or i == len(nums):
                return float("-inf")
            return nums[i]

        while not (
            getValue(idx - 1) < getValue(idx) and getValue(idx) > getValue(idx + 1)
        ):
            if getValue(idx) < getValue(idx + 1):
                idx += 1
            else:
                idx -= 1
        return idx


# 163 - Missing Ranges - EASY - PREMIUM
class Solution:
    def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[str]:
        def helper(lower: int, upper: int) -> str:
            if lower == upper:
                return str(lower)
            return f"{lower}->{upper}"

        ans = []
        if not nums:
            return helper(lower, upper)
        if lower < nums[0]:
            ans.append(lower, nums[0] - 1)
        for i in range(len(nums) - 1):
            if nums[i] + 1 < nums[i + 1]:
                ans.append(helper(nums[i] + 1, nums[i + 1] - 1))
        if nums[-1] < upper:
            ans.append(nums[-1] + 1, upper)
        return ans


# 165 - Compare Version Numbers - MEDIUM
class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        v1 = version1.split(".")
        v2 = version2.split(".")
        i = j = 0
        while i < len(v1) and j < len(v2):
            if int(v1[i]) > int(v2[j]):
                return 1
            elif int(v1[i]) < int(v2[j]):
                return -1
            else:
                i += 1
                j += 1
        # whether the rest part of longer string are all '0'
        if i < len(v1):
            return int(any(int(s) > 0 for s in v1[i:]))
        if j < len(v2):
            return -int(any(int(s) > 0 for s in v2[j:]))
        return 0

    def compareVersion(self, version1: str, version2: str) -> int:
        versions1 = [int(v) for v in version1.split(".")]
        versions2 = [int(v) for v in version2.split(".")]
        for i in range(max(len(versions1), len(versions2))):
            v1 = versions1[i] if i < len(versions1) else 0
            v2 = versions2[i] if i < len(versions2) else 0
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1
        return 0

    # compare number between each '.'
    def compareVersion(self, v1: str, v2: str) -> int:
        m, n = len(v1), len(v2)
        i = j = 0
        while i < m or j < n:
            a = b = 0
            while i < m and v1[i] != ".":
                a = 10 * a + int(v1[i])
                i += 1
            while j < n and v2[j] != ".":
                b = 10 * b + int(v2[j])
                j += 1
            if a > b:
                return 1
            elif a < b:
                return -1
            i += 1
            j += 1
        return 0


# 167 - Two Sum II - Input Array Is Sorted - EASY
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for i in range(len(nums)):
            if nums[i] in dic:
                return [dic[nums[i]] + 1, i + 1]
            else:
                dic[target - nums[i]] = i
        return [-1, -1]

    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1
        while left < right:
            summ = numbers[left] + numbers[right]
            if summ < target:
                left += 1
            elif summ > target:
                right -= 1
            else:
                return [left + 1, right + 1]
        return [-1, -1]


# 169 - Majority Element - EASY
class Solution:
    # O(n) / O(1)
    def majorityElement(self, nums: List[int]) -> int:
        ans = cnt = 0
        for n in nums:
            if cnt == 0:
                ans = n
            if ans == n:
                cnt += 1
            else:
                cnt -= 1
        return ans

    # O(n) / O(n)
    def majorityElement(self, nums: List[int]) -> int:
        cnt = collections.Counter(nums)
        return max(cnt.keys(), key=cnt.get)

    # O(nlogn) / O(logn)
    def majorityElement(self, nums: List[int]) -> int:
        nums.sort()
        return nums[len(nums) // 2]


# 171 - Excel Sheet Column Number - EASY
class Solution:
    def titleToNumber(self, columnTitle: str) -> int:
        ans = 0
        for ch in columnTitle:
            ans = ans * 26 + ord(ch) - ord("A") + 1
        return ans


# 172 - Factorial Trailing Zeroes - MEDIUM
class Solution:
    # O(n) / O(1)
    def trailingZeroes(self, n: int) -> int:
        ans = 0
        for i in range(5, n + 1, 5):
            while i % 5 == 0:
                i //= 5
                ans += 1
        return ans

    # O(logn) / O(1)
    def trailingZeroes(self, n: int) -> int:
        if n // 5 == 0:
            return 0
        return n // 5 + self.trailingZeroes(n // 5)

    def trailingZeroes(self, n: int) -> int:
        ans = 0
        while n:
            n //= 5
            ans += n
        return ans

    def trailingZeroes(self, n: int) -> int:
        ans = 0
        p = 5
        while p <= n:
            ans += n // p
            p *= 5
        return ans

    # O(1) / O(1)
    def trailingZeroes(self, n: int) -> int:
        return n // 5 + n // 25 + n // 125 + n // 625 + n // 3125


# 173 - Binary Search Tree Iterator - MEDIUM
# save all node.val by inorder traversal
class BSTIterator:
    # O(n) / O(n)
    def __init__(self, root: Optional[TreeNode]):
        self.dq = collections.deque()
        self.inorder(root)

    def next(self) -> int:
        return self.dq.popleft()

    def hasNext(self) -> bool:
        return len(self.dq) > 0

    def inorder(self, root: TreeNode) -> None:
        if not root:
            return
        self.inorder(root.left)
        self.dq.append(root.val)
        self.inorder(root.right)
        return


# iterate. save all left nodes while pop each node
class BSTIterator:
    # O(n) / O(h), where h is the height of the tree
    def __init__(self, root: Optional[TreeNode]):
        self.st = []
        while root:
            self.st.append(root)
            root = root.left

    def next(self) -> int:
        cur = self.st.pop()
        node = cur.right
        while node:
            self.st.append(node)
            node = node.left
        return cur.val

    def hasNext(self) -> bool:
        return len(self.st) > 0


# Abstract the putting into stack operation into a function
class BSTIterator:
    def __init__(self, root: TreeNode):
        self.stack = []
        self.pushAllLeftNodes(root)

    def next(self):
        cur = self.stack.pop()
        node = cur.right
        self.pushAllLeftNodes(node)
        return cur.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0

    def pushAllLeftNodes(self, root: TreeNode) -> None:
        while root:
            self.stack.append(root)
            root = root.left
        return


# 179 - Largest Number - MEDIUM
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        # cmp: Callable[[str, str], Literal[1, -1]] = (lambda x, y: 1 if y + x > x + y else -1)
        cmp: Callable[[str, str], int] = lambda x, y: int(y + x) - int(x + y)
        ans = "".join(sorted(map(str, nums), key=functools.cmp_to_key(cmp)))
        return "0" if ans[0] == "0" else ans


# 188 - Best Time to Buy and Sell Stock IV - HARD
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        buy = [-math.inf] * (k + 1)
        sell = [-math.inf] * (k + 1)
        buy[0] = 0
        sell[0] = 0
        for p in prices:
            for i in range(k, 0, -1):
                buy[i] = bu
        return


# 189 - Rotate Array - MEDIUM
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        k %= len(nums)
        """
        Input: [1], 0
        Wrong: nums[:k], nums[k:] = nums[-k:], nums[:-k]
        Assignment Visualization: [], [1] = [1], []
        Conclusion: assignment from left to right
        """
        nums[k:], nums[:k] = nums[:-k], nums[-k:]
        # nums[:] = nums[-k:] + nums[:-k]
        return


# 190 - Reverse Bits - EASY
class Solution:
    def reverseBits(self, n: int) -> int:
        ans = 0
        for _ in range(31):
            if n & 1:
                ans += 1
            n >>= 1
            ans <<= 1
        if n & 1:
            ans += 1
        return ans

    def reverseBits(self, n: int):
        ans = 0
        for _ in range(32):
            ans = (ans << 1) + (n & 1)
            # ans = (ans << 1) ^ (n & 1)
            # ans = (ans << 1) | (n & 1)
            n >>= 1
        return ans

    def reverseBits(self, n: int):
        oribin = "{0:032b}".format(n)
        reversebin = oribin[::-1]
        return int(reversebin, 2)


# 191 - Number of 1 Bits - EASY
class Solution:
    def hammingWeight(self, n: int) -> int:
        ans = 0
        while n:
            if n & 1:
                ans += 1
            n >>= 1
        return ans

    def hammingWeight(self, n: int) -> int:
        ans = 0
        while n:
            n &= n - 1
            ans += 1
        return ans

    def hammingWeight(self, n: int) -> int:
        return bin(n).count("1")


# 198 - House Robber - MEDIUM
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) < 3:
            return max(nums)
        dp = [0] * len(nums)
        dp[0], dp[1] = nums[0], max(nums[:2])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])
        return dp[-1]

    def rob(self, nums: List[int]) -> int:
        pre, cur = 0, 0
        for i in range(len(nums)):
            pre, cur = cur, max(pre + nums[i], cur)
        return cur


# 199 - Binary Tree Right Side View - MEDIUM
class Solution:
    # dfs
    def rightSideView(self, root: TreeNode) -> List[int]:
        def dfs(root, l):
            if not root:
                return
            if len(ans) == l:
                ans.append(root.val)
            dfs(root.right, l + 1)
            dfs(root.left, l + 1)
            return

        ans = []
        dfs(root, 0)
        return ans

    # bfs
    def rightSideView(self, root: TreeNode) -> List[int]:
        dq, ans = collections.deque(), []
        if root:
            dq.append(root)
        while dq:
            ans.append(dq[-1].val)
            for _ in range(len(dq)):
                node = dq.popleft()
                if node.left:
                    dq.append(node.left)
                if node.right:
                    dq.append(node.right)
        return ans
