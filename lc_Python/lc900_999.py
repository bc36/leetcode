import collections, heapq, functools
from typing import List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 911 - Online Election - MEDIUM


# 913 - Cat and Mouse — HARD
class Solution:
    # 思路：
    # 将该问题视为状态方程，状态公式[mouse,cat,turn]代表老鼠的位置，猫的位置，下一步轮到谁走
    # 猫胜利的状态为[i,i,1]或[i,i,2]（i!=0），1代表老鼠走，2代表猫走
    # 老鼠胜利的状态为[0,i,1]或[0,i,2]
    # 用0代表未知状态，1代表老鼠赢，2代表猫赢
    # 由最终的胜利状态，回推
    # 假如当前父节点轮次是1（父节点轮次是2同样的道理）
    # 父节点=1 if 子节点是1
    # 或者
    # 父节点=2 if 所有子节点是2
    def catMouseGame(self, graph: List[List[int]]) -> int:
        n = len(graph)
        degrees = [[[0] * 2 for _ in range(n)] for _ in range(n)]  #(m,c,t)
        for i in range(n):
            for j in range(n):
                if j == 0:
                    continue
                degrees[i][j][0] += len(graph[i])
                degrees[i][j][1] += len(graph[j]) - (0 in graph[j])

        dp = [[[0] * 2 for _ in range(n)] for _ in range(n)]  #(m,c,t)
        queue = collections.deque()
        for i in range(1, n):
            states = [(i, i, 0), (i, i, 1), (0, i, 0), (0, i, 1)]
            results = [2, 2, 1, 1]
            for (m, c, t), rv in zip(states, results):
                dp[m][c][t] = rv
            queue.extend(states)

        while queue:
            m, c, t = queue.popleft()
            rv = dp[m][c][t]
            if t == 0:  # mouse
                for pre in graph[c]:
                    if pre == 0 or dp[m][pre][1] != 0:
                        continue
                    if rv == 2:
                        dp[m][pre][1] = 2
                        queue.append((m, pre, 1))
                    else:
                        degrees[m][pre][1] -= 1
                        if degrees[m][pre][1] == 0:
                            dp[m][pre][1] = 1
                            queue.append((m, pre, 1))
            else:
                for pre in graph[m]:
                    if dp[pre][c][0] != 0:
                        continue
                    if rv == 1:
                        dp[pre][c][0] = 1
                        queue.append((pre, c, 0))
                    else:
                        degrees[pre][c][0] -= 1
                        if degrees[pre][c][0] == 0:
                            dp[pre][c][0] = 2
                            queue.append((pre, c, 0))

        return dp[1][2][0]

    def catMouseGame(self, graph: List[List[int]]) -> int:
        n = len(graph)
        # search(step,cat,mouse) 表示步数=step，猫到达位置cat，鼠到达位置mouse的情况下最终的胜负情况
        @functools.lru_cache(None)
        def search(mouse, cat, step):
            # mouse到达洞最多需要n步(初始step=1) 说明mouse走n步还没达洞口 且cat也没抓住mouse
            if step == 2 * (n):
                return 0
            # cat抓住mouse
            if cat == mouse:
                return 2
            # mouse入洞
            if mouse == 0:
                return 1
            # 奇数步：mouse走
            if step % 2 == 0:
                # 对mouse最优的策略: 先看是否能mouse赢 再看是否能平 如果都不行则cat赢
                drawFlag = False
                for nei in graph[mouse]:
                    ans = search(nei, cat, step + 1)
                    if ans == 1:
                        return 1
                    elif ans == 0:
                        drawFlag = True
                if drawFlag:
                    return 0
                return 2
            else:
                # 对cat最优的策略: 先看是否能cat赢 再看是否能平 如果都不行则mouse赢
                drawFlag = False
                for nei in graph[cat]:
                    if nei == 0:
                        continue
                    ans = search(mouse, nei, step + 1)
                    if ans == 2:
                        return 2
                    elif ans == 0:
                        drawFlag = True
                if drawFlag:
                    return 0
                return 1

        return search(1, 2, 0)


# 918 - Maximum Sum Circular Subarray - MEDIUM
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        dpx = dpn = 0  # mx = mi = 0 -> wrong
        mx = mi = nums[0]  # help solve for all elements being negative
        for i in range(len(nums)):
            dpx = nums[i] + max(dpx, 0)
            mx = max(mx, dpx)
            dpn = nums[i] + min(dpn, 0)
            mi = min(mi, dpn)
        return max(sum(nums) - mi, mx) if mx > 0 else mx

    # reducing the number of times 'max()' and 'min()' are used will reduce the runtime
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        dpx = dpn = 0
        mx = mi = nums[0]
        for n in nums:
            dpx = n + dpx if dpx > 0 else n
            if dpx > mx: mx = dpx
            dpn = n + dpn if dpn < 0 else n
            if dpn < mi: mi = dpn
        return max(sum(nums) - mi, mx) if mx > 0 else mx


# 921 - Minimum Add to Make Parentheses Valid - MEDIUM
class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        while "()" in s:
            s = s.replace("()", "")
        return len(s)

    # stack
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


# 935 - Knight Dialer - MEDIUM
# 0         -> 4 6
# 1 3 7 9   -> 2 8 / 4 6
# 2 8       -> 1 3 7 9
# 4 6       -> 0 / 1 3 7 9
class Solution:
    def knightDialer(self, n: int) -> int:
        if n == 1:
            return 10
        a, b, c, d, mod = 1, 4, 2, 2, 10**9 + 7
        for _ in range(n - 1):
            a, b, c, d = d, (2 * c + 2 * d) % mod, b, (2 * a + b) % mod
        return (a + b + c + d) % mod

    def knightDialer(self, n: int) -> int:
        x1 = x2 = x3 = x4 = x5 = x6 = x7 = x8 = x9 = x0 = 1
        for _ in range(n - 1):
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x0 = \
                x6 + x8, \
                x7 + x9, \
                x4 + x8, \
                x3 + x9 + x0, \
                0, \
                x1 + x7 + x0, \
                x2 + x6, \
                x1 + x3, \
                x2 + x4, \
                x4 + x6
        return (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x0) % (10**9 + 7)


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
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        if not root:
            return 0
        val = root.val if root.val >= low and root.val <= high else 0
        return val + self.rangeSumBST(root.left, low, high) + self.rangeSumBST(
            root.right, low, high)

    # since its a 'binary search tree' which means that left.val < root.val < right.val
    # so we can speed up by jump some unqualified node (the value greater than high or smalller than low)
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

    def isAlienSorted(self, words: List[str], order: str) -> bool:
        m = {c: i for i, c in enumerate(order)}
        words = [[m[c] for c in w] for w in words]
        return all(w1 <= w2 for w1, w2 in zip(words, words[1:]))

    def isAlienSorted(self, words: List[str], order: str) -> bool:
        return words == sorted(words, key=lambda w: map(order.index, w))

    # compare each character in word[i] and word[i+1]
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        order_map = {val: index for index, val in enumerate(order)}
        # check the next word letter one by one
        for i in range(len(words) - 1):
            for j in range(len(words[i])):
                # find a mismatch letter between words[i] and words[i + 1],
                if j >= len(words[i + 1]):  # ("apple", "app")
                    return False
                if words[i][j] != words[i + 1][j]:
                    if order_map[words[i][j]] > order_map[words[i + 1][j]]:
                        return False
                    break
        return True


# 973 - K Closest Points to Origin - MEDIUM
class Solution:
    # Pay attention that if the points are at the same distance,
    # different coordinates should be returned.
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        # [info[0]: square, info[1]: position index]
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
        ans, i = [], 0
        while len(ans) < k:
            if distance[order[i]]:
                ans.append(points[distance[order[i]].pop()])
            else:
                i += 1
        return ans

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        points.sort(key=lambda x: (x[0]**2 + x[1]**2))
        return points[:k]

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap = []
        for (x, y) in points:
            dist = -(x * x + y * y)
            if len(heap) == k:
                heapq.heappushpop(heap, (dist, x, y))
            else:
                heapq.heappush(heap, (dist, x, y))
        return [(x, y) for (_, x, y) in heap]

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        q = [(-x**2 - y**2, i) for i, (x, y) in enumerate(points[:k])]
        heapq.heapify(q)
        for i in range(k, len(points)):
            x, y = points[i]
            dist = -x**2 - y**2
            heapq.heappushpop(q, (dist, i))
        ans = [points[idx] for (_, idx) in q]
        return ans


# 977 - Squares of a Sorted Array - EASY
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        return sorted([num**2 for num in nums])

    def sortedSquares(self, nums: List[int]) -> List[int]:
        left, right = 0, len(nums) - 1
        ans = [0] * len(nums)
        while left <= right:
            al, ar = abs(nums[left]), abs(nums[right])
            if al > ar:
                ans[right - left] = al**2
                left += 1
            else:
                ans[right - left] = ar**2
                right -= 1
        return ans


# 994 - Rotting Oranges - MEDIUM
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        starts, m, n, fresh = [], len(grid), len(grid[0]), 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    starts.append((i, j))
                elif grid[i][j] == 1:
                    fresh += 1
        ans, dq = 0, collections.deque(starts)
        while dq:
            for _ in range(len(dq)):
                x, y = dq.popleft()
                for i, j in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                    if 0 <= i < m and 0 <= j < n and grid[i][j] == 1:
                        grid[i][j] = 2
                        fresh -= 1
                        dq.append((i, j))
            if not dq:
                break
            ans += 1
        return ans if fresh == 0 else -1


# 997 - Find the Town Judge - EASY
class Solution:
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        in_d = [0] * n
        out_d = [0] * n
        for t in trust:
            out_d[t[0] - 1] += 1
            in_d[t[1] - 1] += 1
        for i in range(n):
            if in_d[i] == n - 1 and out_d[i] == 0:
                return i + 1
        return -1