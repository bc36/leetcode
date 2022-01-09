import collections, math, copy, bisect, heapq, datetime
from typing import List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 1005 - Maximize Sum Of Array After K Negations - EASY
class Solution:
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        neg = [x for x in nums if x < 0]
        pos = [x for x in nums if x > 0]
        if len(neg) >= k:
            neg.sort()
            return sum(pos) - sum(neg[:k]) + sum(neg[k:])
        k -= len(neg)
        if not k & 1:
            return sum(pos) - sum(neg)
        tmp = min([abs(x) for x in nums])
        return sum(pos) - sum(neg) - 2 * tmp


# 1009 - Complement of Base 10 Integer - EASY
class Solution:
    def bitwiseComplement(self, n: int) -> int:
        ans = ''
        while n:
            ans += '0' if n & 1 else '1'
            n >>= 1
        return int(ans[::-1], 2) if ans else 1

    def bitwiseComplement(self, n: int) -> int:
        x = 1
        while n > x:
            x = x * 2 + 1
        return x ^ n  # XOR

    def bitwiseComplement(self, n: int) -> int:
        if not n: return 1
        mask = n
        mask |= mask >> 1
        mask |= mask >> 2
        mask |= mask >> 4
        mask |= mask >> 8
        mask |= mask >> 16
        return n ^ mask


# 1010 - Pairs of Songs With Total Durations Divisible by 60 - MEDIUM
class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        c, ret = [0] * 60, 0
        for t in time:
            ret += c[-t % 60]
            c[t % 60] += 1
        return ret

    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        dic, ans = collections.defaultdict(int), 0
        for t in time:
            mod = t % 60
            dic[mod] += 1
            if mod == 30 or mod == 0:
                ans += dic[mod] - 1
            elif 60 - mod in dic:
                ans += dic[60 - mod]
        return ans

    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        ans, cnt = 0, collections.Counter()
        for t in time:
            theOther = -t % 60
            ans += cnt[theOther]
            cnt[t % 60] += 1
        return ans


# 1015 - Smallest Integer Divisible by K - MEDIUM
class Solution:
    # time consuming, just because Python supports arbitrarily large numbers
    def smallestRepunitDivByK(self, k: int) -> int:
        if not k % 2 or not k % 5: return -1
        ans, l = 1, 1
        while True:
            if ans % k == 0: return l
            l += 1
            ans = 10 * ans + 1

    # At every iteration, n = kq + r for some quotient q and remainder r.
    # Therefore, 10*n + 1 = 10(kq + r) + 1 = 10kq + 10r + 1.
    # 10kq is divisible by k, so for 10*n + 1 to be divisible by k, it all depends on if 10r + 1 is divisible by k.
    # Therefore, we only have to keep track of r!
    def smallestRepunitDivByK(self, k: int) -> int:
        if not k % 2 or not k % 5: return -1
        r = length = 1
        while True:
            r = r % k
            if not r: return length
            length += 1
            r = 10 * r + 1

    # k possible remainders from 0 to k-1
    def smallestRepunitDivByK(self, k: int) -> int:
        if k % 2 == 0 or k % 5 == 0: return -1
        n = 1
        for i in range(k):
            r = n % k
            if r == 0: return i + 1
            n = r * 10 + 1


# 1026 - Maximum Difference Between Node and Ancestor - MEDIUM
class Solution:
    # down to top, calculate the minimum and maximum values then pass them to the root
    def maxAncestorDiff(self, root: TreeNode) -> int:
        self.ans = 0

        def dfs(root):
            if not root:
                return float('inf'), -float('inf')
            lmin, lmax = dfs(root.left)
            rmin, rmax = dfs(root.right)
            rootmin = min(root.val, lmin, rmin)
            rootmax = max(root.val, lmax, rmax)
            self.ans = max(self.ans, abs(root.val - rootmin),
                           abs(root.val - rootmax))
            return rootmin, rootmax

        dfs(root)
        return self.ans

    # top to down, pass the minimum and maximum values to the children
    def maxAncestorDiff(self, root: TreeNode) -> int:
        def dfs(root: TreeNode, low: int, high: int) -> int:
            if root is None:
                return high - low
            low = min(root.val, low)
            high = max(root.val, high)
            return max(dfs(root.left, low, high), dfs(root.right, low, high))

        return dfs(root, root.val, root.val)

    # top to down, pass the minimum and maximum values to the children
    def maxAncestorDiff(self,
                        root: TreeNode,
                        mn: int = 100000,
                        mx: int = 0) -> int:
        return max(self.maxAncestorDiff(root.left, min(mn, root.val), max(mx, root.val)), \
            self.maxAncestorDiff(root.right, min(mn, root.val), max(mx, root.val))) \
            if root else mx - mn

    def maxAncestorDiff(self, root: TreeNode) -> int:
        if not root: return 0
        stack = [(root, root.val, root.val)]  #stack, parent, child
        res = 0
        while stack:
            node, parent, child = stack.pop()
            res = max(res, abs(parent - child))
            if node.left:
                stack.append(
                    (node.left, max(parent,
                                    node.left.val), min(child, node.left.val)))
            if node.right:
                stack.append((node.right, max(parent, node.right.val),
                              min(child, node.right.val)))
        return res


# 1034 - Coloring A Border - MEDIUM
class Solution:
    # bfs
    def colorBorder(self, grid: List[List[int]], row: int, col: int,
                    color: int) -> List[List[int]]:
        position, borders, originalColor = [(row, col)], [], grid[row][col]
        visited = [[False] * len(grid[0]) for _ in range(len(grid))]
        visited[row][col] = True
        while position:
            x, y = position.pop()
            isBorder = False
            for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if not (0 <= nx < len(grid) and 0 <= ny < len(grid[0])
                        and grid[nx][ny] == originalColor):
                    isBorder = True
                elif not visited[nx][ny]:
                    visited[nx][ny] = True
                    position.append((nx, ny))
            if isBorder:
                borders.append((x, y))
        for i, j in borders:
            grid[i][j] = color
        return grid

    # bfs
    def colorBorder(self, grid: List[List[int]], row: int, col: int,
                    color: int) -> List[List[int]]:
        m, n = len(grid), len(grid[0])
        bfs, component, border = [[row, col]], set([(row, col)]), set()
        while bfs:
            r, c = bfs.pop()
            for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                x, y = r + i, c + j
                if 0 <= x < m and 0 <= y < n and grid[x][y] == grid[r][c]:
                    if (x, y) not in component:
                        bfs.append([x, y])
                        component.add((x, y))
                else:
                    border.add((r, c))
        for x, y in border:
            grid[x][y] = color
        return grid

    # dfs
    def colorBorder(self, grid: List[List[int]], row: int, col: int,
                    color: int) -> List[List[int]]:
        visited, m, n = set(), len(grid), len(grid[0])

        def dfs(x: int, y: int) -> bool:
            if (x, y) in visited:
                return True
            if not (0 <= x < m and 0 <= y < n
                    and grid[x][y] == grid[row][col]):
                return False
            visited.add((x, y))
            if dfs(x + 1, y) + dfs(x - 1, y) + dfs(x, y + 1) + dfs(x,
                                                                   y - 1) < 4:
                grid[x][y] = color
            return True

        dfs(row, col)
        return grid


# 1041 - Robot Bounded In Circle - MEDIUM
class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        x, y, dx, dy = 0, 0, 0, 1
        for i in instructions:
            if i == 'R': dx, dy = dy, -dx
            if i == 'L': dx, dy = -dy, dx
            if i == 'G': x, y = x + dx, y + dy
        return (x, y) == (0, 0) or not (dx == 0 and dy > 0)


# 1047 - Remove All Adjacent Duplicates In String - EASY
class Solution:
    # stack
    def removeDuplicates(self, s: str) -> str:
        stack = [s[0]]
        for i in range(1, len(s)):
            if stack and s[i] == stack[-1]:
                stack.pop()
            else:
                stack.append(s[i])
        return "".join(stack)

    # two pointers
    def removeDuplicates(self, s: str) -> str:
        # pointers: 'ch' and 'end',
        # change 'ls' in-place.
        ls, end = list(s), -1
        for ch in ls:
            if end >= 0 and ls[end] == ch:
                end -= 1
            else:
                end += 1
                ls[end] = ch
        return "".join(ls[:end + 1])


# 1078 - Occurrences After Bigram - EASY
class Solution:
    def findOcurrences(self, text: str, first: str, second: str) -> List[str]:
        text, ans = text.split(), []
        for i in range(2, len(text)):
            if text[i - 2] == first and text[i - 1] == second:
                ans.append(text[i])
        return ans


# 1185 - Day of the Week - EASY
class Solution:
    def dayOfTheWeek(self, day: int, month: int, year: int) -> str:
        return datetime.date(year, month, day).strftime('%A')

    # Zelle formula
    days = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ]

    def dayOfTheWeek(self, d, m, y):
        if m < 3:
            m += 12
            y -= 1
        c, y = y // 100, y % 100
        w = (c // 4 - 2 * c + y + y // 4 + 13 * (m + 1) // 5 + d - 1) % 7
        return self.days[w]


# 1091 - Shortest Path in Binary Matrix - MEDIUM
class Solution:
    # TLE, not suitable for 'dfs', be careful with 'visited2'(cycle)
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if grid[-1][-1] == 1 or grid[0][0] == 1: return -1

        def dfs(x, y, step, visited):
            n = len(grid)
            if x == y == n - 1:
                self.ans = min(self.ans, step)
                return
            for i, j in ((x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
                         (x, y - 1), (x, y + 1), (x + 1, y - 1), (x + 1, y),
                         (x + 1, y + 1)):
                if 0 <= i < n and 0 <= j < n and grid[i][j] == 0 and (
                        i, j) not in visited:
                    visited.add((i, j))
                    visited2 = copy.deepcopy(visited)
                    dfs(i, j, step + 1, visited2)
            return

        self.ans = math.inf
        dfs(0, 0, 1, set())
        return self.ans if self.ans != math.inf else -1

    # bfs, O(n^2) + O(n)
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if grid[-1][-1] == 1 or grid[0][0] == 1: return -1
        dq, n = collections.deque([(0, 0, 1)]), len(grid)
        # no need to use 'visited', space complexity from O(n^2) to O(n)
        grid[0][0] = 1
        while dq:
            x, y, step = dq.popleft()
            if x == y == n - 1: return step
            for i, j in ((x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
                         (x, y - 1), (x, y + 1), (x + 1, y - 1), (x + 1, y),
                         (x + 1, y + 1)):
                if 0 <= i < n and 0 <= j < n and grid[i][j] == 0:
                    grid[i][j] = 1
                    dq.append((i, j, step + 1))
        return -1