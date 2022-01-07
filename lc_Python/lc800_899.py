import itertools
from typing import List, Optional
import collections, math


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 807 - Max Increase to Keep City Skyline - MEDIUM
class Solution:
    def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
        row_maxes = [max(row) for row in grid]
        col_maxes = [max(col) for col in zip(*grid)]

        return sum(
            min(row_maxes[r], col_maxes[c]) - val for r, row in enumerate(grid)
            for c, val in enumerate(row))

    def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
        rows, cols = list(map(max, grid)), list(map(max, zip(*grid)))
        return sum(min(i, j) for i in rows for j in cols) - sum(map(sum, grid))


# 825 - Friends Of Appropriate Ages - MEDIUM
class Solution:
    def numFriendRequests(self, ages: List[int]) -> int:
        def request(a: int, b: int) -> bool:
            return not (b <= 0.5 * a + 7 or b > a or b > 100 and a < 100)

        c = collections.Counter(ages)
        return sum(
            request(a, b) * c[a] * (c[b] - (a == b)) for a in c for b in c)

    def numFriendRequests(self, ages: List[int]) -> int:
        cnt = collections.Counter(ages)
        ans = 0
        for a in cnt:
            for b in cnt:
                if not (b <= 0.5 * a + 7 or b > a or b > 100 and a < 100):
                    if a != b:
                        ans += cnt[a] * cnt[b]
                    else:
                        ans += cnt[a] * (cnt[a] - 1)
        return ans


# 827 - Making A Large Island - HARD
class Solution:
    # STEP 1: Explore every island using DFS, count its area
    #         give it an island index and save the result to a {index: area} map.
    # STEP 2: Loop every cell == 0,
    #         check its connected islands and calculate total islands area.
    def largestIsland(self, grid: List[List[int]]) -> int:
        N = len(grid)

        # move(int x, int y), return all possible next position in 4 directions.
        def move(x: int, y: int):
            for i, j in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= x + i < N and 0 <= y + j < N:
                    yield x + i, y + j

        # Change the value of grid[x][y] to its index so act as an area
        def dfs(x: int, y: int, index: int) -> int:
            ret = 0
            grid[x][y] = index
            for i, j in move(x, y):
                if grid[i][j] == 1:
                    ret += dfs(i, j, index)
            return ret + 1

        # Since the grid has elements 0 or 1.
        # The island index is initialized with 2
        index = 2
        areas = {0: 0}
        # DFS every island and give it an index of island
        for x in range(N):
            for y in range(N):
                if grid[x][y] == 1:
                    areas[index] = dfs(x, y, index)
                    index += 1
        # Traverse every 0 cell and count biggest island it can conntect
        # The 'possible' connected island index is stored in a set to remove duplicate index.
        ret = max(areas.values())
        for x in range(N):
            for y in range(N):
                if grid[x][y] == 0:
                    possible = set(grid[i][j] for i, j in move(x, y))
                    # '+1' means grid[x][y] itself
                    ret = max(ret, sum(areas[index] for index in possible) + 1)
        return ret


# 844 - Backspace String Compare - EASY
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        a, b = [], []
        for ch in s:
            if ch == '#':
                if a: a.pop()
            else: a.append(ch)
        for ch in t:
            if ch == '#':
                if b: b.pop()
            else: b.append(ch)
        return a == b

    # O(n) + O(1): reversed, save '#'
    def backspaceCompare(self, s: str, t: str) -> bool:
        def build(s: str) -> str:
            skip, a = 0, ''
            for ch in reversed(s):
                if ch == '#':
                    skip += 1
                elif skip:
                    skip -= 1
                else:
                    a += ch
            return a

        return build(s) == build(t)


# 846 - Hand of Straights - MEDIUM
class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        if len(hand) % groupSize > 0:
            return False
        cnt = collections.Counter(hand)
        for start in sorted(hand):
            if cnt[start] == 0:
                continue
            for num in range(start, start + groupSize):
                if cnt[num] == 0:
                    return False
                cnt[num] -= 1
        return True


# 851 - Loud and Rich - MEDIUM
class Solution:
    def loudAndRich(self, richer: List[List[int]],
                    quiet: List[int]) -> List[int]:
        n = len(quiet)
        g, ans = [[] for _ in range(n)], [-1] * n
        for r in richer:
            g[r[1]].append(r[0])

        def dfs(x: int):
            if ans[x] != -1:
                return
            ans[x] = x
            for y in g[x]:
                dfs(y)
                if quiet[ans[y]] < quiet[ans[x]]:
                    ans[x] = ans[y]

        for i in range(n):
            dfs(i)
        return ans

    def loudAndRich(self, richer: List[List[int]],
                    quiet: List[int]) -> List[int]:
        n = len(quiet)
        g = [[] for _ in range(n)]
        inDeg = [0] * n
        for r in richer:
            g[r[0]].append(r[1])
            inDeg[r[1]] += 1

        ans = list(range(n))
        q = collections.deque(i for i, deg in enumerate(inDeg) if deg == 0)
        while q:
            x = q.popleft()
            for y in g[x]:
                if quiet[ans[x]] < quiet[ans[y]]:
                    ans[y] = ans[x]
                inDeg[y] -= 1
                if inDeg[y] == 0:
                    q.append(y)
        return ans


# 859 - Buddy Strings - EASY
class Solution:
    def buddyStrings(self, s: str, goal: str) -> bool:
        if len(s) != len(goal):
            return False
        if s == goal:
            if len(set(s)) < len(s):
                return True
            else:
                return False
        diff = [(a, b) for a, b in zip(s, goal) if a != b]
        return len(diff) == 2 and diff[0] == diff[1][::-1]


# 863 - All Nodes Distance K in Binary Tree - MEDIUM
class Solution:
    # find parent of each node, then dfs
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        def findParent(root: TreeNode):
            if root.left:
                dic[root.left.val] = root
                findParent(root.left)
            if root.right:
                dic[root.right.val] = root
                findParent(root.right)
            return

        def dfs(root: TreeNode, visited: TreeNode, k: int):
            if k == 0:
                ans.append(root.val)
                return
            if root.left and root.left != visited:
                dfs(root.left, root, k - 1)
            if root.right and root.right != visited:
                dfs(root.right, root, k - 1)
            if root.val in dic and dic[root.val] != visited:
                dfs(dic[root.val], root, k - 1)
            return

        ans, dic = [], {}  # k:node.value v:node.parent
        findParent(root)
        dfs(target, None, k)
        return ans

    # bfs
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        def connect(parent: TreeNode, child: TreeNode):
            if parent and child:
                conn[parent.val].append(child.val)
                conn[child.val].append(parent.val)
            if child.left:
                connect(child, child.left)
            if child.right:
                connect(child, child.right)

        conn = collections.defaultdict(list)  # undirected graph
        connect(None, root)
        bfs = [target.val]
        seen = set(bfs)
        for _ in range(k):
            new_level = []
            for node_val in bfs:
                for connected_node_val in conn[node_val]:
                    if connected_node_val not in seen:
                        new_level.append(connected_node_val)
            bfs = new_level
            # seen = set(bfs).union(seen) # '.intersection()' <=> '&'
            seen |= set(bfs)
        return bfs

    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        def connected(node: TreeNode):
            if node.left:
                adj[node].append(node.left)
                adj[node.left].append(node)
                connected(node.left)
            if node.right:
                adj[node].append(node.right)
                adj[node.right].append(node)
                connected(node.right)

        def dfs(node: TreeNode, step: int):
            if step < k:
                visited.add(node)
                for v in adj[node]:
                    if v not in visited:
                        dfs(v, step + 1)
            else:
                res.append(node.val)

        adj, res, visited = collections.defaultdict(list), [], set()
        connected(root)
        dfs(target, 0)
        return res


# 876 - Middle of the Linked List - EASY
class Solution:
    # recursive. O(n)+ stack space / O(3)
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        self.n = 0
        self.head = ListNode(-1)
        dummy = ListNode(-1, head)

        def helper(head: Optional[ListNode]):
            if head:
                self.n += 1
                helper(head.next)
            else:
                self.n >>= 1
                return
            self.n -= 1
            if self.n == 0:
                self.head = head
                self.n -= 1
                return
            return

        helper(dummy)
        return self.head

    # compute the length of linked list. O(1.5n) / O(2)
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        n, cp = 0, head
        while head:
            head = head.next
            n += 1
        head = cp
        n >>= 1
        while n:
            head = head.next
            n -= 1
        return head

    # two pointers. O(n) / O(2)
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow


# 878 - Nth Magical Number - HARD
class Solution:
    def nthMagicalNumber(self, N: int, a: int, b: int) -> int:
        def check(n):
            return n // a + n // b - n // c >= N

        c = a * b // math.gcd(a, b)
        if a > b:
            a, b = b, a
        l, r = a * N // 2, a * N
        while l < r:
            mid = (l + r) >> 1
            if check(mid):
                r = mid
            else:
                l = mid + 1
        return l % (10**9 + 7)
