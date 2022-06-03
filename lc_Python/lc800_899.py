from typing import List, Optional
import collections, math, itertools, re


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 804 - Unique Morse Code Words - EASY
class Solution:
    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        dic = {
            i: v
            for i, v in enumerate(
                [
                    ".-",
                    "-...",
                    "-.-.",
                    "-..",
                    ".",
                    "..-.",
                    "--.",
                    "....",
                    "..",
                    ".---",
                    "-.-",
                    ".-..",
                    "--",
                    "-.",
                    "---",
                    ".--.",
                    "--.-",
                    ".-.",
                    "...",
                    "-",
                    "..-",
                    "...-",
                    ".--",
                    "-..-",
                    "-.--",
                    "--..",
                ]
            )
        }
        s = set()
        for w in words:
            s.add("".join([dic[ord(ch) - ord("a")] for ch in w]))
        return len(s)

    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        morse = [
            ".-",
            "-...",
            "-.-.",
            "-..",
            ".",
            "..-.",
            "--.",
            "....",
            "..",
            ".---",
            "-.-",
            ".-..",
            "--",
            "-.",
            "---",
            ".--.",
            "--.-",
            ".-.",
            "...",
            "-",
            "..-",
            "...-",
            ".--",
            "-..-",
            "-.--",
            "--..",
        ]
        return len(
            set("".join(morse[ord(ch) - ord("a")] for ch in word) for word in words)
        )


# 806 - Number of Lines To Write String - EASY
class Solution:
    def numberOfLines(self, widths: List[int], s: str) -> List[int]:
        mx = 100
        ans = 1
        cur = 0
        for ch in s:
            w = widths[ord(ch) - ord("a")]
            if cur + w > mx:
                ans += 1
                cur = w
            else:
                cur += w
        return [ans, cur]


# 807 - Max Increase to Keep City Skyline - MEDIUM
class Solution:
    def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
        row_maxes = [max(row) for row in grid]
        col_maxes = [max(col) for col in zip(*grid)]

        return sum(
            min(row_maxes[r], col_maxes[c]) - val
            for r, row in enumerate(grid)
            for c, val in enumerate(row)
        )

    def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
        rows, cols = list(map(max, grid)), list(map(max, zip(*grid)))
        return sum(min(i, j) for i in rows for j in cols) - sum(map(sum, grid))


# 812 - Largest Triangle Area - EASY
class Solution:
    # heron's formula
    def largestTriangleArea(self, points: List[List[int]]) -> float:
        return max(
            abs(x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2) / 2
            for (x1, y1), (x2, y2), (x3, y3) in itertools.combinations(points, 3)
        )


# 819 - Most Common Word - EASY
class Solution:
    def mostCommonWord(self, p: str, banned: List[str]) -> str:
        p = p.lower() + "."
        w = ""
        freq = collections.defaultdict(int)
        banned = set(banned)
        for ch in p:
            if ch.isalpha():
                w += ch
            else:
                if w not in banned and len(w) > 0:
                    freq[w] += 1
                w = ""
        return sorted(freq.keys(), key=freq.get)[-1]

    # '\w' matches [a-zA-Z0-9_]
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        return collections.Counter(
            w for w in re.split(r"[^\w]+", paragraph.lower()) if w and w not in banned
        ).most_common(1)[0][0]

    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        ban = set(banned)
        words = re.findall(r"\w+", paragraph.lower())
        return collections.Counter(w for w in words if w not in ban).most_common(1)[0][
            0
        ]

    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        return max(
            collections.Counter(re.split(r"[ !?',;.]", paragraph.lower())).items(),
            key=lambda x: (len(x) > 0, x[0] not in set(banned + [""]), x[1]),
        )[0]


# 821 - Shortest Distance to a Character - EASY
class Solution:
    # O(n) / O(n)
    def shortestToChar(self, s: str, c: str) -> List[int]:
        n = len(s)
        ans = [1e4] * n
        i = 0
        while i < n:
            while i < n and s[i] != c:
                i += 1
            if i < n and s[i] == c:
                ans[i] = 0
                k = 1
                j = i - 1
                while j > -1 and ans[j] > k:
                    ans[j] = k
                    j -= 1
                    k += 1
                k = 1
                j = i + 1
                while j < n and ans[j] > k:
                    ans[j] = k
                    j += 1
                    k += 1
                i += 1
        return ans

    def shortestToChar(self, s: str, c: str) -> List[int]:
        pre = float("-inf")
        ans = []
        for i in range(len(s)):
            if s[i] == c:
                pre = i
            ans.append(i - pre)
        pre = float("inf")
        for i in range(len(s) - 1, -1, -1):
            if s[i] == c:
                pre = i
            ans[i] = min(ans[i], pre - i)
        return ans

    def shortestToChar(self, s: str, c: str) -> List[int]:
        n = len(s)
        ans = [0 if ch == c else n for ch in s]
        for i in range(1, n):
            ans[i] = min(ans[i], ans[i - 1] + 1)
        for i in range(n - 2, -1, -1):
            ans[i] = min(ans[i], ans[i + 1] + 1)
        return ans

    def shortestToChar(self, s: str, c: str) -> List[int]:
        p = [i for i in range(len(s)) if c == s[i]]
        return [min(abs(x - i) for i in p) for x in range(len(s))]


# 824 - Goat Latin - EASY
class Solution:
    def toGoatLatin(self, sentence: str) -> str:
        arr = []
        for i, s in enumerate(sentence.split()):
            if s[0] in "aeiouAEIOU":
                arr.append(s + "ma" + "a" * (i + 1))
            else:
                arr.append(s[1:] + s[0] + "ma" + "a" * (i + 1))
        return " ".join(arr)


# 825 - Friends Of Appropriate Ages - MEDIUM
class Solution:
    def numFriendRequests(self, ages: List[int]) -> int:
        def request(a: int, b: int) -> bool:
            return not (b <= 0.5 * a + 7 or b > a or b > 100 and a < 100)

        c = collections.Counter(ages)
        return sum(request(a, b) * c[a] * (c[b] - (a == b)) for a in c for b in c)

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


# 829 - Consecutive Numbers Sum - HARD
class Solution:
    # O(sqrt(2N)) / O(1)
    def consecutiveNumbersSum(self, n: int) -> int:
        # N -1 -2 -3 -4 -5.......-k = 0
        # N = 1 + 2 + 3 +.........+ k
        # (1 + k)k / 2 = N
        # k = sqrt(2N)
        ans = 0
        i = 1
        while n > 0:
            ans += n % i == 0
            n -= i
            i += 1
        return ans

    def consecutiveNumbersSum(self, n: int) -> int:
        ans = 0
        end = n << 1
        i = 1
        while i * i <= end:
            ans += (n - (i + 1) * i // 2) % i == 0
            i += 1
        return ans

    # N = (x + 1) + (x + 2) + ... + (x + k) = kx + k * (k + 1) / 2
    # 2 * N = k(2x + k + 1)
    # 'k' is odd or '2x + k + 1' is odd
    # TODO
    # def consecutiveNumbersSum(self, n: int) -> int:
    #     ans = 1
    #     for i in range(2, int(math.sqrt(2 * n)) + 1):
    #         if (n - i * (i - 1) // 2) % i == 0:
    #             ans += 1
    #     return ans

    # def consecutiveNumbersSum(self, n: int) -> int:
    #     ans = 1
    #     i = 3
    #     while n % 2 == 0:
    #         n //= 2
    #     while i * i <= n:
    #         count = 0
    #         while n % i == 0:
    #             n //= i
    #             count += 1
    #         ans *= count + 1
    #         i += 2
    #     if n > 1:
    #         ans *= 2
    #     return ans


# 838 - Push Dominoes - MEDIUM
class Solution:
    def pushDominoes(self, dominoes: str) -> str:
        while 1:
            new = dominoes.replace("R.L", "S")
            new = new.replace(".L", "LL").replace("R.", "RR")
            if new == dominoes:
                break
            else:
                dominoes = new
        return dominoes.replace("S", "R.L")

    def pushDominoes(self, d: str) -> str:
        n = len(d)
        # [L_dist, R_dist]
        records = [[math.inf, math.inf] for _ in range(n)]
        cur = -math.inf
        for i in range(n):
            if d[i] == "R":
                cur = i
            elif d[i] == "L":
                cur = -math.inf
            records[i][1] = i - cur
        cur = math.inf
        for i in range(n - 1, -1, -1):
            if d[i] == "L":
                cur = i
            elif d[i] == "R":
                cur = math.inf
            records[i][0] = cur - i
        return "".join("." if l == r else ("R" if l > r else "L") for l, r in records)


# 844 - Backspace String Compare - EASY
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        a, b = [], []
        for ch in s:
            if ch == "#":
                if a:
                    a.pop()
            else:
                a.append(ch)
        for ch in t:
            if ch == "#":
                if b:
                    b.pop()
            else:
                b.append(ch)
        return a == b

    # O(n) + O(1): reversed, save '#'
    def backspaceCompare(self, s: str, t: str) -> bool:
        def build(s: str) -> str:
            skip, a = 0, ""
            for ch in reversed(s):
                if ch == "#":
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


# 849 - Maximize Distance to Closest Person - MEDIUM
class Solution:
    def maxDistToClosest(self, seats: List[int]) -> int:
        a = pre = seats.index(1)
        b = seats[::-1].index(1)
        c = 0
        for i in range(len(seats)):
            if seats[i] == 1:
                c = max(c, (i - pre) // 2)
                pre = i
        return max(a, b, c)

    def maxDistToClosest(self, seats: List[int]) -> int:
        i, j, ans, n = 0, len(seats) - 1, 0, len(seats)
        while seats[j] == 0:
            j -= 1
        ans = max(ans, n - 1 - j)
        while seats[i] == 0:
            i += 1
        ans, pre = max(ans, i), i
        while i <= j:
            if seats[i] == 1:
                ans = max(ans, (i - pre) // 2)
                pre = i
            i += 1
        return ans

    def maxDistToClosest(self, seats: List[int]) -> int:
        prev, ans = 0, 0
        for cur, seat in enumerate(seats):
            if seat:
                if seats[prev]:
                    ans = max(ans, (cur - prev) // 2)
                else:
                    ans = max(ans, (cur - prev))
                prev = cur
        if seats[prev]:
            ans = max(ans, len(seats) - 1 - prev)
        return ans


# 851 - Loud and Rich - MEDIUM
class Solution:
    def loudAndRich(self, richer: List[List[int]], quiet: List[int]) -> List[int]:
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

    def loudAndRich(self, richer: List[List[int]], quiet: List[int]) -> List[int]:
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

        ans = []
        dic = {}  # k:node.value v:node.parent
        findParent(root)
        dfs(target, None, k)
        return ans

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
                ans.append(node.val)
            return

        adj = collections.defaultdict(list)
        ans = []
        visited = set()
        connected(root)
        dfs(target, 0)
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
        ans = [target.val]
        seen = set(ans)
        for _ in range(k):
            new = []
            for node_val in ans:
                for connected_node_val in conn[node_val]:
                    if connected_node_val not in seen:
                        new.append(connected_node_val)
            ans = new
            # seen = set(ans).union(seen) # '.intersection()' <=> '&'
            seen |= set(ans)
        return ans


# 868 - Binary Gap - EASY
class Solution:
    def binaryGap(self, n: int) -> int:
        ans = c = 0
        while n:
            if n & 1:
                ans = max(ans, c)
                c = 1
            else:
                c += 1 if c > 0 else 0
            n >>= 1
        return ans


# 875 - Koko Eating Bananas - MEDIUM
class Solution:
    # O(n * logm), O(1)
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        def check(mid: int, h: int) -> bool:
            for b in piles:
                if not b % mid:
                    h -= b // mid
                else:
                    h -= b // mid + 1
                # or
                # h -= (b + mid - 1) // mid
                # or
                # h -= (b - 1) / mid + 1
                # equal to
                # h -= math.ceil(b / mid)
            return h >= 0

        lo, hi = 1, max(piles)
        while lo < hi:
            mid = (lo + hi) >> 1
            if not check(mid, h):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        lo, hi = 1, max(piles)
        while lo < hi:
            middle = (lo + hi) // 2
            need = 0
            for pile in piles:
                need += math.ceil(pile / middle)
            if need <= h:
                hi = middle
            else:
                lo = middle + 1
        return hi


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


# 883 - Projection Area of 3D Shapes - EASY
class Solution:
    def projectionArea(self, grid: List[List[int]]) -> int:
        n = len(grid)
        a = sum(1 if x != 0 else 0 for g in grid for x in g)
        b = sum(max(g) for g in grid)
        c = sum(max(grid[i][j] for i in range(n)) for j in range(n))
        return a + b + c

    # zip + '*' unpack: row -> column
    def projectionArea(self, grid: List[List[int]]) -> int:
        a = sum(v > 0 for row in grid for v in row)
        b = sum(map(max, grid))
        c = sum(map(max, zip(*grid)))
        return a + b + c

    def projectionArea(self, grid: List[List[int]]) -> int:
        return sum(map(max, grid + list(zip(*grid)))) + sum(
            v > 0 for r in grid for v in r
        )

    def projectionArea(self, grid: List[List[int]]) -> int:
        return sum(
            [
                sum(map(max, grid)),
                sum(map(max, zip(*grid))),
                sum(v > 0 for row in grid for v in row),
            ]
        )


# 884 - Uncommon Words from Two Sentences - EASY
class Solution:
    def uncommonFromSentences(self, s1: str, s2: str) -> List[str]:
        cnt = collections.Counter(s1.split() + s2.split())
        ans = []
        for k in cnt:
            if cnt[k] == 1:
                ans.append(k)
        return ans
