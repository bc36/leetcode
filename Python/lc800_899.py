import bisect, collections, functools, heapq, itertools, math, operator, string, re
from typing import List, Optional, Tuple
import sortedcontainers


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


# 801 - Minimum Swaps To Make Sequences Increasing - HARD
class Solution:
    # O(n) / O(n)
    def minSwap(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        # f[i][0]: 不交换 nums1[i] 和 nums2[i], f[i][1]: 交换
        f = [[math.inf, math.inf] for _ in range(n)]
        f[0][0] = 0
        f[0][1] = 1
        for i in range(1, n):
            # if (
            #     nums1[i - 1] < nums1[i]
            #     and nums2[i - 1] < nums2[i]
            #     and nums1[i - 1] < nums2[i]
            #     and nums2[i - 1] < nums1[i]
            # ):
            #     f[i][0] = min(f[i - 1][0], f[i - 1][1])
            #     f[i][1] = min(f[i - 1][0], f[i - 1][1]) + 1
            # elif nums1[i - 1] < nums1[i] and nums2[i - 1] < nums2[i]:
            #     f[i][0] = f[i - 1][0]
            #     f[i][1] = f[i - 1][1] + 1
            # elif nums1[i - 1] < nums2[i] and nums2[i - 1] < nums1[i]:
            #     f[i][0] = f[i - 1][1]
            #     f[i][1] = f[i - 1][0] + 1

            # 上述可化简为
            if nums1[i - 1] < nums1[i] and nums2[i - 1] < nums2[i]:
                f[i][0] = f[i - 1][0]
                f[i][1] = f[i - 1][1] + 1
            if nums1[i - 1] < nums2[i] and nums2[i - 1] < nums1[i]:
                f[i][0] = min(f[i - 1][1], f[i][0])
                f[i][1] = min(f[i - 1][0] + 1, f[i][1])

            # 或者为
            # if nums1[i] > nums1[i - 1] and nums2[i] > nums2[i - 1]:
            #     f[i][0] = f[i - 1][0]
            #     f[i][1] = f[i - 1][1] + 1
            #     # 如果当前的 nums1 和 nums2 都是严格单调递增的, 那么交换的状态就必须和 i-1 一样才行
            #     # 比如当前 i 时刻进行了交换, 那么 i-1 必须也进行交换, 因为 i-1 到 i 这一段是严格单调递增的
            #     # 那么 i 时刻如果是进行了交换才变得单增, 那么 i-1 时刻一定也是进行了交换了的
            #     # 同理, i 时刻没有变化, i-1 也是没有变化的
            # if nums1[i] > nums2[i - 1] and nums2[i] > nums1[i - 1]:
            #     f[i][0] = min(f[i][0], f[i - 1][1])
            #     f[i][1] = min(f[i][1], f[i - 1][0] + 1)

        return min(f[-1])

    # O(n) / O(1)
    def minSwap(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        # x 不交换当前 对应的最小交换次数, y 交换当前 对应的最小交换次数
        x, y = 0, 1
        for i in range(1, n):
            prex, prey = x, y
            x = y = n
            if nums1[i] > nums1[i - 1] and nums2[i] > nums2[i - 1]:  # 能不能不交换
                x = prex
                y = prey + 1  # 前面交换, 当前也需要交换
            if nums1[i] > nums2[i - 1] and nums2[i] > nums1[i - 1]:  # 能不能交换
                x = min(x, prey)  # 等价于前面交换了, 当前没交换
                y = min(y, prex + 1)  # 前面没交换, 当前交换
        return min(x, y)

    def minSwap(self, nums1: List[int], nums2: List[int]) -> int:
        prex, prey = 0, 1
        for (a1, b1), (a2, b2) in pairwise(zip(nums1, nums2)):
            x = y = math.inf
            if a1 < a2 and b1 < b2:
                x = prex
                y = prey + 1
            if b1 < a2 and a1 < b2:
                x = min(x, prey)
                y = min(y, prex + 1)
            prex, prey = x, y
        return min(prex, prey)

    def minSwap(self, nums1: List[int], nums2: List[int]) -> int:
        x = y = 0  # x 不交换当前 对应的最小交换次数, y 交换当前 对应的最小交换次数
        a1 = b1 = -1  # a1, b1 前一个数据
        for a2, b2 in zip(nums1, nums2):
            if a1 >= a2 or b1 >= b2:  # 这一步一定交换
                x, y = y, x + 1  # 状态反转
            elif a1 >= b2 or b1 >= a2:  # 状态跟随
                y = y + 1
            else:  # 可交换 也可不交换, 满足 a1, b1 < a2, b2
                x, y = min(x, y), min(x, y) + 1
            a1, b1 = a2, b2
        return min(x, y)

    def minSwap(self, nums1: List[int], nums2: List[int]) -> int:
        x, y = 0, 1
        for i in range(1, len(nums1)):
            # In this case, the ith manipulation should be same as the i-1th manipulation
            if nums1[i - 1] >= nums2[i] or nums2[i - 1] >= nums1[i]:
                y = y + 1
            # In this case, the ith manipulation should be the opposite of the i-1th manipulation
            elif nums1[i - 1] >= nums1[i] or nums2[i - 1] >= nums2[i]:
                x, y = y, x + 1
            else:
                x, y = min(x, y), min(x, y) + 1
        return min(x, y)


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


# 805 - Split Array With Same Average - HARD
class Solution:
    # meet in the middle, O(2 ** n) 降至 O(2 ** (n // 2)), 降低搜索空间
    def splitArraySameAverage(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return False
        summ = sum(nums)
        for i in range(n):
            nums[i] = nums[i] * n - summ
        m = n // 2
        left = set()
        for i in range(1, 1 << m):
            t = sum(v for j, v in enumerate(nums[:m]) if i >> j & 1)
            if t == 0:
                return True
            left.add(t)
        rsum = sum(nums[m:])
        for i in range(1, 1 << (n - m)):
            t = sum(v for j, v in enumerate(nums[m:]) if i >> j & 1)
            if t == 0 or rsum != t and -t in left:
                return True
        return False

    def splitArraySameAverage(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return False
        m = n // 2
        summ = sum(nums)
        # 把问题转化为在整个和为 0 的数组中, 求一个和为零的子序列(其中子序列不能为空或全部)
        nums = [n * v - summ for v in nums]
        left = set()
        right = set()

        # TODO flag1 flag2 什么意思
        def dfs1(idx: int, summ: int, flag1: bool, flag2: bool) -> None:
            if idx == m:
                if flag1 and flag2:
                    left.add(summ)
                return
            dfs1(idx + 1, summ, flag1, True)
            dfs1(idx + 1, summ + nums[idx], True, flag2)
            return

        def dfs2(idx: int, summ: int, flag1: bool, flag2: bool) -> None:
            if idx == n - m:
                if flag1 and flag2:
                    right.add(summ)
                return
            dfs2(idx + 1, summ, flag1, True)
            dfs2(idx + 1, summ + nums[m + idx], True, flag2)
            return

        dfs1(0, 0, False, False)
        dfs2(0, 0, False, False)
        if 0 in left or 0 in right:
            return True
        for v in left:
            if -v in right:
                return True
        return sum(nums[:m]) == 0  # nums 已经变了


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


# 809 - Expressive Words - MEDIUM
class Solution:
    def expressiveWords(self, s: str, words: List[str]) -> int:
        a = []
        for c, grp in itertools.groupby(s):
            a.append((c, len(list(grp))))
        ans = 0
        for w in words:
            i = 0
            ok = True
            for c, grp in itertools.groupby(w):
                l = len(list(grp))
                if (
                    i == len(a)
                    or c != a[i][0]
                    or (l != a[i][1] and (a[i][1] < l or a[i][1] < 3))
                ):
                    ok = False
                    break
                i += 1
            if ok and i == len(a):
                ans += 1
        return ans


# 811 - Subdomain Visit Count - MEDIUM
class Solution:
    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        d = {}
        for c in cpdomains:
            x, y = c.split()
            x = int(x)
            z = y.split(".")
            if len(z) == 2:
                d[z[1]] = d.get(z[1], 0) + x
                d[y] = d.get(y, 0) + x
            else:
                d[z[2]] = d.get(z[2], 0) + x
                d[z[1] + "." + z[2]] = d.get(z[1] + "." + z[2], 0) + x
                d[y] = d.get(y, 0) + x
        return [str(v) + " " + k for k, v in d.items()]

    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        cnt = collections.Counter()
        for d in cpdomains:
            c, s = d.split()
            c = int(c)
            cnt[s] += c
            while "." in s:
                s = s[s.index(".") + 1 :]
                cnt[s] += c
        return [f"{c} {s}" for s, c in cnt.items()]


# 812 - Largest Triangle Area - EASY
class Solution:
    # heron's formula
    def largestTriangleArea(self, points: List[List[int]]) -> float:
        return max(
            abs(x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2) / 2
            for (x1, y1), (x2, y2), (x3, y3) in itertools.combinations(points, 3)
        )


# 813 - Largest Sum of Averages - MEDIUM
class Solution:
    # O(k * n^2) / O(n)
    def largestSumOfAverages(self, nums: List[int], k: int) -> float:
        @functools.lru_cache(None)
        def dfs(i: int, k: int):
            if i == len(nums):
                return -math.inf
            if k == 1:
                return sum(nums[i:]) / (len(nums) - i)
            score = summ = 0
            for j in range(i, len(nums)):
                summ += nums[j]
                score = max(score, summ / (j - i + 1) + dfs(j + 1, k - 1))
            return score

        return dfs(0, k)

    def largestSumOfAverages(self, nums: List[int], k: int) -> float:
        # f[i][k]: 将前i个元素(包含i)分成k份, 所能得到的最大分数
        n = len(nums)
        f = [[0] * (k + 1) for _ in range(n + 1)]
        p = [0] * (n + 1)
        for i, v in enumerate(nums):
            p[i + 1] = p[i] + v
        for i in range(1, n + 1):
            f[i][1] = p[i] / i
            for j in range(2, k + 1):
                for h in range(i):
                    f[i][j] = max(f[i][j], f[h][j - 1] + (p[i] - p[h]) / (i - h))
        return f[-1][-1]


# 814 - Binary Tree Pruning - MEDIUM
class Solution:
    def pruneTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def postorder(root: TreeNode) -> int:
            if not root:
                return 0
            l = postorder(root.left)
            r = postorder(root.right)
            if not l:
                root.left = None
            if not r:
                root.right = None
            return l + r + root.val

        return root if postorder(root) else None

    def pruneTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        root.left = self.pruneTree(root.left)
        root.right = self.pruneTree(root.right)
        if root.val == 0 and not root.left and not root.right:
            return None
        return root


# 816 - Ambiguous Coordinates - MEDIUM
class Solution:
    # O(n ** 3) / O(n ** 3)
    def ambiguousCoordinates(self, s: str) -> List[str]:
        def calc(i: int, j: int) -> List[str]:
            res = []
            for k in range(1, j - i + 1):
                l = s[i : i + k]
                r = s[i + k : j]
                ok = (l == "0" or not l.startswith("0")) and not r.endswith("0")
                if ok:
                    res.append(l + ("." if k < j - i else "") + r)
            return res

        n = len(s)
        return [
            f"({x}, {y})"
            for i in range(2, n - 1)
            for x in calc(1, i)
            for y in calc(i, n - 1)
        ]


# 817 - Linked List Components - MEDIUM
class Solution:
    def numComponents(self, head: Optional[ListNode], nums: List[int]) -> int:
        ans = 0
        s = set(nums)
        isHead = False
        while head:
            if head.val not in s:
                isHead = False
            elif not isHead:
                ans += 1
                isHead = True
            head = head.next
        return ans


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


# 822 - Card Flipping Game - MEDIUM
class Solution:
    def flipgame(self, fronts: List[int], backs: List[int]) -> int:
        same = {f for f, b in zip(fronts, backs) if f == b}
        ans = 2001
        for x in itertools.chain(fronts, backs):
            if x not in same:
                ans = min(ans, x)
        return ans % 2001


# 823 - Binary Trees With Factors - MEDIUM
class Solution:
    def numFactoredBinaryTrees(self, arr: List[int]) -> int:
        s = set(arr)

        @functools.cache
        def dfs(rootVal: int) -> int:
            res = 1
            for v in arr:
                if rootVal % v == 0 and rootVal // v in s:
                    res += dfs(v) * dfs(rootVal // v)
            return res

        return sum(dfs(v) for v in arr) % (10**9 + 7)

    def numFactoredBinaryTrees(self, arr: List[int]) -> int:
        arr.sort()
        idx = {v: i for i, v in enumerate(arr)}

        @functools.cache
        def dfs(i: int) -> int:
            res = 1
            rootVal = arr[i]
            for j in range(i):
                v = arr[j]
                if rootVal % v == 0 and rootVal // v in idx:
                    res += dfs(j) * dfs(idx[rootVal // v])
            return res

        return sum(dfs(i) for i in range(len(arr))) % (10**9 + 7)

    def numFactoredBinaryTrees(self, arr: List[int]) -> int:
        arr.sort()
        idx = {v: i for i, v in enumerate(arr)}
        f = [1] * len(arr)
        for i, rootVal in enumerate(arr):
            for j in range(i):
                if rootVal % arr[j] == 0 and rootVal // arr[j] in idx:
                    f[i] += f[j] * f[idx[rootVal // arr[j]]]
        return sum(f) % (10**9 + 7)


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
    def largestIsland(self, grid: List[List[int]]) -> int:
        def move(x: int, y: int) -> None:
            for i, j in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= x + i < n and 0 <= y + j < n:
                    yield x + i, y + j
            return

        def dfs(x: int, y: int, index: int) -> int:
            area = 0
            grid[x][y] = index
            for i, j in move(x, y):
                if grid[i][j] == 1:
                    area += dfs(i, j, index)
            return area + 1

        n = len(grid)
        index = 2
        areas = {0: 0}
        for x in range(n):
            for y in range(n):
                if grid[x][y] == 1:
                    areas[index] = dfs(x, y, index)
                    index += 1
        ans = max(areas.values())
        for x in range(n):
            for y in range(n):
                if grid[x][y] == 0:
                    possible = set(grid[i][j] for i, j in move(x, y))
                    ans = max(ans, 1 + sum(areas[index] for index in possible))
        return ans

    def largestIsland(self, grid: List[List[int]]) -> int:
        def bfs(i, j, l) -> int:
            area = 1
            grid[i][j] = l
            dq = collections.deque([(i, j)])
            while dq:
                for _ in range(len(dq)):
                    x, y = dq.popleft()
                    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                        if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 1:
                            area += 1
                            grid[nx][ny] = l
                            dq.append((nx, ny))
            return area

        n = len(grid)
        l = 2
        areas = {}
        ans = 0
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v == 1:
                    areas[l] = bfs(i, j, l)
                    ans = max(ans, areas[l])
                    l += 1
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v == 0:
                    can = set()
                    for x, y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                        if 0 <= x < n and 0 <= y < n:
                            can.add(grid[x][y])
                    ans = max(ans, 1 + sum(areas.get(l, 0) for l in can))
        return ans

    def largestIsland(self, grid: List[List[int]]) -> int:
        def dfs(i: int, j: int, label: int) -> None:
            grid[i][j] = label
            area[label] += 1
            for x, y in (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1):
                if 0 <= x < n and 0 <= y < n and grid[x][y] == 1:
                    dfs(x, y, label)
            return

        n = len(grid)
        area = collections.Counter()
        label = 2
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v == 1:
                    dfs(i, j, label)
                    label += 1
        ans = max(area.values(), default=0)
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v == 0:
                    new = 1
                    connected = {0}
                    for x, y in (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1):
                        if 0 <= x < n and 0 <= y < n and grid[x][y] not in connected:
                            new += area[grid[x][y]]
                            connected.add(grid[x][y])
                    ans = max(ans, new)
        return ans


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
        end = n * 2
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


# 830 - Positions of Large Groups - EASY
class Solution:
    def largeGroupPositions(self, s: str) -> List[List[int]]:
        ans = []
        p = ""
        l = 0
        for r, c in enumerate(s + "#"):
            if c != p:
                if l + 3 <= r:
                    ans.append([l, r - 1])
                l = r
            p = c
        return ans


# 831 - Masking Personal Information - MEDIUM
class Solution:
    def maskPII(self, s: str) -> str:
        at = s.find("@")
        if at != -1:
            return (s[0] + "*" * 5 + s[at - 1 :]).lower()
        s = "".join(i for i in s if i.isdigit())
        return ["", "+*-", "+**-", "+***-"][len(s) - 10] + "***-***-" + s[-4:]


# 832 - Flipping an Image - EASY
class Solution:
    def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
        for i, r in enumerate(image):
            image[i] = r[::-1]
        for i, r in enumerate(image):
            for j, v in enumerate(r):
                image[i][j] = 1 - v
        return image

    def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
        for r in image:
            for j in range((len(r) + 1) // 2):
                if r[j] == r[-1 - j]:
                    r[j] = r[-1 - j] = 1 - r[j]
        return image

    def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
        return [[v ^ 1 for v in r[::-1]] for r in image]
        return [[1 - v for v in r[::-1]] for r in image]


# 833 - Find And Replace in String - MEDIUM
class Solution:
    def findReplaceString(
        self, s: str, indices: List[int], sources: List[str], targets: List[str]
    ) -> str:
        replace = [(c, 1) for c in s]
        for i, src, tar in zip(indices, sources, targets):
            if s.startswith(src, i):  # a bit faster than slicing
                replace[i] = (tar, len(src))

        ans = []
        i = 0
        while i < len(s):
            ans.append(replace[i][0])
            i += replace[i][1]
        return "".join(ans)

    def findReplaceString(
        self, s: str, indices: List[int], sources: List[str], targets: List[str]
    ) -> str:
        ans = list(s)
        for i, src, tar in zip(indices, sources, targets):
            if s.startswith(src, i):
                ans[i : i + len(src)] = [tar] + [""] * (len(src) - 1)
        return "".join(ans)


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


# 841 - Keys and Rooms - MEDIUM
class Solution:
    # O(m + n) / O(n), m is the total number of keys
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        can = [True] + [False] * (len(rooms) - 1)
        dq = collections.deque([0])
        while dq:
            n = dq.popleft()
            for x in rooms[n]:
                if not can[x]:
                    can[x] = True
                    dq.append(x)
        return all(can)

    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        def dfs(n: int):
            for x in rooms[n]:
                if not can[x]:
                    can[x] = True
                    dfs(x)
            return

        can = [True] + [False] * (len(rooms) - 1)
        dfs(0)
        return all(can)


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
        for i, v in enumerate(seats):
            if v == 1:
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
        pre = ans = 0, 0
        for i, v in enumerate(seats):
            if v:
                if seats[pre]:
                    ans = max(ans, (i - pre) // 2)
                else:
                    ans = max(ans, (i - pre))
                pre = i
        if seats[pre]:
            ans = max(ans, len(seats) - 1 - pre)
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


# 854 - K-Similar Strings - HARD
class Solution:
    def kSimilarity(self, s1: str, s2: str) -> int:
        def next(s: str) -> List[str]:
            i = 0
            while s[i] == s2[i]:
                i += 1
            res = []
            for j in range(i + 1, len(s1)):
                if s[j] == s2[i] and s[j] != s2[j]:
                    res.append(s2[: i + 1] + s[i + 1 : j] + s[i] + s[j + 1 :])
            return res

        q = collections.deque([s1])
        vis = {s1}
        ans = 0
        while 1:
            for _ in range(len(q)):
                s = q.popleft()
                if s == s2:
                    return ans
                for nxt in next(s):
                    if nxt not in vis:
                        vis.add(nxt)
                        q.append(nxt)
            ans += 1

    def kSimilarity(self, s1: str, s2: str) -> int:
        step, n = 0, len(s1)
        q, vis = [(s1, 0)], {s1}
        while True:
            tmp = q
            q = []
            for s, i in tmp:
                if s == s2:
                    return step
                while i < n and s[i] == s2[i]:
                    i += 1
                for j in range(i + 1, n):
                    if s[j] == s2[i] != s2[j]:  # 剪枝，只在 s[j] != s2[j] 时去交换
                        t = list(s)
                        t[i], t[j] = t[j], t[i]
                        t = "".join(t)
                        if t not in vis:
                            vis.add(t)
                            q.append((t, i + 1))
            step += 1


# 855 - Exam Room - MEDIUM
class ExamRoom:
    def __init__(self, n: int):
        self.n = n
        self.hp = []
        self.pairs = {}
        self.add(0, n)

    def add(self, l: int, r: int) -> None:
        # 处理前两个人, 坐在两端
        if l == 0 or r == self.n:
            d = r - l
        else:
            d = (r - l + 1) >> 1
        heapq.heappush(self.hp, (-d, l, r))
        self.pairs[l] = r
        self.pairs[r] = l
        return

    def seat(self) -> int:
        l = r = -1
        while self.hp:
            _, l, r = heapq.heappop(self.hp)
            # check for lasy deletion
            if l == self.pairs.get(r, -1) and r == self.pairs.get(l, -1):
                break

        # try:  # check full
        #     assert l != r
        # except AssertionError:
        #     print("No free seats.")
        #     self.add(l, r)
        #     return 0

        # 处理前两个人, 坐在两端
        if l == 0:
            p = 0
        elif r == self.n:
            p = self.n - 1
        else:
            p = (l + r - 1) >> 1
        self.add(l, p)
        self.add(p + 1, r)
        return p

    def leave(self, p: int) -> None:
        l = self.pairs.pop(p)
        r = self.pairs.pop(p + 1)
        self.add(l, r)
        return


# 856 - Score of Parentheses - MEDIUM
class Solution:
    # O(n^2) / O(n^2)
    def scoreOfParentheses(self, s: str) -> int:
        if len(s) == 2:
            return 1
        cur = l = p = 0
        for r, c in enumerate(s):
            p += 1 if c == "(" else -1
            if p == 0:
                if l + 1 == r:
                    cur += 1
                else:
                    cur += 2 * self.scoreOfParentheses(s[l + 1 : r])
                l = r + 1
        return cur

    def scoreOfParentheses(self, s: str) -> int:
        if len(s) == 2:
            return 1
        p = 0
        for i, c in enumerate(s):
            p += 1 if c == "(" else -1
            if p == 0:
                if i == len(s) - 1:
                    return 2 * self.scoreOfParentheses(s[1:-1])
                return self.scoreOfParentheses(s[: i + 1]) + self.scoreOfParentheses(
                    s[i + 1 :]
                )

    # O(n) / O(n)
    def scoreOfParentheses(self, s: str) -> int:
        st = [0]
        for c in s:
            if c == ")":
                cur = st.pop()
                st[-1] += max(2 * cur, 1)
            else:
                st.append(0)
        return st[-1]

    # O(n) / O(1)
    def scoreOfParentheses(self, s: str) -> int:
        ans = dep = 0
        for i, c in enumerate(s):
            if c == ")":
                dep -= 1
                if s[i - 1] == "(":
                    ans += 1 << dep
            else:
                dep += 1
        return ans


# 857 - Minimum Cost to Hire K Workers - HARD
class Solution:
    # O(nlogn) / O(n)
    def mincostToHireWorkers(
        self, quality: List[int], wage: List[int], k: int
    ) -> float:
        qw = sorted(zip(quality, wage), key=lambda x: x[1] / x[0])
        hp = [-q for q, _ in qw[:k]]
        heapq.heapify(hp)
        summ = -sum(hp)
        ans = summ * qw[k - 1][1] / qw[k - 1][0]
        for q, w in qw[k:]:
            if q < -hp[0]:
                summ += heapq.heapreplace(hp, -q) + q
                ans = min(ans, summ * w / q)
        return ans

    def mincostToHireWorkers(
        self, quality: List[int], wage: List[int], k: int
    ) -> float:
        ans = math.inf
        hp = []
        summ = 0
        for q, w in sorted(zip(quality, wage), key=lambda x: x[1] / x[0]):
            summ += q
            heapq.heappush(hp, -q)
            if len(hp) > k:
                summ += heapq.heappop(hp)
            if len(hp) == k:
                ans = min(ans, summ * w / q)
        return ans

    def mincostToHireWorkers(
        self, quality: List[int], wage: List[int], k: int
    ) -> float:
        qw = sorted(zip(quality, wage), key=lambda p: p[1] / p[0])
        ans = math.inf
        summ = 0
        hp = []
        for q, w in qw[: k - 1]:
            summ += q
            heapq.heappush(hp, -q)
        for q, w in qw[k - 1 :]:
            summ += q
            heapq.heappush(hp, -q)
            ans = min(ans, summ * w / q)
            summ += heapq.heappop(hp)
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


# 860 - Lemonade Change - EASY
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        a = b = 0
        for x in bills:
            if x == 5:
                a += 1
            elif x == 10:
                a -= 1
                b += 1
            elif b:
                a -= 1
                b -= 1
            else:
                a -= 3
            if a < 0:
                return False
        return True


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


# 864 - Shortest Path to Get All Keys - HARD
class Solution:
    # 只有在拿到一个新的钥匙后, 才可以走回头路
    # BFS 中的每个状态应该是由'坐标'以及'此时所持有的钥匙情况'所决定的
    # 位运算技巧 -> 状态压缩, 1010 表示找到第 2, 4 把钥匙
    # O(m * n * 2 ** k) / O(m * n * 2 ** k), k 是钥匙数量
    def shortestPathAllKeys(self, grid: List[str]) -> int:
        m = len(grid)
        n = len(grid[0])
        x = y = cnt = 0  # 先找到起点, 以及钥匙的数量
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v == "@":
                    x = i
                    y = j
                elif v.islower():
                    cnt += 1
        q = [(x, y, 0)]
        vis = {(x, y, 0)}
        getAll = (1 << cnt) - 1  # 所有钥匙都找到的状态
        step = 0
        while q:
            nxt = []
            for x, y, cur in q:
                if cur == getAll:
                    return step
                for nx, ny in ((x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)):
                    if not 0 <= nx < m or not 0 <= ny < n:
                        continue
                    v = grid[nx][ny]
                    key = cur  # 注意四个方向分别更新钥匙状态, 需要一个新变量
                    if v == "#":
                        continue
                    # if v.isupper() and (key >> (ord(v) - ord("A"))) & 1 == 0:
                    if v.isupper() and key & (1 << (ord(v) - ord("A"))) == 0:
                        continue
                    if v.islower():
                        key |= 1 << (ord(v) - ord("a"))
                    state = (nx, ny, key)
                    if state not in vis:
                        nxt.append(state)
                        vis.add(state)
            step += 1
            q = nxt
        return -1


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


# 870 - Advantage Shuffle - MEDIUM
class Solution:
    # O(nlogn) / O(n)
    def advantageCount(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        idx = sorted(range(len(nums1)), key=lambda i: nums2[i])
        i = 0
        can = collections.defaultdict(list)
        cannot = list()
        for v in nums1:
            if v > nums2[idx[i]]:
                can[nums2[idx[i]]].append(v)
                i += 1
            else:
                cannot.append(v)
        for i, v in enumerate(nums2):
            if v in can and can[v]:
                nums1[i] = can[v].pop()
            else:
                nums1[i] = cannot.pop()
        return nums1

    def advantageCount(self, nums1: List[int], nums2: List[int]) -> List[int]:
        n = len(nums1)
        nums1.sort()
        idx = sorted(range(n), key=lambda i: nums2[i])
        l = 0
        r = n - 1
        for v in nums1:
            if v > nums2[idx[l]]:
                nums2[idx[l]] = v  # 用下等马比下等马
                l += 1
            else:
                nums2[idx[r]] = v  # 用下等马比上等马
                r -= 1
        return nums2


# 873 - Length of Longest Fibonacci Subsequence - MEDIUM
class Solution:
    def lenLongestFibSubseq(self, arr: List[int]) -> int:
        s = set(arr)
        dp = {}
        for i in range(2, len(arr)):
            vi = arr[i]
            for j in range(i - 1, 0, -1):
                vj = arr[j]
                vk = vi - vj
                if vk >= vj:
                    break
                if vk in s:
                    dp[(vj, vi)] = dp.get((vk, vj), 2) + 1
        if dp:
            return max(dp.values())
        return 0

    def lenLongestFibSubseq(self, arr: List[int]) -> int:
        def dfs(i: int, j: int) -> int:
            if memo[i][j] != 0:
                return memo[i][j]
            v = arr[i] + arr[j]
            if v in m:
                memo[i][j] = dfs(j, m[v]) + 1
                return memo[i][j]
            else:
                memo[i][j] = 2
                return 2

        m = {v: i for i, v in enumerate(arr)}
        n = len(arr)
        # TLE: don't know the reason
        #   1. @cache
        #   2. memo = collections.defaultdict(int)
        memo = [[0] * n for _ in range(n)]
        ans = max(dfs(i, j) for i in range(len(arr)) for j in range(i + 1, len(arr)))
        return 0 if ans == 2 else ans


# 874 - Walking Robot Simulation - MEDIUM
class Solution:
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        obs = {(x, y) for x, y in obstacles}
        ans = x = y = d = 0  # 0: up, 1: right, 2: down, 3: left
        for c in commands:
            if c == -1:
                d = (d + 1) % 4
            elif c == -2:
                d = (d - 1) % 4
            else:
                for _ in range(c):
                    nx, ny = x + (d % 2) * (2 - d), y + (d % 2 == 0) * (1 - d)
                    if (nx, ny) in obs:
                        break
                    x, y = nx, ny
                if x * x + y * y > ans:
                    ans = x * x + y * y
        return ans

    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        obs = set(map(tuple, obstacles))
        d = (0, 1, 0, -1, 0)
        ans = x = y = i = 0
        for c in commands:
            if c == -2:
                i = (i + 3) % 4
            elif c == -1:
                i = (i + 1) % 4
            else:
                for _ in range(c):
                    nx, ny = x + d[i], y + d[i + 1]
                    if (nx, ny) in obs:
                        break
                    x, y = nx, ny
                    ans = max(ans, x * x + y * y)
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
        l = 1
        r = max(piles)
        while l < r:
            m = (l + r) // 2
            need = 0
            for p in piles:
                need += math.ceil(p / m)
            if need <= h:
                r = m
            else:
                l = m + 1
        return r

    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        l = 0
        r = max(piles)
        while l < r:
            m = (l + r) >> 1
            t = 0
            if m == 0:  # cuz l init as 0, can be omitted if l init as 1, see above
                return 1
            for p in piles:
                t += (p + m - 1) // m  # trick
            if t > h:
                l = m + 1
            else:
                r = m
        return l

    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        return bisect.bisect_left(
            range(max(piles)),
            x=-h,
            lo=1,
            key=lambda x: -sum((p + x - 1) // x for p in piles),
        )

    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        return (
            bisect.bisect_left(
                range(1, max(piles)),
                x=-h,
                key=lambda x: -sum((p + x - 1) // x for p in piles),
            )
            + 1
        )

    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        return (
            bisect.bisect_left(
                range(1, max(piles)),
                x=True,
                key=lambda x: sum((p + x - 1) // x for p in piles) <= h,
            )
            + 1
        )

    # wrong answer, why?
    # bisect_left seems to require sequence sorted in ascending order,
    # at here True > False
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        return (
            bisect.bisect_left(
                range(1, max(piles)),
                x=False,
                key=lambda x: sum((p + x - 1) // x for p in piles) > h,
            )
            + 1
        )


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


# 882 - Reachable Nodes In Subdivided Graph - HARD
class Solution:
    # djikstra
    def reachableNodes(self, edges: List[List[int]], maxMoves: int, n: int) -> int:
        g = [[] for _ in range(n)]
        for x, y, t in edges:
            g[x].append((y, t))
            g[y].append((x, t))
        ans = 0
        h = [(0, 0)]
        vis = set()
        used = {}
        while h and h[0][0] <= maxMoves:
            moved, x = heapq.heappop(h)
            if x in vis:
                continue
            vis.add(x)
            ans += 1
            for y, t in g[x]:
                if y not in vis and moved + t + 1 <= maxMoves:
                    heapq.heappush(h, (moved + t + 1, y))
                used[(x, y)] = min(t, maxMoves - moved)
        for x, y, t in edges:
            ans += min(t, used.get((x, y), 0) + used.get((y, x), 0))
        return ans

    def reachableNodes(self, edges: List[List[int]], maxMoves: int, n: int) -> int:
        g = collections.defaultdict(dict)
        for x, y, t in edges:
            g[x][y] = g[y][x] = t
        h = [(0, 0)]
        d = {0: 0}  # 维护到每个节点的最短距离
        used = collections.defaultdict(int)  #  记录每条边的使用情况
        ans = 0
        while h:
            moved, x = heapq.heappop(h)
            if moved > d[x]:
                continue
            ans += 1
            for y in g[x]:
                used[(x, y)] = min(g[x][y], maxMoves - moved)
                new = moved + g[x][y] + 1
                if new < d.get(y, maxMoves + 1):
                    heapq.heappush(h, (new, y))
                    d[y] = new
        for x, y, t in edges:
            ans += min(t, used[(x, y)] + used[(y, x)])
        return ans

    def reachableNodes(self, edges: List[List[int]], maxMoves: int, n: int) -> int:
        # 返回从 start 到每个点的最短路
        def dijkstra(g: List[List[Tuple[int]]], start: int) -> List[int]:
            dist = [math.inf] * len(g)
            dist[start] = 0
            h = [(0, start)]
            while h:
                d, x = heapq.heappop(h)
                if d > dist[x]:
                    continue
                for y, w in g[x]:
                    new = dist[x] + w
                    if new < dist[y]:
                        dist[y] = new
                        heapq.heappush(h, (new, y))
            return dist

        g = [[] for _ in range(n)]
        for u, v, cnt in edges:
            g[u].append((v, cnt + 1))
            g[v].append((u, cnt + 1))
        dist = dijkstra(g, 0)
        ans = sum(d <= maxMoves for d in dist)
        for u, v, cnt in edges:
            a = max(maxMoves - dist[u], 0)
            b = max(maxMoves - dist[v], 0)
            ans += min(a + b, cnt)
        return ans


# 883 - Projection Area of 3D Shapes - EASY
class Solution:
    def projectionArea(self, grid: List[List[int]]) -> int:
        n = len(grid)
        a = sum(v > 0 for r in grid for v in r)
        b = sum(max(r) for r in grid)
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


# 886 - Possible Bipartition - MEDIUM
class Solution:
    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        g = [[] * n for _ in range(n)]
        for a, b in dislikes:
            g[a - 1].append(b - 1)
            g[b - 1].append(a - 1)
        color = [0] * n

        def dfs(x: int, c: int) -> bool:
            color[x] = c
            return all(color[y] != c and (color[y] or dfs(y, 3 - c)) for y in g[x])

        return all(c or dfs(i, 1) for i, c in enumerate(color))

    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        g = [[] for _ in range(n)]
        for a, b in dislikes:
            g[a - 1].append(b - 1)
            g[b - 1].append(a - 1)
        color = [0] * n
        for i, c in enumerate(color):
            if not c:
                q = collections.deque([i])
                color[i] = 1
                while q:
                    x = q.popleft()
                    for y in g[x]:
                        if color[y] == color[x]:
                            return False
                        if not color[y]:
                            color[y] = 3 - color[x]
                            q.append(y)
        return True

    def possibleBipartition(self, n: int, dislikes: List[List[int]]) -> bool:
        class UnionFind:
            def __init__(self, n: int) -> None:
                self.p = [i for i in range(n)]
                self.sz = [1] * n

            def find(self, x: int) -> int:
                if self.p[x] != x:
                    self.p[x] = self.find(self.p[x])
                return self.p[x]

            def union(self, x: int, y: int) -> None:
                px = self.find(x)
                py = self.find(y)
                if px == py:
                    return
                self.p[px] = py
                self.sz[py] += self.sz[px]
                return

            def is_connected(self, x: int, y: int) -> bool:
                return self.find(x) == self.find(y)

        g = [[] for _ in range(n)]
        for a, b in dislikes:
            g[a - 1].append(b - 1)
            g[b - 1].append(a - 1)
        uf = UnionFind(n)
        for x, adj in enumerate(g):
            for y in adj:
                uf.union(y, adj[0])
                if uf.is_connected(x, y):
                    return False
        return True


# 888 - Fair Candy Swap - EASY
class Solution:
    def fairCandySwap(self, aliceSizes: List[int], bobSizes: List[int]) -> List[int]:
        aliceSizes.sort()
        bobSizes.sort()
        a = sum(aliceSizes)
        b = sum(bobSizes)
        # sumA - x + y == sumB + x - y
        # (sumA - sumB - 2 * x) // 2
        for x in aliceSizes:
            yy = b - a + 2 * x
            if yy & 1 or yy < 0:
                continue
            y = yy // 2
            p = bisect.bisect_left(bobSizes, y)
            if p < len(bobSizes) and bobSizes[p] == y:
                return [x, bobSizes[p]]
        return []

    def fairCandySwap(self, aliceSizes: List[int], bobSizes: List[int]) -> List[int]:
        d = (sum(aliceSizes) - sum(bobSizes)) // 2
        a = set(aliceSizes)
        ans = []
        for y in bobSizes:
            x = y + d
            if x in a:
                ans = [x, y]
                break
        return ans


# 889 - Construct Binary Tree from Preorder and Postorder Traversal - MEDIUM
class Solution:
    def constructFromPrePost(
        self, preorder: List[int], postorder: List[int]
    ) -> Optional[TreeNode]:
        if not preorder:
            return None
        root = TreeNode(preorder[0])
        if len(postorder) == 1:
            return root
        i = postorder.index(preorder[1])
        root.left = self.constructFromPrePost(preorder[1 : i + 2], postorder[: i + 1])
        root.right = self.constructFromPrePost(preorder[i + 2 :], postorder[i + 1 : -1])
        return root


# 891 - Sum of Subsequence Widths - HARD
class Solution:
    # 子序列不要求连续 + 最大/最小 -> 顺序无关, 可以排序
    # 以 v 作为最大值的子序列, v 左侧元素可以选也可以不选 2**i * v
    # 以 v 作为最小值的子序列, v 右侧元素贡献 - 2**(n-1-i) * v
    def sumSubseqWidths(self, nums: List[int]) -> int:
        nums.sort()
        mod = 10**9 + 7
        ans = 0
        for i, v in enumerate(nums):
            ans = (ans + (pow(2, i, mod) - pow(2, len(nums) - 1 - i, mod)) * v) % mod
        return ans


# 890 - Find and Replace Pattern - MEDIUM
class Solution:
    def findAndReplacePattern(self, words: List[str], pattern: str) -> List[str]:
        ans = []
        for word in words:
            m2p = {}
            p2m = {}
            f = True
            for w, p in zip(word, pattern):
                if w not in m2p and p not in p2m:
                    m2p[w] = p
                    p2m[p] = w
                # elif w not in m2p or p not in p2m or m2p[w] != p or p2m[p] != w:
                elif m2p.get(w, p) != p or p2m.get(p, w) != w:
                    f = False
                    break
            if f:
                ans.append(word)
        return ans

    def findAndReplacePattern(self, words: List[str], pattern: str) -> List[str]:
        def match(word: str, pattern: str) -> bool:
            m = {}
            for x, y in zip(word, pattern):
                if x not in m:
                    m[x] = y
                elif m[x] != y:
                    return False
            return True

        return [w for w in words if match(w, pattern) and match(pattern, w)]


# 895 - Maximum Frequency Stack - HARD
class FreqStack:
    # O(1) / O(n)
    def __init__(self):
        self.cnt = collections.defaultdict(int)
        self.st = []  # 把出现次数不同的元素, 压入不同的栈中

    def push(self, val: int) -> None:
        if self.cnt[val] == len(self.st):
            self.st.append([val])
        else:
            self.st[self.cnt[val]].append(val)
        self.cnt[val] += 1
        return

    def pop(self) -> int:
        v = self.st[-1].pop()
        if len(self.st[-1]) == 0:
            self.st.pop()
        self.cnt[v] -= 1
        return v


class FreqStack:
    # O(1) / O(n)
    def __init__(self):
        self.cnt = collections.defaultdict(int)
        self.st = collections.defaultdict(list)
        self.mx = 0

    def push(self, val: int) -> None:
        self.cnt[val] += 1
        f = self.cnt[val]
        self.st[f].append(val)
        self.mx = max(self.mx, f)

    def pop(self) -> int:
        v = self.st[self.mx].pop()
        self.cnt[v] -= 1
        self.mx -= 1 if len(self.st[self.mx]) == 0 else 0
        return v


class FreqStack:
    # O(logn) / O(n)
    def __init__(self):
        self.cnt = collections.defaultdict(int)
        self.q = []
        self.ts = 0  # time stamp

    def push(self, val: int) -> None:
        self.ts += 1
        self.cnt[val] += 1
        heapq.heappush(self.q, (-self.cnt[val], -self.ts, val))

    def pop(self) -> int:
        v = heapq.heappop(self.q)[2]
        self.cnt[v] -= 1
        return v


# 899 - Orderly Queue - HARD
class Solution:
    def orderlyQueue(self, s: str, k: int) -> str:
        if k == 1:
            return min(s[i:] + s[:i] for i in range(len(s)))
        return "".join(sorted(s))
