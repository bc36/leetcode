import bisect, collections, functools, heapq, itertools, math, operator, string
from typing import List, Optional, Tuple, Union
import sortedcontainers


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 1401 - Circle and Rectangle Overlapping - MEDIUM
class Solution:
    # 分解成两个问题, 圆心到 x1 <= x <= x2 的最小值 和 圆心到 y1 <= y <= y2 的最小值
    def checkOverlap(
        self,
        radius: int,
        xCenter: int,
        yCenter: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> bool:
        dist = 0
        if xCenter < x1 or xCenter > x2:
            dist += min((x1 - xCenter) ** 2, (x2 - xCenter) ** 2)
        if yCenter < y1 or yCenter > y2:
            dist += min((y1 - yCenter) ** 2, (y2 - yCenter) ** 2)
        return dist <= radius**2

    def checkOverlap(
        self,
        radius: int,
        xCenter: int,
        yCenter: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> bool:
        dx = max(0, x1 - xCenter, xCenter - x2)
        dy = max(0, y1 - yCenter, yCenter - y2)
        return dx * dx + dy * dy <= radius * radius


# 1403 - Minimum Subsequence in Non-Increasing Order - MEDIUM
class Solution:
    def minSubsequence(self, nums: List[int]) -> List[int]:
        nums.sort(reverse=True)
        arr = [0]
        for v in nums:
            arr.append(v + arr[-1])
        for i in range(len(nums)):
            if arr[i + 1] > arr[-1] - arr[i + 1]:
                return nums[: i + 1]

    def minSubsequence(self, nums: List[int]) -> List[int]:
        nums.sort(reverse=True)
        summ = sum(nums)
        cur = 0
        for i, v in enumerate(nums):
            cur += v
            if cur > summ - cur:
                return nums[: i + 1]


# 1405 - Longest Happy String - MEDIUM
class Solution:
    def longestDiverseString(self, a: int, b: int, c: int) -> str:
        ans, pq = "", []
        if a:
            pq.append([-a, "a"])
        if b:
            pq.append([-b, "b"])
        if c:
            pq.append([-c, "c"])
        heapq.heapify(pq)
        while pq:
            n = len(ans)
            cur = heapq.heappop(pq)
            if n >= 2 and ans[-1] == ans[-2] == cur[1]:
                if not pq:
                    break
                nxt = heapq.heappop(pq)
                ans += nxt[1]
                if nxt[0] + 1 < 0:
                    nxt[0] += 1
                    heapq.heappush(pq, nxt)
                heapq.heappush(pq, cur)
            else:
                ans += cur[1]
                if cur[0] + 1 < 0:
                    cur[0] += 1
                    heapq.heappush(pq, cur)
        return ans

    def longestDiverseString(self, a: int, b: int, c: int) -> str:
        def generate(a, b, c, aa, bb, cc):
            if a < b:
                return generate(b, a, c, bb, aa, cc)
            if b < c:
                return generate(a, c, b, aa, cc, bb)
            if b == 0:
                return min(2, a) * aa
            usedA = min(2, a)
            usedB = 1 if a - usedA >= b else 0
            return (
                usedA * aa + usedB * bb + generate(a - usedA, b - usedB, c, aa, bb, cc)
            )

        return generate(a, b, c, "a", "b", "c")

    def longestDiverseString(
        self, a: int, b: int, c: int, aa="a", bb="b", cc="c"
    ) -> str:
        if a < b:
            return self.longestDiverseString(b, a, c, bb, aa, cc)
        elif b < c:
            return self.longestDiverseString(a, c, b, aa, cc, bb)
        elif b == 0:
            return min(2, a) * aa
        usedA = min(2, a)
        usedB = 1 if a - usedA >= b else 0
        return (
            usedA * aa
            + usedB * bb
            + self.longestDiverseString(a - usedA, b - usedB, c, aa, bb, cc)
        )


# 1408 - String Matching in an Array - EASY
class Solution:
    def stringMatching(self, words: List[str]) -> List[str]:
        ans = []
        s = " ".join(words)
        for w in words:
            if s.count(w) != 1:
                ans.append(w)
        return ans

    def stringMatching(self, words: List[str]) -> List[str]:
        ans = []
        for x in words:
            for y in words:
                if x != y and x in y:
                    ans.append(x)
                    break
        return ans


# 1413 - Minimum Value to Get Positive Step by Step Sum - EASY
class Solution:
    def minStartValue(self, nums: List[int]) -> int:
        ans = cur = 0
        for v in nums:
            cur += v
            ans = min(ans, cur)
        return 1 - ans if ans else 1


# 1414 - Find the Minimum Number of Fibonacci Numbers Whose Sum Is K - MEDIUM
class Solution:
    def findMinFibonacciNumbers(self, k: int) -> int:
        s = [1, 1]
        while s[-1] <= k:
            s.append(s[-2] + s[-1])
        ans = 0
        i = len(s) - 1
        while k != 0:
            if k >= s[i]:
                k -= s[i]
                ans += 1
            i -= 1
        return ans

    def findMinFibonacciNumbers(self, k: int) -> int:
        if k == 0:
            return 0
        f1 = f2 = 1
        while f1 + f2 <= k:
            f1, f2 = f2, f1 + f2
        return self.findMinFibonacciNumbers(k - f2) + 1


# 1417 - Reformat The String - EASY
class Solution:
    def reformat(self, s: str) -> str:
        a = [c for c in s if c.isdigit()]
        b = [c for c in s if c.isalpha()]
        if abs(len(a) - len(b)) >= 2:
            return ""
        if len(a) < len(b):
            a, b = b, a

        ans = [""] * len(s)
        ans[::2], ans[1::2] = a, b
        return "".join(ans)

        ans = ""
        while a or b:
            if a:
                ans += a.pop()
            if b:
                ans += b.pop()
        return ans


# 1419 - Minimum Number of Frogs Croaking - MEDIUM
class Solution:
    def minNumberOfFrogs(self, croakOfFrogs: str) -> int:
        mx = c = r = o = a = 0
        for ch in croakOfFrogs:
            if ch == "c":
                c += 1
            elif ch == "r":
                c -= 1
                r += 1
            elif ch == "o":
                r -= 1
                o += 1
            elif ch == "a":
                o -= 1
                a += 1
            else:
                a -= 1
            if c < 0 or r < 0 or o < 0 or a < 0:
                return -1
            mx = max(mx, c + r + o + a)
        if c + r + o + a != 0:
            return -1
        return mx

    def minNumberOfFrogs(self, croakOfFrogs: str) -> int:
        # pre = dict(pairwise("croakc"[::-1]))
        pre = dict(zip("roakc", "croak"))
        d = collections.defaultdict(int)
        for ch in croakOfFrogs:
            p = pre[ch]
            if d[p]:
                d[p] -= 1
            elif ch != "c":
                return -1
            d[ch] += 1
        if any(d[ch] for ch in "croa"):
            return -1
        return d["k"]


# 1422 - Maximum Score After Splitting a String - EASY
class Solution:
    def maxScore(self, s: str) -> int:
        l = s[0] == "0"
        r = sum(c == "1" for c in s[1:])
        mx = l + r
        for i in range(1, len(s) - 1):
            l += s[i] == "0"
            r -= s[i] == "1"
            mx = max(mx, l + r)
        return mx


# 1428 - Leftmost Column with at Least a One - MEDIUM


# 1431 - Kids With the Greatest Number of Candies - EASY
class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        mx = max(candies)
        return [(c + extraCandies) >= mx for c in candies]


# 1437 - Check If All 1's Are at Least Length K Places Away - EASY
class Solution:
    def kLengthApart(self, nums: List[int], k: int) -> bool:
        pre = -1e5
        for i, v in enumerate(nums):
            if v == 1:
                if i - pre - 1 < k:
                    return False
                pre = i
        return True


# 1438 - Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit - MEDIUM
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        s = sortedcontainers.SortedList()
        l = r = ans = 0
        while r < len(nums):
            s.add(nums[r])
            while s[-1] - s[0] > limit:
                s.remove(nums[l])
                l += 1
            ans = max(ans, r - l + 1)
            r += 1
        return ans

    # O(n) / O(n), monotonic queue
    # think about the difference of two solutions below
    # print the third example in leetcode to figure out why
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        mx = collections.deque()
        mi = collections.deque()
        l = r = ans = 0
        while r < len(nums):
            while mx and nums[mx[-1]] < nums[r]:
                mx.pop()
            while mi and nums[mi[-1]] > nums[r]:
                mi.pop()
            mx.append(r)
            mi.append(r)
            while nums[mx[0]] - nums[mi[0]] > limit:
                if l == mx[0]:
                    mx.popleft()
                if l == mi[0]:
                    mi.popleft()
                l += 1
            ans = max(ans, r - l + 1)
            r += 1
        return ans

    def longestSubarray(self, nums: List[int], limit: int) -> int:
        mx = collections.deque()
        mi = collections.deque()
        l = 0
        for r in range(len(nums)):
            while mx and nums[r] > nums[mx[-1]]:
                mx.pop()
            while mi and nums[r] < nums[mi[-1]]:
                mi.pop()
            mx.append(r)
            mi.append(r)
            if nums[mx[0]] - nums[mi[0]] > limit:
                if mx[0] == l:
                    mx.popleft()
                if mi[0] == l:
                    mi.popleft()
                l += 1
        # the window will keep the same length and slide 1 step right if not satisfied
        # when the window meet the answer, it will keep the length to the end
        return r - l + 1
        # or
        return len(nums) - l

    # O(nlogn) / O(n)
    def longestSubarray(self, A, limit):
        maxq, minq = [], []
        res = i = 0
        for j, a in enumerate(A):
            heapq.heappush(maxq, [-a, j])
            heapq.heappush(minq, [a, j])
            while -maxq[0][0] - minq[0][0] > limit:
                i = min(maxq[0][1], minq[0][1]) + 1
                while maxq[0][1] < i:
                    heapq.heappop(maxq)
                while minq[0][1] < i:
                    heapq.heappop(minq)
            res = max(res, j - i + 1)
        return res


# 1439 - Find the Kth Smallest Sum of a Matrix With Sorted Rows - HARD
class Solution:
    # O(m * (nk + nk*lognk)), brute force
    def kthSmallest(self, mat: List[List[int]], k: int) -> int:
        arr = [0]
        for row in mat:
            arr = sorted([x + r for r in row for x in arr])[:k]
        return arr[-1]

    # O(k * m * logk) / O(k)
    def kthSmallest(self, mat: List[List[int]], k: int) -> int:
        m, n = len(mat), len(mat[0])
        pq = [(sum(r[0] for r in mat), (0,) * m)]
        seen = {(0,) * m}
        while k - 1:
            score, pos = heapq.heappop(pq)
            for i in range(m):
                new = pos[:i] + (pos[i] + 1,) + pos[i + 1 :]
                if pos[i] + 1 < n and new not in seen:
                    seen.add(new)
                    sc = score - mat[i][pos[i]] + mat[i][pos[i] + 1]
                    heapq.heappush(pq, (sc, new))
            k -= 1
        return pq[0][0]

    def kthSmallest(self, mat: List[List[int]], k: int) -> int:
        m = len(mat)
        n = len(mat[0])
        pq = [(sum(mat[i][0] for i in range(m)), [0] * m)]
        seen = {tuple([0] * m)}
        while k > 1:
            score, pos = heapq.heappop(pq)
            for i in range(m):
                pos[i] += 1
                if pos[i] < n and tuple(pos) not in seen:
                    seen.add(tuple(pos))
                    sc = score - mat[i][pos[i] - 1] + mat[i][pos[i]]
                    heapq.heappush(pq, (sc, pos[:]))  # pos[:], copy
                pos[i] -= 1
            k -= 1
        return pq[0][0]

    # O(m * k * logm) / O(m)
    # check() can run up to (m - i + 1) * min(k, n ^ i), 1 <= i <= m times.
    # And n ^ i can go up to k very quickly, so time complexity will be O(m * k)
    def kthSmallest(self, mat: List[List[int]], k: int) -> int:
        def check(target, i, cur):
            if i == len(mat):
                return 1  # why 1: TODO
            count = 0
            for j in range(len(mat[0])):
                val = cur + mat[i][j] - mat[i][0]
                if val <= target:
                    count += check(target, i + 1, val)
                    if count >= k:
                        break
                else:
                    break
            return count

        l = r = 0
        for row in mat:
            l += row[0]
            r += row[-1]
        start = l
        while l < r:
            m = (l + r) // 2
            if check(m, 0, start) >= k:
                r = m
            else:
                l = m + 1
        return l

    def kthSmallest(self, mat: List[List[int]], k: int) -> int:
        m, n = len(mat), len(mat[0])
        left = right = 0
        for i in range(m):
            left += mat[i][0]
            right += mat[i][-1]

        def dfs(mid, i, pre):
            nonlocal num
            if pre > mid or i == m or num > k:
                return
            dfs(mid, i + 1, pre)  # not choose element in this row
            for j in range(1, n):  # choose
                cur = pre - mat[i][0] + mat[i][j]
                if cur <= mid:
                    num += 1
                    dfs(mid, i + 1, cur)
                else:
                    break
            return

        init = left
        while left < right:
            mid = (left + right) // 2
            num = 1
            dfs(mid, 0, init)
            if num >= k:
                right = mid
            else:
                left = mid + 1
        return left


# 1441 - Build an Array With Stack Operations - EASY
class Solution:
    def buildArray(self, target: List[int], n: int) -> List[str]:
        ans = []
        j = 0
        for i in range(1, n + 1):
            ans.append("Push")
            if i < target[j]:
                ans.append("Pop")
            if i == target[j]:
                j += 1
            if j == len(target):
                break
        return ans

    def buildArray(self, target: List[int], n: int) -> List[str]:
        ans = []
        i = 1
        for v in target:
            while i != v:
                ans.extend(["Push", "Pop"])
                i += 1
            ans.append("Push")
            i += 1
        return ans


# 1446 - Consecutive Characters - EASY
class Solution:
    def maxPower(self, s: str) -> int:
        tmp, ans = 1, 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                tmp += 1
                ans = max(tmp, ans)
            else:
                tmp = 1
        return ans

    def maxPower(self, s: str) -> int:
        i, ans = 0, 1
        while i < len(s):
            j = i
            while j < len(s) and s[i] == s[j]:
                j += 1
            ans = max(ans, j - i)
            i = j
        return ans


# 1447 - Simplified Fractions - MEDIUM
class Solution:
    def simplifiedFractions(self, n: int) -> List[str]:
        ans = []
        while n > 1:
            for t in range(1, n):
                if math.gcd(t, n) == 1:
                    ans.append(str(t) + "/" + str(n))
            n -= 1
        return ans

    def gcd(self, a: int, b: int):
        if not b:
            return a
        return self.gcd(b, a % b)

    def gcd(self, a: int, b: int):
        while b:
            a, b = b, a % b
        return a


# 1448 - Count Good Nodes in Binary Tree - MEDIUM
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        def inorder(root: TreeNode, mx: Union[int, float]) -> None:
            if not root:
                return
            if root.val >= mx:
                self.cnt += 1
                mx = root.val
            inorder(root.left, mx)
            inorder(root.right, mx)
            return

        self.cnt = 0
        inorder(root, float("-inf"))
        return self.cnt

    def goodNodes(self, root: TreeNode) -> int:
        def inorder(root: TreeNode, mx: int) -> None:
            if root.val >= mx:
                self.cnt += 1
            if root.left:
                inorder(root.left, max(mx, root.val))
            if root.right:
                inorder(root.right, max(mx, root.val))
            return

        self.cnt = 0
        inorder(root, root.val)
        return self.cnt

    def goodNodes(self, root: TreeNode) -> int:
        def inorder(root: TreeNode, mx: int) -> int:
            if not root:
                return 0
            mx = max(root.val, mx)
            return (root.val >= mx) + inorder(root.left, mx) + inorder(root.right, mx)

        return inorder(root, root.val)

    def goodNodes(self, root: TreeNode, mx=-(10**4)) -> int:
        if not root:
            return 0
        ans = 0
        if root.val >= mx:
            mx = root.val
            ans += 1
        return ans + self.goodNodes(root.left, mx) + self.goodNodes(root.right, mx)


# 1450 - Number of Students Doing Homework at a Given Time - EASY
class Solution:
    def busyStudent(
        self, startTime: List[int], endTime: List[int], queryTime: int
    ) -> int:
        return sum(s <= queryTime <= e for s, e in zip(startTime, endTime))


# 1455 - Check If a Word Occurs As a Prefix of Any Word in a Sentence - EASY
class Solution:
    def isPrefixOfWord(self, sentence: str, searchWord: str) -> int:
        for i, s in enumerate(sentence.split()):
            if s.startswith(searchWord):
                return i + 1
        return -1


# 1457 - Pseudo-Palindromic Paths in a Binary Tree - MEDIUM
class Solution:
    # O(9n) / O(9n)
    def pseudoPalindromicPaths(self, root: Optional[TreeNode]) -> int:
        def dfs(root: TreeNode, cnt: List[int]):
            nonlocal ans
            if not root:
                return
            cnt[root.val] += 1
            dfs(root.left, cnt[::])
            dfs(root.right, cnt[::])
            if not root.left and not root.right:
                odd = 0
                ok = True
                for v in cnt:
                    if v & 1:
                        if odd:
                            ok = False
                            break
                        odd = 1
                ans += ok
            return

        ans = 0
        dfs(root, [0] * 10)
        return ans

    # O(n) / O(n)
    def pseudoPalindromicPaths(self, root: TreeNode) -> int:
        # hashmap
        return

    # O(n) / O(n)
    def pseudoPalindromicPaths(self, root: TreeNode) -> int:
        def dfs(r: TreeNode, mask: int) -> int:
            if not r:
                return 0
            mask ^= 1 << (r.val - 1)
            if not r.left and not r.right:
                return int(not mask & (mask - 1))
            return dfs(r.left, mask) + dfs(r.right, mask)

        return dfs(root, 0)

    def pseudoPalindromicPaths(self, root: TreeNode, mask=0):
        if not root:
            return 0
        mask ^= 1 << (root.val - 1)
        ans = self.pseudoPalindromicPaths(
            root.left, mask
        ) + self.pseudoPalindromicPaths(root.right, mask)
        if root.left == root.right == None:
            if mask & (mask - 1) == 0:
                ans += 1
        return ans


# 1460 - Make Two Arrays Equal by Reversing Sub-arrays - EASY
class Solution:
    def canBeEqual(self, target: List[int], arr: List[int]) -> bool:
        return collections.Counter(target) == collections.Counter(arr)
        return sorted(target) == sorted(arr)


# 1464 - Maximum Product of Two Elements in an Array - EASY
class Solution:
    # O(n^2) / O(1)
    def maxProduct(self, nums: List[int]) -> int:
        return max(
            (v - 1) * (u - 1)
            for i, v in enumerate(nums)
            for j, u in enumerate(nums)
            if i != j
        )

    # O(nlogn) / O(1)
    def maxProduct(self, nums: List[int]) -> int:
        s = sorted(nums)
        return (s[-1] - 1) * (s[-2] - 1)

    def maxProduct(self, nums: List[int]) -> int:
        f = max(nums[0], nums[1])
        s = min(nums[0], nums[1])
        for v in nums[2:]:
            if v > f:
                f, s = v, f
            elif v > s:
                s = v
        return (f - 1) * (s - 1)


# 1470 - Shuffle the Array - EASY
class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        ans = [0] * (2 * n)
        ans[::2] = nums[:n:]
        ans[1::2] = nums[n::]
        return ans

    def shuffle(self, nums: List[int], n: int) -> List[int]:
        nums[::2], nums[1::2] = nums[:n], nums[n:]
        return nums


# 1475 - Final Prices With a Special Discount in a Shop - EASY
class Solution:
    # O(n^2) / O(1)
    def finalPrices(self, prices: List[int]) -> List[int]:
        for i, p in enumerate(prices):
            for j in range(i + 1, len(prices)):
                if p >= prices[j]:
                    prices[i] = p - prices[j]
                    break
        return prices

    # O(n) / O(n)
    def finalPrices(self, prices: List[int]) -> List[int]:
        ans = [0] * len(prices)
        st = [0]
        for i in range(len(prices) - 1, -1, -1):
            while st[-1] != 0 and st[-1] > prices[i]:
                st.pop()
            ans[i] = prices[i] - st[-1]
            st.append(prices[i])
        return ans


# 1483 - Kth Ancestor of a Tree Node - HARD
class TreeAncestor:
    # O(nk) / O(n), 暴力, 显然是 TLE 的
    def __init__(self, n: int, parent: List[int]):
        self.p = parent

    def getKthAncestor(self, node: int, k: int) -> int:
        for _ in range(k):
            node = self.p[node]
            if node == -1:
                return -1
        return node


class TreeAncestor:
    # 一个节点的所有祖先节点 (从根节点到该节点的路径上经过的所有点) 的dfs序 (从根节点以dfs访问的顺序) 都在该节点前
    # 通过BFS对树进行层次遍历, 确定每层有哪些节点
    # 假设要找节点 A 的第 k 个父节点, 节点 A 在第 L 层, 即只需找到第 L - k 层的所有节点中 A 的祖先节点.
    # 又由于 A 的祖先节点在每一层只有一个, 且 dfs 序必然在节点 A 前,
    # 所以可以通过二分查找在第 L - k 层中找比 A 小且距离 A 最近的元素即为答案
    # O(n + klogn) / O(n)
    def __init__(self, n: int, parent: List[int]):
        self.tree = [[] for _ in range(n)]
        for i in range(1, n):
            self.tree[parent[i]].append(i)

        # 构建 dfs 序
        self.o = 0  # order
        self.do = [0] * n  # node X's dfs order
        self.dor = [0] * n  # order to index i, dfs order reverse
        self.dfs(0)

        # 存下每一层的所有节点 (dfs序号) 及每个节点对应的层号, 每一层是一层 dfs 序
        self.layers = []
        self.do2l = [0] * n  # node (X's)'s dfs order to bfs level
        q = [self.do[0]]
        lv = 0
        while q:
            self.layers.append(q)
            new = []
            for do in q:
                self.do2l[do] = lv
                for nei in self.tree[self.dor[do]]:
                    new.append(self.do[nei])
            q = new
            lv += 1
        return

    def dfs(self, x: int) -> None:
        self.do[x] = self.o
        self.dor[self.o] = x
        self.o += 1
        for y in self.tree[x]:
            self.dfs(y)
        return

    def getKthAncestor(self, node: int, k: int) -> int:
        # 先找到要去的层级
        do = self.do[node]
        lv = self.do2l[do]
        if lv - k < 0:
            return -1
        layer = self.layers[lv - k]
        l = 0
        r = len(layer) - 1
        while l < r:
            m = l + r + 1 >> 1
            if layer[m] < do:
                l = m
            else:
                r = m - 1
        return self.dor[layer[l]]


class TreeAncestor:
    # 一个可以优化的点, 在 dfs 时可以同时求当前节点在第几层
    # O(n + klogn) / O(n)
    def __init__(self, n: int, parent: List[int]):
        self.tree = [[] for _ in range(n)]
        for i in range(1, n):
            self.tree[parent[i]].append(i)
        self.o = 0  # order
        self.do = [0] * n  # node X's dfs order
        self.level = [0] * n  # node X's bfs level
        self.layers = [[] for _ in range(n)]  # node X's
        self.dor = [0] * n
        self.dfs(0, 0)

    def dfs(self, x: int, l: int) -> None:
        self.do[x] = self.o
        self.level[x] = l
        self.layers[l].append(self.o)
        self.dor[self.o] = x
        self.o += 1
        for y in self.tree[x]:
            self.dfs(y, l + 1)
        return

    def getKthAncestor(self, node: int, k: int) -> int:
        if self.level[node] - k < 0:
            return -1
        layer = self.layers[self.level[node] - k]
        l = 0
        r = len(layer) - 1
        while l < r:
            m = l + r + 1 >> 1
            if layer[m] < self.do[node]:
                l = m
            else:
                r = m - 1
        return self.dor[layer[l]]

    # TODO dp 版本
    # https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/


# 1486 - XOR Operation in an Array - EASY
class Solution:
    # O(n) / O(n)
    def xorOperation(self, n: int, start: int) -> int:
        return functools.reduce(operator.xor, [start + 2 * i for i in range(n)])
        return functools.reduce(operator.xor, range(start, start + (n << 1) - 1, 2))

    # O(n) / O(1)
    def xorOperation(self, n: int, start: int) -> int:
        ans = 0
        for i in range(n):
            ans ^= start + i * 2
        return ans


# 1487 - Making File Names Unique - MEDIUM
class Solution:
    def getFolderNames(self, names: List[str]) -> List[str]:
        d = collections.defaultdict(int)
        for i, v in enumerate(names):
            if v in d:
                k = d[v]
                while f"{v}({k})" in d:
                    k += 1
                d[v] = k + 1
                names[i] = f"{v}({k})"
            d[names[i]] = 1
        return names


# 1491 - Average Salary Excluding the Minimum and Maximum Salary - EASY
class Solution:
    def average(self, s: List[int]) -> float:
        return (sum(s) - max(s) - min(s)) / (len(s) - 2)
