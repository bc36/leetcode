from bisect import bisect
from typing import List
import collections, heapq, functools, itertools, math, sortedcontainers


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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


# 1428 - Leftmost Column with at Least a One - MEDIUM


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

    # O(n * logn) / O(n)
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


class Solution:
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
        def inorder(root, premax):
            if not root:
                return
            if root.val >= premax:
                self.cnt += 1
                premax = root.val
            inorder(root.left, premax)
            inorder(root.right, premax)
            return

        self.cnt = 0
        inorder(root, float("-inf"))
        return self.cnt

    def goodNodes(self, root: TreeNode) -> int:
        def inorder(root, premax):
            if root.val >= premax:
                self.cnt += 1
            if root.left:
                inorder(root.left, max(premax, root.val))
            if root.right:
                inorder(root.right, max(premax, root.val))
            return

        self.cnt = 0
        inorder(root, root.val)
        return self.cnt

    def goodNodes(self, root: TreeNode) -> int:
        def inorder(root, premax):
            if not root:
                return 0
            premax = max(root.val, premax)
            return (
                (root.val >= premax)
                + inorder(root.left, premax)
                + inorder(root.right, premax)
            )

        return inorder(root, root.val)


# 1460 - Make Two Arrays Equal by Reversing Sub-arrays - EASY
class Solution:
    def canBeEqual(self, target: List[int], arr: List[int]) -> bool:
        return collections.Counter(target) == collections.Counter(arr)
        return sorted(target) == sorted(arr)


# 1491 - Average Salary Excluding the Minimum and Maximum Salary - EASY
class Solution:
    def average(self, s: List[int]) -> float:
        return (sum(s) - max(s) - min(s)) / (len(s) - 2)
