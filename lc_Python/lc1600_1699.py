import collections, itertools, heapq, functools, math
from typing import List


class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# 1606 - Find Servers That Handled Most Number of Requests - HARD
class Solution:
    # O(nlogk + k)
    def busiestServers(self, k: int, a: List[int], l: List[int]) -> List[int]:
        free = list(range(k))
        busy = []
        req = [0] * k
        for i in range(len(a)):
            while busy and busy[0][0] <= a[i]:
                _, x = heapq.heappop(busy)
                heapq.heappush(free, x + ((i - x - 1) // k + 1) * k)
                # heapq.heappush(free, i + (x - i) % k)
                # # the same as below
                # while x < i:
                #     x += k
                # heapq.heappush(free, x)
            if free:
                x = heapq.heappop(free) % k
                req[x] += 1
                heapq.heappush(busy, (a[i] + l[i], x))
        m = max(req)
        return [i for i in range(len(req)) if req[i] == m]


# 1608 - Special Array With X Elements Greater Than or Equal X - EASY
class Solution:
    def specialArray(self, nums: List[int]) -> int:
        nums.sort(reverse=True)
        for i, v in enumerate(nums):
            if v < i:
                return i
            if v == i:
                return -1
        return len(nums)


# 1609 - Even Odd Tree - MEDIUM
class Solution:
    # bfs
    def isEvenOddTree(self, root: TreeNode) -> bool:
        dq = collections.deque([root])
        is_even = True
        while dq:
            pre = None
            for _ in range(len(dq)):
                n = dq.popleft()
                if is_even:
                    if n.val % 2 == 0:
                        return False
                    if pre and pre.val >= n.val:
                        return False
                else:
                    if n.val % 2 == 1:
                        return False
                    if pre and pre.val <= n.val:
                        return False
                if n.left:
                    dq.append(n.left)
                if n.right:
                    dq.append(n.right)
                pre = n
            is_even = not is_even  # bool value cannot use '~' to inverse
        return True

    def isEvenOddTree(self, root: TreeNode) -> bool:
        l, nodes = 0, [root]
        while nodes:
            nxt, cur = [], float("inf") if l % 2 else 0
            for n in nodes:
                if (
                    (l % 2 == n.val % 2)
                    or (l % 2 and cur <= n.val)
                    or ((not l % 2) and cur >= n.val)
                ):
                    return False
                cur = n.val
                if n.left:
                    nxt.append(n.left)
                if n.right:
                    nxt.append(n.right)
            nodes = nxt
            l += 1
        return True


# 1614 - Maximum Nesting Depth of the Parentheses - EASY
class Solution:
    def maxDepth(self, s: str) -> int:
        ans = left = 0
        for ch in s:
            if ch == "(":
                left += 1
                ans = max(ans, left)
            elif ch == ")":
                left -= 1
        return ans


# 1619 - Mean of Array After Removing Some Elements - EASY
class Solution:
    def trimMean(self, arr: List[int]) -> float:
        arr.sort()
        n = len(arr)
        return sum(arr[n // 20 : -n // 20]) / (n * 0.9)


# 1620 - Coordinate With Maximum Network Quality - MEDIUM
class Solution:
    def bestCoordinate(self, towers: List[List[int]], radius: int) -> List[int]:
        def calc(q: int, x: int, y: int, a: int, b: int) -> int:
            d = ((x - a) ** 2 + (y - b) ** 2) ** 0.5
            return q // (1 + d) if d <= radius else 0

        m = max(v[0] for v in towers)
        n = max(v[1] for v in towers)
        cx = cy = mx = 0
        for x in range(m + 1):
            for y in range(n + 1):
                s = sum(calc(q, x, y, a, b) for a, b, q in towers)
                if s > mx:
                    mx = s
                    cx = x
                    cy = y
        return [cx, cy]


# 1624 - Largest Substring Between Two Equal Characters - EASY
class Solution:
    def maxLengthBetweenEqualCharacters(self, s: str) -> int:
        ans = -1
        arr = [-1] * 26
        for i, c in enumerate(s):
            if arr[ord(c) - 97] == -1:
                arr[ord(c) - 97] = i
            else:
                ans = max(ans, i - arr[ord(c) - 97] - 1)
        return ans


# 1629 - Slowest Key - EASY
class Solution:
    def slowestKey(self, rT: List[int], keys: str) -> str:
        ans, time = keys[0], rT[0]
        for i in range(len(rT) - 1):
            if (
                rT[i + 1] - rT[i] > time
                or rT[i + 1] - rT[i] == time
                and keys[i + 1] > ans
            ):
                time = rT[i + 1] - rT[i]
                ans = keys[i + 1]
        return ans


# 1636 - Sort Array by Increasing Frequency - EASY
class Solution:
    def frequencySort(self, nums: List[int]) -> List[int]:
        # return sorted(nums, key=lambda x: (nums.count(x), -x))
        cnt = collections.Counter(nums)
        return sorted(nums, key=lambda x: (cnt[x], -x))
        return sorted(nums, key=lambda x: (cnt.get(x), -x))


# 1640 - Check Array Formation Through Concatenation - EASY
class Solution:
    def canFormArray(self, arr: List[int], pieces: List[List[int]]) -> bool:
        s = set(map(tuple, pieces))
        l = 0
        for r in range(len(arr)):
            if tuple(arr[l : r + 1]) in s:
                l = r + 1
        return l == len(arr)

    def canFormArray(self, arr: List[int], pieces: List[List[int]]) -> bool:
        try:
            pieces.sort(key=lambda x: arr.index(x[0]))  # 元素不匹配, index ValueError
        except ValueError:
            return False
        i = 0
        for p in pieces:
            for v in p:
                if v == arr[i]:
                    i += 1
                else:
                    return False
        return True


# 1648 - Sell Diminishing-Valued Colored Balls - MEDIUM
class Solution:
    # O(nlogC) / O(1), C = max(inventory)
    def maxProfit(self, inventory: List[int], orders: int) -> int:
        l = 0
        r = max(inventory)
        # find a value where all the balls end up with v or v+1
        while l < r:
            mid = (l + r) // 2
            count = sum(n - mid for n in inventory if n >= mid)
            if count <= orders:
                r = mid
            else:
                l = mid + 1
        fn = lambda x, y: (x + y) * (y - x + 1) // 2
        rest = orders - sum(n - l for n in inventory if n >= l)
        ans = 0
        for n in inventory:
            if n >= l:
                if rest > 0:
                    ans += fn(l, n)
                    rest -= 1
                else:
                    ans += fn(l + 1, n)
        return ans % (10**9 + 7)

    # O(nlogn) / O(1)
    def maxProfit(self, inv: List[int], orders: int) -> int:
        fn = lambda s, e: (e + s) * (e - s + 1) // 2
        inv.sort(reverse=True)
        inv.append(0)
        ans = 0
        cnt = 1  # the number of maximum values
        for i in range(len(inv) - 1):
            if inv[i] > inv[i + 1]:
                if cnt * (inv[i] - inv[i + 1]) <= orders:
                    ans += cnt * fn(inv[i + 1] + 1, inv[i])
                    orders -= cnt * (inv[i] - inv[i + 1])
                else:
                    a, b = divmod(orders, cnt)
                    ans += cnt * fn(inv[i] - a + 1, inv[i])
                    ans += b * (inv[i] - a)
                    break
            cnt += 1
        return ans % (10**9 + 7)


# 1650 - Lowest Common Ancestor of a Binary Tree III - MEDIUM
class Solution:
    def lowestCommonAncestor(self, p: "Node", q: "Node") -> "Node":
        path = set()
        while p:
            path.add(p)
            p = p.parent
        while q not in path:
            q = q.parent
        return q

    # like running in a cycle
    def lowestCommonAncestor(self, p: "Node", q: "Node") -> "Node":
        p1, p2 = p, q
        while p1 != p2:
            p1 = p1.parent if p1.parent else q
            p2 = p2.parent if p2.parent else p
        return p1


# 1652 - Defuse the Bomb - EASY
class Solution:
    def decrypt(self, code: List[int], k: int) -> List[int]:
        n = len(code)
        ans = []
        for i, v in enumerate(code):
            if k > 0:
                if k + i >= n:
                    ans.append(sum(code[i + 1 :] + code[: k + i + 1 - n]))
                else:
                    ans.append(sum(code[i + 1 : i + k + 1]))
            elif k < 0:
                if i + k < 0:
                    ans.append(sum(code[:i] + code[n + k + i :]))
                else:
                    ans.append(sum(code[i + k : i]))
            else:
                ans.append(0)
        return ans

    def decrypt(self, code: List[int], k: int) -> List[int]:
        n = len(code)
        ans = [0 for _ in range(n)]
        c2 = code * 2
        for i in range(n):
            if k > 0:
                ans[i] = sum(c2[i + 1 : i + k + 1])
            if k < 0:
                ans[i] = sum(c2[i + n + k : i + n])
        return ans


# 1656 - Design an Ordered Stream - EASY
class OrderedStream:
    def __init__(self, n: int):
        self.arr = [""] * (n + 2)
        self.ptr = 1

    def insert(self, idKey: int, value: str) -> List[str]:
        self.arr[idKey] = value
        ans = []
        while self.arr[self.ptr]:
            ans.append(self.arr[self.ptr])
            self.ptr += 1
        return ans


# 1662 - Check If Two String Arrays are Equivalent - EASY
class Solution:
    def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
        return "".join(word1) == "".join(word2)


# 1668 - Maximum Repeating Substring - EASY
class Solution:
    # O(n ** 2) / O(n)
    def maxRepeating(self, sequence: str, word: str) -> int:
        s = word
        ans = 0
        while s in sequence:
            ans += 1
            s += word
        return ans

    # O(n ** 2) / O(1)
    def maxRepeating(self, sequence: str, word: str) -> int:
        ans = 0
        for i in range(len(sequence)):
            t = k = 0
            j = i
            while j < len(sequence):
                if sequence[j] == word[k]:
                    j += 1
                    k += 1
                else:
                    break
                if k == len(word):
                    t += 1
                    k = 0
            ans = max(ans, t)
        return ans

    # O(n * m) / O(n)
    def maxRepeating(self, sequence: str, word: str) -> int:
        n = len(sequence)
        m = len(word)
        f = [0] * n
        for i in range(m - 1, n):
            valid = True
            for j in range(m):
                if sequence[i - m + j + 1] != word[j]:
                    valid = False
                    break
            if valid:
                f[i] = (0 if i == m - 1 else f[i - m]) + 1
        return max(f)


# 1672 - Richest Customer Wealth - EASY
class Solution:
    def maximumWealth(self, accounts: List[List[int]]) -> int:
        # return max(sum(a) for a in accounts)
        return max(map(sum, accounts))


# 1676 - Lowest Common Ancestor of a Binary Tree IV - MEDIUM
class Solution:
    def lowestCommonAncestor(
        self, root: "TreeNode", nodes: "List[TreeNode]"
    ) -> "TreeNode":
        nodes = set(nodes)

        def lca(root):
            """Return LCA of nodes."""
            if not root or root in nodes:
                return root
            left, right = lca(root.left), lca(root.right)
            if left and right:
                return root
            return left or right

        return lca(root)


# 1678 - Goal Parser Interpretation - EASY
class Solution:
    def interpret(self, c: str) -> str:
        return c.replace("()", "o").replace("(al)", "al")


# 1679 - Max Number of K-Sum Pairs - MEDIUM
class Solution:
    # O(nlogn) / O(logn)
    def maxOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        ans = l = 0
        r = len(nums) - 1
        while l < r:
            if nums[l] + nums[r] > k:
                r -= 1
            elif nums[l] + nums[r] < k:
                l += 1
            else:
                ans += 1
                l += 1
                r -= 1
        return ans

    # O(n) / O(n)
    def maxOperations(self, nums: List[int], k: int) -> int:
        ans = 0
        cnt = collections.Counter(nums)
        for key, val in cnt.items():
            if key * 2 == k:
                ans += val // 2
            elif key * 2 < k and k - key in cnt:
                ans += min(val, cnt[k - key])
        return ans

    def maxOperations(self, nums: List[int], k: int) -> int:
        cnt = collections.Counter(nums)
        ans = 0
        for key in cnt:
            if key * 2 < k:
                ans += min(cnt[key], cnt.get(k - key, 0))
            elif key * 2 == k:
                ans += cnt[key] // 2
        return ans


# 1688 - Count of Matches in Tournament - EASY
class Solution:
    def numberOfMatches(self, n: int) -> int:
        ans = 0
        while n > 1:
            if n & 1:
                ans += (n - 1) // 2
                n += 1
            else:
                ans += n // 2
            n //= 2
        return ans

    def numberOfMatches(self, n: int) -> int:
        return n - 1


# 1694 - Reformat Phone Number - EASY
class Solution:
    def reformatNumber(self, number: str) -> str:
        number = list(number.replace(" ", "").replace("-", ""))
        ans = []
        t = 0
        p = ""
        for i, v in enumerate(number):
            if t == 0 and i >= len(number) - 4:
                break
            p += v
            t += 1
            if t >= 3:
                ans.append(p)
                t = 0
                p = ""
        if i == len(number) - 4:
            ans.extend([number[-4] + number[-3], number[-2] + number[-1]])
        else:
            ans.append("".join(number[i:]))
        return "-".join(ans)

    def reformatNumber(self, number: str) -> str:
        number = number.replace(" ", "").replace("-", "")
        ans = []
        i = 0
        while i < len(number) - 4:
            ans.append(number[i : i + 3])
            i += 3
        number = number[i:]
        if len(number) == 4:
            ans.extend([number[:2], number[2:]])
        else:
            ans.append(number)
        return "-".join(ans)
