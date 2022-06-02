import collections, bisect, functools, math, heapq
from typing import List, Optional


# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 400 - Nth Digit - MEDIUM
class Solution:
    def findNthDigit(self, n: int) -> int:
        k = 1
        while k * (10**k) < n:
            n += 10**k
            k += 1
        return int(str(n // k)[n % k])

    def findNthDigit(self, n):
        n -= 1
        for digits in range(1, 11):
            first = 10 ** (digits - 1)
            if n < 9 * first * digits:
                return int(str(first + n / digits)[n % digits])
            n -= 9 * first * digits


# 402 - Remove K Digits - MEDIUM
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []
        for d in num:
            while stack and k and stack[-1] > d:
                stack.pop()
                k -= 1
            stack.append(d)
        if k > 0:
            stack = stack[:-k]
        return "".join(stack).lstrip("0") or "0"


# 408 - Valid Word Abbreviation - EASY
class Solution:
    def validWordAbbreviation(self, word: str, abbr: str) -> bool:
        i = j = 0
        while i < len(word) and j < len(abbr):
            if abbr[j].isalpha():
                if word[i] != abbr[j]:
                    return False
                i += 1
                j += 1
            else:
                if abbr[j] == "0":
                    return False
                tmp = ""
                while j < len(abbr) and abbr[j].isdigit():
                    tmp = tmp + abbr[j]
                    j += 1
                i += int(tmp)
        return i == len(word) and j == len(abbr)


# 409 - Longest Palindrome - EASY
class Solution:
    def longestPalindrome(self, s: str) -> int:
        cnt = collections.Counter(s)
        ans, odd = 0, False
        for k in cnt:
            if cnt[k] & 1:
                odd = True
            ans += cnt[k] // 2 * 2
        return ans + 1 if odd else ans

    def longestPalindrome(self, s: str) -> int:
        arr = [0] * 128
        for ch in s:
            arr[ord(ch) - ord("a")] += 1
        odd = 0
        for n in arr:
            odd += n & 1
        return len(s) - odd + 1 if odd else len(s)


# 413 - Arithmetic Slices - MEDIUM
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        dp = [0] * len(nums)
        for i in range(1, len(nums) - 1):
            if nums[i - 1] + nums[i + 1] == nums[i] * 2:
                dp[i] = dp[i - 1] + 1
        return sum(dp)

    # (1,2,3)->1 (1,2,3,4)->3 (1,2,3,4,5)->6 (1,2,3,4,5,6)->10
    # add a number to an Arithmetic Slices, each increment is added by 1
    # so there are two ways of understanding:
    # 1. the rule of equal variance series / 2. space optimized dp
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        if len(nums) < 3:
            return 0
        ans = add = 0
        for i in range(2, len(nums)):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                add += 1
                ans += add
            else:
                add = 0
        return ans


# 415 - Add Strings - EASY
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        ans = ""
        i, j, carry = len(num1) - 1, len(num2) - 1, 0
        while i >= 0 or j >= 0 or carry != 0:
            n1 = int(num1[i]) if i >= 0 else 0
            n2 = int(num2[j]) if j >= 0 else 0
            tmp = n1 + n2 + carry
            carry = tmp // 10
            ans = str(tmp % 10) + ans
            i, j = i - 1, j - 1
        return ans


# 417 - Pacific Atlantic Water Flow - MEDIUM
class Solution:
    # O(mn) / O(mn)
    def pacificAtlantic(self, h: List[List[int]]) -> List[List[int]]:
        def dfs(x, y, ocean):
            s.add((x, y))
            if ocean == "p":
                pac[x][y] = 1
            else:
                atl[x][y] = 1
            for nx, ny in [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]:
                if (
                    0 <= nx < m
                    and 0 <= ny < n
                    and (nx, ny) not in s
                    and h[nx][ny] >= h[x][y]
                ):
                    dfs(nx, ny, ocean)

        m = len(h)
        n = len(h[0])
        s = set()
        ans = []
        pac = [[0] * n for _ in range(m)]
        atl = [[0] * n for _ in range(m)]
        for i in range(m):
            dfs(i, 0, "p")
        for j in range(1, n):
            dfs(0, j, "p")
        s.clear()
        for i in range(m):
            dfs(i, n - 1, "a")
        for j in range(n - 1):
            dfs(m - 1, j, "a")
        for i in range(m):
            for j in range(n):
                if pac[i][j] == atl[i][j] == 1:
                    ans.append([i, j])
        return ans

    def pacificAtlantic(self, h: List[List[int]]) -> List[List[int]]:
        m = len(h)
        n = len(h[0])
        pac = [[0] * n for _ in range(m)]  # the point that pac water can reach
        atl = [[0] * n for _ in range(m)]
        ans = []

        def dfs(h, can, r, c):
            if can[r][c]:
                return
            can[r][c] = 1
            if pac[r][c] and atl[r][c]:
                ans.append([r, c])
            if r - 1 >= 0 and h[r - 1][c] >= h[r][c]:
                dfs(h, can, r - 1, c)
            if r + 1 < m and h[r + 1][c] >= h[r][c]:
                dfs(h, can, r + 1, c)
            if c - 1 >= 0 and h[r][c - 1] >= h[r][c]:
                dfs(h, can, r, c - 1)
            if c + 1 < n and h[r][c + 1] >= h[r][c]:
                dfs(h, can, r, c + 1)
            return

        for i in range(m):
            dfs(h, pac, i, 0)  # left
            dfs(h, atl, i, n - 1)  # right
        for j in range(n):
            dfs(h, pac, 0, j)  # up
            dfs(h, atl, m - 1, j)  # down
        return ans

    def pacificAtlantic(self, h: List[List[int]]) -> List[List[int]]:
        m = len(h)
        n = len(h[0])
        p_vis = set()
        a_vis = set()

        def dfs(s: set, x: int, y: int):
            s.add((x, y))
            for nx, ny in [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]:
                if (
                    0 <= nx < m
                    and 0 <= ny < n
                    and (nx, ny) not in s
                    and h[nx][ny] >= h[x][y]
                ):
                    dfs(s, nx, ny)

        for i in range(m):
            dfs(p_vis, i, 0)
            dfs(a_vis, i, n - 1)
        for j in range(n):
            dfs(p_vis, 0, j)
            dfs(a_vis, m - 1, j)
        return list(p_vis.intersection(a_vis))


# 419 - Battleships in a Board - MEDIUM
class Solution:
    def countBattleships(self, board):
        total = 0
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == "X":
                    flag = 1
                    if j > 0 and board[i][j - 1] == "X":
                        flag = 0
                    if i > 0 and board[i - 1][j] == "X":
                        flag = 0
                    total += flag
        return total


# 421 - Maximum XOR of Two Numbers in an Array - MEDIUM
class Solution:
    # 1000ms
    def findMaximumXOR(self, nums: List[int]) -> int:
        ans = 0
        for i in reversed(range(32)):
            prefixes = set([x >> i for x in nums])
            ans <<= 1
            candidate = ans + 1
            for p in prefixes:
                if candidate ^ p in prefixes:
                    ans = candidate
                    break
        return ans

    # 300ms
    def findMaximumXOR(self, nums: List[int]) -> int:
        ans = 0
        max_len = max(nums).bit_length()
        if not max_len:
            return 0
        k = 1 << (max_len - 1)
        while k:
            seen = set()
            ans ^= k
            for num in nums:
                seen.add(cur := (ans & num))
                if cur ^ ans in seen:
                    break
            else:
                ans ^= k
            k >>= 1
        return ans


# 423 - Reconstruct Original Digits from English - MEDIUM
class Solution:
    def originalDigits(self, s: str) -> str:
        n0 = s.count("z")
        n2 = s.count("w")
        n4 = s.count("u")
        n6 = s.count("x")
        n8 = s.count("g")
        n1 = s.count("o") - n0 - n2 - n4
        n3 = s.count("t") - n2 - n8
        n5 = s.count("f") - n4
        n7 = s.count("s") - n6
        n9 = s.count("i") - n5 - n6 - n8

        ns = (n0, n1, n2, n3, n4, n5, n6, n7, n8, n9)
        return "".join((str(i) * n for i, n in enumerate(ns)))


# 426 - Convert Binary Search Tree to Sorted Doubly Linked List - MEDIUM
# inorder, bfs
class Solution:
    def treeToDoublyList(self, root: "Node") -> "Node":
        if not root:
            return None
        dummy = Node(-1)
        pre = dummy
        stack, node = [], root
        while stack or node:
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            # node.left, prev.right, prev = prev, node, node
            node.left = pre
            pre.right = node
            pre = node
            node = node.right
        dummy.right.left, pre.right = pre, dummy.right
        return dummy.right


# 427 - Construct Quad Tree - MEDIUM
# Definition for a QuadTree node.
class QNode:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight


class Solution:
    def construct(self, grid: List[List[int]]) -> "QNode":
        def dfs(r0: int, c0: int, r1: int, c1: int) -> "QNode":
            if all(
                grid[i][j] == grid[r0][c0] for i in range(r0, r1) for j in range(c0, c1)
            ):
                return QNode(grid[r0][c0], True, None, None, None, None)
            return QNode(
                None,  # 'val' can be anything if the node is not a leaf
                False,
                dfs(r0, c0, (r0 + r1) // 2, (c0 + c1) // 2),
                dfs(r0, (c0 + c1) // 2, (r0 + r1) // 2, c1),
                dfs((r0 + r1) // 2, c0, r1, (c0 + c1) // 2),
                dfs((r0 + r1) // 2, (c0 + c1) // 2, r1, c1),
            )

        return dfs(0, 0, len(grid), len(grid))

    def construct(self, grid: List[List[int]]) -> "QNode":
        n = len(grid)
        pre = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                pre[i][j] = (
                    pre[i - 1][j]
                    + pre[i][j - 1]
                    - pre[i - 1][j - 1]
                    + grid[i - 1][j - 1]
                )

        def getSum(r0: int, c0: int, r1: int, c1: int) -> int:
            return pre[r1][c1] - pre[r1][c0] - pre[r0][c1] + pre[r0][c0]

        def dfs(r0: int, c0: int, r1: int, c1: int) -> "QNode":
            total = getSum(r0, c0, r1, c1)
            if total == 0:
                return QNode(False, True)
            if total == (r1 - r0) * (c1 - c0):
                return QNode(True, True)
            return QNode(
                True,  # 'val' can be anything if the node is not a leaf
                False,
                dfs(r0, c0, (r0 + r1) // 2, (c0 + c1) // 2),
                dfs(r0, (c0 + c1) // 2, (r0 + r1) // 2, c1),
                dfs((r0 + r1) // 2, c0, r1, (c0 + c1) // 2),
                dfs((r0 + r1) // 2, (c0 + c1) // 2, r1, c1),
            )

        return dfs(0, 0, n, n)


# 429 - N-ary Tree Level Order Traversal - MEDIUM
class Solution:
    def levelOrder(self, root: "Node") -> List[List[int]]:
        ans = []

        def dfs(root: "Node", depth: int):
            if not root:
                return
            if len(ans) <= depth:
                ans.append([])
            ans[depth].append(root.val)
            for ch in root.children:
                dfs(ch, depth + 1)
            return

        dfs(root, 0)
        return ans

    def levelOrder(self, root: "Node") -> List[List[int]]:
        ans = []

        def dfs(root: "Node", lv: int):
            if not root:
                return
            if lv == len(ans):
                ans.append([root.val])
            else:
                ans[lv].append(root.val)
            for ch in root.children:
                dfs(ch, lv + 1)
            return

        dfs(root, 0)
        return ans

    def levelOrder(self, root: "Node") -> List[List[int]]:
        if not root:
            return []
        q = [root]
        ans = []
        while q:
            nxt = []
            cur = []
            for node in q:
                cur.append(node.val)
                nxt += [child for child in node.children]
            q = nxt
            ans.append(cur)
        return ans

    def levelOrder(self, root: "Node") -> List[List[int]]:
        if not root:
            return []
        q = [root]
        ans = []
        while q:
            ans.append([node.val for node in q])
            q = [ch for node in q for ch in node.children]
        return ans


# 432 - All O`one Data Structure - HARD
# TODO


# 433 - Minimum Genetic Mutation - MEDIUM
class Solution:
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        bk = set(bank)
        seen = set()
        q = [start]
        step = 0
        if end not in bk:
            return -1
        while q:
            nxt = []
            for n in q:
                for i in range(8):
                    for g in ["A", "T", "C", "G"]:
                        a = n[:i] + g + n[i + 1 :]
                        if a == end:
                            return step + 1
                        if a in bk and a not in seen:
                            nxt.append(a)
                            seen.add(a)
            q = nxt
            step += 1
        return -1


# 435 - Non-overlapping Intervals - MEDIUM
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: x[1])
        ans = 0
        mxr = -math.inf
        for l, r in intervals:
            if mxr <= l:
                mxr = r
            # elif mxr <= r:
            else:
                ans += 1
        return ans

    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals = sorted(intervals, key=lambda x: x[1])
        mxr = intervals[0][1]
        can = 1
        for i in range(1, len(intervals)):
            if intervals[i][0] >= mxr:
                can += 1
                mxr = intervals[i][1]
        return len(intervals) - can


# 437 - Path Sum III - MEDIUM
class Solution:
    # O(n ^ 2) / O(n)
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        if not root:
            return 0

        def dfs(root: TreeNode, t: int):
            if not root:
                return 0
            ans = 0
            if root.val == t:
                ans += 1
            ans += dfs(root.left, t - root.val) + dfs(root.right, t - root.val)
            return ans

        ans = dfs(root, targetSum)
        ans += self.pathSum(root.left, targetSum)
        ans += self.pathSum(root.right, targetSum)
        return ans

    # O(n) / O(n), prefix sum
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        pre = collections.defaultdict(int)
        pre[0] = 1

        def dfs(root: TreeNode, cur: int):
            if not root:
                return 0
            ans = 0
            cur += root.val
            ans += pre[cur - targetSum]
            pre[cur] += 1
            ans += dfs(root.left, cur) + dfs(root.right, cur)
            pre[cur] -= 1
            return ans

        return dfs(root, 0)


# 438 - Find All Anagrams in a String - MEDIUM
class Solution:
    # sliding window + list
    def findAnagrams(self, s: str, p: str) -> List[int]:
        if len(s) < len(p):
            return []
        ans, ss, pp = [], [0] * 26, [0] * 26
        for i in range(len(p)):
            ss[ord(s[i]) - ord("a")] += 1
            pp[ord(p[i]) - ord("a")] += 1
        if ss == pp:
            ans.append(0)
        k = len(p)
        for i in range(len(p), len(s)):
            ss[ord(s[i]) - ord("a")] += 1
            ss[ord(s[i - k]) - ord("a")] -= 1
            if ss == pp:
                ans.append(i - k + 1)
        return ans

    # sliding window + two pointers
    def findAnagrams(self, s: str, p: str) -> List[int]:
        ans = []
        if len(s) < len(p):
            return ans
        p_cnt = [0] * 26
        s_cnt = [0] * 26
        for i in range(len(p)):
            p_cnt[ord(p[i]) - ord("a")] += 1

        left = 0
        for right in range(len(s)):
            cur_right = ord(s[right]) - ord("a")
            s_cnt[cur_right] += 1
            while s_cnt[cur_right] > p_cnt[cur_right]:
                # move left pointer to satisfy 's_cnt[cur_right] == p_cnt[cur_right]'
                cur_left = ord(s[left]) - ord("a")
                s_cnt[cur_left] -= 1
                left += 1
            if right - left + 1 == len(p):
                ans.append(left)
        return ans


# 440 - K-th Smallest in Lexicographical Order - HARD
class Solution:
    # O(logn * logn) / O(1)
    def findKthNumber(self, n: int, k: int) -> int:
        def getCnt(first: int) -> int:
            cnt = 0
            last = first
            while first <= n:
                cnt += min(last, n) - first + 1
                first *= 10
                last = last * 10 + 9
            return cnt

        cnt = prefix = 1
        while cnt < k:
            add = getCnt(prefix)
            if cnt + add > k:
                prefix *= 10
                cnt += 1
            else:
                prefix += 1
                cnt += add
        return prefix

    def findKthNumber(self, n: int, k: int) -> int:
        def cal_steps(n1, n2):
            step = 0
            while n1 <= n:
                step += min(n2, n + 1) - n1
                n1 *= 10
                n2 *= 10
            return step

        cur = 1
        k -= 1
        while k:
            steps = cal_steps(cur, cur + 1)
            if steps <= k:
                k -= steps
                cur += 1
            else:
                k -= 1
                cur *= 10
        return cur


# 441 - Arranging Coins - EASY
class Solution:
    def arrangeCoins(self, n: int) -> int:
        for i in range(n):
            i += 1
            n -= i
            if n == 0:
                return i
            if n < 0:
                return i - 1


# 442 - Find All Duplicates in an Array - MEDIUM
class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        s = set()
        ans = list()
        for n in nums:
            if n in s:
                ans.append(n)
            else:
                s.add(n)
        return ans

    def findDuplicates(self, nums: List[int]) -> List[int]:
        ans = []
        for n in nums:
            # using the sign of number in nums as a marker
            # using the input array itself as a hash
            n = abs(n)
            if nums[n - 1] > 0:
                nums[n - 1] = -nums[n - 1]  # visited
            else:
                ans.append(n)  # the number is already negative
        return ans

    def findDuplicates(self, nums: List[int]) -> List[int]:
        for i in range(len(nums)):
            while nums[i] != nums[nums[i] - 1]:
                nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
        return [num for i, num in enumerate(nums) if num - 1 != i]


# 443 - String Compression - MEDIUM
class Solution:
    def compress(self, chars: List[str]) -> int:
        i = j = 0  # read / write
        while i < len(chars):
            chars[j] = chars[i]
            j += 1
            count = 0
            while i < len(chars) and chars[i] == chars[j - 1]:
                count += 1
                i += 1
            if count > 1:
                for ch in str(count):
                    chars[j] = ch
                    j += 1
        return j

    def compress(self, chars: List[str]) -> int:
        l = r = 0
        while r < len(chars):
            count = 1
            while r + 1 < len(chars) and chars[r] == chars[r + 1]:
                count += 1
                r += 1
            chars[l] = chars[r]
            if count > 1:
                s = str(count)
                chars[l + 1 : l + 1 + len(s)] = s
                l += len(s)
            l += 1
            r += 1
        return l


# 448 - Find All Numbers Disappeared in an Array - EASY
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        n = [0] * len(nums)
        for i in range(len(nums)):
            n[nums[i] - 1] = 1
        ans = []
        for i in range(len(n)):
            if n[i] == 0:
                ans.append(i + 1)
        return ans

    # marker the scaned number as negative
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        for i in range(len(nums)):
            index = abs(nums[i]) - 1
            nums[index] = -abs(nums[index])
        return [i + 1 for i in range(len(nums)) if nums[i] > 0]


# 449 - Serialize and Deserialize BST - MEDIUM
class Codec:
    # O(nlogn) / O(n)
    def serialize(self, root: TreeNode) -> str:
        """Encodes a tree to a single string."""
        arr = []

        def preorder(root: TreeNode):
            if not root:
                return
            arr.append(str(root.val))
            preorder(root.left)
            preorder(root.right)
            return

        preorder(root)
        return ",".join(arr)

    def deserialize(self, data: str) -> TreeNode:
        """Decodes your encoded data to tree."""
        if not data:
            return

        def build(p: List[int], i: List[int]) -> TreeNode:
            if not p:
                return
            mid = p[0]
            idx = i.index(mid)
            root = TreeNode(mid)
            root.left = build(p[1 : idx + 1], i[:idx])
            root.right = build(p[idx + 1 :], i[idx + 1 :])
            return root

        preorder = list(map(int, data.split(",")))
        inorder = sorted(preorder)
        return build(preorder, inorder)


class Codec:
    # O(n) / O(n)
    def serialize(self, root: TreeNode) -> str:
        arr = []

        def postorder(root: TreeNode) -> None:
            if root is None:
                return
            postorder(root.left)
            postorder(root.right)
            arr.append(root.val)
            return

        postorder(root)
        return " ".join(map(str, arr))

    def deserialize(self, data: str) -> TreeNode:
        arr = list(map(int, data.split()))

        def build(lower: int, upper: int) -> TreeNode:
            if arr == [] or arr[-1] < lower or arr[-1] > upper:
                return None
            val = arr.pop()
            root = TreeNode(val)
            root.right = build(val, upper)
            root.left = build(lower, val)
            return root

        return build(-math.inf, math.inf)


# 450 - Delete Node in a BST - MEDIUM
class Solution:
    # O(n ^ 2) / O(n), similar to 105. Construct Binary Tree from Preorder and Inorder Traversal
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        preorder = []

        def helper(root: TreeNode, k: int):
            if not root:
                return
            if root.val != k:
                preorder.append(root.val)
            helper(root.left, k)
            helper(root.right, k)
            return

        helper(root, key)
        inorder = sorted(preorder)

        def build(preorder: List[int], inorder: List[int]) -> TreeNode:
            if not preorder:
                return
            root = TreeNode(preorder[0])
            idx = inorder.index(root.val)  # slow
            root.left = build(preorder[1 : idx + 1], inorder[:idx])
            root.right = build(preorder[idx + 1 :], inorder[idx + 1 :])
            return root

        return build(preorder, inorder)

    # O(nlogn) / O(n)
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        preorder = []

        def helper(root: TreeNode, k: int):
            if not root:
                return
            if root.val != k:
                preorder.append(root.val)
            helper(root.left, k)
            helper(root.right, k)
            return

        helper(root, key)
        inorder = sorted(preorder)
        d = {v: i for i, v in enumerate(inorder)}

        def helper(p: int, ileft: int, iright: int) -> TreeNode:
            if ileft > iright:
                return None
            root = TreeNode(preorder[p])
            idx = d[preorder[p]]
            root.left = helper(p + 1, ileft, idx - 1)
            root.right = helper(p + idx - ileft + 1, idx + 1, iright)
            return root

        return helper(0, 0, len(inorder) - 1)

    # O(n) / O(n)
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root:
            return
        if root.val == key:
            if not root.left or not root.right:
                return root.right or root.left
            # Replaces root with a successor
            # (the smallest node larger than root, that is, the smallest node in its right subtree)
            # as the new root
            p = root.right
            while p.left:
                p = p.left
            p.right = self.deleteNode(root.right, p.val)
            p.left = root.left
            root = p
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            root.left = self.deleteNode(root.left, key)
        return root

    # O(n) / O(n)
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        def successor(root: TreeNode) -> int:
            # One step right and then always left
            root = root.right
            while root.left:
                root = root.left
            return root.val

        def predecessor(root: TreeNode) -> int:
            # One step left and then always right
            root = root.left
            while root.right:
                root = root.right
            return root.val

        if not root:
            return None
        if key > root.val:
            root.right = self.deleteNode(root.right, key)
        elif key < root.val:
            root.left = self.deleteNode(root.left, key)
        else:
            if not (root.left or root.right):
                root = None
            elif root.right:
                root.val = successor(root)
                root.right = self.deleteNode(root.right, root.val)
            else:
                root.val = predecessor(root)
                root.left = self.deleteNode(root.left, root.val)
        return root


# 452 - Minimum Number of Arrows to Burst Balloons - MEDIUM
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points = sorted(points, key=lambda x: x[1])
        ans, end = 0, float("-inf")
        for p in points:
            if p[0] > end:
                ans += 1
                end = p[1]
        return ans


# 454 - 4Sum II - MEDIUM
class Solution:
    def fourSumCount(
        self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]
    ) -> int:
        dic = collections.defaultdict(int)
        for n1 in nums1:
            for n2 in nums2:
                dic[n1 + n2] += 1
        ans = 0
        for n3 in nums3:
            for n4 in nums4:
                ans += dic[-n3 - n4]
        return ans

    def fourSumCount(
        self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]
    ) -> int:
        ab = collections.Counter(a + b for a in nums1 for b in nums2)
        return sum(ab[-c - d] for c in nums3 for d in nums4)


# 461 - Hamming Distance - EASY
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        ans = 0
        while x != 0 and y != 0:
            if x & 1 != y & 1:
                ans += 1
            x >>= 1
            y >>= 1
        while x != 0:
            if x & 1:
                ans += 1
            x >>= 1
        while y != 0:
            if y & 1:
                ans += 1
            y >>= 1
        return ans


# 467 - Unique Substrings in Wraparound String - MEDIUM
class Solution:
    # O(n) / O(26)
    def findSubstringInWraproundString(self, p: str) -> int:
        d = collections.defaultdict(int)
        l = 0
        for i, c in enumerate(p):
            # if i > 0 and (ord(c) - ord(p[i - 1])) % 26 == 1:
            if i > 0 and ord(p[i - 1]) + 1 == ord(c) or (p[i - 1] == "z" and c == "a"):
                l += 1
            else:
                l = 1
            d[c] = max(d[c], l)
        return sum(d.values())


# 473 - Matchsticks to Square - MEDIUM
class Solution:
    # O(4 ** n) / O(n)
    def makesquare(self, m: List[int]) -> bool:
        if sum(m) % 4 != 0:
            return False
        t = sum(m) // 4
        self.ans = False
        m = sorted(m, reverse=True)

        def dfs(a, b, c, d, i):
            if self.ans or i == len(m) and a == b == c == d == t:
                self.ans = True
                return
            if a + m[i] <= t:
                dfs(a + m[i], b, c, d, i + 1)
            if b + m[i] <= t:
                dfs(a, b + m[i], c, d, i + 1)
            if c + m[i] <= t:
                dfs(a, b, c + m[i], d, i + 1)
            if d + m[i] <= t:
                dfs(a, b, c, d + m[i], i + 1)
            return

        dfs(0, 0, 0, 0, 0)
        return self.ans

    # O(4 ** n) / O(n)
    def makesquare(self, m: List[int]) -> bool:
        if sum(m) % 4:
            return False
        t = sum(m) // 4
        m.sort(reverse=True)  # speed up, if not reverse, TLE
        edges = [0] * 4

        """
        @functools.lru_cache(None)
        failed in this case: [5,5,5,5,4,4,4,4,3,3,3,3]
        the decorator seems to prevent subsequent computation results to modify previous results
        however, the memorized search is originally intended to be used to compute constant results, but in the case of heavy computations.
        """

        def dfs(i) -> bool:
            if i == len(m):
                return True
            for e in range(4):
                if edges[e] + m[i] <= t:
                    # speed up, why?
                    # cuz if the former failed, the same value will fail again
                    if e == 0 or edges[e] != edges[e - 1]:
                        edges[e] += m[i]
                        if dfs(i + 1):
                            return True
                        edges[e] -= m[i]
            return False

        return dfs(0)

    # O(n * 2^n) / O(2^n), hard
    def makesquare(self, m: List[int]) -> bool:
        if sum(m) % 4 != 0:
            return False
        t = sum(m) // 4

        @functools.lru_cache(None)
        def dfs(state: int, cur: int):
            if cur == t:
                cur = 0
                if state == (1 << len(m)) - 1:
                    return True
            for i in range(len(m)):
                if not 1 << i & state and cur + m[i] <= t:
                    if dfs(1 << i | state, cur + m[i]):
                        return True
            return False

        return dfs(0, 0)


# 479 - Largest Palindrome Product - HARD
class Solution:
    def largestPalindrome(self, n: int) -> int:
        if n == 1:
            return 9
        upper = 10**n - 1
        for left in range(upper, upper // 10, -1):
            p = x = left
            while x:
                p = p * 10 + x % 10  # 'p' is a Palindrome
                x //= 10
            x = upper
            while x * x >= p:
                if p % x == 0:
                    return p % 1337
                x -= 1
        return


# 475 - Heaters - MEDIUM
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        heaters = heaters + [float("-inf"), float("inf")]
        houses.sort()
        heaters.sort()
        ans, i = 0, 0
        for h in houses:
            while h > heaters[i + 1]:
                i += 1
            dis = min(h - heaters[i], heaters[i + 1] - h)
            ans = max(ans, dis)
        return ans


# 476 - Number Complement - EASY
class Solution:
    def findComplement(self, num: int) -> int:
        mask = num
        mask |= mask >> 1
        mask |= mask >> 2
        mask |= mask >> 4
        mask |= mask >> 8
        mask |= mask >> 16
        return num ^ mask

    def findComplement(self, num: int) -> int:
        a = 1  # sum is -1
        while True:
            if num >= a:
                a <<= 1
            else:
                return a - num - 1


# 489 - Robot Room Cleaner - HARD - PREMIUM
class Solution:
    def cleanRoom(self, robot):
        dirs = [-1, 0, 1, 0, -1]
        seen = set()

        def dfs(x, y, d):
            robot.clean()
            seen.add((x, y))
            for i in range(4):
                cur = (i + d) % 4
                nx, ny = x + dirs[cur], y + dirs[cur + 1]
                if (nx, ny) not in seen and robot.move():
                    dfs(nx, ny, cur)
                    robot.turnRight()
                    robot.turnRight()
                    robot.move()
                    robot.turnLeft()
                    robot.turnLeft()
                robot.turnRight()

        dfs(robot.row, robot.col, 0)


# 495 - Teemo Attacking - EASY
class Solution:
    def findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:
        ans = 0
        for i in range(1, len(timeSeries)):
            ans += min(duration, timeSeries[i] - timeSeries[i - 1])
        return ans + duration

    # reduce the number of function calls can speed up the operation
    def findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:
        ans = 0
        lastTime = timeSeries[0]
        for i in timeSeries[1:]:
            if i - lastTime > duration:
                ans += duration
            else:
                ans += i - lastTime
            lastTime = i
        return ans + duration


# 496 - Next Greater Element I - EASY
class Solution:
    # brutal-force solution
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        m, n = len(nums1), len(nums2)
        ret = [0] * m
        for i in range(m):
            j = nums2.index(nums1[i])
            k = j + 1
            while k < n and nums2[k] < nums2[j]:
                k += 1
            ret[i] = nums2[k] if k < n else -1
        return ret

    # stack
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack = []
        dic = {}  # save the next greater element
        for i in range(len(nums2) - 1, -1, -1):
            while stack and nums2[i] > stack[-1]:
                stack.pop()
            dic[nums2[i]] = -1 if len(stack) == 0 else stack[-1]
            stack.append(nums2[i])
        return [dic[n1] for n1 in nums1]

    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack = []
        dic = {}  # save the next greater element
        for n in nums2[::-1]:
            while stack and n > stack[-1]:
                stack.pop()
            if stack:
                dic[n] = stack[-1]
            stack.append(n)
        return [dic.get(n, -1) for n in nums1]
        # stack, dic = [], {}
        # for n in nums2:
        #     while len(stack) and stack[-1] < n:
        #         dic[stack.pop()] = n
        #     stack.append(n)
        # for i in range(len(nums1)):
        #     nums1[i] = dic.get(nums1[i], -1)
        # return nums1
