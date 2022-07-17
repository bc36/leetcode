import collections, math, random, bisect, itertools, functools, heapq, re
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


# 500 - Keyboard Row - EASY
# The '<' and '>' operators are testing for strict subsets
class Solution:
    def findWords(self, words):
        line1, line2, line3 = set("qwertyuiop"), set("asdfghjkl"), set("zxcvbnm")
        ret = []
        for word in words:
            w = set(word.lower())
            if w <= line1 or w <= line2 or w <= line3:
                ret.append(word)
        return ret


"""lc 501
BST: internal nodes each store a key greater than all the keys in the node’s left subtree
     and less than those in its right subtree.
Inorder Traversal of BST: ordered sequence
"""


# 501 - Find Mode in Binary Search Tree - EASY
class Solution:
    def findMode(self, root: TreeNode) -> List[int]:
        def inorderBST(root: TreeNode):
            if not root:
                return
            nonlocal ans, count, pre, mx
            inorderBST(root.left)
            if root.val == pre:
                count += 1
            else:
                pre = root.val
                count = 1
            if count == mx:
                ans.append(root.val)
            if count > mx:
                mx = count
                ans = [root.val]
            inorderBST(root.right)
            return

        count = pre = mx = 0
        ans = []
        inorderBST(root)
        return ans


# 504 - Base 7 - EASY
class Solution:
    def convertToBase7(self, num: int) -> str:
        if num == 0:
            return "0"
        arr = []
        f = "-" if num < 0 else ""
        num = abs(num)
        while num:
            arr.append(str(num % 7))
            num //= 7
        return f + "".join(reversed((arr)))


# 506 - Relative Ranks - EASY
class Solution:
    def findRelativeRanks(self, score: List[int]) -> List[str]:
        dic = {n: idx for idx, n in enumerate(sorted(score, reverse=True))}
        medals = ["Gold Medal", "Silver Medal", "Bronze Medal"]
        return [str(dic[i] + 1) if dic[i] >= 3 else medals[dic[i]] for i in score]


# 507 - Perfect Number - EASY
class Solution:
    def checkPerfectNumber(self, num: int) -> bool:
        if num == 1:
            return False
        ans = 0
        for i in range(1, int(math.sqrt(num) + 1)):
            if num % i == 0:
                ans += num // i + i
        return ans == num * 2

    # [0, 10**8] has only 5 perfect numbers
    def checkPerfectNumber(self, num: int) -> bool:
        return num == 6 or num == 28 or num == 496 or num == 8128 or num == 33550336


# 508 - Most Frequent Subtree Sum - MEDIUM
class Solution:
    def findFrequentTreeSum(self, root: TreeNode) -> List[int]:
        def post(root: TreeNode) -> int:
            if not root:
                return 0
            t = root.val + post(root.left) + post(root.right)
            d[t] += 1
            return t

        d = collections.defaultdict(int)
        post(root)
        m = max(d.values())
        return [k for k, v in d.items() if v == m]


# 509 - Fibonacci Number - EASY
class Solution:
    def fib(self, n: int) -> int:
        if n < 2:
            return n
        one, two, ans = 0, 1, 0
        for _ in range(1, n):
            ans = one + two
            one, two = two, ans
        return ans


# 513 - Find Bottom Left Tree Value - MEDIUM
class Solution:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        dq = collections.deque([root])
        ans = 2**31
        while dq:
            ans = dq[0].val
            for _ in range(len(dq)):
                n = dq.popleft()
                if n.left:
                    dq.append(n.left)
                if n.right:
                    dq.append(n.right)
        return ans

    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        def dfs(root: TreeNode, h: int):
            if not root:
                return
            h += 1
            dfs(root.left, h)
            dfs(root.right, h)
            if h > height[0]:
                ans[0] = root.val
                height[0] = h
            return

        ans = [2**32]
        height = [0]
        dfs(root, 0)
        return ans[0]


# 515 - Find Largest Value in Each Tree Row - MEDIUM
class Solution:
    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        ans = []
        dq = collections.deque([root])
        while dq:
            m = -math.inf
            for _ in range(len(dq)):
                n = dq.popleft()
                m = n.val if n.val > m else m
                if n.left:
                    dq.append(n.left)
                if n.right:
                    dq.append(n.right)
            ans.append(m)
        return ans

    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(root: TreeNode, h: int):
            if not root:
                return
            if len(ans) == h:
                ans.append(root.val)
            else:
                ans[h] = root.val if root.val > ans[h] else ans[h]
            dfs(root.left, h + 1)
            dfs(root.right, h + 1)
            return

        ans = []
        dfs(root, 0)
        return ans


# 516 - Longest Palindromic Subsequence - MEDIUM
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        dp = [[0] * len(s) for _ in range(len(s))]
        for i in range(len(s) - 1, -1, -1):
            dp[i][i] = 1
            for j in range(i + 1, len(s)):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        return dp[0][-1]

    # reverse, LCS
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        ss = s[::-1]
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if s[i - 1] == ss[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
        return dp[n][n]


# 518 - Coin Change 2 - MEDIUM
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [1] + [0] * amount
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]
        return dp[-1]


# 519 - Random Flip Matrix - MEDIUM
# TLE
class Solution:
    def __init__(self, m: int, n: int):
        self.zero = [i for i in range(n * m)]
        self.m = m
        self.n = n
        self.total = m * n - 1

    def flip(self) -> List[int]:
        index = random.randint(0, self.total)
        self.total -= 1
        val = self.zero.pop(index)
        return [val // self.n, val % self.n]

    def reset(self) -> None:
        self.zero = [i for i in range(self.n * self.m)]
        self.total = self.m * self.n - 1
        return


# Single sampling
class Solution:
    def __init__(self, m: int, n: int):
        self.total = n * m - 1
        self.m = m
        self.n = n
        self.map = {}

    def flip(self) -> List[int]:
        x = random.randint(0, len(self.zero) - 1)
        self.total -= 1
        index = self.map.get(x, x)
        self.map[x] = self.map.get(self.total, self.total)
        return [index // self.n, index % self.n]

    def reset(self) -> None:
        self.total = self.m * self.n - 1
        self.map.clear()
        return


# Multiple sampling
class Solution:
    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n
        self.total = m * n
        self.flipped = set()

    def flip(self) -> List[int]:
        while (x := random.randint(0, self.total - 1)) in self.flipped:
            pass
        self.flipped.add(x)
        return [x // self.n, x % self.n]

    def reset(self) -> None:
        self.total = self.m * self.n
        self.flipped.clear()


# 520 - Detect Capital - EASY
class Solution:
    # brutal-force
    def detectCapitalUse(self, word: str) -> bool:
        if ord(word[0]) >= 97 and ord(word[0]) <= 122:
            for ch in word:
                if ord(ch) < 97 or ord(ch) > 122:
                    return False
        else:
            if len(word) > 1:
                if ord(word[1]) >= 65 and ord(word[1]) <= 90:
                    for ch in word[1:]:
                        if ord(ch) < 65 or ord(ch) > 90:
                            return False
                else:
                    for ch in word[1:]:
                        if ord(ch) < 97 or ord(ch) > 122:
                            return False
        return True

    def detectCapitalUse(self, word: str) -> bool:
        # Solution 1: word.istitle()
        return word.istitle() or word.isupper() or word.islower()
        # Solution 2:
        # return word[1:] == word[1:].lower() or word == word.upper()


# 521 - Longest Uncommon Subsequence I - EASY
class Solution:
    def findLUSlength(self, a: str, b: str) -> int:
        if a == b:
            return -1
        return max(len(a), len(b))


# 522 - Longest Uncommon Subsequence II - MEDIUM
class Solution:
    # O(n ^ 2 * l) / O(1), l is the average length of each str
    def findLUSlength(self, strs: List[str]) -> int:
        def is_subseq(s1: str, s2: str) -> bool:
            i = j = 0
            while i < len(s1) and j < len(s2):
                if s1[i] == s2[j]:
                    i += 1
                j += 1
            return i == len(s1)

        ans = -1
        for i, s1 in enumerate(strs):
            ok = True
            for j, s2 in enumerate(strs):
                if i != j and is_subseq(s1, s2):
                    ok = False
                    break
            if ok:
                ans = max(ans, len(s1))
        return ans

    def findLUSlength(self, strs: List[str]) -> int:
        def is_subseq(s1: str, s2: str) -> bool:
            # len(s1) <= len(s2)
            i = j = 0
            while i < len(s1) and j < len(s2):
                if s1[i] == s2[j]:
                    i += 1
                j += 1
            return i == len(s1)

        cnt = collections.Counter(strs)
        pairs = sorted(cnt.items(), key=lambda x: len(x[0]), reverse=True)
        vis = set()
        for k, v in pairs:
            if v == 1:
                ok = True
                for x in vis:
                    if is_subseq(k, x):
                        ok = False
                if ok:
                    return len(k)
            else:
                vis.add(k)
        return -1


# 523 - Continuous Subarray Sum - MEDIUM
class Solution:
    # 'cur' calculate the prefix sum remainder of input array 'nums'
    # 'seen' will record the first occurrence of the remainder.
    # If we have seen the same remainder before,
    # it means the subarray sum is a multiple of k
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        seen, cur = {0: -1}, 0
        for i, num in enumerate(nums):
            cur = (cur + num) % abs(k) if k else cur + num
            if i - seen.setdefault(cur, i) > 1:
                return True
        return False

    # Idea: if sum(nums[i:j]) % k == 0 for some i < j
    # then sum(nums[:j]) % k == sum(nums[:i]) % k
    # So we just need to use a dictionary to keep track of
    # sum(nums[:i]) % k and the corresponding index i
    # Once some later sum(nums[:j]) % k == sum(nums[:i]) % k and j - i > 1
    # we return True
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        rmd, sumRmd = {0: -1}, 0
        # why {0: -1}: sum(nums) % k == 0
        for i, num in enumerate(nums):
            sumRmd = (num + sumRmd) % k
            if sumRmd not in rmd:
                rmd[sumRmd] = i
            else:
                if i - rmd[sumRmd] > 1:
                    return True
        return False

    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        presum = itertools.accumulate(nums)
        dic = {0: -1}
        for index, num in enumerate(presum):
            if num % k in dic:
                if index - dic[num % k] > 1:
                    return True
                # do not update the value in dic, or use 'set()'
                continue
            dic[num % k] = index
        return

    # the required length is at least 2,
    # so we just need to insert the mod one iteration later.
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        numSum, pre = 0, 0
        s = set()
        for num in nums:
            numSum += num
            mod = numSum % k
            if mod in s:
                return True
            s.add(pre)
            pre = mod
        return False

    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        preSum = [0]  # length = len(nums) + 1
        for num in nums:
            preSum.append(preSum[-1] + num)
        s = set()
        for i in range(2, len(preSum)):
            s.add(preSum[i - 2] % k)
            if preSum[i] % k in s:
                return True
        return False


# 525 - Contiguous Array - MEDIUM
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        ans = presum = 0
        dic = {}
        for i in range(len(nums)):
            presum += 1 if nums[i] else -1
            if presum == 0:
                ans = i + 1
            elif presum in dic:
                ans = max(i - dic[presum], ans)
            else:
                dic[presum] = i
        return ans

    def findMaxLength(self, nums: List[int]) -> int:
        d = {0: -1}  # trick for presum
        presum = ans = 0
        for i in range(len(nums)):
            presum = presum + 1 if nums[i] else presum - 1
            if presum in d:
                ans = max(ans, i - d[presum])
            else:
                d[presum] = i
        return ans


# 528 - Random Pick with Weight - MEDIUM
# prefix sum + binary search
# seperate [1, total] in len(w) parts, each part has w[i] elements
class Solution:
    def __init__(self, w: List[int]):
        # Calculate the prefix sum to generate a random number
        # The coordinates of the distribution correspond to the size of the number
        self.presum = list(itertools.accumulate(w))

    def pickIndex(self) -> int:
        rand = random.randint(1, self.presum[-1])
        return bisect.bisect_left(self.presum, rand)


class Solution:
    def __init__(self, w: List[int]):
        def pre(w: List[int]) -> List[int]:
            sum = 0
            ans = []
            for i in range(len(w)):
                ans.append(sum + w[i])
                sum += w[i]
            return ans

        self.presum = pre(w)

    def pickIndex(self) -> int:
        rand = random.randint(1, self.presum[-1])
        left, right = 0, len(self.presum) - 1
        while left < right:
            mid = (left + right) // 2
            if self.presum[mid] >= rand:
                right = mid
            else:
                left = mid + 1
        return left


# 532 - K-diff Pairs in an Array - MEDIUM
class Solution:
    # like two sum (lc 1)
    def findPairs(self, nums: List[int], k: int) -> int:
        seen = set()
        pairs = set()
        for n in nums:
            if n - k in seen:
                pairs.add((n - k, n))
            if n + k in seen:
                pairs.add((n, n + k))
            seen.add(n)
        return len(pairs)

    def findPairs(self, nums: List[int], k: int) -> int:
        ans = 0
        cnt = collections.Counter(nums)
        for i in cnt:
            if (k > 0 and i + k in cnt) or (k == 0 and cnt[i] > 1):
                ans += 1
        return ans

    def findPairs(self, nums: List[int], k: int) -> int:
        if k == 0:
            cnt = collections.Counter(nums)
            return sum(True for v in cnt.values() if v >= 2)
        s = set(nums)
        return sum(True for v in set(nums) if v + k in s)


# 535 - Encode and Decode TinyURL - MEDIUM
class Codec:
    def __init__(self):
        self.db = {}
        self.id = 0

    def encode(self, longUrl: str) -> str:
        self.id += 1
        self.db[self.id] = longUrl
        return "http://tinyurl.com/" + str(self.id)

    def decode(self, shortUrl: str) -> str:
        i = shortUrl.rfind("/")
        id = int(shortUrl[i + 1 :])
        return self.db[id]


class Codec:
    def encode(self, longUrl: str) -> str:
        return longUrl

    def decode(self, shortUrl: str) -> str:
        return shortUrl


# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(url))
# 537 - Complex Number Multiplication - MEDIUM
class Solution:
    def complexNumberMultiply(self, num1: str, num2: str) -> str:
        a, b = num1[:-1].split("+")
        c, d = num2[:-1].split("+")
        e = int(a) * int(c) - int(b) * int(d)
        f = int(a) * int(d) + int(b) * int(c)
        return f"{e}+{f}i"


# 539 - Minimum Time Difference - MEDIUM
class Solution:
    def findMinDifference(self, timePoints: List[str]) -> int:
        if len(timePoints) > 1440:
            return 0
        timePoints.sort()
        firstTime = preTime = int(timePoints[0][:2]) * 60 + int(timePoints[0][3:])
        ans = float("inf")
        for t in timePoints[1:]:
            time = int(t[:2]) * 60 + int(t[3:])
            ans = min(ans, time - preTime)
            if ans == 0:
                break
            preTime = time
        ans = min(ans, firstTime + 1440 - preTime)
        return ans


# 540 - Single Element in a Sorted Array - MEDIUM
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            nums[i] ^= nums[i - 1]
        return nums[-1]

    # mid is even: mid + 1 = mid ^ 1
    # mid is odd: mid - 1 = mid ^ 1
    def singleNonDuplicate(self, nums: List[int]) -> int:
        low, high = 0, len(nums) - 1
        while low < high:
            mid = (low + high) // 2
            if nums[mid] == nums[mid ^ 1]:
                low = mid + 1
            else:
                high = mid
        return nums[low]

    # the answer must have an even number of index
    def singleNonDuplicate(self, nums: List[int]) -> int:
        low, high = 0, len(nums) - 1
        while low < high:
            mid = (low + high) // 2
            mid -= mid & 1
            if nums[mid] == nums[mid + 1]:
                low = mid + 2
            else:
                high = mid
        return nums[low]

    def singleNonDuplicate(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        l = 0
        r = len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if mid % 2 == 0 and mid + 1 < len(nums):  # mid是偶数(nums[mid]前面有偶数个元素)
                if nums[mid] == nums[mid + 1]:  # mid前面没有单一元素
                    l = mid + 1
                else:  # mid前面有单一元素
                    r = mid - 1
            elif mid % 2 != 0 and mid + 1 < len(nums):  # mid是奇数(nums[mid]前面有奇数个元素)
                if nums[mid] == nums[mid + 1]:  # mid前面有单一元素
                    r = mid - 1
                else:  # mid前面没有单一元素
                    l = mid + 1
            else:
                return nums[mid]
        return nums[l]

    def singleNonDuplicate(self, nums: List[int]) -> int:
        return nums[
            bisect.bisect_left(
                range(len(nums) - 1), True, key=lambda x: nums[x] != nums[x ^ 1]
            )
        ]


# 542 - 01 Matrix - MEDIUM
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m = len(mat)
        n = len(mat[0])
        dq = collections.deque()
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    dq.append((i, j))
                else:
                    mat[i][j] = -1
        while dq:
            x, y = dq.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < m and 0 <= ny < n and mat[nx][ny] == -1:
                    dq.append((nx, ny))
                    mat[nx][ny] = mat[x][y] + 1
        return mat

    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        dq = collections.deque()
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    dq.append((i, j, 1))
                else:
                    mat[i][j] = -1
        while dq:
            x, y, t = dq.popleft()
            for nx, ny in [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]:
                if 0 <= nx < m and 0 <= ny < n and mat[nx][ny] == -1:
                    mat[nx][ny] = t
                    dq.append((nx, ny, t + 1))
        return mat


# 543 - Diameter of Binary Tree - EASY
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.maxL = 0

        def dfs(root: TreeNode) -> int:
            if not root:
                return 0
            left = dfs(root.left)
            right = dfs(root.right)
            self.maxL = max(self.maxL, left + right)
            return max(left, right) + 1

        dfs(root)
        return self.maxL


# 547 - Number of Provinces - MEDIUM
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        seen, circle = set(), 0

        def dfs(person: int):
            for friend, isFriend in enumerate(isConnected[person]):
                if isFriend and friend not in seen:
                    seen.add(friend)
                    dfs(friend)
            return

        for person in range(len(isConnected)):
            if person not in seen:
                dfs(person)
                circle += 1
        return circle


# Union—Find
class UnionFind:
    def __init__(self):
        self.father = {}
        self.num_of_sets = 0

    def find(self, x):
        root = x
        while self.father[root] != None:
            root = self.father[root]
        while x != root:
            original_father = self.father[x]
            self.father[x] = root
            x = original_father
        return root

    def merge(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            self.father[root_x] = root_y
            self.num_of_sets -= 1

    def add(self, x):
        if x not in self.father:
            self.father[x] = None
            self.num_of_sets += 1


class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        uf = UnionFind()
        for i in range(len(isConnected)):
            uf.add(i)
            for j in range(i):
                if isConnected[i][j]:
                    uf.merge(i, j)

        return uf.num_of_sets


# 553 - Optimal Division - MEDIUM
class Solution:
    def optimalDivision(self, nums: List[int]) -> str:
        res = ""
        if len(nums) > 2:
            res = "/(" + "/".join(str(i) for i in nums[1:]) + ")"
        elif len(nums) == 2:
            res = "/" + "/".join(str(i) for i in nums[1:]) + ""
        return str(nums[0]) + res


# 556 - Next Greater Element III - MEDIUM
class Solution:
    def nextGreaterElement(self, n: int) -> int:
        nums = list(str(n))
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i < 0:
            return -1
        j = len(nums) - 1
        while j >= 0 and nums[i] >= nums[j]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
        nums[i + 1 :] = nums[i + 1 :][::-1]
        ans = int("".join(nums))
        return ans if ans < 2**31 else -1


# 557 - Reverse Words in a String III - EASY
class Solution:
    def reverseWords(self, s: str) -> str:
        split = s.split()  # default delimiter: " ", whitespace
        for i in range(len(split)):
            split[i] = split[i][::-1]
        return " ".join(split)

        # return ' '.join(x[::-1] for x in s.split())
        # return ' '.join(s.split()[::-1])[::-1]


# 558 - Logical OR of Two Binary Grids Represented as Quad-Trees - MEDIUM
class Node:
    # Definition for a QuadTree node.
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight


class Solution:
    def intersect(self, q1: "Node", q2: "Node") -> "Node":
        if (q1.isLeaf and q1.val) or (q2.isLeaf and not q2.val):
            return q1
        if (q2.isLeaf and q2.val) or (q1.isLeaf and not q1.val):
            return q2
        l1 = self.intersect(q1.topLeft, q2.topLeft)
        l2 = self.intersect(q1.topRight, q2.topRight)
        l3 = self.intersect(q1.bottomLeft, q2.bottomLeft)
        l4 = self.intersect(q1.bottomRight, q2.bottomRight)
        if (
            l1.isLeaf
            and l2.isLeaf
            and l3.isLeaf
            and l4.isLeaf
            and l1.val == l2.val == l3.val == l4.val
        ):
            return l1
        return Node(None, False, l1, l2, l3, l4)


# 559 - Maximum Depth of N-ary Tree - EASY
class Solution:
    # root.children is a list
    # bfs
    def maxDepth(self, root: "Node") -> int:
        if not root:
            return 0
        dq = collections.deque([root])
        ans = 0
        while dq:
            for _ in range(len(dq)):
                node = dq.popleft()
                for ch in node.children:
                    dq.append(ch)
            ans += 1
        return ans

    # dfs
    def maxDepth(self, root: "Node") -> int:
        if not root:
            return 0
        return max([self.maxDepth(child) for child in root.children], default=0) + 1


# 560 - Subarray Sum Equals K - MEDIUM
# Why not sliding window?
# The next element might be negative
# Moving pointer to the right cannot guarantee the sum will become larger
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        presum, ans = 0, 0
        # 'dic' used to store the number of occurences of each 'presum'
        # 'dic[0] = 1' indicating that the successive subarray of sum is 0 occured 1 time
        dic = collections.defaultdict(int)
        dic[0] = 1
        for i in range(len(nums)):
            presum += nums[i]
            ans += dic[presum - k]
            dic[presum] += 1
        return ans

    def subarraySum(self, nums: List[int], k: int) -> int:
        dic = {0: 1}
        ans = presum = 0
        for n in nums:
            presum += n
            if presum - k in dic:
                ans += dic[presum - k]
            dic[presum] = dic.get(presum, 0) + 1
        return ans


# 563 - Binary Tree Tilt - EAST
class Solution:
    def findTilt(self, root: Optional[TreeNode]) -> int:
        self.ans = 0

        # return sum of right subtree and left subtree
        def helper(root: TreeNode) -> int:
            if not root:
                return 0
            vl = helper(root.left)
            vr = helper(root.right)
            self.ans += abs(vl - vr)
            return vl + vr + root.val

        helper(root)
        return self.ans


# 564 - Find the Closest Palindrome - HARD
class Solution:
    def nearestPalindromic(self, n: str) -> str:
        if int(n) < 10 or int(n[::-1]) == 1:
            return str(int(n) - 1)
        if n == "11":
            return "9"
        # if set(n) == {'9'}:
        #     return str(int(n) + 2)
        l, r = n[: (len(n) + 1) // 2], n[(len(n) + 1) // 2 :]
        temp = [str(int(l) - 1), l, str(int(l) + 1)]
        temp = [i + i[len(r) - 1 :: -1] for i in temp]
        # float('inf'), deal with the case like '88'
        return min(temp, key=lambda x: abs(int(x) - int(n)) or float("inf"))

    def nearestPalindromic(self, n: str) -> str:
        if int(n) < 10 or int(n[::-1]) == 1:
            return str(int(n) - 1)
        if n == "11":
            return "9"
        left = n[: (len(n) + 1) // 2]
        # N-1,N-1 / N, N / N+1, N+1
        tmp = [str(int(left) - 1), left, str(int(left) + 1)]
        tmp = [i + i[: (len(n)) // 2][::-1] for i in tmp]
        return min(tmp, key=lambda x: abs(int(x) - int(n)) or float("inf"))


# 566 - Reshape the Matrix - EASY
class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        if r * c != len(mat) * len(mat[0]):
            return mat
        tmp = [0] * (r * c)
        ans = [[0] * c for _ in range(r)]
        n = len(mat[0])
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                tmp[i * n + j] = mat[i][j]
        for i in range(r):
            for j in range(c):
                ans[i][j] = tmp[i * c + j]
        return ans

    # not use tmp, save space
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        if r * c != len(mat) * len(mat[0]):
            return mat
        ans = [[0] * c for _ in range(r)]
        m = n = 0
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                ans[m][n] = mat[i][j]
                n += 1
                if n == c:
                    m += 1
                    n = 0
        return ans

    # better
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        if r * c != m * n:
            return mat
        ans = [[0] * c for _ in range(r)]
        for x in range(m * n):
            ans[x // c][x % c] = mat[x // n][x % n]
        return ans


# 567 - Permutation in String - MEDIUM
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False
        arr1 = [0] * 26
        arr2 = [0] * 26
        for i in range(len(s1)):
            arr1[ord(s1[i]) - ord("a")] += 1
            arr2[ord(s2[i]) - ord("a")] += 1
        if arr1 == arr2:
            return True
        for i in range(len(s1), len(s2)):
            arr2[ord(s2[i - len(s1)]) - ord("a")] -= 1
            arr2[ord(s2[i]) - ord("a")] += 1
            if arr1 == arr2:
                return True
        return False

    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False
        c = [0] * 26
        m, n = len(s1), len(s2)
        for i in range(m):
            c[ord(s1[i]) - ord("a")] -= 1
            c[ord(s2[i]) - ord("a")] += 1
        diff = 0
        for i in c:
            diff += 1 if i else 0
        if diff == 0:
            return True
        for i in range(m, n):
            if c[ord(s2[i - m]) - ord("a")] == 1:
                diff -= 1
            elif c[ord(s2[i - m]) - ord("a")] == 0:
                diff += 1
            c[ord(s2[i - m]) - ord("a")] -= 1
            if c[ord(s2[i]) - ord("a")] == -1:
                diff -= 1
            elif c[ord(s2[i]) - ord("a")] == 0:
                diff += 1
            c[ord(s2[i]) - ord("a")] += 1
            if diff == 0:
                return True
        return False


# 572 - Subtree of Another Tree - EASY
class Solution:
    def isSubtree(self, root: TreeNode, sub: TreeNode) -> bool:
        if not sub and not root:
            return True
        if not sub or not root:
            return False
        return (
            self.isSameTree(sub, root)
            or self.isSubtree(root.left, sub)
            or self.isSubtree(root.right, sub)
        )

    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not q and not p:
            return True
        if not q or not p:
            return False
        return (
            q.val == p.val
            and self.isSameTree(q.left, p.left)
            and self.isSameTree(q.right, p.right)
        )


# 575 - Distribute Candies - EASY
class Solution:
    # counter
    def distributeCandies(self, candyType: List[int]) -> int:
        return min(len(collections.Counter(candyType)), int(len(candyType) / 2))

    # set
    def distributeCandies(self, candyType: List[int]) -> int:
        return min(len(set(candyType)), len(candyType) // 2)


# 583 - Delete Operation for Two Strings - MEDIUM
class Solution:
    # 1143 - Longest Common Subsequence
    def minDistance(self, word1: str, word2: str) -> int:
        len1, len2 = len(word1), len(word2)
        dp = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return len1 + len2 - dp[-1][-1] * 2


# 587 - Erect the Fence - HARD
# Jarvis Algorithm                    - O(n ^ 2) / O(n)
# Graham Scan                         - O(nlogn) / O(n)
# Monotone Chain / Andrew Algorithm   - O(nlogn) / O(n)
class Solution:
    def outerTrees(self, points: List[List[int]]) -> List[List[int]]:
        """Computes the convex hull of a set of 2D points.
        Input: an iterable sequence of (x, y) pairs representing the points.
        Output: a list of vertices of the convex hull in counter-clockwise order,
          starting from the vertex with the lexicographically smallest coordinates.
        Implements Andrew's monotone chain algorithm. O(n log n) complexity.
        """
        points = sorted(map(tuple, points), key=lambda p: (p[0], p[1]))

        # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
        # Returns a positive value, if OAB makes a counter-clockwise turn,
        # negative for clockwise turn, and zero if the points are collinear.
        def sign(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and sign(lower[-2], lower[-1], p) < 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and sign(upper[-2], upper[-1], p) < 0:
                upper.pop()
            upper.append(p)

        # Concatenation of the lower and upper hulls gives the convex hull.
        # Last point of each list is omitted because it is repeated at the
        # beginning of the other list.
        return list(set(lower + upper))


# 589 - N-ary Tree Preorder Traversal - EASY
class Solution:
    def preorder(self, root: "Node") -> List[int]:
        if not root:
            return []
        t = [root.val]
        for ch in root.children:
            t += self.preorder(ch)
        return t

    def preorder(self, root: "Node") -> List[int]:
        def dfs(root: "Node"):
            if not root:
                return
            ans.append(root.val)
            for ch in root.children:
                dfs(ch)
            return

        ans = []
        dfs(root)
        return ans

    def preorder(self, root: "Node") -> List[int]:
        if not root:
            return []
        s = [root]
        ans = []
        while s:
            r = s.pop()
            ans.append(r.val)
            s.extend(reversed(r.children))
            # for ch in r.children[::-1]:
            #     s.append(ch)
        return ans


# 590 - N-ary Tree Postorder Traversal - EASY
class Solution:
    def postorder(self, root: "Node") -> List[int]:
        if not root:
            return []
        t = []
        for ch in root.children:
            t += self.postorder(ch)
        t.append(root.val)
        return t

    def postorder(self, root: "Node") -> List[int]:
        def dfs(root: Node):
            if not root:
                return
            for ch in root.children:
                dfs(ch)
            ans.append(root.val)
            return

        ans = []
        dfs(root)
        return ans

    def postorder(self, root: "Node") -> List[int]:
        if not root:
            return []
        q = [root]
        ans = []
        while q:
            r = q.pop()
            ans.append(r.val)
            q.extend(r.children)
            # for ch in r.children:
            #     q.append(ch)
        return reversed(ans)


# 591 - Tag Validator - HARD
class Solution:
    def isValid(self, code: str) -> bool:
        code = re.sub(r"<!\[CDATA\[.*?\]\]>|t", "-", code)
        prev = None
        while code != prev:
            prev = code
            code = re.sub(r"<([A-Z]{1,9})>[^<]*</\1>", "t", code)
        return code == "t"


# 594 - Longest Harmonious Subsequence - EASY
class Solution:
    def findLHS(self, nums: List[int]) -> int:
        nums.sort()
        i = ans = 0
        for j in range(len(nums)):
            while nums[j] - nums[i] > 1:
                i += 1
            if nums[j] - nums[i] == 1:
                ans = max(ans, j - i + 1)
        return ans


# 598 - Range Addition II - EASY
class Solution:
    def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
        if not ops:
            return m * n
        return min(op[0] for op in ops) * min(op[1] for op in ops)


# 599 - Minimum Index Sum of Two Lists - EASY
class Solution:
    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        ans = []
        d = {v: i for i, v in enumerate(list1)}
        mx = math.inf
        for i, v in enumerate(list2):
            if v in d and i + d[v] < mx:
                mx = i + d[v]
                ans = [v]
            elif v in d and i + d[v] == mx:
                ans.append(v)
        return ans
