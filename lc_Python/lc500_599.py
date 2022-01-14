import collections, math, random, bisect, itertools
import functools
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
        line1, line2, line3 = set('qwertyuiop'), set('asdfghjkl'), set(
            'zxcvbnm')
        ret = []
        for word in words:
            w = set(word.lower())
            if w <= line1 or w <= line2 or w <= line3:
                ret.append(word)
        return ret


# 506 - Relative Ranks - EASY
class Solution:
    def findRelativeRanks(self, score: List[int]) -> List[str]:
        dic = {n: idx for idx, n in enumerate(sorted(score, reverse=True))}
        medals = ["Gold Medal", "Silver Medal", "Bronze Medal"]
        return [
            str(dic[i] + 1) if dic[i] >= 3 else medals[dic[i]] for i in score
        ]


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


# 509 - Fibonacci Number - EASY
class Solution:
    def fib(self, n: int) -> int:
        if n < 2: return n
        one, two, ans = 0, 1, 0
        for _ in range(1, n):
            ans = one + two
            one, two = two, ans
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
        for i in range(len(coins)):
            for j in range(coins[i], amount + 1):
                dp[j] += dp[j - coins[i]]
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


# 542 - 01 Matrix - MEDIUM
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        dq = collections.deque([])
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    dq.append((i, j))
                else:
                    mat[i][j] = '#'  # not visited yet
        while dq:
            x, y = dq.popleft()
            for i, j in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if 0 <= i < m and 0 <= j < n and mat[i][j] == '#':
                    mat[i][j] = mat[x][y] + 1
                    dq.append((i, j))
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


# 557 - Reverse Words in a String III - EASY
class Solution:
    def reverseWords(self, s: str) -> str:
        split = s.split()  # default delimiter: " ", whitespace
        for i in range(len(split)):
            split[i] = split[i][::-1]
        return " ".join(split)

        # return ' '.join(x[::-1] for x in s.split())
        # return ' '.join(s.split()[::-1])[::-1]


# 558


# 559 - Maximum Depth of N-ary Tree - EASY
class Solution:
    # root.children is a list
    # bfs
    def maxDepth(self, root: 'Node') -> int:
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
    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        return max([self.maxDepth(child) for child in root.children],
                   default=0) + 1


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


# 567 - Permutation in String - MEDIUM
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False
        arr1 = [0] * 26
        arr2 = [0] * 26
        for i in range(len(s1)):
            arr1[ord(s1[i]) - ord('a')] += 1
            arr2[ord(s2[i]) - ord('a')] += 1
        if arr1 == arr2:
            return True
        for i in range(len(s1), len(s2)):
            arr2[ord(s2[i - len(s1)]) - ord('a')] -= 1
            arr2[ord(s2[i]) - ord('a')] += 1
            if arr1 == arr2:
                return True
        return False


# 572 - Subtree of Another Tree - EASY
class Solution:
    def isSubtree(self, root: TreeNode, sub: TreeNode) -> bool:
        if not sub and not root:
            return True
        if not sub or not root:
            return False
        return self.isSameTree(sub, root) or self.isSubtree(
            root.left, sub) or self.isSubtree(root.right, sub)

    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not q and not p:
            return True
        if not q or not p:
            return False
        return q.val == p.val and self.isSameTree(
            q.left, p.left) and self.isSameTree(q.right, p.right)


# 575 - Distribute Candies - EASY
class Solution:
    # counter
    def distributeCandies(self, candyType: List[int]) -> int:
        return min(len(collections.Counter(candyType)),
                   int(len(candyType) / 2))

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
