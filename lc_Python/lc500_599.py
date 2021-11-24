import collections
from operator import le
from os import pread
import random, bisect, itertools
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


# 520 - Detect Capital - EASY
# brutal-force
class Solution:
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


class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        # Solution 1: word.istitle()
        return word.istitle() or word.isupper() or word.islower()
        # Solution 2:
        # return word[1:] == word[1:].lower() or word == word.upper()


# 523 - Continuous Subarray Sum - MEDUIM
# 'cur' calculate the prefix sum remainder of input array 'nums'
# 'seen' will record the first occurrence of the remainder.
# If we have seen the same remainder before,
# it means the subarray sum is a multiple of k
class Solution:
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
class Solution:
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


class Solution:
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
class Solution:
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


class Solution:
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


# 559 - Maximum Depth of N-ary Tree - EASY
# root.children is a list
# bfs
class Solution:
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
class Solution:
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
# depth-first search
class Solution:
    def findTilt(self, root: Optional[TreeNode]) -> int:
        self.ans = 0  # ans = []

        # return sum of right subtree and left subtree
        def dfs(root: TreeNode):
            if not root:
                return 0
            vl = dfs(root.left)
            vr = dfs(root.right)
            self.ans += abs(vl - vr)  # ans.append(abs(vl - vr))
            return vl + vr + root.val

        dfs(root)
        return self.ans  # sum(ans)


# 575 - Distribute Candies - EASY
# counter
class Solution:
    def distributeCandies(self, candyType: List[int]) -> int:
        return min(len(collections.Counter(candyType)),
                   int(len(candyType) / 2))


# set
class Solution:
    def distributeCandies(self, candyType: List[int]) -> int:
        return min(len(set(candyType)), len(candyType) // 2)


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
        return 1