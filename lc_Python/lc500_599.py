import collections
import random, bisect, itertools
from typing import List


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


# 523 - Continuous Subarray Sum - MEDUIM
# brutal-force: Time Limit Exceeded
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        presumList = []
        for i in range(len(nums)):
            presum = [j for j in itertools.accumulate(nums[i:])]
            if len(presum[1:]) > 0:
                presumList.append(presum[1:])
        for presum in presumList:
            for p in presum:
                if p % k == 0:
                    return True
        return False


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
        rmd, presumRmd = {0: -1}, 0
        # why {0: -1}:
        # -1 is just before you start the index.
        # So if you get the first 2 elements sum to k,
        # your current i is 1. So 1 - (-1) = 2 still satisfies the return True condition.
        for i, num in enumerate(nums):
            if k != 0:
                presumRmd = (num + presumRmd) % k
            else:
                presumRmd += num
            if presumRmd not in rmd:
                rmd[presumRmd] = i
            else:
                if i - rmd[presumRmd] > 1:
                    return True
        return False


# the required length is at least 2,
# so we just need to insert the mod one iteration later.
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        summ, pre = 0, 0
        s = set()
        for num in nums:
            summ += num
            mod = summ % k
            if mod in s:
                return True
            s.add(pre)
            pre = mod
        return False


# 528 - Random Pick with Weight - MEDIUM
# prefix sum + binary search
# seperate [1, total] in len(w) parts, each part has w[i] elements
class Solution:
    def __init__(self, w: List[int]):
        # Calculate the prefix sum to generate a random number
        # The coordinates of the distribution correspond to the size of the number
        # 计算前缀和，这样可以生成一个随机数，根据数的大小对应分布的坐标
        self.presum = list(itertools.accumulate(w))

    def pickIndex(self) -> int:
        rand = random.randint(1, self.presum[-1])
        return bisect.bisect_left(self.presum, rand)


# 575 - Distribute Candies - EASY
# counter
class Solution:
    def distributeCandies(self, candyType: List[int]) -> int:
        # len(candyType) // 2
        return min(len(collections.Counter(candyType)),
                   int(len(candyType) / 2))


# set
class Solution:
    def distributeCandies(self, candyType: List[int]) -> int:
        return min(len(set(candyType)), len(candyType) // 2)


# 598 - Range Addition II - EASY
class Solution:
    def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
        return 1