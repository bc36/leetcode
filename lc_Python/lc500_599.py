import collections
import random, bisect, itertools
from typing import List

# 500 - Keyboard Row - EASY
# The '<' and '>' operators are testing for strict subsets
class Solution:
    def findWords(self, words):
        line1, line2, line3 = set('qwertyuiop'), set('asdfghjkl'), set('zxcvbnm')
        ret = []
        for word in words:
            w = set(word.lower())
            if w <= line1 or w <= line2 or w <= line3:
                ret.append(word)
        return ret


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
        return min(len(collections.Counter(candyType)), int(len(candyType) / 2))
        
# set
class Solution:
    def distributeCandies(self, candyType: List[int]) -> int:
        return min(len(set(candyType)), len(candyType) // 2)