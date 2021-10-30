import random, bisect, itertools
from typing import List

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