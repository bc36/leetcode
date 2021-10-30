from typing import List
import functools, operator

# 121 - Best Time to Buy and Sell Stock - EASY
# Dynamic Programming
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        hisLowPrice, ans = prices[0], 0
        for price in prices:
            ans = max(ans, price - hisLowPrice)
            hisLowPrice = min(hisLowPrice, price)
        return ans

# 136 - Single Number - EASY
# XOR operation
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ans = 0
        for i in nums:
            ans ^= i
        return ans

# lambda arguments: expression
# reduce(func, seq)
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        # return functools.reduce(operator.xor, nums)
        return functools.reduce(lambda x, y: x ^ y, nums)