from typing import List

# 121 - Best Time to Buy and Sell Stock - EASY
# Dynamic Programming
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        hisLowPrice, ans = prices[0], 0
        for price in prices:
            ans = max(ans, price - hisLowPrice)
            hisLowPrice = min(hisLowPrice, price)
        return ans