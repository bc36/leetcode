from typing import List


# 1762 - Buildings With an Ocean View - MEDIUM
class Solution:
    def findBuildings(self, heights: List[int]) -> List[int]:
        ans = []
        maxH = 0
        for i in range(len(heights) - 1, -1, -1):
            # print(i)
            if heights[i] > maxH:
                ans.append(i)
                maxH = heights[i]
        ans.reverse()
        return ans
