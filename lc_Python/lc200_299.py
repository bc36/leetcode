from typing import List
import collections

# 260 - Single Number III - MEDUIM
# Hash / O(n) + O(n)
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        cnt = collections.Counter(nums)
        ans = [num for num, times in cnt.items() if times == 1]
        return ans

# "lsb" is the last 1 of its binary representation, means that two numbers are different in that bit
# split nums[] into two lists, one with that bit as 0 and the other with that bit as 1.
# separately perform XOR operation, find the number that appears once in each list.
# O(n) + O(1)
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        xorSum = 0
        for i in nums:
            xorSum ^= i
        lsb = xorSum & -xorSum
        # mask = 1
        # while(xorSum & mask == 0):
        #     mask = mask << 1
        ans1, ans2 = 0, 0
        for i in nums:
            if i & lsb > 0:
                ans1 ^= i
            else:
                ans2 ^= i
        return [ans1, ans2]