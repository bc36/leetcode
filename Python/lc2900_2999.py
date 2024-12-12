import bisect, collections, functools, heapq, itertools, math, operator, string, sys
from typing import List, Optional, Tuple
import sortedcontainers


# 2908 - Minimum Sum of Mountain Triplets I - EASY
class Solution:
    # O(n^3) / O(1)
    def minimumSum(self, nums: List[int]) -> int:
        return min(
            (
                nums[i] + nums[j] + nums[k]
                for i in range(len(nums) - 2)
                for j in range(i + 1, len(nums) - 1)
                for k in range(j + 1, len(nums))
                if nums[i] < nums[j] and nums[j] > nums[k]
            ),
            default=-1,
        )

    # O(n) / O(1)
    def minimumSum(self, nums: List[int]) -> int:
        n = len(nums)
        pre = [nums[0]] + [0] * (n - 1)
        for i in range(1, n - 1):
            pre[i] = min(pre[i - 1], nums[i])
        suf = [0] * (n - 1) + [nums[-1]]
        for i in range(n - 2, 0, -1):
            suf[i] = min(suf[i + 1], nums[i])
        ans = math.inf
        for i in range(1, n - 1):
            if pre[i] < nums[i] and nums[i] > suf[i]:
                ans = min(ans, pre[i - 1] + nums[i] + suf[i + 1])
        return -1 if ans == math.inf else ans


# 2917 - Find the K-or of an Array - EASY
class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        ans = 0
        for i in range(31):
            cnt = 0
            for x in nums:
                cnt += 1 << i & x == 1 << i
            if cnt >= k:
                ans |= 1 << i
        return ans

    def findKOr(self, nums: List[int], k: int) -> int:
        ans = 0
        for i in range(31):
            ans |= (sum(x >> i & 1 for x in nums) >= k) << i
        return ans

    def findKOr(self, nums: List[int], k: int) -> int:
        return functools.reduce(
            operator.or_, ((sum(x >> i & 1 for x in nums) >= k) << i for i in range(31))
        )


# 2931 - Maximum Spending After Buying Items - HARD
class Solution:
    def maxSpending(self, values: List[List[int]]) -> int:
        h = [(v[-1], i) for i, v in enumerate(values)]
        heapq.heapify(h)
        ans = 0
        for d in range(1, len(values) * len(values[0]) + 1):
            v, i = heapq.heappop(h)
            ans += v * d
            values[i].pop()
            if values[i]:
                heapq.heappush(h, (values[i][-1], i))
        return ans

    def maxSpending(self, values: List[List[int]]) -> int:
        arr = sorted(x for row in values for x in row)
        return sum(x * i for i, x in enumerate(arr, 1))


# 2951 - Find the Peaks - EASY
class Solution:
    def findPeaks(self, mountain: List[int]) -> List[int]:
        return list(
            i
            for i in range(1, len(mountain) - 1)
            if mountain[i - 1] < mountain[i] and mountain[i] > mountain[i + 1]
        )
        return list(
            i
            for i in range(1, len(mountain) - 1)
            if mountain[i - 1] < mountain[i] > mountain[i + 1]
        )


# 2952 - Minimum Number of Coins to be Added - MEDIUM
class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        coins.sort()
        ans = i = 0
        s = 1
        while s <= target:
            if i < len(coins) and coins[i] <= s:
                s += coins[i]
                i += 1
            else:
                s += s  # 无法得到 s, 需要添加一个 s
                ans += 1
        return ans


# 2956 - Find Common Elements Between Two Arrays - EASY
class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        s1, s2 = set(nums1), set(nums2)
        return [sum(x in s2 for x in nums1), sum(y in s1 for y in nums2)]
