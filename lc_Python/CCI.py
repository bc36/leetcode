import collections, heapq
from typing import List

"""
Cracking the Coding Interview
"""

# https://leetcode.cn/problems/is-unique-lcci/
# 面试题 01.01. 判定字符是否唯一 - EASY
class Solution:
    def isUnique(self, astr: str) -> bool:
        return len(astr) == len(set(astr))

    def isUnique(self, astr: str) -> bool:
        mask = 0
        for c in astr:
            shift = ord(c) - ord("a")
            if mask & 1 << shift != 0:
                return False
            mask |= 1 << shift
        return True


# https://leetcode.cn/problems/check-permutation-lcci/
# 面试题 01.02. 判定是否互为字符重排 - EASY
class Solution:
    def CheckPermutation(self, s1: str, s2: str) -> bool:
        return collections.Counter(s1) == collections.Counter(s2)
        return sorted(s1) == sorted(s2)


# https://leetcode.cn/problems/string-to-url-lcci/
# 面试题 01.03. String to URL LCCI - EASY
class Solution:
    def replaceSpaces(self, S: str, length: int) -> str:
        return S[:length].replace(" ", "%20")


# https://leetcode.cn/problems/get-kth-magic-number-lcci/
# 面试题 17.09. 第 k 个数 - MEDIUM
class Solution:
    # O(log3k + 3log3k) / O(3k)
    def getKthMagicNumber(self, k: int) -> int:
        q = [1]
        s = {1}
        for _ in range(k - 1):
            x = heapq.heappop(q)
            for p in 3, 5, 7:
                if p * x not in s:
                    heapq.heappush(q, p * x)
                    s.add(p * x)
        return q[0]

    # O(k) / O(k)
    def getKthMagicNumber(self, k: int) -> int:
        # x >= y >= z, 是三个指针, 当前丑数一定是某一个旧丑数的倍增结果
        # x3, y5, z7 是之前不同丑数的倍增结果. f 是递增的丑数序列
        # 倍增结果可能一样大, 一样大时, 同时移动多个指针, 去重
        f = [0] * (k + 1)
        f[1] = 1
        x = y = z = 1
        for i in range(2, k + 1):
            x3, y5, z7 = f[x] * 3, f[y] * 5, f[z] * 7
            f[i] = min(x3, y5, z7)
            if f[i] == x3:
                x += 1
            if f[i] == y5:
                y += 1
            if f[i] == z7:
                z += 1
        return f[k]


# https://leetcode.cn/problems/missing-two-lcci/
# 面试题 17.19. 消失的两个数字 - HARD
class Solution:
    # O(n) / O(n), 原地哈希, 改变出现的值所映射下标的符号, 作为标记
    def missingTwo(self, nums: List[int]) -> List[int]:
        n = len(nums) + 2
        nums.extend([1] * 2)
        for i in range(n - 2):
            nums[abs(nums[i]) - 1] *= -1
        return [i + 1 for i in range(n) if nums[i] > 0]
