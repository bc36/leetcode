import bisect, collections, functools, heapq, itertools, math, operator, string
from typing import List, Optional, Tuple
import sortedcontainers

"""
Cracking the Coding Interview
"""


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


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


# https://leetcode.cn/problems/palindrome-permutation-lcci/
# 面试题 01.04. 回文排列 - EASY
class Solution:
    def canPermutePalindrome(self, s: str) -> bool:
        cnt = collections.Counter(s)
        odd = sum(v & 1 for v in cnt.values())
        return True if odd <= 1 else False

    def canPermutePalindrome(self, s: str) -> bool:
        st = set()
        for c in s:
            if c in st:
                st.remove(c)
            else:
                st.add(c)
        return len(st) <= 1

    def canPermutePalindrome(self, s: str) -> bool:
        mask = 0
        for c in s:
            if mask & 1 << ord(c):
                mask &= ~(1 << ord(c))
                # mask ^= 1 << ord(c)
            else:
                mask |= 1 << ord(c)
        return sum(1 for i in range(128) if mask & 1 << i) <= 1


# https://leetcode.cn/problems/compress-string-lcci/
# 面试题 01.06. 字符串压缩 - EASY
class Solution:
    def compressString(self, S: str) -> str:
        ans = pre = ""
        cnt = 0
        for c in S + "#":
            if c == pre:
                cnt += 1
            else:
                ans += pre + str(cnt) if cnt > 0 else ""
                cnt = 1
            pre = c
        return ans if len(ans) < len(S) else S

    def compressString(self, S: str) -> str:
        if S == "":
            return ""
        ans = ""
        pre = S[0]
        cnt = 0
        for c in S + "#":
            if c == pre:
                cnt += 1
            else:
                ans += pre + str(cnt)
                cnt = 1
            pre = c
        return ans if len(ans) < len(S) else S


# https://leetcode.cn/problems/zero-matrix-lcci/
# 面试题 01.08. 零矩阵 - MEDIUM
class Solution:
    # O(mn) / O(m + n)
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        row = set()
        col = set()
        for i, r in enumerate(matrix):
            for j, v in enumerate(r):
                if v == 0:
                    row.add(i)
                    col.add(j)
        for r in row:
            for j in range(len(matrix[0])):
                matrix[r][j] = 0
        for c in col:
            for i in range(len(matrix)):
                matrix[i][c] = 0
        return

    # O(mn) / O(1)
    def setZeroes(self, matrix: List[List[int]]) -> None:
        m = len(matrix)
        n = len(matrix[0])
        flag_col0 = any(matrix[i][0] == 0 for i in range(m))
        flag_row0 = any(matrix[0][j] == 0 for j in range(n))
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        if flag_col0:
            for i in range(m):
                matrix[i][0] = 0
        if flag_row0:
            for j in range(n):
                matrix[0][j] = 0
        return


# https://leetcode.cn/problems/string-rotation-lcci/
# 面试题 01.09. 字符串轮转 - EASY
class Solution:
    def isFlipedString(self, s1: str, s2: str) -> bool:
        if collections.Counter(s1) != collections.Counter(s2):
            return False
        if s1 == s2 == "":
            return True
        q1 = collections.deque(s1)
        q2 = collections.deque(s2)
        i = len(s1)
        while i:
            if q1[0] == q2[0] and q1 == q2:
                return True
            q1.append(q1.popleft())
            i -= 1
        return False

    def isFlipedString(self, s1: str, s2: str) -> bool:
        return len(s1) == len(s2) and s2 in s1 + s1

    def isFlipedString(self, s1: str, s2: str) -> bool:
        m = len(s1)
        n = len(s2)
        if m != n:
            return False
        if m == 0 and n == 0:
            return True
        # python 切片还是很快的
        for i in range(m):
            if s1[i] == s2[0]:
                if s2 == s1[i:] + s1[:i]:
                    return True
        return False


# https://leetcode.cn/problems/kth-node-from-end-of-list-lcci/
# 面试题 02.02. 返回倒数第 k 个节点 - EASY
class Solution:
    def kthToLast(self, head: ListNode, k: int) -> int:
        tail = head
        for _ in range(k):
            head = head.next
        while head:
            head = head.next
            tail = tail.next
        return tail.val


# https://leetcode.cn/problems/delete-middle-node-lcci/
# 面试题 02.03. 删除中间节点 - EASY
class Solution:
    def deleteNode(self, node: ListNode) -> None:
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next
        return


# https://leetcode.cn/problems/bianry-number-to-string-lcci/
# 面试题 05.02. 二进制数转字符串 - MEDIUM
class Solution:
    # 任何进制表示的小数, 乘上进制等价于小数点往右移一位
    # num 如果可以表示为有限位二进制小数, 那么可以表示为一个形如 b / (2^k) 的最简分数
    def printBin(self, num: float) -> str:
        s = ["0."]
        for _ in range(6):
            num *= 2
            if num < 1:
                s.append("0")
            else:
                num -= 1
                s.append("1")
                if num == 0:
                    return "".join(s)
        return "ERROR"

    # 十进制小数转二进制小数:
    # 小数部分乘以 2, 取整数部分(1 或 0)作为二进制小数的下一位(然后置零),
    # 小数部分作为下一次乘法的被乘数, 直到小数部分为 0 或者二进制小数的长度超过 32 位
    def printBin(self, num: float) -> str:
        ans = "0."
        while len(ans) < 32 and num:
            num *= 2
            x = int(num)
            ans += str(x)
            num -= x
        return "ERROR" if num else ans


# https://leetcode.cn/problems/pond-sizes-lcci/
# 面试题 16.19. 水域大小 - MEDIUM
class Solution:
    def pondSizes(self, land: List[List[int]]) -> List[int]:
        def bfs(i: int, j: int) -> int:
            q = [(i, j)]
            t = 1
            land[i][j] = 1
            while q:
                new = []
                for x, y in q:
                    for dx, dy in (
                        (0, 1),
                        (0, -1),
                        (1, 0),
                        (-1, 0),
                        (1, 1),
                        (1, -1),
                        (-1, 1),
                        (-1, -1),
                    ):
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < m and 0 <= ny < n and land[nx][ny] == 0:
                            t += 1
                            land[nx][ny] = 1
                            new.append((nx, ny))
                q = new
            return t

        def dfs(i: int, j: int) -> int:
            if land[i][j]:
                return 0
            land[i][j] = 1
            res = 1
            for dx, dy in (
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ):
                nx = i + dx
                ny = j + dy
                if 0 <= nx < m and 0 <= ny < n and land[nx][ny] == 0:
                    res += dfs(nx, ny)
            return res

        m, n = len(land), len(land[0])
        return sorted(
            bfs(i, j) for i, row in enumerate(land) for j, v in enumerate(row) if v == 0
        )
        return sorted(
            dfs(i, j) for i, row in enumerate(land) for j, v in enumerate(row) if v == 0
        )


# https://leetcode.cn/problems/add-without-plus-lcci/
# 面试题 17.01. 不用加号的加法 - EASY
class Solution:
    def add(self, a: int, b: int) -> int:
        return sum((a, b))

    def add(self, a: int, b: int) -> int:
        return a - (-b)

    # python 左移位都不会溢出, 负数会超时
    # def add(self, a: int, b: int) -> int:
    #     summ = carry = 0
    #     while b:
    #         summ = a ^ b
    #         carry = (a & b) << 1
    #         a, b = summ, carry
    #     return a

    def add(self, a: int, b: int) -> int:
        a %= 2**32
        b %= 2**32
        while b != 0:
            carry = ((a & b) << 1) % (2**32)
            a = (a ^ b) % (2**32)
            b = carry
        if a & (2**31):  # 负数
            return ~((a ^ (2**31)) ^ (2**31) - 1)
        else:  # 正数
            return a


# https://leetcode.cn/problems/missing-number-lcci/
# 面试题 17.04. 消失的数字 - EASY
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        return set(range(len(nums) + 1)).difference(set(nums)).pop()

    def missingNumber(self, nums: List[int]) -> int:
        return len(nums) * (len(nums) + 1) // 2 - sum(nums)

    def missingNumber(self, nums: List[int]) -> int:
        return sum(range(len(nums) + 1)) - sum(nums)

    def missingNumber(self, nums: List[int]) -> int:
        return functools.reduce(operator.xor, nums) ^ functools.reduce(
            operator.xor, range(len(nums) + 1)
        )

    def missingNumber(self, nums: List[int]) -> int:
        x = 0
        for i, v in enumerate(nums):
            x ^= i ^ v
        return x ^ len(nums)


# https://leetcode.cn/problems/find-longest-subarray-lcci/
# 面试题 17.05.  字母与数字 - MEDIUM
class Solution:
    def findLongestSubarray(self, array: List[str]) -> List[str]:
        s = set(string.ascii_letters)
        d = {0: -1}
        pre = mx = l = r = 0
        for i, v in enumerate(array):
            pre += 1 if v in s else -1
            if pre in d:
                if i - d[pre] > mx:
                    mx = i - d[pre]
                    l = d[pre] + 1
                    r = i + 1
            else:
                d[pre] = i
        return array[l:r]

    def findLongestSubarray(self, array: List[str]) -> List[str]:
        pre = list(itertools.accumulate((1 if v.isalpha() else -1 for v in array), 0))
        begin = end = 0
        first = {}
        for r, v in enumerate(pre):
            l = first.get(v, -1)
            if l < 0:
                first[v] = r
            elif r - l > end - begin:
                begin, end = l, r
        return array[begin:end]


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
