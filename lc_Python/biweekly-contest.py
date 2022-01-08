import collections, itertools, functools
from typing import List

# 68 / 2021.12.25

# https://leetcode-cn.com/problems/check-if-a-parentheses-string-can-be-valid/
# 5948 判断一个括号字符串是否有效. 正反遍历, 可能的左括号最大最小值. 类似678
class Solution:
    def canBeValid(self, s: str, locked: str) -> bool:
        if len(s) % 2 == 1: return False
        # 正序遍历: 未匹配的左括号 ( 的最大数目
        cnt = 0
        for ch, b in zip(s, locked):
            if ch == '(' and b == '1':
                cnt += 1
            elif ch == ')' and b == '1':
                cnt -= 1
            elif b == '0':
                cnt += 1
            if cnt < 0: return False
        # 逆序遍历: 未匹配的右括号 ) 的最大数目
        cnt = 0
        for ch, b in zip(s[::-1], locked[::-1]):
            if ch == ')' and b == '1':
                cnt += 1
            elif ch == '(' and b == '1':
                cnt -= 1
            elif b == '0':
                cnt += 1
            if cnt < 0: return False
        return True

    def canBeValid(self, s: str, locked: str) -> bool:
        if len(s) % 2 == 1: return False
        # 未匹配的左括号的最大, 最小值
        max_left = min_left = 0
        for ch, b in zip(s, locked):
            # locked[i]==1时, 无法改变字符, 直接加减
            if ch == '(' and b == '1':
                max_left += 1
                min_left += 1
            elif ch == ')' and b == '1':
                max_left -= 1
                min_left -= 1
            # locked[i]==0时, 可作为通配符,
            # 贪心地将: 未匹配的左括号的最大值+1, 最小值-1
            elif b == '0':
                max_left += 1
                min_left -= 1
            # 保持当前未匹配的左括号的最小值>=0
            min_left = max(0, min_left)
            # 未匹配的左括号的最大值不能为负
            if max_left < 0:
                return False
        return min_left == 0  # 最终未匹配的左括号的最小值应为0


# 69 / 2022.1.8
# https://leetcode-cn.com/problems/stamping-the-grid/
# https://leetcode.com/problems/stamping-the-grid/
# 5931. 用邮票贴满网格图
# 直接check, 更改矩阵会超时 -> 二维前缀和
class Solution:
    def possibleToStamp(self, grid: List[List[int]], stampHeight: int, stampWidth: int) -> bool:
        m, n = len(grid), len(grid[0])
        sum = [[0] * (n + 1) for _ in range(m + 1)]
        diff = [[0] * (n + 1) for _ in range(m + 1)]
        for i, row in enumerate(grid):
            for j, v in enumerate(row):  # grid 的二维前缀和
                sum[i + 1][j + 1] = sum[i + 1][j] + sum[i][j + 1] - sum[i][j] + v

        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v == 0:
                    x, y = i + stampHeight, j + stampWidth  # 注意这是矩形右下角横纵坐标都 +1 后的位置
                    if x <= m and y <= n and sum[x][y] - sum[x][j] - sum[i][y] + sum[i][j] == 0:
                        diff[i][j] += 1
                        diff[i][y] -= 1
                        diff[x][j] -= 1
                        diff[x][y] += 1  # 更新二维差分

        # 还原二维差分矩阵对应的计数矩阵，这里用滚动数组实现
        cnt, pre = [0] * (n + 1), [0] * (n + 1)
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                cnt[j + 1] = cnt[j] + pre[j + 1] - pre[j] + diff[i][j]
                if cnt[j + 1] == 0 and v == 0:
                    return False
            cnt, pre = pre, cnt
        return True