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
