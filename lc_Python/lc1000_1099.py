import collections
from typing import List


# 1047 - Remove All Adjacent Duplicates In String - EASY
# stack
class Solution:
    def removeDuplicates(self, s: str) -> str:
        stack = [s[0]]
        for i in range(1, len(s)):
            if stack and s[i] == stack[-1]:
                stack.pop()
            else:
                stack.append(s[i])
        return "".join(stack)


# two pointers
class Solution:
    def removeDuplicates(self, s: str) -> str:
        # pointers: 'ch' and 'end',
        # change 'ls' in-place.
        ls, end = list(s), -1
        for ch in ls:
            if end >= 0 and ls[end] == ch:
                end -= 1
            else:
                end += 1
                ls[end] = ch
        return "".join(ls[:end + 1])
