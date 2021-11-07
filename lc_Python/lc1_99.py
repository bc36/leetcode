import bisect, collections, functools, random, operator
from typing import Iterable
'''
Function usually used

bit operation
&   bitwise AND
|   bitwise OR
^   bitwise XOR
~   bitwise NOT Inverts all the bits (~x = -x-1)
<<  left shift
>>  right shift
'''

# For viewing definitions
bisect.bisect_left()
collections.Counter(dict)
collections.deque(Iterable)
random.randint()
functools.reduce()
operator.xor()


# 3 - Longest Substring Without Repeating Characters - MEDIUM
# sliding window + hashmap
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        left, right, ans = 0, 0, 0
        dic = {}
        while right < len(s):
            if s[right] in dic and dic[s[right]] >= left:
                left = dic[s[right]] + 1
            dic[s[right]] = right
            right += 1
            ans = max(ans, right - left)
        return ans


# ord(), chr() / byte -> position
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        exist = [0 for _ in range(256)]
        left, right, ans = 0, 0, 0
        while right < len(s):
            if exist[ord(s[right]) - 97] == 0:
                exist[ord(s[right]) - 97] += 1
                right += 1
            else:
                exist[ord(s[right]) - 97] -= 1
                left += 1

            ans = max(ans, right - left)
        return ans


# 43 - Multiply Strings - MEDIUM
class Solution:
    def multiply(self, num1, num2):
        res = [0] * (len(num1) + len(num2))
        for i, e1 in enumerate(reversed(num1)):
            for j, e2 in enumerate(reversed(num2)):
                res[i + j] += int(e1) * int(e2)
                res[i + j + 1] += res[i + j] // 10
                res[i + j] %= 10
        # reverse, prepare to output
        while len(res) > 1 and res[-1] == 0:
            res.pop()
        return ''.join(map(str, res[::-1]))
