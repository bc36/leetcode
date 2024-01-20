"""
给定一个文本 t 和一个字符串 s, 尝试找到并展示 s 在 t 中的所有出现



https://www.zhihu.com/question/21923021/answer/37475572
https://oi-wiki.org/string/kmp/
https://cp-algorithms.com/string/prefix-function.html

lc 3008 https://leetcode.cn/problems/find-beautiful-indices-in-the-given-array-ii/
"""

import bisect
from typing import List


class KMP:
    """
    pa = KMP().find(s, a)
    pb = KMP().find(s, b)
    """

    def __init__(self):
        return

    @staticmethod
    def prefix_function(s: str) -> List[int]:
        """calculate the longest common true prefix and true suffix for s [:i] and s [:i]"""
        pi = [0] * len(s)
        for i in range(1, len(s)):
            j = pi[i - 1]
            while j > 0 and s[i] != s[j]:
                j = pi[j - 1]
            if s[i] == s[j]:
                j += 1
            pi[i] = j  # pi[i]<=i
        # pi[0] = 0
        return pi

    def find(self, s1: str, s2: str):
        """find the index position of s2 in s1"""
        n, m = len(s1), len(s2)
        pi = self.prefix_function(s2 + "#" + s1)
        ans = []
        for i in range(m + 1, m + n + 1):
            if pi[i] == m:
                ans.append(i - m - m)
        return ans


def prep(p: str) -> List[int]:
    pi = [0] * len(p)
    j = 0
    for i in range(1, len(p)):
        while j != 0 and p[j] != p[i]:
            j = pi[j - 1]
        if p[j] == p[i]:
            j += 1
        pi[i] = j
    return pi


class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        pa = prep(a + "#" + s)
        pb = prep(b + "#" + s)
        ia = [i - len(a) * 2 for i in range(len(pa)) if pa[i] == len(a)]
        ib = [i - len(b) * 2 for i in range(len(pb)) if pb[i] == len(b)]


class KMP:
    """
    i1 = KMP().match(s, a)
    i2 = KMP().match(s, b)
    """

    def partial(self, s: str) -> List[int]:
        g, pi = 0, [0] * len(s)
        for i in range(1, len(s)):
            while g and (s[g] != s[i]):
                g = pi[g - 1]
            pi[i] = g = g + (s[g] == s[i])
        return pi

    def match(self, s: str, pat: str) -> List[int]:
        pi = self.partial(pat)

        g, idx = 0, []
        for i in range(len(s)):
            while g and pat[g] != s[i]:
                g = pi[g - 1]
            g += pat[g] == s[i]
            if g == len(pi):
                idx.append(i + 1 - g)
                g = pi[g - 1]

        return idx


def kmp(text: str, pattern: str) -> List[int]:
    """
    pos_a = self.kmp(s, a)
    pos_b = self.kmp(s, b)
    """
    m = len(pattern)
    pi = [0] * m
    c = 0
    for i in range(1, m):
        v = pattern[i]
        while c and pattern[c] != v:
            c = pi[c - 1]
        if pattern[c] == v:
            c += 1
        pi[i] = c

    res = []
    c = 0
    for i, v in enumerate(text):
        v = text[i]
        while c and pattern[c] != v:
            c = pi[c - 1]
        if pattern[c] == v:
            c += 1
        if c == len(pattern):
            res.append(i - m + 1)
            c = pi[c - 1]
    return res
