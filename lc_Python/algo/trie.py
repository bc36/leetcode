"""trie

数据结构
"""


import collections, functools
from typing import List


def build_trie(words: List[str]) -> None:
    TRIE = lambda: collections.defaultdict(TRIE)
    trie = TRIE()
    for w in words:
        functools.reduce(dict.__getitem__, w, trie)["#"] = True
    return


def build_trie(words: List[List[int]]) -> None:
    trie = {}
    for w in words:
        r = trie
        for c in w:
            if c not in r:
                r[c] = {}
            r = r[c]
        r["end"] = True
    return


# wc 311 T4, 2416
class Solution:
    # 1.3s
    def sumPrefixScores(self, words: List[str]) -> List[int]:
        Trie = lambda: collections.defaultdict(Trie)
        CNT = "#"

        trie = Trie()
        for w in words:
            r = trie
            for ch in w:
                r = r[ch]
                r[CNT] = r.get(CNT, 0) + 1
        ans = []
        for w in words:
            r = trie
            score = 0
            for ch in w:
                r = r[ch]
                score += r[CNT]
            ans.append(score)
        return ans


def build_trie(words: List[List[int]]) -> None:
    # (1) 小慢 2s
    trie = {}
    for w in words:
        r = trie
        for c in w:
            if c not in r:
                r[c] = {}
                r[(c, "#")] = 1  # 打标记, 当前多有少个以 word[: i + 1] 为前缀, wc 311 T4
            else:
                r[(c, "#")] += 1
            r = r[c]
        r["end"] = True

    # (2) 小慢 2s
    trie = [None] * 27  # 最后一位用于计数
    trie[26] = 0
    for w in words:
        r = trie
        for ch in w:
            c = ord(ch) - ord("a")
            if r[c] is None:
                r[c] = [None] * 27
                r[c][26] = 0
            r = r[c]
            r[26] += 1

    # (3) 1s
    trie = {}
    for w in words:
        r = trie
        for c in w:
            if c not in r:
                r[c] = {}
                r[c]["cnt"] = 1  # 打标记, 当前多有少个以 word[: i + 1] 为前缀, wc 311 T4
            else:
                r[c]["cnt"] += 1
            r = r[c]
        r["end"] = True

    # (4)
    trie = {}
    for w in words:
        r = trie
        for c in w:
            if c not in r:
                r[c] = {}
            r = r[c]
            r["cnt"] = r.get("cnt", 0) + 1  # 打标记, 当前多有少个以 word[: i + 1] 为前缀, wc 311 T4
        r["end"] = True
    return
