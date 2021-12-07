import collections
from typing import List


# 1816 - Truncate Sentence - EASY
class Solution:
    def truncateSentence(self, s: str, k: int) -> str:
        return " ".join(s.split()[:k])