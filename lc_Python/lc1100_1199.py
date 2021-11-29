import bisect, functools
from typing import List


# 1137 - N-th Tribonacci Number - EASY
class Solution:
    @functools.lru_cache()
    def tribonacci(self, n: int) -> int:
        if n == 0: return 0
        if n == 1: return 1
        if n == 2: return 1
        return self.tribonacci(n -1) \
                + self.tribonacci(n - 2) \
                + self.tribonacci(n - 3)


class Solution:
    def __init__(self):
        self.cache = {0: 0, 1: 1, 2: 1}

    def tribonacci(self, n: int) -> int:
        if n in self.cache: return self.cache[n]
        self.cache[n] = self.tribonacci(n - 1) \
                        + self.tribonacci(n - 2) \
                        + self.tribonacci(n - 3)
        return self.cache[n]


class Solution:
    def tribonacci(self, n: int) -> int:
        if n == 0:
            return 0
        elif n < 3:
            return 1
        one, two, three, ans = 0, 1, 1, 0
        for _ in range(2, n):
            ans = one + two + three
            one, two, three = two, three, ans
        return ans


# 1146 - Snapshot Array - MEDIUM
# only update the change of each element, rather than record the whole arr
class SnapshotArray:
    def __init__(self, length: int):
        self.arr = [{0: 0} for _ in range(length)]
        self.snap_id = 0

    def set(self, index: int, val: int) -> None:
        self.arr[index][self.snap_id] = val
        return

    def snap(self) -> int:
        self.snap_id += 1
        return self.snap_id - 1

    def get(self, index: int, snap_id: int) -> int:
        d = self.arr[index]
        if snap_id in d:
            return d[snap_id]
        k = list(d.keys())
        i = bisect.bisect_left(k, snap_id)
        return d[k[i - 1]]


# 1178 - Number of Valid Words for Each Puzzle - HARD
# 超时
class Solution:
    def findNumOfValidWords(self, words: List[str],
                            puzzles: List[str]) -> List[int]:
        puzzleSet = [set(p) for p in puzzles]
        wordSet = [set(w) for w in words]
        # firstLetters = set([p[0] for p in puzzles])
        ans = []
        for i, puzzle in enumerate(puzzles):
            num = 0
            for j in range(len(words)):
                # contain the first letter of puzzle
                if puzzle[0] in wordSet[j]:
                    # every letter is in puzzle
                    if wordSet[j] <= puzzleSet[i]:
                        num += 1
            ans.append(num)

        return ans