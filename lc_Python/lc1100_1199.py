import bisect
from typing import List


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
                        num +=1
            ans.append(num)
                
        return ans