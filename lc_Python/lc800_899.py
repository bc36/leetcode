from typing import List
import collections


# 827 - Making A Large Island - HARD
# STEP 1: Explore every island using DFS, count its area
#         give it an island index and save the result to a {index: area} map.
# STEP 2: Loop every cell == 0,
#         check its connected islands and calculate total islands area.
class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        N = len(grid)

        # move(int x, int y), return all possible next position in 4 directions.
        def move(x: int, y: int):
            for i, j in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= x + i < N and 0 <= y + j < N:
                    yield x + i, y + j

        # Change the value of grid[x][y] to its index so act as an area
        def dfs(x: int, y: int, index: int) -> int:
            ret = 0
            grid[x][y] = index
            for i, j in move(x, y):
                if grid[i][j] == 1:
                    ret += dfs(i, j, index)
            return ret + 1

        # Since the grid has elements 0 or 1.
        # The island index is initialized with 2
        index = 2
        areas = {0: 0}
        # DFS every island and give it an index of island
        for x in range(N):
            for y in range(N):
                if grid[x][y] == 1:
                    areas[index] = dfs(x, y, index)
                    index += 1
        # Traverse every 0 cell and count biggest island it can conntect
        # The 'possible' connected island index is stored in a set to remove duplicate index.
        ret = max(areas.values())
        for x in range(N):
            for y in range(N):
                if grid[x][y] == 0:
                    possible = set(grid[i][j] for i, j in move(x, y))
                    # '+1' means grid[x][y] itself
                    ret = max(ret, sum(areas[index] for index in possible) + 1)
        return ret


# 859 - Buddy Strings - EASY
class Solution:
    def buddyStrings(self, s: str, goal: str) -> bool:
        if len(s) != len(goal):
            return False
        if s == goal:
            if len(set(s)) < len(s):
                return True
            else:
                return False
        diff = [(a, b) for a, b in zip(s, goal) if a != b]
        return len(diff) == 2 and diff[0] == diff[1][::-1]