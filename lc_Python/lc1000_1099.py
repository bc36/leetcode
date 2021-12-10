import collections
from typing import List


# 1005 - Maximize Sum Of Array After K Negations - EASY
class Solution:
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        neg = [x for x in nums if x < 0]
        pos = [x for x in nums if x > 0]
        if len(neg) >= k:
            neg.sort()
            return sum(pos) - sum(neg[:k]) + sum(neg[k:])
        k -= len(neg)
        if not k & 1:
            return sum(pos) - sum(neg)
        tmp = min([abs(x) for x in nums])
        return sum(pos) - sum(neg) - 2 * tmp


# 1010 - Pairs of Songs With Total Durations Divisible by 60 - MEDIUM
class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        c, ret = [0] * 60, 0
        for t in time:
            ret += c[-t % 60]
            c[t % 60] += 1
        return ret


class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        ans, cnt = 0, collections.Counter()
        for t in time:
            theOther = -t % 60
            ans += cnt[theOther]
            cnt[t % 60] += 1
        return ans


class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        ans, cnt = 0, collections.Counter()
        for t in time:
            theOther = -t % 60
            ans += cnt[theOther]
            cnt[t % 60] += 1
        return ans


# 1034 - Coloring A Border - MEDIUM
class Solution:
    # bfs
    def colorBorder(self, grid: List[List[int]], row: int, col: int,
                    color: int) -> List[List[int]]:
        position, borders, originalColor = [(row, col)], [], grid[row][col]
        visited = [[False] * len(grid[0]) for _ in range(len(grid))]
        visited[row][col] = True
        while position:
            x, y = position.pop()
            isBorder = False
            for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if not (0 <= nx < len(grid) and 0 <= ny < len(grid[0])
                        and grid[nx][ny] == originalColor):
                    isBorder = True
                elif not visited[nx][ny]:
                    visited[nx][ny] = True
                    position.append((nx, ny))
            if isBorder:
                borders.append((x, y))
        for i, j in borders:
            grid[i][j] = color
        return grid

    # bfs
    def colorBorder(self, grid: List[List[int]], row: int, col: int,
                    color: int) -> List[List[int]]:
        m, n = len(grid), len(grid[0])
        bfs, component, border = [[row, col]], set([(row, col)]), set()
        while bfs:
            r, c = bfs.pop()
            for i, j in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
                x, y = r + i, c + j
                if 0 <= x < m and 0 <= y < n and grid[x][y] == grid[r][c]:
                    if (x, y) not in component:
                        bfs.append([x, y])
                        component.add((x, y))
                else:
                    border.add((r, c))
        for x, y in border:
            grid[x][y] = color
        return grid

    # dfs
    def colorBorder(self, grid: List[List[int]], row: int, col: int,
                    color: int) -> List[List[int]]:
        visited, m, n = set(), len(grid), len(grid[0])

        def dfs(x: int, y: int) -> bool:
            if (x, y) in visited:
                return True
            if not (0 <= x < m and 0 <= y < n
                    and grid[x][y] == grid[row][col]):
                return False
            visited.add((x, y))
            if dfs(x + 1, y) + dfs(x - 1, y) + dfs(x, y + 1) + dfs(x,
                                                                   y - 1) < 4:
                grid[x][y] = color
            return True

        dfs(row, col)
        return grid


# 1047 - Remove All Adjacent Duplicates In String - EASY
class Solution:
    # stack
    def removeDuplicates(self, s: str) -> str:
        stack = [s[0]]
        for i in range(1, len(s)):
            if stack and s[i] == stack[-1]:
                stack.pop()
            else:
                stack.append(s[i])
        return "".join(stack)

    # two pointers
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