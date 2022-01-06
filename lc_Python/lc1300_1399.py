import collections
from typing import List


# 1306 - Jump Game III - MEDIUM
class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        visited, self.ans = set(), False

        def dfs(idx: int):
            visited.add(idx)
            if 0 <= idx + arr[idx] < len(
                    arr) and idx + arr[idx] not in visited:
                dfs(idx + arr[idx])
            if 0 <= idx - arr[idx] < len(
                    arr) and idx - arr[idx] not in visited:
                dfs(idx - arr[idx])
            if not arr[idx]:
                self.ans = True
            return

        dfs(start)
        return self.ans

    def canReach(self, arr: List[int], start: int) -> bool:
        dq, seen = collections.deque([start]), {start}
        while dq:
            cur = dq.popleft()
            if arr[cur] == 0:
                return True
            for child in cur - arr[cur], cur + arr[cur]:
                if 0 <= child < len(arr) and child not in seen:
                    seen.add(child)
                    dq.append(child)
        return False


# 1345 - Jump Game IV - HARD
class Solution:
    def minJumps(self, arr: List[int]) -> int:
        # +1 / -1 / same value
        n, idx = len(arr), {}  # defaultdict(list)
        for i in range(n):
            # save left and right endpoints of the interval with the same value appearing consecutively
            if i in (0, n - 1):
                idx[arr[i]] = idx.get(arr[i], []) + [i]
            elif arr[i] != arr[i - 1] or arr[i] != arr[i + 1]:
                idx[arr[i]] = idx.get(arr[i], []) + [i]
        visited = [True] + [False] * (n - 1)
        queue = [(0, 0)]  # deque
        while queue:
            i, step = queue.pop(0)
            for j in (idx.get(arr[i], []) + [i - 1, i + 1]):
                if 0 <= j < n and not visited[j]:
                    if j == n - 1:
                        return step + 1
                    visited[j] = True
                    queue.append((j, step + 1))
            idx[arr[i]] = []  # has visited
        return 0