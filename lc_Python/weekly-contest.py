from typing import List
import collections, functools

# 274


# 5969
# https://leetcode-cn.com/problems/destroying-asteroids/
# https://leetcode-cn.com/contest/weekly-contest-274/problems/destroying-asteroids/
class Solution:
    def asteroidsDestroyed(self, mass: int, asteroids: List[int]) -> bool:
        asteroids.sort()
        for asteroid in asteroids:
            if mass < asteroid:
                return False
            mass += asteroid
        return True


# 5970
# https://leetcode-cn.com/problems/maximum-employees-to-be-invited-to-a-meeting/
# https://leetcode-cn.com/contest/weekly-contest-274/problems/maximum-employees-to-be-invited-to-a-meeting/
class Solution:
    def maximumInvitations(self, f: List[int]) -> int:
        # functional graph
        # either table consists of one cycle
        # or multiple (two chains + cycle of two)
        # bidirectional edges
        edges = collections.defaultdict(list)
        n = len(f)
        for i in range(n):
            edges[f[i]].append(i)
            edges[i].append(f[i])
        seen = [False] * n
        self.l = 0

        def dfs(v, p, d):
            self.l = max(self.l, d)
            seen[v] = True
            for i in edges[v]:
                if i != p and not seen[i]:
                    dfs(i, v, d + 1)

        total = 0
        best = 0
        for i in range(n):
            if seen[i]:
                continue
            if f[f[i]] == i:  # cycle of two
                self.l = 0
                dfs(i, f[i], 0)
                cur = self.l
                self.l = 0
                dfs(f[i], i, 0)
                cur += self.l
                total += cur + 2
        for i in range(n):
            if not seen[i]:
                dfs(i, i, 0)
                l = [i]
                j = i
                cur = 0
                s = set()
                s.add(i)
                while True:
                    j = f[j]
                    if j in s:
                        while True:
                            cur += 1
                            if l.pop() == j:
                                break
                        best = max(best, cur)
                        break
                    s.add(j)
                    l.append(j)
        return max(total, best)

    def maximumInvitations(self, favorite: List[int]) -> int:
        # there must be one or more cycle in this graph
        # each connected region in the graph will lead to 1 cycle
        # Either
        #  all cycles of size 2 fulfilled, each node can have 1 edge
        #  one cycle fulfilled, nothing else
        visited = [False] * len(favorite)
        depth = [None] * len(favorite)
        represent = [None] * len(favorite)
        cycles = []

        def dfs(n, d=0):
            if visited[n]:
                if represent[n] is None:
                    # found cycle
                    represent[n] = n
                    cycles.append((n, d - depth[n]))
                    return n
                else:
                    # revisit
                    return represent[n]
            visited[n] = True
            depth[n] = d
            ans = dfs(favorite[n], d + 1)
            represent[n] = ans
            return ans

        for i in range(len(favorite)):
            dfs(i)
        ans = 0
        for i, j in cycles:
            ans = max(ans, j)
        rev = []
        for i in favorite:
            rev.append([])
        for index, i in enumerate(favorite):
            rev[i].append(index)

        def michi(x, nashi):
            answer = 0
            for i in rev[x]:
                if i == nashi:
                    continue
                answer = max(answer, michi(i, nashi) + 1)
            return answer

        ans2 = 0
        for i0, j in cycles:
            if j != 2:
                continue
            i1 = favorite[i0]
            assert i0 == favorite[i1]
            ans2 += 2 + michi(i0, i1) + michi(i1, i0)
        return max(ans, ans2)

    def maximumInvitations(self, favorite: List[int]) -> int:
        n = len(favorite)
        f = [0] * n
        e = [set() for i in range(n)]
        num = n + 1
        for i in range(n):
            e[favorite[i]].add(i)

        def findquan(x, num):
            if f[x] == num:
                return
            f[x] = num
            findquan(favorite[x], num)

        def find(x, y):
            if f[x] == y:
                nonlocal num
                findquan(x, num)
                num += 1
                return
            if f[x] != 0:
                return

            f[x] = y
            if f[favorite[x]] == 0 or f[favorite[x]] == y:
                find(favorite[x], y)

        def finddeep(x):
            d = 0
            nonlocal n
            for i in e[x]:
                if f[i] <= n:
                    d = max(d, finddeep(i))
            return d + 1

        for i in range(n):
            if f[i] == 0:
                find(i, i + 1)
        ans = 0
        ans2 = 0
        done = set()
        for i in range(n):
            if f[i] > n and f[i] not in done:
                done.add(f[i])
                p = i
                num = 1
                while favorite[p] != i:
                    num += 1
                    p = favorite[p]
                if num > 2:
                    ans = max(ans, num)
                if num == 2:
                    ans2 += finddeep(p) + finddeep(i)
        return max(ans, ans2)

    def maximumInvitations(self, A: List[int]) -> int:
        res = 0
        pairs = []
        d = collections.defaultdict(list)
        for i, a in enumerate(A):
            if A[a] == i:
                pairs.append(i)
            else:
                d[a].append(i)
        # print(pairs, d)
        @functools.lru_cache(None)
        def cal(x):
            if d[x] == []:
                return 1
            mx = 0
            for c in d[x]:
                mx = max(mx, cal(c))
            return mx + 1

        for a in pairs:
            # print(a, cal(a))
            res += cal(a)
        # print(res)

        def check(i):
            dd = {}
            cur = i
            while True:
                dd[cur] = len(dd)
                cur = A[cur]
                if cur in d:
                    for k in dd:
                        d[k] = -1
                    break
                if cur in dd:
                    length = len(dd) - dd[cur]
                    for k, v in dd.items():
                        if v < len(dd) - length:
                            d[k] = -1
                        else:
                            d[k] = length
                    break

        d = {}
        for i, a in enumerate(A):
            if i in d:
                continue
            else:
                check(i)
        return max(res, max(d.values()))

    def maximumInvitations(self, f: List[int]) -> int:
        def dfs(cur):
            res = 0
            mark[cur] = True
            for nxt in e[cur]:
                if nxt != f[cur]:
                    res = max(res, dfs(nxt))
            res += 1
            return res

        def dfs2(cur, root):
            nonlocal res
            nonlocal counter
            nxt = f[cur]
            if rank[nxt] >= rank[root]:
                res = max(res, depth[cur] + 1 - depth[nxt])
            elif rank[nxt] < 0:
                rank[nxt] = counter
                counter += 1
                depth[nxt] = depth[cur] + 1
                dfs2(nxt, root)

        n = len(f)
        e = [set() for _ in range(n)]
        for i in range(n):
            e[f[i]].add(i)
        q = []
        for i in range(n):
            if f[f[i]] == i:
                q.append(i)
        mark = [False] * n
        res = 0
        for root in q:
            res += dfs(root)
        depth = [-1] * n
        rank = [-1] * n
        counter = 0
        for i in range(n):
            if not mark[i]:
                depth[i] = 0
                rank[i] = counter
                counter += 1
                dfs2(i, i)
        return res
