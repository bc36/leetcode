from typing import List
import collections, heapq, functools, math, random, queue, bisect, itertools


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# https://leetcode-cn.com/problems/na-ying-bi/
# LCP 06. 拿硬币 - EASY
class Solution:
    def minCount(self, coins: List[int]) -> int:
        ans = 0
        for c in coins:
            if c & 1:
                ans += c // 2 + 1
            else:
                ans += c // 2
        return ans

    def minCount(self, coins: List[int]) -> int:
        return sum([(x + 1) // 2 for x in coins])


# https://leetcode-cn.com/problems/chuan-di-xin-xi/
# LCP 07. 传递信息 - EASY
class Solution:
    # bfs, O(n^k), O(n + len(relation) + n^k), length of queue:n^k
    def numWays(self, n: int, relation: List[List[int]], k: int) -> int:
        dic = collections.defaultdict(set)
        for r in relation:
            dic[r[0]].add(r[1])
        dq = collections.deque([0])
        ans = 0
        while dq and k:
            size = len(dq)
            for _ in range(size):
                cur = dq.popleft()
                for i in dic[cur]:
                    if k == 1 and i == n - 1:
                        ans += 1
                        continue
                    dq.append(i)
            k -= 1
        return ans

    # dfs, O(n^k), O(n + len(relation) + k), depth of stack:k
    def numWays(self, n: int, relation: List[List[int]], k: int) -> int:
        def dfs(start: int, k: int):
            if not k:
                if start == n - 1:
                    self.ans += 1
                return
            for i in edge[start]:
                dfs(i, k - 1)
            return

        edge = collections.defaultdict(list)
        for o, i in relation:
            edge[o].append(i)
        self.ans = 0
        dfs(0, k)
        return self.ans

    # dp, O(k * len(relation)), O(k * n),  dp[i][j] 为经过 i 轮传递到编号 j 的玩家的方案数
    def numWays(self, n: int, relation: List[List[int]], k: int) -> int:
        dp = [[0] * n for _ in range(k + 1)]
        dp[0][0] = 1
        for i in range(k):
            for src, dst in relation:
                dp[i + 1][dst] += dp[i][src]
        return dp[-1][-1]


# https://leetcode-cn.com/problems/ju-qing-hong-fa-shi-jian/
# LCP 08. 剧情触发时间
class Solution:
    # O(n * log(m)), O(n)
    def getTriggerTime(self, increase: List[List[int]],
                       requirements: List[List[int]]) -> List[int]:
        n = len(increase)
        c = [0] * (n + 1)
        r = [0] * (n + 1)
        h = [0] * (n + 1)
        for i in range(n):
            c[i + 1] = c[i] + increase[i][0]
            r[i + 1] = r[i] + increase[i][1]
            h[i + 1] = h[i] + increase[i][2]
        ans = []
        for need in requirements:
            t = max(bisect.bisect_left(c, need[0]),
                    bisect.bisect_left(r, need[1]),
                    bisect.bisect_left(h, need[2]))
            if t == len(c):
                t = -1
            ans.append(t)
        return ans


class Solution:
    def getTriggerTime(self, increase: List[List[int]],
                       requirements: List[List[int]]) -> List[int]:
        m = 100001
        c = [1e9] * m
        r = [1e9] * m
        h = [1e9] * m
        pre = [0, 0, 0]
        for i, v in enumerate(increase):
            pre[0] += v[0]
            pre[1] += v[1]
            pre[2] += v[2]
            for j in range(pre[0], -1, -1):
                if c[j] != 1e9:
                    break
                c[j] = i + 1
            for j in range(pre[1], -1, -1):
                if r[j] != 1e9:
                    break
                r[j] = i + 1
            for j in range(pre[2], -1, -1):
                if h[j] != 1e9:
                    break
                h[j] = i + 1
        ans = []
        for v in requirements:
            if sum(v) == 0:
                t = 0
            else:
                t = max(c[v[0]], r[v[1]], h[v[2]])
                if t == 1e9:
                    t = -1
            ans.append(t)
        return ans


# https://leetcode-cn.com/problems/zui-xiao-tiao-yue-ci-shu/
# LCP 09. 最小跳跃次数
class Solution:
    def minJump(self, jump: List[int]) -> int:
        n = len(jump)
        visited = [True] + [False] * (n - 1)
        queue = [(0, 0)]
        cur = far = 0
        while cur < len(queue):
            idx, step = queue[cur]
            if idx + jump[idx] >= n:
                return step + 1
            if not visited[idx + jump[idx]]:
                queue.append((idx + jump[idx], step + 1))
                visited[idx + jump[idx]] = True
            for j in range(far, idx):
                if not visited[j]:
                    queue.append((j, step + 1))
                    visited[j] = True
            far = max(far, idx + 1)
            cur += 1
        return -1

    def minJump(self, jump: List[int]) -> int:
        n = len(jump)
        dp = [0] * n
        for i in range(n - 1, -1, -1):
            if i + jump[i] >= n:
                dp[i] = 1
            else:
                dp[i] = 1 + dp[i + jump[i]]
            for j in range(i + 1, n):  # 后面可以往前跳
                if dp[j] > dp[i]:  # 下标比j还大的后面的就可以选择从j跳, 不用从i跳
                    dp[j] = dp[i] + 1
                else:
                    break
        return dp[0]


# https://leetcode-cn.com/problems/er-cha-shu-ren-wu-diao-du/
# LCP 10. 二叉树任务调度
class Solution(object):
    def minimalExecTime(self, root: TreeNode) -> float:
        def dfs(root):
            if root is None:
                return 0., 0.
            a, b = dfs(root.left)
            c, d = dfs(root.right)
            total = root.val + a + c
            if a < c:
                a, c = c, a
                b, d = d, b
            if a - 2 * b <= c:
                p = (a + c) / 2
            else:
                p = c + b
            return total, p

        total, p = dfs(root)
        return total - p

    # TODO
    def minimalExecTime(self, root: TreeNode) -> float:
        def dfs(root: TreeNode):
            """
            return:
                [0]: 该结点及子树的最短运行时间, 左右子并行执行，或四孙的并行执行
                [1]: 该结点及子树的最大运行时间, 令左右子串行执行，让父节点去分配并行
            """
            if not root: return 0, 0
            left = dfs(root.left)
            right = dfs(root.right)
            return max(left[0], right[0], (left[1] + right[1]) /
                       2) + root.val, left[1] + right[1] + root.val

        return dfs(root)[0]


# https://leetcode-cn.com/problems/qi-wang-ge-shu-tong-ji/
# LCP 11. 期望个数统计
class Solution:
    def expectNumber(self, scores: List[int]) -> int:
        return len(set(scores))


# https://leetcode-cn.com/problems/xiao-zhang-shua-ti-ji-hua/
# LCP 12. 小张刷题计划
class Solution:
    def minTime(self, time: List[int], m: int) -> int:
        def check(limit: int, m: int) -> bool:
            total = mx = 0
            for t in time:
                mx = max(mx, t)
                total += t
                if total - mx > limit:
                    m -= 1
                    total = mx = t
            return m > 0

        lo, hi = 0, sum(time) // m
        while lo < hi:
            mid = (lo + hi) >> 1
            if check(mid, m):
                hi = mid
            else:
                lo = mid + 1
        return lo


# https://leetcode-cn.com/problems/xun-bao/
# LCP 13. 寻宝 - HARD
class Solution:
    def minimalSteps(self, maze: List[str]) -> int:
        # 计算（x, y）到maze中其他点的距离，结果保存在ret中
        def bfs(x: int, y: int) -> List[List[int]]:
            ret = [[-1] * n for _ in range(m)]
            ret[x][y] = 0
            dq = collections.deque([(x, y)])
            while dq:
                i, j = dq.popleft()
                for nx, ny in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                    if 0 <= nx < m and 0 <= ny < n and maze[nx][
                            ny] != '#' and ret[nx][ny] == -1:
                        ret[nx][ny] = ret[i][j] + 1
                        dq.append((nx, ny))
            return ret

        m, n = len(maze), len(maze[0])
        startX = startY = endX = endY = -1
        buttons, stones = [], []  # 机关 & 石头
        # 记录所有特殊信息的位置
        for i in range(m):
            for j in range(n):
                if maze[i][j] == 'S':
                    startX = i
                    startY = j
                elif maze[i][j] == 'T':
                    endX = i
                    endY = j
                elif maze[i][j] == 'O':
                    stones.append((i, j))
                elif maze[i][j] == 'M':
                    buttons.append((i, j))
        nb, ns = len(buttons), len(stones)
        startToAnyPos = bfs(startX, startY)
        # 若没有机关，最短距离就是(startX, startY)到(endX, endY)的距离
        if nb == 0:
            return startToAnyPos[endX][endY]
        # 记录第i个机关到第j个机关的最短距离
        # dist[i][nb]表示到起点的距离， dist[i][nb+1]表示到终点的距离
        dist = [[-1] * (nb + 2) for _ in range(nb)]
        # 遍历所有机关，计算其和其他点的距离
        buttonsToAnyPos = []
        for i in range(nb):
            bx, by = buttons[i]
            # 记录第i个机关到其他点的距离
            iToAnyPos = bfs(bx, by)
            buttonsToAnyPos.append(iToAnyPos)
            # 第i个机关到终点的距离就是(bx, by)到(endX, endY)的距离
            dist[i][nb + 1] = iToAnyPos[endX][endY]
        for i in range(nb):
            # 计算第i个机关到(startX, startY)的距离
            # 即从第i个机关出发，通过每个石头(sx, sy)，到(startX, startY)的最短距离
            tmp = -1
            for j in range(ns):
                sx, sy = stones[j]
                if buttonsToAnyPos[i][sx][sy] != -1 and startToAnyPos[sx][
                        sy] != -1:
                    if tmp == -1 or tmp > buttonsToAnyPos[i][sx][
                            sy] + startToAnyPos[sx][sy]:
                        tmp = buttonsToAnyPos[i][sx][sy] + startToAnyPos[sx][sy]
            dist[i][nb] = tmp
            # 计算第i个机关到第j个机关的距离
            # 即从第i个机关出发，通过每个石头(sx, sy)，到第j个机关的最短距离
            for j in range(i + 1, nb):
                mn = -1
                for k in range(ns):
                    sx, sy = stones[k]
                    if buttonsToAnyPos[i][sx][sy] != -1 and buttonsToAnyPos[j][
                            sx][sy] != -1:
                        if mn == -1 or mn > buttonsToAnyPos[i][sx][
                                sy] + buttonsToAnyPos[j][sx][sy]:
                            mn = buttonsToAnyPos[i][sx][sy] + buttonsToAnyPos[
                                j][sx][sy]
                # 距离是无向图，对称的
                dist[i][j] = mn
                dist[j][i] = mn
        # 若有任意一个机关 到起点或终点没有路径(即为-1),则说明无法达成，返回-1
        for i in range(nb):
            if dist[i][nb] == -1 or dist[i][nb + 1] == -1:
                return -1
        # dp数组， -1代表没有遍历到, 1<<nb表示题解中提到的mask, dp[mask][j]表示当前处于第j个机关，总的触发状态为mask所需要的最短路径, 由于有2**nb个状态，因此1<<nb的开销必不可少
        dp = [[-1] * nb for _ in range(1 << nb)]
        # 初识状态，即从start到第i个机关，此时mask的第i位为1，其余位为0
        for i in range(nb):
            dp[1 << i][i] = dist[i][nb]
        # 二进制中数字大的mask的状态肯定比数字小的mask的状态多，所以直接从小到大遍历更新即可
        for mask in range(1, (1 << nb)):
            for i in range(nb):
                # 若当前位置是正确的，即mask的第i位是1
                if mask & (1 << i) != 0:
                    for j in range(nb):
                        # 选择下一个机关j,要使得机关j目前没有到达，即mask的第j位是0
                        if mask & (1 << j) == 0:
                            nextMask = mask | (1 << j)
                            if dp[nextMask][j] == -1 or dp[nextMask][
                                    j] > dp[mask][i] + dist[i][j]:
                                dp[nextMask][j] = dp[mask][i] + dist[i][j]
        # 最后一个机关到终点
        ans = -1
        finalMask = (1 << nb) - 1
        for i in range(nb):
            if ans == -1 or ans > dp[finalMask][i] + dist[i][nb + 1]:
                ans = dp[finalMask][i] + dist[i][nb + 1]
        return ans

    def minimalSteps(self, maze: List[str]) -> int:
        n = len(maze)
        m = len(maze[0])
        buttons = []
        stones = []
        sx, sy, tx, ty = None, None, None, None
        for i, r in enumerate(maze):
            for j, c in enumerate(r):
                if c == 'M':
                    buttons.append((i, j))
                elif c == 'O':
                    stones.append((i, j))
                elif c == 'S':
                    sx = i
                    sy = j
                elif c == 'T':
                    tx = i
                    ty = j

        def dfs(x, y):
            ret = [[-1] * m for _ in range(n)]
            ret[x][y] = 0
            Q = collections.deque()
            Q.append((x, y))

            def inBound(x, y):
                #print(x, y, n, m)
                return x >= 0 and x < n and y >= 0 and y < m

            while Q:
                cx, cy = Q.popleft()
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    x = cx + dx
                    y = cy + dy
                    if inBound(x, y) and maze[x][y] != '#' and ret[x][y] == -1:
                        ret[x][y] = ret[cx][cy] + 1
                        Q.append((x, y))
            return ret

        nb = len(buttons)
        ns = len(stones)
        startDist = dfs(sx, sy)
        if nb == 0:  #没有机关
            return startDist[tx][ty]
        dd = []  #每个机关出发，到其他所有点的距离:中间结果
        dist = [[-1] * (nb + 2) for _ in range(nb)]  #每个机关到其他机关的距离，到起点，终点的距离
        for i in range(nb):
            x, y = buttons[i]
            d = dfs(x, y)
            dd.append(d)
            dist[i][nb + 1] = d[tx][ty]
            if dist[i][nb + 1] == -1:
                return -1
        for i in range(nb):
            #起点-->第k个石头的距离 + 第k个石头--->第i个机关的距离
            tmp = -1
            ddi = dd[i]
            for k in range(ns):
                x, y = stones[k]
                if ddi[x][y] != -1 and startDist[x][y] != -1:
                    if tmp == -1 or tmp > startDist[x][y] + ddi[x][y]:
                        tmp = startDist[x][y] + ddi[x][y]
            dist[i][nb] = tmp
            if dist[i][nb] == -1:
                return -1
            #第i个机关-->某个石头的距离 + 这个石头--->第j个机关的距离
            for j in range(i + 1, nb):
                tmp = -1
                ddj = dd[j]
                for k in range(ns):
                    x, y = stones[k]
                    if ddi[x][y] != -1 and ddj[x][y] != -1:
                        t = ddi[x][y] + ddj[x][y]
                        if tmp == -1 or tmp > t:
                            tmp = t
                dist[i][j] = tmp
                dist[j][i] = tmp
        nb1 = 1 << nb
        dp = [[-1] * nb for _ in range(nb1)]
        for i in range(nb):
            dp[1 << i][i] = dist[i][nb]
        for mask in range(1, nb1):
            for i in range(nb):
                if (mask & (1 << i)) == 0:
                    continue
                for j in range(nb):
                    if (mask & (1 << j)) != 0:
                        continue
                    next = mask | (1 << j)
                    t = dp[mask][i] + dist[i][j]
                    if dp[next][j] == -1 or dp[next][j] > t:
                        dp[next][j] = t
        result = -1
        finalMask = nb1 - 1
        for i in range(nb):
            t = dp[finalMask][i] + dist[i][nb + 1]
            if result == -1 or result > t:
                result = t
        return result