from typing import List
import collections, heapq, functools, math, random, queue


# https://leetcode-cn.com/problems/na-ying-bi/
# LCP 06. 拿硬币
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


# https://leetcode-cn.com/problems/xun-bao/
# LCP 13. 寻宝
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
        if nb == 0: return startToAnyPos[endX][endY]
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
        # 四个方向
        dd = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        # 计算（x, y）到maze中其他点的距离，结果保存在ret中
        def bfs(x, y, maze, m, n):
            ret = [[-1] * n for _ in range(m)]
            ret[x][y] = 0
            q = queue.Queue()
            q.put((x, y))
            while q.qsize():
                curx, cury = q.get()
                for dx, dy in dd:
                    nx = curx + dx
                    ny = cury + dy
                    if 0 <= nx < m and 0 <= ny < n and maze[nx][
                            ny] != '#' and ret[nx][ny] == -1:
                        ret[nx][ny] = ret[curx][cury] + 1
                        q.put((nx, ny))
            return ret

        m = len(maze)
        n = len(maze[0])
        startX = -1
        startY = -1
        endX = -1
        endY = -1
        # 机关 & 石头
        buttons = []
        stones = []
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
                else:
                    pass
        nb = len(buttons)
        ns = len(stones)
        startToAnyPos = bfs(startX, startY, maze, m, n)
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
            iToAnyPos = bfs(bx, by, maze, m, n)
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