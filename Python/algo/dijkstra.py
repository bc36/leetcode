"""dijkstra

图论, 最短路

(贪心)求解 非负权图 上单源最短路径的算法
加边 O(1) 操作多, 查询操作少

时间复杂度: 
    暴力实现: O(n^2 + m) = O(n^2)
    优先队列: O(mlogm), 堆内元素个数是 m
稀疏图中 m = O(n), 稠密图中 m = O(n^2)
稀疏图 -> 暴力, 稠密图 -> 优先队列

"""

import heapq, math
from typing import List, Tuple


def dijkstra(
    g: List[List[Tuple[int]]], start: int, d1: int = -1, d2: int = -1
) -> List[int]:
    """
    返回从 start 到每个点的最短路 dist: List[int]:

    Dijkstra 求(有向)(带权)最短路, (路)变成负数求可以求最长路(还是正权值)
    d1, d2 用于去除图中的环

    O(mlogm) / O(m), 稀疏图中 m = O(n), 稠密图中 m = (n^2)
    """
    dist = [math.inf] * len(g)
    dist[start] = 0
    q = [(0, start)]
    while q:
        cur, x = heapq.heappop(q)
        if cur > dist[x]:
            continue
        for y, w in g[x]:
            # if x == d1 and y == d2 or x == d2 and y == d1:  # 去除环
            #     continue
            new = dist[x] + w
            if new < dist[y]:
                dist[y] = new
                heapq.heappush(q, (new, y))
    return dist


def dijkstra(g: List[List[Tuple[int]]], start: int, end: int) -> int:
    """
    返回从 start 到 end 的最短路
    如果路径不存在, 返回 math.inf

    O(mlogm) / O(m), 稀疏图中 m = O(n), 稠密图中 m = (n^2)
    """
    dist = [math.inf] * len(g)
    dist[start] = 0
    q = [(0, start)]
    while q:
        cur, x = heapq.heappop(q)
        if cur > dist[x]:
            continue
        if x == end:
            return cur
        for y, w in g[x]:
            new = cur + w
            if new < dist[y]:
                dist[y] = new
                heapq.heappush(q, (new, y))
    return math.inf


def dijkstra(g: List[List[Tuple[int]]], start: int, end: int) -> int:
    """
    返回从 start 到 end 的最短路
    如果路径不存在, 返回 math.inf

    O(mlogm) / O(m), 稀疏图中 m = O(n), 稠密图中 m = (n^2)
    """
    vis = [False] * len(g)
    q = [(0, start)]
    while q:
        cur, x = heapq.heappop(q)
        if vis[x]:
            continue
        if x == end:
            return cur
        vis[x] = True
        for y, w in g[x]:
            heapq.heappush(q, (cur + w, y))
    return math.inf


edges = [[1, 2, 3], [1, 3, 4]]  # [x, y, w]
n = 3
g = [[] for _ in range(n)]
for x, y, w in edges:
    g[x].append((y, w))
    g[y].append((x, w))


def dijkstra(g: List[List[Tuple[int]]], start: int, end: int) -> int:
    dist = [math.inf] * len(g)
    dist[start] = 0
    q = [(0, start)]
    while q:
        cur, x = heapq.heappop(q)
        if cur > dist[x]:
            continue
        if x == end:
            return cur
        for y in g[x]:
            new = cur + g[x][y]
            if new < dist[y]:
                dist[y] = new
                heapq.heappush(q, (new, y))
    return math.inf


edges = [[1, 2, 3], [1, 3, 4]]  # [x, y, w]
n = 3
g = [dict() for _ in range(n)]
for x, y, w in edges:
    g[x][y] = g[y][x] = w


# 朴素 Dijkstra 算法每次求最短路的时间复杂度为 O(n^2)
# 在稠密图中, 比堆的实现要快, (实际测试一般题还是稀疏图多...)
def dijkstra(g: List[List[Tuple[int]]], start: int, end: int) -> int:
    n = len(g)
    dist = [math.inf] * n
    dist[start] = 0
    vis = [False] * n
    while True:  # 最多 n 次
        x = -1
        for i in range(n):
            if not vis[i] and (x < 0 or dist[i] < dist[x]):
                x = i
        if x < 0 or dist[x] == math.inf:
            return -1
        if x == end:  # 巨大优化
            return dist[x]
        vis[x] = True
        for y, w in enumerate(g[x]):
            if dist[x] + w < dist[y]:
                dist[y] = dist[x] + w
