package src;

import java.util.*;
import java.util.stream.*;

public class Lc1200_1299 {
    // 1222. Queens That Can Attack the King - MEDIUM
    class Solution1222a {
        private final static int[][] directions = { { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 }, { -1, -1 },
                { 0, -1 }, { 1, -1 } };

        public List<List<Integer>> queensAttacktheKing(int[][] queens, int[] king) {
            boolean[][] isQueen = new boolean[8][8]; // 数组效率比哈希表高
            for (int[] q : queens) {
                isQueen[q[0]][q[1]] = true;
            }
            List<List<Integer>> ans = new ArrayList<>();
            for (int[] d : directions) {
                int x = king[0] + d[0];
                int y = king[1] + d[1];
                while (0 <= x && x < 8 && 0 <= y && y < 8) {
                    if (isQueen[x][y]) {
                        ans.add(List.of(x, y));
                        break;
                    }
                    x += d[0];
                    y += d[1];
                }
            }
            return ans;
        }
    }

    // 1253. Reconstruct a 2-Row Binary Matrix - MEDIUM
    class Solution1253a {
        public List<List<Integer>> reconstructMatrix(int upper, int lower, int[] colsum) {
            List<List<Integer>> ans = new ArrayList<>();
            int[] a = new int[colsum.length];
            int[] b = new int[colsum.length];
            for (int i = 0; i < colsum.length; i++) {
                if (colsum[i] > 1) {
                    upper--;
                    lower--;
                    a[i] = 1;
                    b[i] = 1;
                }
            }
            if (upper < 0) {
                return ans;
            }
            for (int i = 0; i < colsum.length; i++) {
                if (colsum[i] == 1) {
                    if (upper > 0) {
                        upper--;
                        a[i] = 1;
                    } else {
                        lower--;
                        b[i] = 1;
                    }
                }
            }
            if (lower == 0) {
                // 4ms
                // ans.add(a);
                // ans.add(b);
                // 11ms
                ans.add(IntStream.of(a).boxed().toList());
                ans.add(IntStream.of(b).boxed().toList());
                // 14ms
                // ans.add(Arrays.stream(a).boxed().toList());
                // ans.add(Arrays.stream(b).boxed().toList());
            }
            return ans;
        }
    }

    class Solution1253b { // 13ms
        public List<List<Integer>> reconstructMatrix(int upper, int lower, int[] colsum) {
            List<Integer> a = new ArrayList<>();
            List<Integer> b = new ArrayList<>();
            for (int j = 0; j < colsum.length; ++j) {
                int x = 0, y = 0;
                if (colsum[j] == 2) {
                    x = y = 1;
                    upper--;
                    lower--;
                } else if (colsum[j] == 1) {
                    if (upper > lower) {
                        upper--;
                        x = 1;
                    } else {
                        lower--;
                        y = 1;
                    }
                }
                if (upper < 0 || lower < 0) {
                    break;
                }
                a.add(x);
                b.add(y);
            }
            return upper == 0 && lower == 0 ? List.of(a, b) : List.of();
        }
    }

    // 1254. Number of Closed Islands - MEDIUM
    class Solution1254a {
        // 1. dfs 标记 第/最后 一 行/列 格子为 1
        // 2. 再次遍历, 遇到 0 一定是封闭岛屿, 然后 dfs 标记为 1
        public int closedIsland(int[][] grid) {
            int m = grid.length, n = grid[0].length;
            for (int i = 0; i < m; i++) {
                int step = i == 0 || i == m - 1 ? 1 : n - 1;
                for (int j = 0; j < n; j += step)
                    dfs(grid, i, j);
            }

            int ans = 0;
            for (int i = 1; i < m - 1; i++) {
                for (int j = 1; j < n - 1; j++) {
                    if (grid[i][j] == 0) {
                        ans++;
                        dfs(grid, i, j);
                    }
                }
            }
            return ans;
        }

        private void dfs(int[][] grid, int x, int y) {
            if (x < 0 || x >= grid.length || y < 0 || y >= grid[x].length || grid[x][y] != 0)
                return;
            grid[x][y] = 1;
            dfs(grid, x - 1, y);
            dfs(grid, x + 1, y);
            dfs(grid, x, y - 1);
            dfs(grid, x, y + 1);
            return;
        }
    }

    // 1262. Greatest Sum Divisible by Three - MEDIUM
    class Solution1262a { // 5ms
        public int maxSumDivThree(int[] nums) {
            int[] f = new int[3];
            for (int x : nums) {
                int[] addedVal = new int[] { x + f[0], x + f[1], x + f[2] };
                for (int y : addedVal) {
                    f[y % 3] = Math.max(f[y % 3], y);
                }
            }
            return f[0];
        }
    }

    // 1263. Minimum Moves to Move a Box to Their Target Location - HARD
    class Solution1263a {
        private char[][] grid;
        private int m, n;
        private int[][] dir = new int[][] { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };

        private boolean reachable(int i, int j, int[] state) {
            // state: boxX, boxY, personX, personY
            Queue<int[]> q = new LinkedList<>();
            q.offer(new int[] { state[2], state[3] });
            boolean[][] vis = new boolean[m][n];
            vis[state[0]][state[1]] = true;
            while (!q.isEmpty()) {
                int[] cur = q.poll();
                if (cur[0] == i && cur[1] == j)
                    return true;
                for (int[] d : dir) {
                    int r = cur[0] - d[0], c = cur[1] - d[1]; // box next spots;
                    if (r < 0 || r >= m || c < 0 || c >= n || vis[r][c] || grid[r][c] == '#')
                        continue;
                    vis[r][c] = true;
                    q.offer(new int[] { r, c });
                }
            }
            return false;
        }

        public int minPushBox(char[][] grid) {
            this.grid = grid;
            m = grid.length;
            n = grid[0].length;
            int[] box = new int[] { -1, -1 }, target = new int[] { -1, -1 }, start = new int[] { -1, -1 };
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (grid[i][j] == 'B')
                        box = new int[] { i, j };
                    if (grid[i][j] == 'T')
                        target = new int[] { i, j };
                    if (grid[i][j] == 'S')
                        start = new int[] { i, j };
                }
            }
            Queue<int[]> q = new LinkedList<>();
            q.offer(new int[] { box[0], box[1], start[0], start[1] });
            boolean[][][] vis = new boolean[m][n][4];
            int step = 0;
            while (!q.isEmpty()) {
                for (int i = 0, l = q.size(); i < l; i++) {
                    int[] cur = q.poll();
                    if (cur[0] == target[0] && cur[1] == target[1])
                        return step;
                    for (int j = 0; j < dir.length; j++) {
                        if (vis[cur[0]][cur[1]][j])
                            continue;
                        int[] d = dir[j];
                        int px = cur[0] + d[0], py = cur[1] + d[1]; // where person stands, have room to push
                        if (px < 0 || px >= m || py < 0 || py >= n || grid[px][py] == '#')
                            continue;
                        int bx = cur[0] - d[0], by = cur[1] - d[1]; // box next spots
                        if (bx < 0 || bx >= m || by < 0 || by >= n || grid[bx][by] == '#')
                            continue;
                        if (!reachable(px, py, cur))
                            continue;
                        vis[cur[0]][cur[1]][j] = true;
                        q.offer(new int[] { bx, by, cur[0], cur[1] });
                    }
                }
                step++;
            }
            return -1;
        }
    }

    // 1267. Count Servers that Communicate - MEDIUM
    class Solution1267a { // 2ms
        public int countServers(int[][] grid) {
            int ans = 0, r[] = new int[251], c[] = new int[251];
            for (int i = 0; i < grid.length; i++) {
                for (int j = 0; j < grid[0].length; j++) {
                    if (grid[i][j] == 1) {
                        r[i]++;
                        c[j]++;
                    }
                }
            }

            for (int i = 0; i < grid.length; i++) {
                for (int j = 0; j < grid[0].length; j++) {
                    if (grid[i][j] == 1 && (r[i] > 1 || c[j] > 1)) {
                        ans++;
                    }
                }
            }
            return ans;

        }
    }

    class Solution1267b { // 1ms
        public int countServers(int[][] grid) {
            // 先计算总共有多少服务器 allCount
            // 再计算不能通信的服务器数量 badCount
            // 能够通信的服务器 goodCount = allCount - badCount
            // 不能通信的服务器: 某一行只有一个服务器并且该服务器对应的列也只有一台服务器
            // 如何利用这个性质?
            int allCount = 0, badCount = 0;
            for (int i = 0; i < grid.length; i++) {
                int rowSum = 0, colSum = 0, lastJ = 0;
                for (int j = 0; j < grid[i].length; j++) {
                    if (grid[i][j] == 1) {
                        allCount++;
                        rowSum++;
                        lastJ = j;
                    }
                }
                if (rowSum == 1) {
                    // 行和 为 1, 再扫一遍 lastJ 列, 确定是否 列和 也为 1
                    for (int k = 0; k < grid.length; k++) {
                        if (grid[k][lastJ] == 1) {
                            colSum++;
                        }
                    }
                    if (colSum == 1) {
                        badCount++;
                    }
                }
            }
            return allCount - badCount;
        }
    }

    // 1281. Subtract the Product and Sum of Digits of an Integer - EASY
    class Solution1281a {
        public int subtractProductAndSum(int n) {
            int p = 1, s = 0;
            while (n > 0) {
                p *= n % 10;
                s += n % 10;
                n /= 10;
            }
            return p - s;
        }
    }
}
