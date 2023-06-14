package src;

import java.util.*;

public class Lc1200_1299 {
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
}
