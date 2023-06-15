package src;

import java.util.*;

public class Lc1300_1399 {
    // 1335. Minimum Difficulty of a Job Schedule - HARD
    class Solution1335a {
        // 9ms
        public int minDifficulty(int[] jobDifficulty, int d) {
            int n = jobDifficulty.length;
            if (n < d)
                return -1;
            int[][] f = new int[d][n];
            f[0][0] = jobDifficulty[0];
            for (int i = 1; i < n; i++)
                f[0][i] = Math.max(f[0][i - 1], jobDifficulty[i]);
            for (int i = 1; i < d; i++) {
                for (int j = n - 1; j >= i; j--) {
                    f[i][j] = Integer.MAX_VALUE;
                    int difficulty = 0;
                    for (int k = j; k >= i; k--) {
                        difficulty = Math.max(difficulty, jobDifficulty[k]);
                        f[i][j] = Math.min(f[i][j], f[i - 1][k - 1] + difficulty);
                    }
                }
            }
            return f[d - 1][n - 1];
        }
    }

    class Solution1335b {
        // 10ms
        public int minDifficulty(int[] jobDifficulty, int d) {
            final int inf = 1 << 30;
            int n = jobDifficulty.length;
            int[][] f = new int[n + 1][d + 1];
            for (var g : f) {
                Arrays.fill(g, inf);
            }
            f[0][0] = 0;
            for (int i = 1; i <= n; ++i) {
                for (int j = 1; j <= Math.min(d, i); ++j) {
                    int difficulty = 0;
                    for (int k = i; k > 0; --k) {
                        difficulty = Math.max(difficulty, jobDifficulty[k - 1]);
                        f[i][j] = Math.min(f[i][j], f[k - 1][j - 1] + difficulty); // 溢出
                    }
                }
            }
            return f[n][d] >= inf ? -1 : f[n][d];
        }
    }

    // 1375. Number of Times Binary String Is Prefix-Aligned - MEDIUM
    class Solution {
        public int numTimesAllBlue(int[] flips) {
            int ans = 0, mx = 0;
            for (int i = 0; i < flips.length; i++) {
                mx = Math.max(mx, flips[i]);
                if (mx == i + 1)
                    ans++;
            }
            return ans;
        }
    }

    // 1376. Time Needed to Inform All Employees - MEDIUM
    class Solution1376a {
        public int numOfMinutes(int n, int headID, int[] manager, int[] informTime) {
            int ans = 0;
            for (int i = 0; i < n; i++) {
                ans = Math.max(ans, dfs(i, manager, informTime));
            }
            return ans;
        }

        private int dfs(int x, int[] manager, int[] informTime) {
            if (manager[x] >= 0) {
                informTime[x] += dfs(manager[x], manager, informTime);
                manager[x] = -1;
            }
            return informTime[x];
        }
    }
}
