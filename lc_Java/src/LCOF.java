package src;

import java.util.*;

public class LCOF {
    // https://leetcode.cn/problems/li-wu-de-zui-da-jie-zhi-lcof/
    // 剑指 Offer 47. 礼物的最大价值
    private int[][] grid, memo;

    public int maxValue(int[][] grid) {
        this.grid = grid;
        int m = grid.length, n = grid[0].length;
        memo = new int[m][n];
        return dfs(m - 1, n - 1);
    }

    private int dfs(int i, int j) {
        if (i < 0 || j < 0)
            return 0;
        if (memo[i][j] > 0) // grid[i][j] 都是正数，记忆化的 memo[i][j] 必然为正数
            return memo[i][j];
        return memo[i][j] = Math.max(dfs(i, j - 1), dfs(i - 1, j)) + grid[i][j];
    }

    public int maxValue4(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] memo = new int[m][n];
        class Inner {
            private int dfs(int i, int j) {
                if (i < 0 || j < 0)
                    return 0;
                if (memo[i][j] > 0)
                    return memo[i][j];
                return memo[i][j] = Math.max(dfs(i, j - 1), dfs(i - 1, j)) + grid[i][j];
            }
        }
        Inner in = new Inner();
        return in.dfs(m - 1, n - 1);
    }

    public int maxValue3(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] f = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                f[i][j] = Math.max(f[i - 1][j], f[i][j - 1]) + grid[i - 1][j - 1];
            }
        }
        return f[m][n];
    }

    public int maxValue2(int[][] grid) {
        int n = grid[0].length;
        int[] f = new int[n + 1];
        for (int[] row : grid) {
            for (int j = 1; j <= n; j++) {
                f[j] = Math.max(f[j - 1], f[j]) + row[j - 1];
            }
        }
        return f[n];
    }
}