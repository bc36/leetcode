package src;

import java.util.*;

public class Lc1600_1699 {
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    /**
     * all e in a[:i] have e < x, and all e in a[i:] have e >= x.
     * @param nums
     * @param x
     * @return position <code> i <code>
     */
    private int lowerBound(int[] nums, int x) {
        int l = 0, r = nums.length;
        while (l < r) {
            int m = (l + r) >> 1;
            if (nums[m] < x)
                l = m + 1;
            else
                r = m;
        }
        return l;
    }

    /**
     * all e in a[:i] have e <= x, and all e in a[i:] have e > x.
     * @param nums
     * @param x
     * @return position <code> i <code>
     */
    private int upperBound(int[] nums, int x) {
        int l = 0, r = nums.length;
        while (l < r) {
            int m = (l + r) >> 1;
            if (nums[m] > x)
                r = m;
            else
                l = m + 1;
        }
        return l;
    }

    // 1605. Find Valid Matrix Given Row and Column Sums - M
    public int[][] restoreMatrix(int[] rowSum, int[] colSum) {
        // 7ms
        int[][] ans = new int[rowSum.length][colSum.length];
        for (int i = 0; i < rowSum.length; i++) {
            for (int j = 0; j < colSum.length; j++) {
                int v = Math.min(rowSum[i], colSum[j]);
                ans[i][j] = v;
                rowSum[i] -= v;
                colSum[j] -= v;
            }
        }
        return ans;
    }

    public int[][] restoreMatrix2(int[] rowSum, int[] colSum) {
        // 1ms
        int n = rowSum.length, m = colSum.length, i = 0, j = 0;
        int[][] ans = new int[n][m];
        while (i < n && j < m) {
            int r = rowSum[i];
            int c = colSum[j];
            if (r < c) {
                ans[i][j] = r;
                colSum[j] -= r;
                i++;
            } else {
                ans[i][j] = c;
                rowSum[i] -= c;
                j++;
            }
        }
        return ans;
    }

    // 1615. Maximal Network Rank - M
    // 4ms
    public int maximalNetworkRank(int n, int[][] roads) {
        int[][] conn = new int[n][n];
        int[] deg = new int[n];
        int ans = 0;
        for (int i = 0; i < roads.length; i++) {
            int a = roads[i][0], b = roads[i][1];
            conn[a][b] = 1;
            conn[b][a] = 1;
            deg[a]++;
            deg[b]++;
        }
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) {
                ans = Math.max(ans, deg[a] + deg[b] - conn[a][b]);
            }
        }
        return ans;
    }

    // 2ms
    public int maximalNetworkRank2(int n, int[][] roads) {
        int[][] conn = new int[n][n];
        int[] deg = new int[n];
        for (int[] road : roads) { // 1ms faster
            conn[road[0]][road[1]] = 1;
            conn[road[1]][road[0]] = 1;
            deg[road[0]] += 1;
            deg[road[1]] += 1;
        }
        int ans = Integer.MIN_VALUE;
        for (int a = 0; a < n; a++) {
            for (int b = a + 1; b < n; b++) {
                int cnt = deg[a] + deg[b] - conn[a][b]; // 1ms faster
                ans = ans < cnt ? cnt : ans;
            }
        }
        return ans;
    }

    // 1653. Minimum Deletions to Make String Balanced - M
    public int minimumDeletions(String s) {
        int del = 0; // all 'a'
        char[] arr = s.toCharArray();
        for (char c : arr) {
            del += 'b' - c;
        }
        int ans = del;
        for (char c : arr) {
            del += (c - 'a') * 2 - 1;
            ans = Math.min(ans, del);
        }
        return ans;
    }

    public int minimumDeletions2(String s) {
        int a = 0, b = 0;
        for (int i = 0; i < s.length(); i++) {
            a += (s.charAt(i) - 'a') ^ 1;
            // if (s.charAt(i) == 'a') { // slow
            //     a++;
            // }
        }
        int ans = s.length();
        for (int i = 0; i < s.length(); i++) {
            // a -= s.charAt(i) == 'a' ? 1 : 0; // slow
            char c = s.charAt(i);
            a -= (c - 'a') ^ 1;
            ans = Math.min(ans, a + b);
            // b += s.charAt(i) == 'b' ? 1 : 0;
            b += (c - 'a') ^ 0;
        }
        return ans;
    }

    public int minimumDeletions3(String s) {
        // dp
        int ans = 0, b = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == 'b') {
                b++;
            } else {
                ans = Math.min(ans + 1, b);
            }
        }
        return ans;
    }

    public int minimumDeletions4(String s) {
        // dp, 统计当前数字以 a 或者 b 结尾的最少操作数
        int a = 0, b = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            // 以 b 结尾的最小操作数, 前一个可以是 a, 也可以是 b, 取前一个 a/b 结尾的最小值 + 当前操作数(a + 1, b + 0)
            b = Math.min(b, a) + 'b' - c;
            // 以 a 结尾的最小操作数, 前一个必须是 a, 也就是加上当前位置变化为 b 的操作数
            a = a + c - 'a';
        }
        return Math.min(a, b);
    }
}
