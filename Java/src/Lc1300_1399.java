package src;

import java.util.*;

public class Lc1300_1399 {
    // 1333. Filter Restaurants by Vegan-Friendly, Price and Distance - MEDIUM
    class Solution {
        public List<Integer> filterRestaurants(
                int[][] restaurants, int veganFriendly, int maxPrice, int maxDistance) {
            Arrays.sort(restaurants, (a, b) -> a[1] == b[1] ? b[0] - a[0] : b[1] - a[1]);
            List<Integer> ans = new ArrayList<>();
            for (int[] r : restaurants) {
                if (r[2] >= veganFriendly && r[3] <= maxPrice && r[4] <= maxDistance) {
                    ans.add(r[0]);
                }
            }
            return ans;
        }
    }

    // 1335. Minimum Difficulty of a Job Schedule - HARD
    class Solution1335a { // 9ms
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

    class Solution1335b { // 10ms
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
    class Solution1375a {
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

    // 1388. Pizza With 3n Slices - HARD
    class Solution1388a { // 4ms
        public int maxSizeSlices(int[] slices) {
            int m = slices.length, a[] = new int[m - 1], b[] = new int[m - 1];
            System.arraycopy(slices, 0, a, 0, m - 1);
            System.arraycopy(slices, 1, b, 0, m - 1);
            return Math.max(calc(a), calc(b));
        }

        private int calc(int[] nums) {
            int n = (nums.length + 1) / 3, f[][] = new int[nums.length][n + 1];
            f[0][0] = 0;
            f[1][0] = 0;
            f[0][1] = nums[0];
            f[1][1] = Math.max(nums[0], nums[1]);
            for (int i = 2; i < nums.length; i++) {
                for (int j = 1; j <= n; j++) {
                    f[i][j] = Math.max(f[i - 2][j - 1] + nums[i], f[i - 1][j]);
                }
            }
            return f[nums.length - 1][n];
        }
    }

    class SolutionSolution1388b { // 3ms
        public int maxSizeSlices(int[] slices) {
            int n = slices.length;
            // 通过数学归纳法可知实际上的选择方式可以切换成选n个不相邻的数
            // 首先考虑不选最后一个数的方案, 再考虑不选择第一个数的方案, 从中选出最大的方案；
            // 这样的话就可以避免环状导致的第一个数和最后一个数同时被选择的问题
            return Math.max(calc(slices, 0, n - 2), calc(slices, 1, n - 1));
        }

        public int calc(int[] slices, int begin, int end) {
            // 只能选择 slices.length - 1个披萨
            int n = slices.length - 1;
            int m = slices.length / 3;
            // first[i] 表示间隔将要选择的披萨距离一个位置时选择 i 个披萨的最大值
            int[] first = new int[m + 1];
            // mx[i] 表示到目前为止选择 i 个披萨的最大值
            int[] mx = new int[m + 1];
            first[1] = slices[begin];
            mx[1] = Math.max(slices[begin], slices[begin + 1]);
            for (int i = 2; i < n; i++) {
                for (int j = m; j > 0; j--) {
                    // 下一轮的first, 因为会被替换, 所以临时变量记录
                    int tmp = mx[j];
                    // 考虑这一轮选择的情况和不选的情况
                    mx[j] = Math.max(first[j - 1] + slices[i + begin], mx[j]);
                    first[j] = tmp;
                }
            }
            return mx[m];
        }
    }
}
