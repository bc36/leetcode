package src;

import java.util.*;

public class Lc1300_1399 {
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
