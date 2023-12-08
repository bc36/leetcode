package src;

import java.util.*;

public class Lc2000_2099 {
    // 2008. Maximum Earnings From Taxi - MEDIUM
    class Solution2008a {
        public long maxTaxiEarnings(int n, int[][] rides) {
            List<int[]>[] g = new ArrayList[n + 1];
            for (int[] r : rides) {
                int start = r[0], end = r[1], tip = r[2];
                if (g[end] == null)
                    g[end] = new ArrayList<>();
                g[end].add(new int[] { start, end - start + tip });
            }

            long[] f = new long[n + 1];
            for (int i = 2; i < n + 1; i++) {
                f[i] = f[i - 1];
                if (g[i] != null) {
                    for (int[] p : g[i]) {
                        f[i] = Math.max(f[i], f[p[0]] + p[1]);
                    }
                }
            }
            return f[n];
        }
    }
}
