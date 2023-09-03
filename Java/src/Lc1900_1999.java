package src;

import java.util.*;

public class Lc1900_1999 {
    // 1911. Maximum Alternating Subsequence Sum - MEDIUM
    class Solution1911 {
        public long maxAlternatingSum(int[] nums) {
            long f = 0, g = 0;
            for (int x : nums) {
                long ff = Math.max(g - x, f);
                long gg = Math.max(f + x, g);
                f = ff;
                g = gg;
            }
            return Math.max(f, g);
        }
    }

    // 1921. Eliminate Maximum Number of Monsters - MEDIUM
    class Solution { // 17ms
        public int eliminateMaximum(int[] dist, int[] speed) {
            int n = dist.length, times[] = new int[n];
            for (int i = 0; i < n; ++i) {
                times[i] = (dist[i] - 1) / speed[i];
            }
            Arrays.sort(times);
            for (int i = 0; i < n; ++i) {
                if (times[i] < i) {
                    return i;
                }
            }
            return n;
        }
    }
}
