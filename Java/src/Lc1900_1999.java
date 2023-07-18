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
}
