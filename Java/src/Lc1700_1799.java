package src;

public class Lc1700_1799 {
    // 1749. Maximum Absolute Sum of Any Subarray - MEDIUM
    class Solution1749a {
        // 3ms
        public int maxAbsoluteSum(int[] nums) {
            int ans = 0, p = 0, n = 0;
            for (int v : nums) {
                if (v > 0) {
                    p += v;
                    ans = ans > p ? ans : p;
                    n = n + v > 0 ? 0 : n + v;
                } else {
                    n += v;
                    ans = ans > -n ? ans : -n;
                    p = p + v > 0 ? p + v : 0;
                }
            }
            return ans;
        }
    }

    class Solution1749b {
        // 2ms
        public int maxAbsoluteSum(int[] nums) {
            int ans = 0, p = 0, n = 0;
            for (int v : nums) {
                p = p > 0 ? p + v : v;
                n = n < 0 ? n + v : v;
                ans = ans > p ? ans > -n ? ans : -n : p;
            }
            return ans;
        }
    }

    class Solution1749c {
        // 1ms
        public int maxAbsoluteSum(int[] nums) {
            int sum = 0, mx = 0, mn = 0;
            for (int x : nums) {
                sum += x;
                if (sum > mx) {
                    mx = sum;
                }
                if (sum < mn) {
                    mn = sum;
                }
            }
            return mx - mn;
        }
    }
}
