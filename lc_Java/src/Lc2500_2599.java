package src;

import java.util.*;

public class Lc2500_2599 {
    // 2587. Rearrange Array to Maximize Prefix Score - M
    class Solution2587a {
        public int maxScore(int[] nums) {
            Arrays.sort(nums);
            long count = 0, sum = 0;
            for (int i = nums.length - 1; i >= 0; i--) {
                count += (sum += nums[i]) > 0 ? 1 : 0;
            }
            return (int) count;
        }
    }

    // 2588. Count the Number of Beautiful Subarrays - M
    class Solution2588a {
        // 7ms
        public long beautifulSubarrays(int[] nums) {
            int bits = 0, max = 0;
            for (int x : nums)
                if (x > max)
                    max = x;
            for (; max != 0; bits++)
                max >>= 1;

            long count = 0;

            // long map[] = new long[1 << 20];    // 70ms
            // int map[] = new int[1 << 20];      // 24ms
            // long[] map = new long[1 << bits];  // 13ms
            int[] map = new int[1 << bits]; // 7ms

            for (int i = 0, xor = 0; i < nums.length; i++) {
                map[xor]++;
                count += map[xor ^= nums[i]];
            }
            return count;
        }
    }

    // 2589. Minimum Time to Complete All Tasks - H
    class Solution2589a {
        // O(nU) / O(U), U = max(end), 38ms
        public int findMinimumTime(int[][] tasks) {
            Arrays.sort(tasks, (o, p) -> o[1] - p[1]);
            int ans = 0, vis[] = new int[2001];
            for (int[] t : tasks) {
                for (int i = t[0]; i <= t[1]; i++) {
                    t[2] -= vis[i];
                }
                for (int i = t[1]; i >= t[0] && t[2] > 0; i--) {
                    if (vis[i] == 0) {
                        t[2]--;
                        ans += vis[i] = 1;
                    }
                }
            }
            return ans;
        }
    }
}
