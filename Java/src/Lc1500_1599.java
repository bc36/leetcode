package src;

import java.util.*;

public class Lc1500_1599 {
    // 1574. Shortest Subarray to be Removed to Make Array Sorted - MEDIUM
    class Solution1574a {
        // 1ms
        public int findLengthOfShortestSubarray(int[] arr) {
            int l = 0, n = arr.length, r = n - 1;
            while (l < n - 1 && arr[l] <= arr[l + 1])
                ++l;
            if (l == n - 1)
                return 0;
            int ans = n - l - 1;
            while (l >= 0) {
                while (l < r && arr[l] <= arr[r - 1] && arr[r - 1] <= arr[r])
                    --r;
                if (arr[l] <= arr[r])
                    ans = Math.min(ans, r - l - 1);
                --l;
            }
            while (r > 0 && arr[r - 1] <= arr[r])
                --r;
            return Math.min(ans, r);
        }
    }

    class Solution1574b {
        // 1ms
        public int findLengthOfShortestSubarray(int[] arr) {
            int n = arr.length, l = 0, r = n - 1;
            while (l < n - 1 && arr[l] <= arr[l + 1])
                ++l;
            while (r > 0 && arr[r - 1] <= arr[r])
                --r;
            if (l == n - 1)
                return 0;
            if (arr[l] <= arr[r])
                return r - l - 1;
            int ans = Math.min(r, n - 1 - l), i = 0, j = r;
            // i 增大, j 可能也增大
            while (i <= l && j < n) {
                while (j < n && arr[i] > arr[j])
                    ++j;
                if (j < n)
                    ans = Math.min(ans, j - i - 1);
                ++i;
            }
            return ans;
        }
    }

    // 1590. Make Sum Divisible by P - MEDIUM
    class Solution1590a {
        // O(n) / O(n)
        public int minSubarray(int[] nums, int p) {
            Map<Integer, Integer> mp = new HashMap<Integer, Integer>();
            int ans = nums.length;
            int m = 0, t = 0;
            for (int v : nums)
                m = (m + v) % p;
            mp.put(0, -1);
            for (int i = 0; i < nums.length; i++) {
                t = (t + nums[i]) % p;
                mp.put(t, i);
                ans = Math.min(ans, i - mp.getOrDefault((t - m + p) % p, -nums.length));
            }
            return ans == nums.length ? -1 : ans;
        }
    }

    class Solution1590b {
        public int minSubarray(int[] nums, int p) {
            // java HashMap 扩容耗时影响较大, loadFactor == 0.75
            Map<Integer, Integer> mp = new HashMap<Integer, Integer>(2 * nums.length);
            long sum = 0;
            int ans = nums.length;
            int t = 0;
            for (int v : nums) // 取模过多稍有影响
                sum += v;
            int m = (int) (sum % p);
            if (m == 0) // 影响不大
                return 0;
            mp.put(0, -1);
            for (int i = 0; i < nums.length; i++) {
                t = (t + nums[i]) % p;
                mp.put(t, i);
                ans = Math.min(ans, i - mp.getOrDefault((t - m + p) % p, -nums.length));
            }
            return ans == nums.length ? -1 : ans;
        }
    }

    // 1599. Maximum Profit of Operating a Centennial Wheel - MEDIUM
    class Solution1599a {
        // O(n) / O(1)
        public int minOperationsMaxProfit(int[] customers, int boardingCost, int runningCost) {
            int ans = -1;
            int profit = 0, mx = 0, wait = 0;
            for (int i = 0; i < customers.length; i++) {
                wait += customers[i];
                int onboard = wait >= 4 ? 4 : wait;
                wait -= onboard;
                profit += onboard * boardingCost - runningCost;
                if (profit > mx) {
                    mx = profit;
                    ans = i + 1;
                }
            }
            if (Math.min(4, wait) * boardingCost > runningCost) {
                int loop = wait / 4;
                wait %= 4;
                profit += loop * (4 * boardingCost - runningCost);
                if (profit > mx) {
                    ans = customers.length + loop + (wait * boardingCost > runningCost ? 1 : 0);
                } else if (profit + wait * boardingCost - runningCost > mx) {
                    ans = customers.length + loop + 1;
                }
            }
            return ans;
        }
    }

    class Solution1599b {
        public int minOperationsMaxProfit(int[] customers, int boardingCost, int runningCost) {
            int wait = 0, rotation = 0, total = 0;
            for (int i = 0; i < customers.length; i++) {
                total += customers[i];
                wait += customers[i];
                wait = wait > 4 ? wait - 4 : 0;
                rotation++;
            }
            rotation += wait / 4;
            wait %= 4;
            if (wait * boardingCost > runningCost) {
                rotation++;
            }
            return total * boardingCost - rotation * runningCost <= 0 ? -1 : rotation;
        }
    }
}
