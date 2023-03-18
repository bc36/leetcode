package src;

import java.util.*;

public class Lc1500_1599 {
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

    // 1590. Make Sum Divisible by P - M
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

    public int minSubarray2(int[] nums, int p) {
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

    // 1599. Maximum Profit of Operating a Centennial Wheel - M
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

    public int minOperationsMaxProfit2(int[] customers, int boardingCost, int runningCost) {
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
