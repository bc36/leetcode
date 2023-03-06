package src;

import java.util.*;

public class Lc1500_1599 {
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
