package src;

import java.util.*;

public class Lc2500_2599 {
    class Solution {
        // 2517. Maximum Tastiness of Candy Basket - MEDIUM
        public int maximumTastiness(int[] price, int k) {
            Arrays.sort(price);
            int l = 0, r = (price[price.length - 1] - price[0]) / (k - 1) + 1;
            while (l + 1 < r) {
                int m = l + (r - l) / 2;
                if (f(price, m) >= k)
                    l = m;
                else
                    r = m;
            }
            return l;
        }

        private int f(int[] price, int d) {
            int cnt = 1, pre = price[0];
            for (int p : price) {
                if (p >= pre + d) {
                    cnt++;
                    pre = p;
                }
            }
            return cnt;
        }
    }

    // 2587. Rearrange Array to Maximize Prefix Score - MEDIUM
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

    // 2588. Count the Number of Beautiful Subarrays - MEDIUM
    class Solution2588a { // 7ms
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

    // 2589. Minimum Time to Complete All Tasks - HARD
    class Solution2589a { // O(nU) / O(U), U = max(end), 38ms
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

    // 2591. Distribute Money to Maximum Children - EASY
    public int distMoney(int money, int children) {
        return money == 8 * children ? children
                : money > 8 * children - 8 && money != 8 * children - 4 ? children - 1
                        : money < children ? -1 : Math.min(children - 2, (money - children) / 7);
    }

    // 2592. Maximize Greatness of an Array - MEDIUM
    public int maximizeGreatness(int[] nums) {
        Arrays.sort(nums);
        int j = 0;
        for (int i = 0; i < nums.length; i++) {
            j += nums[i] > nums[j] ? 1 : 0;
        }
        return j;
    }

    // 2593. Find Score of an Array After Marking All Elements - MEDIUM
    public long findScore(int[] nums) {
        TreeSet<Integer> set = new TreeSet<>((o, p) -> nums[o] == nums[p] ? o - p : nums[o] - nums[p]);
        for (int i = 0; i < nums.length; i++) {
            set.add(i);
        }
        long sum = 0;
        for (int i : set) {
            if (nums[i] > 0) {
                sum += nums[i];
                nums[i > 0 ? i - 1 : i] = nums[i < nums.length - 1 ? i + 1 : i] = 0;
            }
        }
        return sum;
    }

    // 2594. Minimum Time to Repair Cars - MEDIUM
    public long repairCars(int[] ranks, int cars) {
        long l = 1, r = 100000000000000L;
        while (l < r) {
            long m = (l + r) / 2, count = 0;
            for (int rank : ranks) {
                count += Math.sqrt(m / rank);
            }
            if (count < cars) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        return l;
    }

    // 2595. Number of Even and Odd Bits - EASY
    public int[] evenOddBit(int n) {
        int[] result = new int[2];
        for (int i = 0; n > 0; i = 1 - i, n /= 2) {
            result[i] += n % 2;
        }
        return result;
    }

    // 2596. Check Knight Tour Configuration - MEDIUM
    public boolean checkValidGrid(int[][] grid) {
        int[][] index = new int[grid.length * grid.length][];
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid.length; j++) {
                index[grid[i][j]] = new int[] { i, j };
            }
        }
        for (int i = 1; i < index.length; i++) {
            if (Math.abs(index[i][0] - index[i - 1][0]) * Math.abs(index[i][1] - index[i - 1][1]) != 2) {
                return false;
            }
        }
        return grid[0][0] == 0;
    }

    // 2597. The Number of Beautiful Subsets - MEDIUM
    public int beautifulSubsets(int[] nums, int k) {
        Arrays.sort(nums);
        return beautifulSubsets(0, new HashMap<>(), nums, k) - 1;
    }

    private int beautifulSubsets(int index, HashMap<Integer, Integer> map, int[] nums, int k) {
        if (index == nums.length) {
            return 1;
        }
        int count = beautifulSubsets(index + 1, map, nums, k);
        if (map.getOrDefault(nums[index] - k, 0) == 0) {
            map.put(nums[index], map.getOrDefault(nums[index], 0) + 1);
            count += beautifulSubsets(index + 1, map, nums, k);
            map.put(nums[index], map.get(nums[index]) - 1);
        }
        return count;
    }

    // 2598. Smallest Missing Non-negative Integer After Operations - MEDIUM
    public int findSmallestInteger(int[] nums, int value) {
        int[] count = new int[value];
        for (int num : nums) {
            count[(num % value + value) % value]++;
        }
        for (int i = 0;; i++) {
            if (--count[i % value] < 0) {
                return i;
            }
        }
    }
}
