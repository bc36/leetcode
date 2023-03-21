package src;

import java.util.*;

public class Lc2400_2499 {
    // 2465. Number of Distinct Averages - E
    // 1ms
    public int distinctAverages(int[] nums) {
        Arrays.sort(nums);
        HashSet<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length / 2; i++) {
            set.add(nums[i] + nums[nums.length - 1 - i]);
        }
        return set.size();
    }

    // 0ms
    public int distinctAverages2(int[] nums) {
        Arrays.sort(nums);
        boolean[] arr = new boolean[201];
        int counts = 0;
        for (int i = 0; i < nums.length / 2; i++) {
            if (!arr[nums[i] + nums[nums.length - 1 - i]]) {
                counts++;
                arr[nums[i] + nums[nums.length - 1 - i]] = true;
            }
        }
        return counts;
    }

    // 2466. Count Ways To Build Good Strings - M
    // 7ms
    public int countGoodStrings(int low, int high, int zero, int one) {
        int f[] = new int[high + 1], ans = 0, mod = (int) (1e9 + 7);
        for (int i = f[0] = 1; i <= high; ++i) {
            if (i >= one) {
                f[i] = (f[i - one] + f[i]) % mod;
            }
            if (i >= zero) {
                f[i] = (f[i - zero] + f[i]) % mod;
            }
        }
        for (int i = low; i <= high; ++i) {
            ans = (ans + f[i]) % mod;
        }
        return ans;
    }

    // 6ms
    public int countGoodStrings2(int low, int high, int zero, int one) {
        int f[] = new int[high + 1], count = 0;
        for (int i = f[0] = 1; i <= high; i++) {
            count = (count + (f[i] = ((i < zero ? 0 : f[i - zero]) + (i < one ? 0 : f[i - one])) % 1000000007)
                    * (i < low ? 0 : 1)) % 1000000007;
        }
        return count;
    }

    // 2469. Convert the Temperature - E
    public double[] convertTemperature(double celsius) {
        return new double[] { celsius + 273.15, celsius * 1.8 + 32 };
    }
}
