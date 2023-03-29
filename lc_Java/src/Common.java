package src;

import java.util.*;

class Sort {
    /**
     * all e in a[:i] have e < x, and all e in a[i:] have e >= x.
     * @param nums
     * @param x
     * @return position <code> i <code>
     */
    public static int lowerBound(int[] nums, int x) {
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
    public static int upperBound(int[] nums, int x) {
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
}

class MathTheory {
    /**
     * [0, n] 区间的合数
     * @param n
     * @return
     */
    public static boolean[] getComposite(int n) {
        boolean[] composite = new boolean[n + 1];
        for (int i = 2; i < n + 1; ++i) {
            for (int j = i + i; j < n + 1; j += i) {
                composite[j] = true;
            }
        }
        return composite;
    }

    /**
     * [0, n] 区间的质数
     * @param n
     * @return
     */
    public static boolean[] getPrimes(int n) {
        boolean[] primes = new boolean[n + 1];
        for (int i = 2; i <= n; ++i) {
            primes[i] = true;
            for (int j = 2; j * j <= i; ++j) {
                if (i % j == 0) {
                    primes[i] = false;
                    break;
                }
            }
        }
        return primes;
    }
}

public class Common {
    void examples() {
        // 声明数组
        int n = 10, seq1[] = new int[n], seq2[] = new int[n];

        // 自定义排序
        // seq1 升序, seq2 升序
        // (1)
        Integer[] order1 = new Integer[n];
        Arrays.sort(order1, (i, j) -> seq1[i] != seq1[j] ? seq1[i] - seq1[j] : seq2[i] - seq2[j]);
        // (2)
        int[][] order2 = new int[n][2];
        for (int i = 0; i < n; ++i)
            order2[i] = new int[] { seq1[i], seq2[i] };
        Arrays.sort(order2, (x, y) -> x[0] == y[0] ? x[1] - y[1] : x[0] - y[0]);

        // 
        int sum = Arrays.stream(seq1).sum();
        int max = Arrays.stream(seq1).max().orElse(0);
        int min = Arrays.stream(seq1).min().orElse(1000_000_000);

        // cs[i] versus s.charAt(i)
        char c;
        String s = new String();
        c = s.charAt(0); // slightly slower
        char[] cs = s.toCharArray();
        c = cs[0]; // slightly faster
    }
}

/**
 * [Obsolete] Because it will add >=1ms of latency <p>
 * 
 * Define functions with common structure and override them to reduce code repetition.
 * Some unique helper functions for problem solving will be implemented using <code>inner class</code>.
 * Such as dfs(), check(), etc.
 */
class Helper {
    public int dfs(int i, int j) {
        return -1;
    }

    public boolean check(int[] arr) {
        return false;
    }

    public boolean check(int[] arr, int x, int y) {
        return false;
    }
}
