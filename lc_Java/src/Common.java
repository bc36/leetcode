package src;

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

public class Common {
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
