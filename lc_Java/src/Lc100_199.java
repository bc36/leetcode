package src;

import java.util.*;

public class Lc100_199 {
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

    // 119. Pascal's Triangle II - E
    public List<Integer> getRow(int rowIndex) {
        List<Integer> pre = new ArrayList<Integer>();
        for (int i = 0; i <= rowIndex; i++) {
            List<Integer> row = new ArrayList<Integer>();
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i) {
                    row.add(1);
                } else {
                    row.add(pre.get(j - 1) + pre.get(j));
                }
            }
            pre = row;
        }
        return pre;
    }

    // 136. Single Number - E
    public int singleNumber(int[] nums) {
        for (int i = 0; i < nums.length - 1; i++) {
            nums[i + 1] ^= nums[i];
        }
        return nums[nums.length - 1];
    }

    public int singleNumber2(int[] nums) {
        int x = 0;
        for (int n : nums) {
            x ^= n;
        }
        return x;
    }

    // 169. Majority Element - E
    // O(n) / O(1)
    public int majorityElement(int[] nums) {
        int ans = 0;
        int cnt = 0;
        for (int n : nums) {
            if (cnt == 0) {
                ans = n;
            }
            // if (ans == n) {
            // cnt++;
            // } else {
            // cnt--;
            // }
            cnt = ans == n ? cnt + 1 : cnt - 1;
        }
        return ans;
    }

    public int majorityElement2(int[] nums) {
        Map<Integer, Integer> cnt = new HashMap<Integer, Integer>();
        for (int n : nums) {
            if (!cnt.containsKey(n)) {
                cnt.put(n, 1);
            } else {
                cnt.put(n, cnt.get(n) + 1);
            }
        }
        // int ans = 0;
        // int c = 0;
        // for (Map.Entry<Integer, Integer> e : cnt.entrySet()) {
        // if (e.getValue() > c) {
        // ans = e.getKey();
        // c = e.getValue();
        // }
        // }
        for (int n : nums) {
            if (cnt.get(n) > nums.length / 2)
                return n;
        }

        return -1;
    }

    // O(nlogn) / O(logn)
    public int majorityElement3(int[] nums) {
        Arrays.sort(nums);
        return nums[nums.length / 2];
    }

    // 189. Rotate Array - M
    // O(n) / O(n)
    public void rotate(int[] nums, int k) {
        int n = nums.length;
        int[] arr = new int[n];
        for (int i = 0; i < n; i++) {
            arr[(i + k) % n] = nums[i];
        }
        System.arraycopy(nums, 0, arr, 0, n);
    }

    public void rotate2(int[] nums, int k) {
        k %= nums.length;
        int[] cp = new int[nums.length];
        System.arraycopy(nums, nums.length - k, cp, 0, k);
        System.arraycopy(nums, 0, cp, k, nums.length - k);
        System.arraycopy(cp, 0, nums, 0, nums.length);
    }

    // O(n) / O(1)
    public void rotate3(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }

    public void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
    }

}
