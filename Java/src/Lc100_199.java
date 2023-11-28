package src;

import java.util.*;

public class Lc100_199 {
    // 110. Balanced Binary Tree - EASY
    class Solution110a {
        public boolean isBalanced(TreeNode root) {
            return dfs(root) != -1;
        }

        public int dfs(TreeNode root) {
            if (root == null)
                return 0;
            int l = dfs(root.left);
            int r = dfs(root.right);
            if (l == -1 || r == -1)
                return -1;
            return Math.abs(l - r) <= 1 ? Math.max(l, r) + 1 : -1;
        }
    }

    // 119. Pascal's Triangle II - EASY
    class Solution119a {
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
    }

    // 136. Single Number - EASY
    class Solution136a {
        public int singleNumber(int[] nums) {
            for (int i = 0; i < nums.length - 1; i++) {
                nums[i + 1] ^= nums[i];
            }
            return nums[nums.length - 1];
        }
    }

    class Solution136b {
        public int singleNumber(int[] nums) {
            int x = 0;
            for (int n : nums) {
                x ^= n;
            }
            return x;
        }
    }

    // 169. Majority Element - EASY
    class Solution169a {
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
    }

    class Solution169b {
        public int majorityElement(int[] nums) {
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
    }

    class Solution169c {
        // O(nlogn) / O(logn)
        public int majorityElement(int[] nums) {
            Arrays.sort(nums);
            return nums[nums.length / 2];
        }
    }

    // 189. Rotate Array - MEDIUM
    class Solution189a {
        // O(n) / O(n)
        public void rotate(int[] nums, int k) {
            int n = nums.length;
            int[] arr = new int[n];
            for (int i = 0; i < n; i++) {
                arr[(i + k) % n] = nums[i];
            }
            System.arraycopy(nums, 0, arr, 0, n);
        }
    }

    class Solution189b {
        public void rotate(int[] nums, int k) {
            k %= nums.length;
            int[] cp = new int[nums.length];
            System.arraycopy(nums, nums.length - k, cp, 0, k);
            System.arraycopy(nums, 0, cp, k, nums.length - k);
            System.arraycopy(cp, 0, nums, 0, nums.length);
        }
    }

    class Solution189c {
        // O(n) / O(1)
        public void rotate(int[] nums, int k) {
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
}
