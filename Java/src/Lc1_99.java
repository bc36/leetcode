package src;

// List / Array
import java.util.Arrays;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
// // Queue
import java.util.Stack;
import java.util.Deque;
import java.util.Queue;
import java.util.ArrayDeque;
import java.util.PriorityQueue;
// // Map / Set
import java.util.Map;
import java.util.Set;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.BitSet;

import java.util.function.Function;
import java.util.function.IntPredicate;

import static java.math.BigInteger.valueOf;
import java.math.BigInteger;

@SuppressWarnings("unused")
public class Lc1_99 {
    // 1. Two Sum - EASY
    class Solution1a {
        public int[] twoSum(int[] nums, int target) {
            Map<Integer, Integer> m = new HashMap<Integer, Integer>();
            for (int i = 0; i < nums.length; i++) {
                if (m.containsKey(target - nums[i])) {
                    return new int[] { m.get(target - nums[i]), i };
                }
                m.put(nums[i], i);
            }
            return new int[0];
        }
    }

    // 9. Palindrome Number - EASY
    class Solution9a {
        public boolean isPalindrome(int x) {
            String s = String.valueOf(x);
            int n = s.length();
            for (int i = 0; i < n / 2; i++) {
                if (s.charAt(i) != s.charAt(n - 1 - i)) {
                    return false;
                }
            }
            return true;
        }
    }

    class Solution9b {
        public boolean isPalindrome(int x) {
            if (x < 0) {
                return false;
            }
            int ori = x;
            int cur = 0;
            while (x != 0) {
                cur = cur * 10 + x % 10;
                x /= 10;
            }
            return cur == ori;
        }
    }

    // 14. Longest Common Prefix - EASY
    class Solution14a {
        public String longestCommonPrefix(String[] strs) {
            if (strs.length == 0) { // strs == null
                return null;
            }
            int row = strs.length;
            int col = strs[0].length();
            for (int j = 0; j < col; j++) {
                char c = strs[0].charAt(j);
                for (int i = 1; i < row; i++) {
                    if (j == strs[i].length() || strs[i].charAt(j) != c) {
                        return strs[0].substring(0, j);
                    }
                }
            }
            return strs[0];
        }
    }

    // 15. 3Sum - MEDIUM
    class Solution15a {
        public List<List<Integer>> threeSum(int[] nums) {
            List<List<Integer>> ans = new ArrayList<>();
            if (nums == null || nums.length < 3) {
                return ans;
            }
            Arrays.sort(nums);
            for (int i = 0; i < nums.length - 2; i++) {
                if (nums[i] > 0)
                    break;
                if (i > 0 && nums[i] == nums[i - 1])
                    continue;
                int target = -nums[i];
                int left = i + 1, right = nums.length - 1;
                while (left < right) {
                    if (nums[left] + nums[right] == target) {
                        ans.add(Arrays.asList(nums[i], nums[left], nums[right]));
                        left++;
                        right--;
                        while (left < right && nums[left] == nums[left - 1]) {
                            left++;
                        }
                        while (left < right && nums[right] == nums[right + 1]) {
                            right--;
                        }
                    } else if (nums[left] + nums[right] < target) {
                        left++;
                    } else {
                        right--;
                    }
                }
            }
            return ans;
        }
    }

    // 48. Rotate Image - EASY
    class Solution48a {
        public void rotate(int[][] matrix) {
            int n = matrix.length;
            for (int i = 0; i < n / 2; i++) {
                for (int j = 0; j < n; j++) {
                    int tmp = matrix[i][j];
                    matrix[i][j] = matrix[n - 1 - i][j];
                    matrix[n - 1 - i][j] = tmp;
                }
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < i; j++) {
                    int tmp = matrix[i][j];
                    matrix[i][j] = matrix[j][i];
                    matrix[j][i] = tmp;
                }
            }
            return;
        }
    }

    // 56. Merge Intervals - MEDIUM
    class Solution56a { // 6ms
        public int[][] merge(int[][] intervals) {
            Arrays.sort(intervals, (i1, i2) -> Integer.compare(i1[0], i2[0]));
            List<int[]> ans = new ArrayList<>();
            int[] temp = intervals[0];
            ans.add(temp);
            for (int[] i : intervals) {
                if (i[0] <= temp[1]) {
                    temp[1] = Math.max(temp[1], i[1]);
                } else {
                    temp = i;
                    ans.add(temp);
                }
            }
            return ans.toArray(new int[ans.size()][]);
        }
    }

    class Solution56b { // 6ms
        public int[][] merge(int[][] intervals) {
            Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
            List<int[]> ans = new ArrayList<>();
            int left = intervals[0][0];
            int right = intervals[0][1];
            for (int i = 1; i < intervals.length; i++) {
                if (intervals[i][0] <= right) {
                    right = Math.max(right, intervals[i][1]);
                } else {
                    ans.add(new int[] { left, right });
                    left = intervals[i][0];
                    right = intervals[i][1];
                }
            }
            ans.add(new int[] { left, right });
            return ans.toArray(new int[0][]);
        }
    }

    // 57. Insert Interval - MEDIUM
    class Solution { // 6ms
        public int[][] insert(int[][] intervals, int[] newInterval) {
            int[][] arr = Stream.concat(Arrays.stream(intervals), Stream.of(newInterval)).toArray(int[][]::new);
            return merge(arr);
        }

        public int[][] merge(int[][] intervals) {
            Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
            List<int[]> ans = new ArrayList<>();
            int left = intervals[0][0];
            int right = intervals[0][1];
            for (int i = 1; i < intervals.length; i++) {
                if (intervals[i][0] <= right) {
                    right = Math.max(right, intervals[i][1]);
                } else {
                    ans.add(new int[] { left, right });
                    left = intervals[i][0];
                    right = intervals[i][1];
                }
            }
            ans.add(new int[] { left, right });
            return ans.toArray(new int[0][]);
        }
    }

    // 94. Binary Tree Inorder Traversal - EASY
    class Solution94a {
        public List<Integer> inorderTraversal(TreeNode root) {
            List<Integer> ans = new ArrayList<Integer>();
            inorder(root, ans);
            return ans;
        }

        public void inorder(TreeNode root, List<Integer> arr) {
            if (root == null) {
                return;
            }
            inorder(root.left, arr);
            arr.add(root.val);
            inorder(root.right, arr);
        }
    }

    class Solution94b {
        public List<Integer> inorderTraversal(TreeNode root) {
            class Inner {
                private void inorder(TreeNode root, List<Integer> arr) {
                    if (root == null)
                        return;
                    inorder(root.left, arr);
                    arr.add(root.val);
                    inorder(root.right, arr);
                }
            }
            Inner inr = new Inner();
            List<Integer> ans = new ArrayList<Integer>();
            inr.inorder(root, ans);
            return ans;
        }
    }
}
