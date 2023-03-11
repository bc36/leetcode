package src;

import java.util.*;

public class LCCI {
    // https://leetcode.cn/problems/find-longest-subarray-lcci/
    // 面试题 17.05.  字母与数字
    // 18ms
    public String[] findLongestSubarray(String[] array) {
        Map<Integer, Integer> d = new HashMap<>();
        d.put(0, -1);
        int pre = 0, mx = 0, start = 0;
        for (int i = 0; i < array.length; i++) {
            pre += array[i].charAt(0) > 'A' ? 1 : -1;
            if (d.containsKey(pre)) {
                int j = d.get(pre);
                if (i - j > mx) {
                    mx = i - j;
                    start = j + 1;
                }
            } else {
                d.put(pre, i);
            }
        }
        String[] ans = new String[mx];
        System.arraycopy(array, start, ans, 0, mx);
        return ans;
    }

    // 11ms, 数组比哈希表快
    public String[] findLongestSubarray3(String[] array) {
        int n = array.length, begin = 0, end = 0;
        int[] pre = new int[n + 1];
        Map<Integer, Integer> d = new HashMap<Integer, Integer>();
        for (int i = 0; i < n; ++i)
            pre[i + 1] = pre[i] + (array[i].charAt(0) >= 'A' ? 1 : -1);
        for (int i = 0; i <= n; ++i) {
            int j = d.getOrDefault(pre[i], -1);
            if (j < 0) {
                d.put(pre[i], i);
            } else if (i - j > end - begin) {
                begin = j;
                end = i;
            }
        }
        // return Arrays.copyOfRange(array, begin, end); // 12ms
        String[] ans = new String[end - begin];
        System.arraycopy(array, begin, ans, 0, ans.length);
        return ans;
    }

    // 7ms
    public String[] findLongestSubarray2(String[] array) {
        int n = array.length, begin = 1, end = 1, s = n;
        int[] first = new int[n * 2 + 1];
        first[s] = 1;
        for (int i = 2; i <= n + 1; ++i) {
            s += array[i - 2].charAt(0) >= 'A' ? 1 : -1;
            int j = first[s];
            if (j == 0) {
                first[s] = i;
            } else if (i - j > end - begin) {
                begin = j;
                end = i;
            }
        }
        // return Arrays.copyOfRange(array, begin, end); // 12ms
        String[] ans = new String[end - begin];
        System.arraycopy(array, begin - 1, ans, 0, ans.length);
        return ans;
    }
}
