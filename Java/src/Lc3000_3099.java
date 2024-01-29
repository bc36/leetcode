package src;

import java.util.*;
import java.math.*;
import java.util.function.*;

import javax.lang.model.util.Elements;

public class Lc3000_3099 {
    // 3014. Minimum Number of Pushes to Type Word I - EASY
    class Solution3014a {
        public int minimumPushes(String word) {
            int map[] = new int[26], count = 0;
            for (char c : word.toCharArray()) {
                map[c - 'a']--;
            }
            Arrays.sort(map);
            for (int i = 0; i < 26; i++) {
                count -= map[i] * (i / 8 + 1);
            }
            return count;
        }
    }

    // 3015. Count the Number of Houses at a Certain Distance I - MEDIUM
    class Solution3015a {
        public int[] countOfPairs(int n, int x, int y) {
            if (x > y) {
                return countOfPairs(n, y, x);
            }
            int[] count = new int[n];
            for (int i = 1; i < n; i++) {
                count[Math.min(i, Math.abs(y - i - 1) + x) - Math.min(i, x)] += 2;
                count[Math.min(i, Math.abs(y - i - 1) + x)] -= 2;
                if (x < i) {
                    count[0] += 2;
                    count[i - Math.max(x, (x + i - Math.abs(y - i - 1) - 1) / 2)] -= 2;
                    if (x + i > Math.abs(y - i - 1) + 2) {
                        count[Math.abs(y - i - 1) + 1] += 2;
                        count[Math.abs(y - i - 1) + 1 + Math.max(0, (x + i - Math.abs(y - i - 1) - 1) / 2 - x)] -= 2;
                    }
                }
            }
            for (int i = 1; i < n; i++) {
                count[i] += count[i - 1];
            }
            return count;
        }
    }

    // 3016. Minimum Number of Pushes to Type Word II - MEDIUM
    class Solution3016a {
        public int minimumPushes(String word) {
            int map[] = new int[26], count = 0;
            for (char c : word.toCharArray()) {
                map[c - 'a']--;
            }
            Arrays.sort(map);
            for (int i = 0; i < 26; i++) {
                count -= map[i] * (i / 8 + 1);
            }
            return count;
        }
    }

    // 3017. Count the Number of Houses at a Certain Distance II - HARD
    class Solution3017a {
        public long[] countOfPairs(int n, int x, int y) {
            if (x > y) {
                return countOfPairs(n, y, x);
            }
            long[] count = new long[n];
            for (int i = 1; i < n; i++) {
                count[Math.min(i, Math.abs(y - i - 1) + x) - Math.min(i, x)] += 2;
                count[Math.min(i, Math.abs(y - i - 1) + x)] -= 2;
                if (x < i) {
                    count[0] += 2;
                    count[i - Math.max(x, (x + i - Math.abs(y - i - 1) - 1) / 2)] -= 2;
                    if (x + i > Math.abs(y - i - 1) + 2) {
                        count[Math.abs(y - i - 1) + 1] += 2;
                        count[Math.abs(y - i - 1) + 1 + Math.max(0, (x + i - Math.abs(y - i - 1) - 1) / 2 - x)] -= 2;
                    }
                }
            }
            for (int i = 1; i < n; i++) {
                count[i] += count[i - 1];
            }
            return count;
        }
    }

    // 3019. Number of Changing Keys - EASY
    class Solution3019a {
        public int countKeyChanges(String s) {
            int count = 0;
            for (int i = 1; i < s.length(); i++) {
                count += Character.toLowerCase(s.charAt(i)) == Character.toLowerCase(s.charAt(i - 1)) ? 0 : 1;
            }
            return count;
        }
    }

    // 3020. Find the Maximum Number of Elements in Subset - MEDIUM
    class Solution3020a {
        public int maximumLength(int[] nums) {
            HashMap<Long, Integer> map = new HashMap<>();
            for (long num : nums) {
                map.put(num, map.getOrDefault(num, 0) + 1);
            }
            int max = 1;
            for (long num : nums) {
                int curr = 0;
                for (; num > 1 && map.getOrDefault(num, 0) > 1; num *= num) {
                    curr += 2;
                }
                max = Math.max(max, num > 1 ? curr + map.getOrDefault(num, -1) : 0);
            }
            return Math.max(max, (map.getOrDefault(1L, 0) - 1) / 2 * 2 + 1);
        }
    }

    // 3021. Alice and Bob Playing Flower Game - MEDIUM
    class Solution3021a {
        public long flowerGame(long n, long m) {
            return (n / 2 + n % 2) * (m / 2) + (m / 2 + m % 2) * (n / 2);
        }
    }

    // 3022. Minimize OR of Remaining Elements Using Operations - HARD
    class Solution3022a {
        public int minOrAfterOperations(int[] nums, int k) {
            int mask = 0;
            for (int t = 29; t >= 0; t--) {
                int p = 1 << t;
                mask |= p;
                int s = nums.length;
                for (int i = 0; i < nums.length && s > k; s--) {
                    int c = nums[i++];
                    for (; i < nums.length && (c & mask) > 0; c &= nums[i++]) {
                    }
                    if ((c & mask) > 0) {
                        break;
                    }
                }
                if (s > k) {
                    mask ^= p;
                }
            }
            return (1 << 30) - 1 ^ mask;
        }
    }
}