package src;

import java.util.*;
import java.math.*;
import java.util.function.*;

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
}