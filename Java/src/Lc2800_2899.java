package src;

import java.math.BigInteger;
import java.util.*;

public class Lc2800_2899 {
    // 2800. Shortest String That Contains Three Strings - MEDIUM
    class Solution2800a {
        public String minimumString(String a, String b, String c) {
            return minimumString("", 0, 0, List.of(a, b, c));
        }

        private String minimumString(String curr, int index, int mask, List<String> list) {
            if (index == 3) {
                return curr;
            }
            String s = null;
            for (int i = 0; i < 3; i++) {
                for (int j = list.get(i).length(); (mask & 1 << i) == 0 && j >= 0; j--) {
                    if (curr.endsWith(list.get(i).substring(0, j)) || curr.contains(list.get(i))) {
                        String n = minimumString(curr + list.get(i).substring(j), index + 1, mask | 1 << i, list);
                        s = s == null || s.length() > n.length() || s.length() == n.length() && s.compareTo(n) > 0 ? n
                                : s;
                        break;
                    }
                }
            }
            return s;
        }
    }

    // 2801. Count Stepping Numbers in Range - HARD
    class Solution2801a {
        private static ArrayList<Long> list = new ArrayList<>() {
            {
                addAll(List.of(0L, 9L, 17L, 32L, 61L, 116L));
                for (int i = 6; i < 100; i++) {
                    add(((get(i - 1) + 4 * get(i - 2) - 3 * get(i - 3) - 3 * get(i - 4) + get(i - 5)) % 1000000007
                            + 1000000007) % 1000000007);
                }
            }
        };

        public int countSteppingNumbers(String low, String high) {
            return (countSteppingNumbers(high) - countSteppingNumbers("" + new BigInteger(low).subtract(BigInteger.ONE))
                    + 1000000007) % 1000000007;
        }

        private int countSteppingNumbers(String s) {
            Integer count = 0, dp[][][] = new Integer[s.length()][10][2];
            for (int i = 1; i < s.length(); i++) {
                count = (count + list.get(i).intValue()) % 1000000007;
            }
            for (int i = 1; i <= s.charAt(0) - '0'; i++) {
                count = (count + countSteppingNumbers(1, i, i < s.charAt(0) - '0', s, dp)) % 1000000007;
            }
            return count;
        }

        private int countSteppingNumbers(int index, int prev, boolean flag, String s, Integer[][][] dp) {
            if (prev < 0 || prev > 9) {
                return 0;
            }
            if (index == s.length()) {
                return 1;
            }
            if (dp[index][prev][flag ? 1 : 0] == null) {
                dp[index][prev][flag ? 1 : 0] = ((!flag && prev >= s.charAt(index) - '0' ? 0
                        : countSteppingNumbers(index + 1, prev + 1, flag || prev + 1 < s.charAt(index) - '0', s, dp))
                        + (!flag && prev - 1 > s.charAt(index) - '0' ? 0
                                : countSteppingNumbers(index + 1, prev - 1, flag || prev <= s.charAt(index) - '0', s,
                                        dp)))
                        % 1000000007;
            }
            return dp[index][prev][flag ? 1 : 0];
        }
    }
}
