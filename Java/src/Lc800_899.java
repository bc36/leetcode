package src;

import java.util.*;

public class Lc800_899 {
    // 823. Binary Trees With Factors - MEDIUM
    class Solution823a {
        public int numFactoredBinaryTrees(int[] arr) {
            Arrays.sort(arr);
            Map<Integer, Integer> m = new HashMap<>();
            for (int i = 0; i < arr.length; i++) {
                m.put(arr[i], i);
            }
            long ans = 0, f[] = new long[arr.length];
            for (int i = 0; i < arr.length; i++) {
                f[i] = 1;
                for (int j = 0; j < i; j++) {
                    if (arr[i] % arr[j] == 0 && m.containsKey(arr[i] / arr[j]))
                        f[i] += f[j] * f[m.get(arr[i] / arr[j])];
                }
                ans += f[i];
            }
            return (int) (ans % 1000000007);
        }
    }

    // 831. Masking Personal Information - MEDIUM
    class Solution831a {
        public String maskPII(String s) {
            int at = s.indexOf("@");
            if (at > 0) {
                s = s.toLowerCase();
                return (s.charAt(0) + "*****" + s.substring(at - 1)).toLowerCase();
            }
            s = s.replaceAll("[^0-9]", "");
            return new String[] { "", "+*-", "+**-", "+***-" }[s.length() - 10] + "***-***-"
                    + s.substring(s.length() - 4);
        }
    }

    // 849. Maximize Distance to Closest Person - MEDIUM
    class Solution849a {
        public int maxDistToClosest(int[] seats) {
            int head = -1, tail = 0, middle = 0, pre = -1;
            for (int i = 0; i < seats.length; i++) {
                if (seats[i] == 1) {
                    if (head == -1) {
                        head = i;
                        pre = i;
                    }
                    tail = seats.length - 1 - i;
                }
            }
            for (int i = 0; i < seats.length; i++) {
                if (seats[i] == 1) {
                    middle = middle > (i - pre) / 2 ? middle : (i - pre) / 2;
                    pre = i;
                }
            }
            return Math.max(Math.max(head, tail), middle);
        }
    }

    // 874. Walking Robot Simulation - MEDIUM
    class Solution874a {
        public int robotSim(int[] commands, int[][] obstacles) {
            Set<Integer> obs = new HashSet<>(obstacles.length);
            for (var e : obstacles) {
                obs.add(func(e[0], e[1]));
            }
            int[] d = { 0, 1, 0, -1, 0 };
            int ans = 0, x = 0, y = 0, i = 0;
            for (int c : commands) {
                if (c == -2) {
                    i = (i + 3) % 4;
                } else if (c == -1) {
                    i = (i + 1) % 4;
                } else {
                    while (c-- > 0) {
                        int nx = x + d[i], ny = y + d[i + 1];
                        if (obs.contains(func(nx, ny))) {
                            break;
                        }
                        x = nx;
                        y = ny;
                        ans = Math.max(ans, x * x + y * y);
                    }
                }
            }
            return ans;
        }

        // 二维坐标 (i, j) 转一维, 常见的方式是 (i * n + j), 这道题 X, Y 长度最大为 60000, 所以代码中 n 取了个 60010
        private int func(int x, int y) {
            return x * 60010 + y;
        }
    }

    // 884. Uncommon Words from Two Sentences - EASY
    class Solution884a {
        public String[] uncommonFromSentences(String s1, String s2) {
            HashMap<String, Integer> m = new HashMap<String, Integer>();
            String[] arr1 = s1.split(" ");
            for (String w : arr1) {
                m.put(w, m.getOrDefault(w, 0) + 1);
            }
            String[] arr2 = s2.split(" ");
            for (String w : arr2) {
                m.put(w, m.getOrDefault(w, 0) + 1);
            }
            ArrayList<String> ans = new ArrayList<String>();
            for (Map.Entry<String, Integer> e : m.entrySet()) {
                if (e.getValue() == 1) {
                    ans.add(e.getKey());
                }
            }
            return ans.toArray(new String[0]);
        }
    }
}
