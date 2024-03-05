package src;

import java.util.*;

@SuppressWarnings("unchecked")
public class Lc1900_1999 {
    // 1911. Maximum Alternating Subsequence Sum - MEDIUM
    class Solution1911 {
        public long maxAlternatingSum(int[] nums) {
            long f = 0, g = 0;
            for (int x : nums) {
                long ff = Math.max(g - x, f);
                long gg = Math.max(f + x, g);
                f = ff;
                g = gg;
            }
            return Math.max(f, g);
        }
    }

    // 1921. Eliminate Maximum Number of Monsters - MEDIUM
    class Solution1921a { // 17ms
        public int eliminateMaximum(int[] dist, int[] speed) {
            int n = dist.length, times[] = new int[n];
            for (int i = 0; i < n; ++i) {
                times[i] = (dist[i] - 1) / speed[i];
            }
            Arrays.sort(times);
            for (int i = 0; i < n; ++i) {
                if (times[i] < i) {
                    return i;
                }
            }
            return n;
        }
    }

    // 1976. Number of Ways to Arrive at Destination - MEDIUM
    class Solution1976a {
        public int countPaths(int n, int[][] roads) {
            ArrayList<int[]>[] list = new ArrayList[n];
            ArrayList<Integer>[] list2 = new ArrayList[n];
            for (int i = 0; i < n; i++) {
                list[i] = new ArrayList<>();
                list2[i] = new ArrayList<>();
            }
            for (int[] road : roads) {
                list[road[0]].add(new int[] { road[1], road[2] });
                list[road[1]].add(new int[] { road[0], road[2] });
            }
            Long[] dist = new Long[n];
            PriorityQueue<long[]> queue = new PriorityQueue<>((o, p) -> Long.compare(o[1], p[1]));
            for (queue.add(new long[2]); !queue.isEmpty();) {
                long[] remove = queue.remove();
                if (dist[(int) remove[0]] == null) {
                    dist[(int) remove[0]] = remove[1];
                    for (int[] i : list[(int) remove[0]]) {
                        queue.add(new long[] { i[0], remove[1] + i[1] });
                    }
                }
            }
            int[] count = new int[n], dp = new int[n];
            for (int[] road : roads) {
                if (dist[road[0]] + road[2] == dist[road[1]]) {
                    list2[road[0]].add(road[1]);
                    count[road[1]]++;
                }
                if (dist[road[1]] + road[2] == dist[road[0]]) {
                    list2[road[1]].add(road[0]);
                    count[road[0]]++;
                }
            }
            ArrayDeque<Integer> deque = new ArrayDeque<>();
            dp[0] = 1;
            for (deque.add(0); !deque.isEmpty();) {
                int remove = deque.remove();
                for (int i : list2[remove]) {
                    dp[i] = (dp[i] + dp[remove]) % 1000000007;
                    if (--count[i] == 0) {
                        deque.add(i);
                    }
                }
            }
            return dp[n - 1];
        }
    }
}
