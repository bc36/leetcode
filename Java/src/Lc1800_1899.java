package src;

import java.util.*;

public class Lc1800_1899 {
    // 1851. Minimum Interval to Include Each Query - HARD
    class Solution1851a { // 120ms
        public int[] minInterval(int[][] intervals, int[] queries) {
            int[][] qidx = new int[queries.length][0];
            for (int i = 0; i < queries.length; ++i) {
                qidx[i] = new int[] { queries[i], i };
            }
            Arrays.sort(qidx, (a, b) -> a[0] - b[0]);
            int[] ans = new int[queries.length];
            Arrays.fill(ans, -1);
            Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
            PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] - b[0]);
            int i = 0;
            for (int[] q : qidx) {
                while (i < intervals.length && intervals[i][0] <= q[0]) {
                    pq.offer(new int[] { intervals[i][1] - intervals[i][0] + 1, intervals[i][1] });
                    i++;
                }
                while (!pq.isEmpty() && pq.peek()[1] < q[0])
                    pq.poll();
                if (!pq.isEmpty())
                    ans[q[1]] = pq.peek()[0];
            }
            return ans;
        }
    }

    class Solution1851b { // 95ms
        public int[] minInterval(int[][] intervals, int[] queries) {
            Map<Integer, Integer> qToAns = new HashMap<>();
            int[] sortedQueries = queries.clone();
            Arrays.sort(sortedQueries);
            Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
            PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[0] - b[0]);
            int i = 0;
            for (int q : sortedQueries) {
                while (i < intervals.length && intervals[i][0] <= q) {
                    pq.offer(new int[] { intervals[i][1] - intervals[i][0] + 1, intervals[i][1] });
                    i++;
                }
                while (!pq.isEmpty() && pq.peek()[1] < q)
                    pq.poll();
                if (pq.isEmpty())
                    qToAns.put(q, -1);
                else
                    qToAns.put(q, pq.peek()[0]);

            }
            int[] ans = new int[queries.length];
            i = 0;
            for (int q : queries) {
                ans[i++] = qToAns.get(q);
            }
            return ans;
        }
    }
}