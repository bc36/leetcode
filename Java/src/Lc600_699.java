package src;

import java.util.*;

public class Lc600_699 {
    // 617. Merge Two Binary Trees - EASY
    class Solution617a {
        public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
            if (root1 != null && root2 != null) {
                root1.val += root2.val;
                root1.left = mergeTrees(root1.left, root2.left);
                root1.right = mergeTrees(root1.right, root2.right);
            }
            return root1 != null ? root1 : root2;
        }
    }

    // 630. Course Schedule III - HARD
    class Solution630a {
        public int scheduleCourse(int[][] courses) {
            Arrays.sort(courses, (x, y) -> x[1] - y[1]);
            PriorityQueue<Integer> pq = new PriorityQueue<>((x, y) -> y - x); // 最大堆
            int total = 0;
            for (int c[] : courses) {
                int d = c[0], lastDay = c[1];
                if (d + total > lastDay && !pq.isEmpty() && d < pq.peek())
                    total -= pq.poll();
                if (d + total <= lastDay) {
                    pq.offer(d);
                    total += d;
                }
            }
            return pq.size();
        }
    }

    class Solution630b {
        public int scheduleCourse(int[][] courses) {
            Arrays.sort(courses, (x, y) -> x[1] - y[1]);
            PriorityQueue<Integer> pq = new PriorityQueue<>((x, y) -> y - x);
            int total = 0;
            for (int[] c : courses) {
                int d = c[0], lastDay = c[1];
                if (total + d <= lastDay) {
                    total += d;
                    pq.offer(d);
                } else if (!pq.isEmpty() && d < pq.peek()) {
                    total -= pq.poll() - d;
                    pq.offer(d);
                }
            }
            return pq.size();
        }
    }
}
