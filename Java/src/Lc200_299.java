package src;

import java.util.*;

@SuppressWarnings("unchecked")
public class Lc200_299 {
    // 210 - Course Schedule II - MEDIUM
    class Solution210a {
        public int[] findOrder(int numCourses, int[][] prerequisites) {
            int indegree[] = new int[numCourses], ans[] = new int[numCourses];
            List<Integer>[] g = new List[numCourses];
            for (int i = 0; i < numCourses; i++) {
                g[i] = new ArrayList<>();
            }
            for (int[] p : prerequisites) {
                indegree[p[0]]++;
                g[p[1]].add(p[0]);
            }
            Deque<Integer> dq = new ArrayDeque<>(); // 3ms
            for (int i = 0; i < numCourses; i++) {
                if (indegree[i] == 0) {
                    dq.add(i);
                }
            }
            // Deque<Integer> dq = IntStream // 6ms
            //         .range(0, numCourses)
            //         .filter(i -> indegree[i] == 0).boxed()
            //         .collect(Collectors.toCollection(ArrayDeque::new));
            int i = 0;
            while (!dq.isEmpty()) {
                int x = dq.poll();
                ans[i++] = x;
                for (int y : g[x]) {
                    indegree[y]--;
                    if (indegree[y] == 0) {
                        dq.add(y);
                    }
                }
            }
            return i == numCourses ? ans : new int[] {};
        }
    }

    class Solution210b { // 2ms
        private int idx, vis[], ans[];
        private List<Integer>[] g;

        private boolean dfs(int x) {
            if (vis[x] == 1)
                return false;
            if (vis[x] == 2)
                return true;
            vis[x] = 1;
            for (int y : g[x])
                if (!dfs(y))
                    return false;
            vis[x] = 2;
            ans[idx++] = x;
            return true;
        }

        public int[] findOrder(int numCourses, int[][] prerequisites) {
            vis = new int[numCourses];
            ans = new int[numCourses];
            g = new List[numCourses];
            for (int i = 0; i < numCourses; i++) {
                g[i] = new ArrayList<>();
            }
            for (int[] p : prerequisites) {
                g[p[0]].add(p[1]); // if build 'g' with successor, return ans[::-1]
            }
            for (int i = 0; i < numCourses; i++) {
                if (!dfs(i))
                    return new int[0];
            }
            return ans;
        }
    }

    // 213. House Robber II - MEDIUM
    class Solution213a {
        public int rob(int[] nums) {
            if (nums.length < 3)
                return nums.length == 1 ? nums[0] : Math.max(nums[0], nums[1]);
            int pre0 = nums[0], cur0 = Math.max(nums[0], nums[1]), pre1 = nums[1], cur1 = Math.max(nums[1], nums[2]);
            for (int i = 2; i < nums.length - 1; i++) {
                int tmp = cur0;
                cur0 = Math.max(pre0 + nums[i], cur0);
                pre0 = tmp;
            }
            for (int i = 3; i < nums.length; i++) {
                int tmp = cur1;
                cur1 = Math.max(pre1 + nums[i], cur1);
                pre1 = tmp;
            }
            return Math.max(Math.max(pre0, cur0), Math.max(pre1, cur1));
        }
    }
}
