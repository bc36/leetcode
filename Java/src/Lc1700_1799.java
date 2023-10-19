package src;

import java.util.*;

@SuppressWarnings("unchecked")
public class Lc1700_1799 {
    // 1726. Tuple with Same Product - MEDIUM
    class Solution1726a {
        public int tupleSameProduct(int[] nums) {
            Map<Integer, Integer> mp = new HashMap<>();
            for (int i = 0; i < nums.length - 1; i++) {
                for (int j = i + 1; j < nums.length; j++) {
                    // mp.put(nums[i] * nums[j], mp.getOrDefault(nums[i] * nums[j], 0) + 1); // 179ms
                    mp.merge(nums[i] * nums[j], 1, Integer::sum); // 161ms
                    // mp.compute(nums[i] * nums[j], (k, v) -> Objects.isNull(v) ? 1 : v + 1); // 160ms
                }
            }
            int ans = 0;
            for (int v : mp.values()) {
                ans += v * (v - 1) / 2;
            }
            return ans * 8;

        }
    }

    // 1749. Maximum Absolute Sum of Any Subarray - MEDIUM
    class Solution1749a { // 3ms
        public int maxAbsoluteSum(int[] nums) {
            int ans = 0, p = 0, n = 0;
            for (int v : nums) {
                if (v > 0) {
                    p += v;
                    ans = ans > p ? ans : p;
                    n = n + v > 0 ? 0 : n + v;
                } else {
                    n += v;
                    ans = ans > -n ? ans : -n;
                    p = p + v > 0 ? p + v : 0;
                }
            }
            return ans;
        }
    }

    class Solution1749b { // 2ms
        public int maxAbsoluteSum(int[] nums) {
            int ans = 0, p = 0, n = 0;
            for (int v : nums) {
                p = p > 0 ? p + v : v;
                n = n < 0 ? n + v : v;
                ans = ans > p ? ans > -n ? ans : -n : p;
            }
            return ans;
        }
    }

    class Solution1749c { // 1ms
        public int maxAbsoluteSum(int[] nums) {
            int sum = 0, mx = 0, mn = 0;
            for (int x : nums) {
                sum += x;
                if (sum > mx) {
                    mx = sum;
                }
                if (sum < mn) {
                    mn = sum;
                }
            }
            return mx - mn;
        }
    }

    // 1761. Minimum Degree of a Connected Trio in a Graph - HARD
    class Solution1761a {
        public int minTrioDegree(int n, int[][] edges) {
            HashSet<Integer>[] graph = new HashSet[n];
            for (int i = 0; i < n; i++) {
                graph[i] = new HashSet<>();
            }
            for (int[] edge : edges) {
                graph[edge[0] - 1].add(edge[1] - 1);
                graph[edge[1] - 1].add(edge[0] - 1);
            }
            int ans = Integer.MAX_VALUE;
            for (int i = 0; i < n; i++) {
                for (int j : graph[i]) {
                    for (int k = j + 1; k < n; k++) {
                        if (graph[i].contains(k) && graph[j].contains(k)) {
                            ans = Math.min(ans, graph[i].size() + graph[j].size() + graph[k].size() - 6);
                        }
                    }
                }
            }
            return ans == Integer.MAX_VALUE ? -1 : ans;
        }
    }

    // 1782. Count Pairs Of Nodes - HARD
    class Solution {
        public int[] countPairs(int n, int[][] edges, int[] queries) {
            int[] d = new int[n + 1];
            var cntE = new HashMap<Integer, Integer>();
            for (var e : edges) {
                int x = e[0], y = e[1];
                if (x > y) {
                    int tmp = x;
                    x = y;
                    y = tmp;
                }
                d[x]++;
                d[y]++;
                cntE.merge(x << 16 | y, 1, Integer::sum);
            }
            int[] ans = new int[queries.length], sd = d.clone();
            Arrays.sort(sd);
            for (int j = 0; j < queries.length; j++) {
                int q = queries[j];
                int left = 1, right = n;
                while (left < right) {
                    if (sd[left] + sd[right] <= q) {
                        left++;
                    } else {
                        ans[j] += right - left;
                        right--;
                    }
                }
                for (var e : cntE.entrySet()) {
                    int k = e.getKey(), c = e.getValue();
                    int s = d[k >> 16] + d[k & 0xffff];
                    if (s > q && s - c <= q) {
                        ans[j]--;
                    }
                }
            }
            return ans;
        }
    }
}
