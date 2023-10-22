package src;

import java.util.*;

@SuppressWarnings("unchecked")
public class Lc1400_1499 {
    // 1401. Circle and Rectangle Overlapping - MEDIUM
    class Solution1401a { // 0ms
        public boolean checkOverlap(int radius, int xCenter, int yCenter, int x1, int y1, int x2, int y2) {
            int dx = Math.max(0, Math.max(x1 - xCenter, xCenter - x2));
            int dy = Math.max(0, Math.max(y1 - yCenter, yCenter - y2));
            return dx * dx + dy * dy <= radius * radius;
        }
    }

    class Solution1401b { // 0ms
        public boolean checkOverlap(int radius, int xCenter, int yCenter, int x1, int y1, int x2, int y2) {
            int dx = Collections.max(Arrays.asList(0, x1 - xCenter, xCenter - x2));
            int dy = Collections.max(Arrays.asList(0, y1 - yCenter, yCenter - y2));
            return dx * dx + dy * dy <= radius * radius;
        }
    }

    // 1402. Reducing Dishes - HARD
    class Solution1402a {
        public int maxSatisfaction(int[] satisfaction) {
            Arrays.sort(satisfaction);
            int f = 0, s = 0;
            for (int i = satisfaction.length - 1; i >= 0; i--) {
                s += satisfaction[i];
                if (s <= 0)
                    break;
                f += s; // f(k) = f(k - 1) + s
            }
            return f;
        }
    }

    // 1414. Find the Minimum Number of Fibonacci Numbers Whose Sum Is K - MEDIUM
    class Solution1414a {
        public int findMinFibonacciNumbers(int k) {
            List<Integer> f = new ArrayList<Integer>();
            f.add(1);
            int f1 = 1, f2 = 1;
            while (f1 + f2 <= k) {
                int f3 = f1 + f2;
                f1 = f2;
                f2 = f3;
                f.add(f3);
            }
            int ans = 0;
            int i = f.size() - 1;
            while (k != 0) {
                if (k >= f.get(i)) {
                    k -= f.get(i);
                    ans++;
                }
                i--;
            }
            return ans;
        }
    }

    class Solution1414b {
        public int findMinFibonacciNumbers(int k) {
            if (k == 0) {
                return 0;
            }
            int f1 = 1, f2 = 1;
            while (f1 + f2 <= k) {
                f2 = f1 + f2;
                f1 = f2 - f1;
            }
            return findMinFibonacciNumbers(k - f2) + 1;
        }
    }

    // 1419. Minimum Number of Frogs Croaking - MEDIUM
    class Solution1419 {
        public int minNumberOfFrogs(String croakOfFrogs) {
            int ans = 0, c = 0, r = 0, o = 0, a = 0;
            for (char ch : croakOfFrogs.toCharArray()) {
                if (ch == 'c') {
                    c++;
                } else if (ch == 'r') {
                    c--;
                    r++;
                } else if (ch == 'o') {
                    r--;
                    o++;
                } else if (ch == 'a') {
                    o--;
                    a++;
                } else {
                    a--;
                }
                if (c < 0 || r < 0 || o < 0 || a < 0) {
                    return -1;
                }
                ans = Math.max(ans, c + r + o + a);
            }
            if (c + r + o + a != 0) {
                return -1;
            }
            return ans;
        }
    }

    // 1448. Count Good Nodes in Binary Tree - MEDIUM
    class Solution1448a {
        public int goodNodes(TreeNode root) {
            return inorder(root, root.val);
        }

        private int inorder(TreeNode root, int mx) {
            if (root == null) {
                return 0;
            }
            mx = Math.max(mx, root.val);
            return (root.val >= mx ? 1 : 0) + inorder(root.left, mx) + inorder(root.right, mx);
        }
    }

    // 1462. Course Schedule IV - MEDIUM
    class Solution1462a { // 8ms
        public List<Boolean> checkIfPrerequisite(int numCourses, int[][] prerequisites, int[][] queries) {
            int[] indegree = new int[numCourses], next = new int[prerequisites.length], head = new int[numCourses];
            for (int i = 0; i < numCourses; ++i)
                head[i] = -1;
            BitSet[] fa = new BitSet[numCourses];
            for (int i = 0; i < numCourses; i++)
                fa[i] = new BitSet(numCourses);
            for (int i = 0; i < prerequisites.length; ++i) {
                int x = prerequisites[i][0], y = prerequisites[i][1];
                indegree[y]++;
                fa[y].set(x);
                next[i] = head[x];
                head[x] = i;
            }
            Deque<Integer> dq = new LinkedList<>();
            for (int i = 0; i < numCourses; i++) {
                if (indegree[i] == 0)
                    dq.offer(i);
            }
            while (!dq.isEmpty()) {
                int x = dq.poll();
                for (int idx = head[x]; idx != -1; idx = next[idx]) {
                    int y = prerequisites[idx][1];
                    fa[y].or(fa[x]);
                    if (--indegree[y] == 0)
                        dq.offer(y);
                }
            }
            List<Boolean> ans = new ArrayList<>(queries.length);
            for (int[] query : queries) {
                if (fa[query[1]].get(query[0]))
                    ans.add(true);
                else
                    ans.add(false);
            }
            return ans;
        }
    }

    class Solution1462b { // 10ms
        private void dfs(List<Integer>[] g, boolean[][] fa, boolean[] vis, int x) {
            if (vis[x])
                return;
            vis[x] = true;
            for (int y : g[x]) {
                dfs(g, fa, vis, y);
                fa[x][y] = true;
                for (int i = 0; i < fa.length; ++i)
                    fa[x][i] = fa[x][i] | fa[y][i];
            }
        }

        public List<Boolean> checkIfPrerequisite(int numCourses, int[][] prerequisites, int[][] queries) {
            List<Integer>[] g = new List[numCourses];
            for (int i = 0; i < numCourses; i++)
                g[i] = new ArrayList<Integer>();
            boolean vis[] = new boolean[numCourses], fa[][] = new boolean[numCourses][numCourses];
            for (int[] p : prerequisites)
                g[p[0]].add(p[1]);
            for (int i = 0; i < numCourses; ++i)
                dfs(g, fa, vis, i);
            List<Boolean> res = new ArrayList<Boolean>(queries.length);
            for (int[] q : queries)
                res.add(fa[q[0]][q[1]]);
            return res;
        }
    }

    // 1483. Kth Ancestor of a Tree Node - HARD
    class TreeAncestor { // 66ms
        // 构建 dfs 序
        int[] i2do;
        int[] do2i;
        int o = 0;

        public void dfs(int x, List<Integer>[] tree) {
            i2do[x] = o;
            do2i[o] = x;
            o++;
            for (int y : tree[x]) {
                dfs(y, tree);
            }
        }

        // 存下每一层的所有节点 (dfs序号) 及每个节点对应的层号
        List<List<Integer>> layers;
        int[] do2l;

        public TreeAncestor(int n, int[] parent) {
            i2do = new int[n];
            do2i = new int[n];
            do2l = new int[n];
            layers = new ArrayList<>();

            List<Integer>[] tree = new List[n];
            for (int i = 0; i < n; i++)
                tree[i] = new ArrayList<>();
            for (int i = 1; i < n; i++) {
                tree[parent[i]].add(i);
            }
            dfs(0, tree);

            List<Integer> q = new ArrayList<>();
            q.add(i2do[0]);
            int lv = 0;
            while (q.size() != 0) {
                layers.add(q);
                List<Integer> nxt = new ArrayList<>();
                for (int x : q) {
                    do2l[x] = lv;
                    for (int y : tree[do2i[x]]) {
                        nxt.add(i2do[y]);
                    }
                }
                q = nxt;
                lv++;
            }

        }

        public int getKthAncestor(int node, int k) {
            // 先找到要去的层级
            int dfsOrder = i2do[node];
            int lv = do2l[dfsOrder];
            int tl = lv - k;
            if (tl < 0)
                return -1;
            List<Integer> layer = layers.get(tl);
            // 二分查找到祖先节点
            int l = 0, r = layer.size() - 1;
            while (l < r) {
                int m = (l + r + 1) >> 1;
                if (layer.get(m) < dfsOrder)
                    l = m;
                else
                    r = m - 1;
            }
            return do2i[layer.get(l)];
        }
    }

    // 1487. Making File Names Unique - MEDIUM
    class Solution1487a {
        public String[] getFolderNames(String[] names) {
            Map<String, Integer> m = new HashMap<>();
            for (int i = 0; i < names.length; ++i) {
                if (m.containsKey(names[i])) {
                    int k = m.get(names[i]);
                    while (m.containsKey(names[i] + "(" + k + ")")) {
                        k++;
                    }
                    m.put(names[i], k + 1);
                    names[i] += "(" + k + ")";
                }
                m.put(names[i], 1);
            }
            return names;
        }
    }
}
