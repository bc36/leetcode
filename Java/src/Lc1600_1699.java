package src;

import java.util.*;

public class Lc1600_1699 {
    // 1604. Alert Using Same Key-Card Three or More Times in a One Hour Period - MEDIUM
    class Solution1604a { // 45ms
        public List<String> alertNames(String[] keyName, String[] keyTime) {
            int n = keyName.length;
            Map<String, List<Integer>> map = new HashMap<String, List<Integer>>();
            for (int i = 0; i < n; ++i) {
                String nm = keyName[i], t = keyTime[i];
                map.putIfAbsent(nm, new ArrayList<Integer>());
                int h = (t.charAt(0) - '0') * 10 + (t.charAt(1) - '0');
                int m = (t.charAt(3) - '0') * 10 + (t.charAt(4) - '0');
                map.get(nm).add(h * 60 + m);
            }
            List<String> ans = new ArrayList<>();
            for (String nm : map.keySet()) {
                List<Integer> arr = map.get(nm);
                if (arr.size() < 3) {
                    continue;
                }
                Collections.sort(arr);
                for (int i = 2; i < arr.size(); ++i) {
                    if (arr.get(i) - arr.get(i - 2) <= 60) {
                        ans.add(nm);
                        break;
                    }
                }
            }
            Collections.sort(ans);
            return ans;
        }
    }

    class Solution1604b { // 39ms
        public List<String> alertNames(String[] keyName, String[] keyTime) {
            int n = keyName.length;
            Map<String, List<Integer>> map = new HashMap<>(n);
            for (int i = 0; i < n; ++i) {
                String nm = keyName[i], t = keyTime[i];
                int minutes = convertTime2Minutes(t);
                map.computeIfAbsent(nm, key -> new ArrayList<>()).add(minutes);
            }
            List<String> ans = new ArrayList<>();
            for (String nm : map.keySet()) {
                List<Integer> arr = map.get(nm);
                if (arr.size() < 3)
                    continue;
                Collections.sort(arr);
                for (int i = 2; i < arr.size(); ++i)
                    if (arr.get(i) - arr.get(i - 2) <= 60) {
                        ans.add(nm);
                        break;
                    }
            }
            Collections.sort(ans);
            return ans;
        }

        private int convertTime2Minutes(String t) {
            char[] cs = t.toCharArray();
            int h = (cs[0] - '0') * 10 + (cs[1] - '0');
            int m = (cs[3] - '0') * 10 + (cs[4] - '0');
            return h * 60 + m;
        }
    }

    // 1605. Find Valid Matrix Given Row and Column Sums - MEDIUM
    class Solution1605a { // 7ms
        public int[][] restoreMatrix(int[] rowSum, int[] colSum) {
            int[][] ans = new int[rowSum.length][colSum.length];
            for (int i = 0; i < rowSum.length; ++i) {
                for (int j = 0; j < colSum.length; ++j) {
                    int v = Math.min(rowSum[i], colSum[j]);
                    ans[i][j] = v;
                    rowSum[i] -= v;
                    colSum[j] -= v;
                }
            }
            return ans;
        }
    }

    class Solution1605b { // 1ms
        public int[][] restoreMatrix(int[] rowSum, int[] colSum) {
            int n = rowSum.length, m = colSum.length, i = 0, j = 0;
            int[][] ans = new int[n][m];
            while (i < n && j < m) {
                int r = rowSum[i];
                int c = colSum[j];
                if (r < c) {
                    ans[i][j] = r;
                    colSum[j] -= r;
                    ++i;
                } else {
                    ans[i][j] = c;
                    rowSum[i] -= c;
                    ++j;
                }
            }
            return ans;
        }
    }

    // 1608. Special Array With X Elements Greater Than or Equal X - EASY
    class Solution1608a { // 0ms
        public int specialArray(int[] nums) {
            int cnt = 0, l = 0;
            for (int v : nums) {
                l = Math.max(l, v);
            }
            int[] arr = new int[l + 1];
            for (int v : nums) {
                ++arr[v];
            }
            for (int i = l; i >= 0; --i) {
                cnt += arr[i];
                if (cnt == i) {
                    return i;
                }
            }
            return -1;
        }
    }

    class Solution1608b { // 2ms
        public int specialArray(int[] nums) {
            int cnt = 0, l = Arrays.stream(nums).max().orElse(0), arr[] = new int[l + 1];
            for (int v : nums) {
                ++arr[v];
            }
            for (int i = l; i >= 0; --i) {
                cnt += arr[i];
                if (cnt == i) {
                    return i;
                }
            }
            return -1;
        }
    }

    class Solution1608c { // 0ms
        public int specialArray(int[] nums) {
            int l = 1, r = nums.length, cnt = 0;
            while (l <= r) {
                cnt = 0;
                int m = (l + r) / 2;
                for (int i = 0; i < nums.length; i++) {
                    if (nums[i] >= m) {
                        cnt++;
                    }
                }
                if (cnt == m) {
                    return cnt;
                } else if (cnt > m) {
                    l = m + 1;
                } else {
                    r = m - 1;
                }
            }
            return -1;
        }
    }

    // 1615. Maximal Network Rank - MEDIUM
    class Solution1615a { // 4ms
        public int maximalNetworkRank(int n, int[][] roads) {
            int[][] conn = new int[n][n];
            int[] deg = new int[n];
            int ans = 0;
            for (int i = 0; i < roads.length; ++i) {
                int a = roads[i][0], b = roads[i][1];
                conn[a][b] = 1;
                conn[b][a] = 1;
                ++deg[a];
                ++deg[b];
            }
            for (int a = 0; a < n; ++a) {
                for (int b = a + 1; b < n; ++b) {
                    ans = Math.max(ans, deg[a] + deg[b] - conn[a][b]);
                }
            }
            return ans;
        }
    }

    class Solution1615b { // 2ms
        public int maximalNetworkRank(int n, int[][] roads) {
            int[][] conn = new int[n][n];
            int[] deg = new int[n];
            for (int[] road : roads) { // 1ms faster
                conn[road[0]][road[1]] = 1;
                conn[road[1]][road[0]] = 1;
                ++deg[road[0]];
                ++deg[road[1]];
            }
            int ans = Integer.MIN_VALUE;
            for (int a = 0; a < n; ++a) {
                for (int b = a + 1; b < n; ++b) {
                    int cnt = deg[a] + deg[b] - conn[a][b]; // 1ms faster
                    ans = ans < cnt ? cnt : ans;
                }
            }
            return ans;
        }
    }

    // 1616. Split Two Strings to Make Palindrome - MEDIUM
    class Solution1616a {
        public boolean checkPalindromeFormation(String a, String b) {
            class Inner {
                private boolean check(String a, String b) {
                    int i = 0, j = a.length() - 1;
                    while (i < j && a.charAt(i) == b.charAt(j)) {
                        ++i;
                        --j;
                    }
                    return isPalindrome(a, i, j) || isPalindrome(b, i, j);
                }

                private boolean isPalindrome(String s, int i, int j) {
                    while (i < j && s.charAt(i) == s.charAt(j)) {
                        ++i;
                        --j;
                    }
                    return i >= j;
                }
            }
            Inner inr = new Inner();
            return inr.check(a, b) || inr.check(b, a);
        }
    }

    // 1625. Lexicographically Smallest String After Applying Operations - MEDIUM
    class Solution1625a {
        public String findLexSmallestString(String s, int a, int b) {
            Deque<String> q = new ArrayDeque<>();
            q.offer(s);
            Set<String> vis = new HashSet<>();
            vis.add(s);
            String ans = s;
            int n = s.length();
            while (!q.isEmpty()) {
                s = q.poll();
                if (ans.compareTo(s) > 0) {
                    ans = s;
                }
                char[] cs = s.toCharArray();
                for (int i = 1; i < n; i += 2) {
                    cs[i] = (char) ('0' + ((cs[i] - '0' + a) % 10));
                }
                String s1 = String.valueOf(cs);
                String s2 = s.substring(n - b) + s.substring(0, n - b);
                for (String x : List.of(s1, s2)) {
                    if (vis.add(x)) {
                        q.offer(x);
                    }
                }
            }
            return ans;
        }
    }

    // 1626. Best Team With No Conflicts - MEDIUM
    class Solution1626a { // O(n^2) / O(n), 48ms
        public int bestTeamScore(int[] scores, int[] ages) {
            int n = scores.length, ans = 0, f[] = new int[n + 1];
            Integer[] order = new Integer[n];
            for (int i = 0; i < n; ++i)
                order[i] = i;
            Arrays.sort(order, (i, j) -> scores[i] != scores[j] ? scores[i] - scores[j] : ages[i] - ages[j]);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < i; ++j)
                    if (ages[order[j]] <= ages[order[i]])
                        f[i] = Math.max(f[i], f[j]);
                f[i] += scores[order[i]];
                ans = Math.max(ans, f[i]);
            }
            return ans;
        }
    }

    class Solution1626b {
        public int bestTeamScore(int[] scores, int[] ages) {
            int n = ages.length, arr[][] = new int[n][2], f[] = new int[n], ans = 0;
            for (int i = 0; i < n; ++i) {
                arr[i] = new int[] { scores[i], ages[i] };
            }
            Arrays.sort(arr, (a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < i; ++j) {
                    if (arr[i][1] >= arr[j][1]) {
                        f[i] = Math.max(f[i], f[j]);
                    }
                }
                f[i] += arr[i][0];
                ans = Math.max(ans, f[i]);
            }
            return ans;
        }
    }

    class Solution1626c { // O(nlogn + nU) / O(n + U), U = max(ages), 48ms
        public int bestTeamScore(int[] scores, int[] ages) {
            int n = scores.length, u = 0, ans = 0;
            Integer[] order = new Integer[n];
            for (int i = 0; i < n; ++i) {
                order[i] = i;
                u = Math.max(u, ages[i]);
            }
            Arrays.sort(order, (i, j) -> scores[i] != scores[j] ? scores[i] - scores[j] : ages[i] - ages[j]);
            int[] f = new int[u + 1];
            for (int i : order) {
                int a = ages[i];
                for (int j = 1; j <= a; ++j)
                    f[a] = Math.max(f[a], f[j]);
                f[a] += scores[i];
                ans = Math.max(ans, f[a]);
            }
            return ans;
        }
    }

    class Solution1626d { // 10ms
        public int bestTeamScore(int[] scores, int[] ages) {
            int n = scores.length;
            Integer[] ids = new Integer[n];
            for (int i = 0; i < n; ++i)
                ids[i] = i;
            Arrays.sort(ids, (i, j) -> scores[i] != scores[j] ? scores[i] - scores[j] : ages[i] - ages[j]);

            for (int i : ids)
                update(ages[i], query(ages[i]) + scores[i]);
            return query(MX);
        }

        private static final int MX = 1000;
        private final int[] t = new int[MX + 1];

        // 返回 max(maxSum[:i+1])
        private int query(int i) {
            int mx = 0;
            for (; i > 0; i &= i - 1)
                mx = Math.max(mx, t[i]);
            return mx;
        }

        // 更新 maxSum[i] 为 mx
        private void update(int i, int mx) {
            for (; i <= MX; i += i & -i)
                t[i] = Math.max(t[i], mx);
        }
    }

    class Solution1626e { // 11ms, BIT TODO
        private int[] tree;
        private int maxAge;

        public int bestTeamScore(int[] scores, int[] ages) {
            int n = scores.length;
            Integer[] idx = new Integer[n];
            maxAge = 0;
            for (int i = 0; i < n; i++) {
                idx[i] = i;
                maxAge = Math.max(maxAge, ages[i]);
            }
            tree = new int[maxAge + 1];
            Arrays.sort(idx, (a, b) -> {
                if (scores[a] == scores[b])
                    return ages[a] - ages[b];
                return scores[a] - scores[b];
            });
            int res = 0;
            for (int i = 0; i < n; i++) {
                int curMax = scores[idx[i]] + find(ages[idx[i]]);
                update(ages[idx[i]], curMax);
                res = Math.max(res, curMax);
            }
            return res;
        }

        private int lowBit(int x) {
            return x & (-x);
        }

        /*更新tree[i]保存的以age[i]为最大年龄的最大得分，由于tree[i]更新，因此包含age[i]的区间的最大值也要随之更新*/
        private void update(int i, int v) {
            for (; i <= maxAge; i += lowBit(i))
                tree[i] = Math.max(tree[i], v);
        }

        /*以区间为单位获取以<=age[i]的每个球员为最大年龄球员的各个组合最大得分
         * 比较<=age[i]的各个区间的最大值，取最大*/
        private int find(int i) {
            int res = 0;
            for (; i > 0; i -= lowBit(i))
                res = Math.max(res, tree[i]);
            return res;
        }
    }

    class Solution1626f {
        int maxAge;
        int[][] sa;
        // 数状数组存储当前age的最大分数和，长度为maxAge
        int[] tree;

        public int bestTeamScore(int[] scores, int[] ages) {
            // 流式处理获取ages的最大值
            this.maxAge = Arrays.stream(ages).max().getAsInt();
            this.tree = new int[maxAge + 1];
            int n = scores.length;
            this.sa = new int[n][2];
            for (int i = 0; i < n; i++) {
                sa[i] = new int[] { scores[i], ages[i] };
            }
            // 按照ages和scores从小到大排序
            Arrays.sort(sa, (a, b) -> a[0] != b[0] ? a[0] - b[0] : a[1] - b[1]);
            int res = 0;
            for (int i = 0; i < n; i++) {
                int curr = query(sa[i][1]) + sa[i][0];
                // update参数代表树状数组的坐标 也就是age
                update(sa[i][1], curr);
                res = Math.max(res, curr);
            }
            return res;
        }

        private int lowbit(int x) {
            return x & (-x);
        }

        // 寻找小于等于当前age的最大分数
        private int query(int i) {
            int res = 0;
            while (i > 0) {
                res = Math.max(res, tree[i]);
                i -= lowbit(i);
            }
            return res;
        }

        // 更新数状数组中当前坐标及其有贡献的父节点的值
        private void update(int i, int val) {
            while (i <= maxAge) {
                tree[i] = Math.max(tree[i], val);
                i += lowbit(i);
            }
        }
    }

    class BinaryIndexedTree { // 10ms
        private int n;
        private int[] c;

        public BinaryIndexedTree(int n) {
            this.n = n;
            c = new int[n + 1];
        }

        public void update(int x, int val) {
            while (x <= n) {
                c[x] = Math.max(c[x], val);
                x += x & -x;
            }
        }

        public int query(int x) {
            int s = 0;
            while (x > 0) {
                s = Math.max(s, c[x]);
                x -= x & -x;
            }
            return s;
        }
    }

    class Solution1626g {
        public int bestTeamScore(int[] scores, int[] ages) {
            int n = ages.length;
            int[][] arr = new int[n][2];
            for (int i = 0; i < n; ++i) {
                arr[i] = new int[] { scores[i], ages[i] };
            }
            Arrays.sort(arr, (a, b) -> a[0] == b[0] ? a[1] - b[1] : a[0] - b[0]);
            int m = 0;
            for (int age : ages) {
                m = Math.max(m, age);
            }
            BinaryIndexedTree tree = new BinaryIndexedTree(m);
            for (int[] x : arr) {
                tree.update(x[1], x[0] + tree.query(x[1]));
            }
            return tree.query(m);
        }
    }

    // 1630. Arithmetic Subarrays - MEDIUM
    class Solution1630a { // O(nlogn * n * m) / O(), 21ms, 暴力排序
        public List<Boolean> checkArithmeticSubarrays(int[] nums, int[] l, int[] r) {
            Helper h = new Helper() {
                public boolean check(int[] arr) {
                    int d = arr[1] - arr[0];
                    for (int j = 2; j < arr.length; ++j) {
                        if (arr[j] - arr[j - 1] != d) {
                            return false;
                        }
                    }
                    return true;
                }
            };

            List<Boolean> ans = new ArrayList<>();
            for (int i = 0; i < l.length; ++i) {
                int len = r[i] - l[i] + 1;
                int[] tmp = new int[len];
                System.arraycopy(nums, l[i], tmp, 0, len);
                Arrays.sort(tmp);
                ans.add(h.check(tmp));
            }
            return ans;
        }
    }

    class Solution1630b { // O(nm) / O(n), 3ms, 无排序 + 数组
        public List<Boolean> checkArithmeticSubarrays(int[] nums, int[] l, int[] r) {
            List<Boolean> result = new ArrayList<>();
            for (int i = 0; i < l.length; ++i) {
                result.add(check(nums, l[i], r[i]));
            }
            return result;
        }

        private boolean check(int[] nums, int l, int r) {
            int min = (int) 1e9, max = (int) -1e9;
            for (int i = l; i <= r; ++i) {
                min = Math.min(min, nums[i]);
                max = Math.max(max, nums[i]);
            }
            if (min == max)
                return true;
            if ((max - min) % (r - l) != 0)
                return false;
            int d = (max - min) / (r - l);
            boolean[] dict = new boolean[r - l + 1];
            for (int i = l; i <= r; ++i) {
                if ((nums[i] - min) % d != 0)
                    return false;
                int j = (nums[i] - min) / d;
                if (dict[j])
                    return false;
                dict[j] = true;
            }
            return true;
        }
    }

    // 1637. Widest Vertical Area Between Two Points Containing No Points - MEDIUM
    class Solution1637a { // 13ms
        public int maxWidthOfVerticalArea(int[][] points) {
            int n = points.length, arr[] = new int[n], ans = 0;
            for (int i = 0; i < n; ++i) {
                arr[i] = points[i][0];
            }
            Arrays.sort(arr);
            for (int i = 1; i < n; ++i) {
                ans = Math.max(ans, arr[i] - arr[i - 1]);
            }
            return ans;
        }
    }

    // 1638. Count Substrings That Differ by One Character - MEDIUM
    class Solution1638a {
        // 6ms
        public int countSubstrings(String s, String t) {
            int n = s.length(), m = t.length(), ans = 0;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    if (s.charAt(i) == t.charAt(j))
                        continue;
                    int l = 0;
                    while (i - (l + 1) >= 0 && j - (l + 1) >= 0 && s.charAt(i - (l + 1)) == t.charAt(j - (l + 1)))
                        ++l;
                    int r = 0;
                    while (i + (r + 1) <= n && j + (r + 1) <= m && s.charAt(i + (r + 1)) == t.charAt(j + (r + 1)))
                        ++r;
                    ans += (l + 1) * (r + 1);
                }
            }
            return ans;
        }
    }

    class Solution1638b { // 3ms
        public int countSubstrings(String s, String t) {
            int n = s.length(), m = t.length(), ans = 0;
            for (int d = -(n - 1); d < m; ++d) {
                for (int i = Math.max(-d, 0), j = i + d, l = 0, r = 1; i <= n && j <= m; ++i, ++j) {
                    if (i == n || j == m || s.charAt(i) != t.charAt(j)) {
                        ans += l * r;
                        l = r;
                        r = 1;
                    } else {
                        ++r;
                    }
                }

            }
            return ans;
        }
    }

    class Solution1638c { // 2ms, cs[i] versus s.charAt(i)
        public int countSubstrings(String s, String t) {
            char[] cs = s.toCharArray(), ct = t.toCharArray();
            int n = cs.length, m = ct.length, ans = 0;
            for (int d = -(n - 1); d < m; ++d) {
                for (int i = Math.max(-d, 0), j = i + d, l = 0, r = 1; i <= n && j <= m; ++i, ++j) {
                    if (i == n || j == m || cs[i] != ct[j]) {
                        ans += l * r;
                        l = r;
                        r = 1;
                    } else {
                        ++r;
                    }
                }
            }
            return ans;
        }
    }

    // 1641. Count Sorted Vowel Strings - MEDIUM
    class Solution1641a { // 1ms
        public int countVowelStrings(int n) {
            int[][] f = new int[n][5];
            for (int i = 0; i < 5; ++i)
                f[0][i] = 1;
            for (int i = 1; i < n; ++i) {
                f[i][0] = f[i - 1][0];
                f[i][1] = f[i - 1][0] + f[i - 1][1];
                f[i][2] = f[i - 1][0] + f[i - 1][1] + f[i - 1][2];
                f[i][3] = f[i - 1][0] + f[i - 1][1] + f[i - 1][2] + f[i - 1][3];
                f[i][4] = f[i - 1][0] + f[i - 1][1] + f[i - 1][2] + f[i - 1][3] + f[i - 1][4];
            }
            return Arrays.stream(f[n - 1]).sum();
        }
    }

    class Solution1641b { // 0ms
        public int countVowelStrings(int n) {
            int a = 1, e = 1, i = 1, o = 1, u = 1;
            for (int j = 1; j < n; ++j) {
                a = a + e + i + o + u;
                e = e + i + o + u;
                i = i + o + u;
                o = o + u;
                // u = u;
            }
            return a + e + i + o + u;
        }
    }

    class Solution1641c { // 0ms
        public int countVowelStrings(int n) {
            return (n + 1) * (n + 2) * (n + 3) * (n + 4) / 24; // C(n + 4, 4)
        }
    }

    // 1653. Minimum Deletions to Make String Balanced - MEDIUM
    class Solution1653a {
        public int minimumDeletions(String s) {
            int del = 0; // all 'a'
            char[] arr = s.toCharArray();
            for (char c : arr) {
                del += 'b' - c;
            }
            int ans = del;
            for (char c : arr) {
                del += (c - 'a') * 2 - 1;
                ans = Math.min(ans, del);
            }
            return ans;
        }
    }

    class Solution1653b {
        public int minimumDeletions(String s) {
            int a = 0, b = 0;
            for (int i = 0; i < s.length(); ++i) {
                a += (s.charAt(i) - 'a') ^ 1;
                // if (s.charAt(i) == 'a') { // slow
                //     ++a;
                // }
            }
            int ans = s.length();
            for (int i = 0; i < s.length(); ++i) {
                // a -= s.charAt(i) == 'a' ? 1 : 0; // slow
                char c = s.charAt(i);
                a -= (c - 'a') ^ 1;
                ans = Math.min(ans, a + b);
                // b += s.charAt(i) == 'b' ? 1 : 0;
                b += (c - 'a') ^ 0;
            }
            return ans;
        }
    }

    class Solution1653c {
        public int minimumDeletions(String s) {
            // dp
            int ans = 0, b = 0;
            for (int i = 0; i < s.length(); ++i) {
                if (s.charAt(i) == 'b') {
                    ++b;
                } else {
                    ans = Math.min(ans + 1, b);
                }
            }
            return ans;
        }
    }

    class Solution1653d {
        public int minimumDeletions(String s) {
            // dp, 统计当前数字以 a 或者 b 结尾的最少操作数
            int a = 0, b = 0;
            for (int i = 0; i < s.length(); ++i) {
                char c = s.charAt(i);
                // 以 b 结尾的最小操作数, 前一个可以是 a, 也可以是 b, 取前一个 a/b 结尾的最小值 + 当前操作数(a + 1, b + 0)
                b = Math.min(b, a) + 'b' - c;
                // 以 a 结尾的最小操作数, 前一个必须是 a, 也就是加上当前位置变化为 b 的操作数
                a = a + c - 'a';
            }
            return Math.min(a, b);
        }
    }

    // 1654. Minimum Jumps to Reach Home - MEDIUM
    class Solution1654a {
        public int minimumJumps(int[] forbidden, int a, int b, int x) {
            HashSet<Integer> set = new HashSet<>();
            for (int i : forbidden) {
                set.add(i);
            }
            HashSet<List<Integer>> visited = new HashSet<>();
            ArrayDeque<List<Integer>> deque = new ArrayDeque<>();
            deque.add(List.of(0, 1));
            for (int i = 0; !deque.isEmpty(); i++) {
                for (int j = deque.size(); j > 0; j--) {
                    List<Integer> poll = deque.poll();
                    if (poll.get(0) == x) {
                        return i;
                    } else if (poll.get(0) >= 0 && poll.get(0) < x << 4 && !set.contains(poll.get(0))
                            && !visited.contains(poll)) {
                        visited.add(poll);
                        deque.add(List.of(poll.get(0) + a, 1));
                        if (poll.get(1) != -1) {
                            deque.add(List.of(poll.get(0) - b, -1));
                        }
                    }
                }
            }
            return -1;
        }
    }

    // 1657. Determine if Two Strings Are Close - MEDIUM
    class Solution1657a { // 15ms
        public boolean closeStrings(String word1, String word2) {
            return Arrays.equals(count(word1), count(word2)) && state(word1) == state(word2);
        }

        public int state(String s) {
            int ans = 0;
            for (char c : s.toCharArray()) {
                ans |= 1 << (c - 'a');
            }
            return ans;
        }

        public int[] count(String s) {
            int[] ans = new int['z' - 'a' + 1];
            for (char c : s.toCharArray()) {
                ans[c - 'a']++;
            }
            Arrays.sort(ans);
            return ans;
        }
    }

    class Solution1657b { // 8ms
        public boolean closeStrings(String word1, String word2) {
            if (word1.length() != word2.length()) {
                return false;
            }
            int[] cnt1 = new int[26], cnt2 = new int[26];
            char[] cs1 = word1.toCharArray(), cs2 = word2.toCharArray();
            for (int i = 0; i < cs1.length; ++i) {
                ++cnt1[cs1[i] - 'a'];
                ++cnt2[cs2[i] - 'a'];
            }
            for (int i = 0; i < 26; ++i) {
                if (cnt1[i] == 0 && (cnt1[i] ^ cnt2[i]) > 0) {
                    return false;
                }
            }
            Arrays.sort(cnt1);
            Arrays.sort(cnt2);
            for (int i = 0; i < 26; ++i) {
                if (cnt1[i] != cnt2[i]) {
                    return false;
                }
            }
            return true;
        }
    }

    // 1685. Sum of Absolute Differences in a Sorted Array - MEDIUM
    class Solution1685a {
        public int[] getSumAbsoluteDifferences(int[] nums) {
            int[] ret = new int[nums.length];
            int sum = 0;
            for (int i = 0; i < nums.length; i++) {
                ret[i] += nums[i] * i - sum;
                sum += nums[i];
            }
            sum = 0;
            for (int i = nums.length - 1; i >= 0; i--) {
                sum += nums[i];
                ret[i] += sum - nums[i] * (nums.length - i);
            }
            return ret;
        }
    }

    class Solution1685b {
        public int longestDecomposition(String s) {
            if (s.isEmpty())
                return 0;
            for (int i = 1, n = s.length(); i <= n / 2; ++i)
                if (s.substring(0, i).equals(s.substring(n - i)))
                    return 2 + longestDecomposition(s.substring(i, n - i));
            return 1;
        }
    }

}
