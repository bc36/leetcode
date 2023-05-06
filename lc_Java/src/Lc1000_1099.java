package src;

import java.util.*;

public class Lc1000_1099 {
    public class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    // 1000. Minimum Cost to Merge Stones - HARD
    class Solution1000a {
        private int k, pre[], memo[][][];

        @SuppressWarnings("unused")
        private int dfs(int l, int r, int p) {
            if (memo[l][r][p] != -1)
                return memo[l][r][p];
            if (p == 1)
                return memo[l][r][p] = l == r ? 0 : dfs(l, r, k) + pre[r + 1] - pre[l];
            int cur = Integer.MAX_VALUE;
            for (int i = l; i < r; i += k - 1)
                cur = Math.min(cur, dfs(l, i, 1) + dfs(i + 1, r, p - 1));
            return memo[l][r][p] = cur;
        }

        private int dfs2(int l, int r, int p) {
            if (memo[l][r][p] != -1)
                return memo[l][r][p];
            if (r - l + 1 < p)
                return memo[l][r][p] = 0;
            if (p == 1)
                return memo[l][r][p] = dfs2(l, r, k) + pre[r + 1] - pre[l];
            int cur = Integer.MAX_VALUE;
            for (int i = l; i < r; i += k - 1)
                cur = Math.min(cur, dfs2(l, i, 1) + dfs2(i + 1, r, p - 1));
            return memo[l][r][p] = cur;
        }

        public int mergeStones(int[] stones, int k) {
            int n = stones.length;
            if ((n - 1) % (k - 1) > 0)
                return -1;
            this.k = k;
            pre = new int[n + 1];
            for (int i = 0; i < n; i++)
                pre[i + 1] = pre[i] + stones[i];
            memo = new int[n][n][k + 1];
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    Arrays.fill(memo[i][j], -1);
            // return dfs(0, n - 1, 1);
            return dfs2(0, n - 1, k);
        }
    }

    // 1003. Check If Word Is Valid After Substitutions - MEDIUM
    class Solution1003a {
        // 6ms
        public boolean isValid(String s) {
            String t = "abc";
            while (s.length() > 0) {
                if (s.indexOf(t) == -1) {
                    return false;
                }
                s = s.replace(t, "");
            }
            return true;
        }
    }

    class Solution1003b {
        // 6ms
        public boolean isValid(String s) {
            Deque<Character> st = new ArrayDeque<>();
            for (char c : s.toCharArray()) {
                if (c == 'c') {
                    if (st.size() < 2)
                        return false;
                    char p1 = st.pop();
                    char p2 = st.pop();
                    if (p1 != 'b' || p2 != 'a')
                        return false;
                } else {
                    st.push(c);
                }
            }
            return st.isEmpty();
        }
    }

    class Solution1003c {
        // 2ms
        public boolean isValid(String S) {
            char[] s = S.toCharArray();
            int i = 0; // 
            for (char c : s) {
                if (c > 'a' && (i == 0 || c - s[--i] != 1))
                    return false;
                if (c < 'c')
                    s[i++] = c;
            }
            return i == 0;
        }
    }

    class Solution1003d {
        // 7ms
        public boolean isValid(String S) {
            char[] s = S.toCharArray();
            Deque<Character> st = new ArrayDeque<>();
            for (char c : s) {
                if (c > 'a' && (st.isEmpty() || c - st.pop() != 1))
                    return false;
                if (c < 'c')
                    st.push(c);
            }
            return st.isEmpty();
        }
    }

    class Solution1003e {
        // 3ms
        public boolean isValid(String s) {
            char[] st = new char[s.length()];
            int i = 0;
            for (char c : s.toCharArray()) {
                st[i++] = c;
                if (i > 2 && st[i - 3] == 'a' && st[i - 2] == 'b' && st[i - 1] == 'c') {
                    i -= 3;
                }
            }
            return i == 0;
        }
    }

    // 1010. Pairs of Songs With Total Durations Divisible by 60 - MEDIUM
    class Solution1010a {
        // 18ms
        public int numPairsDivisibleBy60(int[] time) {
            int ans = 0;
            Map<Integer, Integer> cnt = new HashMap<>();
            for (int t : time) {
                t %= 60;
                ans += cnt.getOrDefault((60 - t) % 60, 0);
                cnt.put(t, cnt.getOrDefault(t, 0) + 1);
            }
            return ans;
        }
    }

    class Solution1010b {
        // 2ms
        public int numPairsDivisibleBy60(int[] time) {
            int ans = 0, cnt[] = new int[60];
            for (int t : time) {
                t %= 60;
                ans += cnt[(60 - t) % 60];
                cnt[t]++;
            }
            return ans;
        }
    }

    class Solution1010c {
        public int numPairsDivisibleBy60(int[] time) {
            int ans = 0, cnt[] = new int[60];
            for (int t : time) {
                cnt[t % 60] += 1;
            }
            ans += combination(cnt[0], 2) + combination(cnt[30], 2);
            int i = 1, j = 59;
            while (i < j) {
                ans += cnt[i++] * cnt[j--];
            }
            return ans;
        }

        // 求组合数
        public long combination(int n, int k) {
            long res = 1;
            for (int i = 1; i <= k; i++) {
                res = res * (n - i + 1) / i;
            }
            return (int) res;
        }
    }

    // 1017. Convert to Base -2 - MEDIUM
    class Solution1017a {
        public String baseNeg2(int n) {
            if (n == 0)
                return String.valueOf(n);
            StringBuilder ans = new StringBuilder();
            while (n != 0) {
                int k = n & 1;
                n -= k;
                n /= -2;
                ans.append(k);
            }
            return ans.reverse().toString();
        }
    }

    // 1019. Next Greater Node In Linked List - MEDIUM
    class Solution1019a {
        // 15ms
        public int[] nextLargerNodes(ListNode head) {
            List<Integer> arr = new ArrayList<>();
            while (head != null) {
                arr.add(head.val);
                head = head.next;
            }
            int[] ans = new int[arr.size()];
            Deque<Integer> st = new ArrayDeque<>();
            for (int i = 0; i < arr.size(); ++i) {
                while (!st.isEmpty() && arr.get(i) > arr.get(st.peek())) {
                    ans[st.poll()] = arr.get(i);
                }
                // st.push(i);
                st.offerFirst(i);
                // st.addFirst(i);
            }
            return ans;
        }
    }

    class Solution1019b {
        // 4ms
        public int[] nextLargerNodes(ListNode head) {
            int n = 0;
            ListNode cur = head;
            while (cur != null) {
                n++;
                cur = cur.next;
            }
            int[] arr = new int[n], ans = new int[n], st = new int[n];
            cur = head;
            for (int i = 0; i < n; ++i) {
                arr[i] = cur.val;
                cur = cur.next;
            }
            int j = -1;
            for (int i = n - 1; i >= 0; --i) {
                while (j != -1 && arr[st[j]] <= arr[i]) {
                    j--;
                }
                ans[i] = j == -1 ? 0 : arr[st[j]];
                st[++j] = i;
            }
            return ans;
        }
    }

    // 1023. Camelcase Matching - MEDIUM
    class Solution1023a {
        private Boolean check(String s, String p) {
            int m = p.length(), j = 0;
            for (int i = 0; i < s.length(); ++i) {
                if (j < m && s.charAt(i) == p.charAt(j)) {
                    ++j;
                } else if (s.charAt(i) < 'a') { // is upper
                    return false;
                }
            }
            return j == p.length();
        }

        public List<Boolean> camelMatch(String[] queries, String pattern) {
            // 0ms
            List<Boolean> ans = new ArrayList<>(pattern.length());
            for (String q : queries) {
                ans.add(check(q, pattern));
            }
            return ans;
            // 2ms
            // return Arrays.stream(queries).map(q -> check(q, pattern)).toList();
        }
    }

    // 1026. Maximum Difference Between Node and Ancestor - MEDIUM
    class Solution1026a {
        // 0ms
        public int maxAncestorDiff(TreeNode root) {
            return dfs(root, root.val, root.val);
        }

        private int dfs(TreeNode root, int mx, int mi) {
            if (root == null) {
                return mx - mi;
            }
            mx = Math.max(root.val, mx);
            mi = Math.min(root.val, mi);
            return Math.max(dfs(root.left, mx, mi), dfs(root.right, mx, mi));
        }
    }

    // 1027. Longest Arithmetic Subsequence - MEDIUM
    class Solution1027a {
        // 24ms
        public int longestArithSeqLength(int[] nums) {
            int n = nums.length, ans = 0, f[][] = new int[n][1001];
            for (int i = 1; i < nums.length; i++) {
                for (int j = i - 1; j >= 0; j--) {
                    int d = nums[i] - nums[j] + 500;
                    if (f[i][d] == 0) {
                        f[i][d] = f[j][d] + 1;
                        ans = Math.max(ans, f[i][d]);
                    }
                }
            }
            return ans + 1;
        }
    }

    class Solution1027b {
        // 35ms
        public int longestArithSeqLength(int[] nums) {
            int n = nums.length, ans = 0, f[][] = new int[n][1001];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < i; j++) {
                    int d = nums[i] - nums[j] + 500;
                    f[i][d] = Math.max(f[i][d], f[j][d] + 1);
                    ans = Math.max(ans, f[i][d]);
                }
            }
            return ans + 1;
        }
    }

    // 1032. Stream of Characters - HARD
    // 倒序建树, 倒序查找, 52ms
    class StreamChecker {
        class Trie {
            Trie[] ch = new Trie[26];
            boolean isEnd = false;
        }

        Trie root = new Trie();
        StringBuilder sb = new StringBuilder();

        void insert(String s) {
            Trie node = root;
            for (int i = s.length() - 1; i >= 0; --i) {
                int p = s.charAt(i) - 'a';
                if (node.ch[p] == null)
                    node.ch[p] = new Trie();
                node = node.ch[p];
            }
            node.isEnd = true;
        }

        public StreamChecker(String[] words) {
            for (String w : words)
                insert(w);
        }

        public boolean query(char letter) {
            sb.append(letter);
            int n = sb.length();
            Trie node = root;
            for (int i = n - 1; i >= Math.max(0, n - 200); --i) {
                int p = sb.charAt(i) - 'a';
                if (node.ch[p] == null) {
                    return false;
                }
                node = node.ch[p];
                if (node.isEnd) {
                    return true;
                }
            }
            return false;
        }

        public boolean query2(char letter) {
            sb.append(letter);
            int n = sb.length();
            Trie node = root;
            for (int i = n - 1; i >= Math.max(0, n - 200) && node != null; --i) {
                int p = sb.charAt(i) - 'a';
                node = node.ch[p];
                if (node != null && node.isEnd) {
                    return true;
                }
            }
            return false;
        }
    }

    // 52ms
    class StreamChecker2 {
        class Trie {
            Trie[] ch;
            boolean isEnd;

            public Trie() {
                this.ch = new Trie[26]; // 注意 constructor 要 new 数组, 否则 default = null
            }
        }

        Trie root;
        StringBuilder sb = new StringBuilder();

        void insert(String word) {
            Trie node = root;
            for (int i = word.length() - 1; i >= 0; --i) {
                char c = word.charAt(i);
                if (node.ch[c - 'a'] == null) {
                    node.ch[c - 'a'] = new Trie();
                }
                node = node.ch[c - 'a'];
            }
            node.isEnd = true;
        }

        public StreamChecker2(String[] words) {
            root = new Trie();
            for (String w : words) {
                insert(w);
            }
        }

        public boolean query(char letter) {
            sb.append(letter);
            Trie node = root;
            for (int i = sb.length() - 1; i >= Math.max(0, sb.length() - 200); --i) {
                char c = sb.charAt(i);
                if (node.ch[c - 'a'] == null) {
                    return false;
                }
                node = node.ch[c - 'a'];
                if (node.isEnd) {
                    return true;
                }
            }
            return false;
        }
    }

    // 49ms
    class StreamChecker3 {
        class TrieNode {
            TrieNode[] ch;
            boolean isEnd = false;

            public TrieNode() {
                ch = new TrieNode[26];
            }
        }

        class Trie {
            TrieNode root;

            public Trie() {
                root = new TrieNode();
            }

            public void insert(CharSequence s) {
                TrieNode node = root;
                for (int i = s.length() - 1; i >= 0; --i) {
                    int p = s.charAt(i) - 'a';
                    if (node.ch[p] == null) {
                        node.ch[p] = new TrieNode();
                    }
                    node = node.ch[p];

                    // As an optimization, this method truncates words, if one ends with the other.
                    // For example, if you add "ball" and "football", only "ball" is kept in the trie.
                    // But it does not seem too obvious
                    // if (node.isEnd) {
                    //     return;
                    // }

                }
                node.isEnd = true;
            }

            public boolean search(char[] ch, int index, int lmt) {
                TrieNode node = root;
                int j = index;
                for (int i = 0; i < lmt; ++i) {
                    if (--j < 0) { // 49ms
                        j += 200;
                    }

                    // j = --j < 0 ? j + 200 : j;     // 49ms
                    // j = ((--j % 200) + 200) % 200; // 54ms, modulo operation is slow

                    node = node.ch[ch[j] - 'a'];
                    if (node == null)
                        return false;
                    if (node.isEnd)
                        return true;
                }
                return false;
            }
        }

        Trie root;
        char[] queries = new char[200]; // rotating array, 200 char limit (for the stream of chars)
        int index = 0, lmt = 0;

        public StreamChecker3(String[] words) {
            root = new Trie();
            for (String w : words)
                root.insert(w);
        }

        public boolean query(char letter) {
            // Wrap around when we hit the end.
            queries[index] = letter;
            index = ++index % 200;
            lmt = lmt < 200 ? ++lmt : lmt;
            return root.search(queries, index, lmt);
        }
    }

    // 1039. Minimum Score Triangulation of Polygon - MEDIUM
    class Solution1039a {
        // 2ms
        private int[] v;
        private int[][] memo;

        public int minScoreTriangulation(int[] values) {
            v = values;
            int n = v.length;
            memo = new int[n][n];
            for (int i = 0; i < n; ++i)
                Arrays.fill(memo[i], -1);
            return dfs(0, n - 1);
        }

        private int dfs(int i, int j) {
            if (i + 1 == j)
                return 0;
            if (memo[i][j] != -1)
                return memo[i][j];
            int res = Integer.MAX_VALUE;
            for (int k = i + 1; k < j; ++k)
                res = Math.min(res, dfs(i, k) + dfs(k, j) + v[i] * v[j] * v[k]);
            return memo[i][j] = res;
        }
    }

    // 1042. Flower Planting With No Adjacent - MEDIUM
    class Solution1042a {
        // 5ms
        public int[] gardenNoAdj(int n, int[][] paths) {
            int[] ans = new int[n];
            int[][] g = new int[n][4];
            // g[x] = [len, a, b, c], at most 3 adjacent points
            for (int[] path : paths) {
                int x = path[0] - 1, y = path[1] - 1;
                g[x][g[x][0]++ + 1] = y;
                g[y][g[y][0]++ + 1] = x;
            }
            for (int i = 0; i < n; i++) {
                int p = 0;
                for (int j = 1; j <= g[i][0]; j++) {
                    p = p | (1 << (ans[g[i][j]] - 1));
                }
                for (int j = 0; j < 4; j++) {
                    if ((p & (1 << j)) == 0) {
                        ans[i] = j + 1;
                        break;
                    }
                }
            }
            return ans;
        }
    }

    // 1053. Previous Permutation With One Swap - MEDIUM
    class Solution1053a {
        // 0ms
        public int[] prevPermOpt1(int[] arr) {
            for (int i = arr.length - 1; i > 0; --i) {
                if (arr[i - 1] > arr[i]) {
                    int t = i;
                    for (int j = i; j < arr.length; ++j) {
                        if (arr[i - 1] > arr[j] && arr[j] > arr[t]) {
                            t = j;
                        }
                    }
                    int tmp = arr[t];
                    arr[t] = arr[i - 1];
                    arr[i - 1] = tmp;
                    break;
                }
            }
            return arr;
        }
    }

    // 1092. Shortest Common Supersequence - HARD
    class Solution1092a {
        // 最长公共子序列
        // f[i][j] 表示字符串 str1 的前 i 个字符和字符串 str2 的前 j 个字符的最长公共子序列的长度
        // 6ms
        public String shortestCommonSupersequence(String str1, String str2) {
            char[] s = str1.toCharArray(), t = str2.toCharArray();
            int m = s.length, n = t.length, f[][] = new int[m + 1][n + 1];
            for (int i = 1; i <= m; ++i) {
                for (int j = 1; j <= n; ++j) {
                    if (s[i - 1] == t[j - 1]) {
                        f[i][j] = f[i - 1][j - 1] + 1;
                    } else {
                        f[i][j] = Math.max(f[i - 1][j], f[i][j - 1]);
                    }
                }
            }
            int i = m, j = n;
            StringBuilder ans = new StringBuilder();
            while (i > 0 || j > 0) {
                if (i == 0) {
                    ans.append(t[--j]);
                } else if (j == 0) {
                    ans.append(s[--i]);
                } else {
                    if (f[i][j] == f[i - 1][j]) {
                        ans.append(s[--i]);
                    } else if (f[i][j] == f[i][j - 1]) {
                        ans.append(t[--j]);
                    } else {
                        --i;
                        --j;
                        ans.append(s[i]);
                    }
                }
            }
            return ans.reverse().toString();
        }
    }

    class Solution1092b {
        // f[i + 1][j + 1] 表示 s 的前 i 个字母和 t 的前 j 个字母的最短公共超序列的长度
        // 6ms
        public String shortestCommonSupersequence(String str1, String str2) {
            char[] s = str1.toCharArray(), t = str2.toCharArray();
            int n = s.length, m = t.length, f[][] = new int[n + 1][m + 1];
            for (int j = 1; j < m; ++j)
                f[0][j] = j;
            for (int i = 1; i < n; ++i)
                f[i][0] = i;
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < m; ++j)
                    if (s[i] == t[j])
                        f[i + 1][j + 1] = f[i][j] + 1;
                    else
                        f[i + 1][j + 1] = Math.min(f[i][j + 1], f[i + 1][j]) + 1;
            int l = f[n][m];
            char[] ans = new char[l];
            for (int i = n - 1, j = m - 1, k = l - 1;;) {
                if (i < 0) {
                    System.arraycopy(t, 0, ans, 0, j + 1);
                    break;
                }
                if (j < 0) {
                    System.arraycopy(s, 0, ans, 0, i + 1);
                    break;
                }
                if (s[i] == t[j]) {
                    ans[k--] = s[i];
                    --i;
                    --j;
                } else if (f[i + 1][j + 1] == f[i][j + 1] + 1)
                    ans[k--] += s[i--];
                else
                    ans[k--] += t[j--];
            }
            return new String(ans);
        }
    }

    class Solution1092c {
        // 6ms
        public String shortestCommonSupersequence(String str1, String str2) {
            char[] s = str1.toCharArray(), t = str2.toCharArray();
            int n = s.length, m = t.length, f[][] = new int[n + 1][m + 1];
            for (int j = 0; j <= m; ++j) {
                f[0][j] = j;
            }
            for (int i = 0; i <= n; ++i) {
                f[i][0] = i;
            }
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < m; ++j) {
                    if (s[i] == t[j]) {
                        f[i + 1][j + 1] = f[i][j] + 1;
                    } else {
                        f[i + 1][j + 1] = Math.min(f[i][j + 1], f[i + 1][j]) + 1;
                    }
                }
            }
            int l = f[n][m], k = l - 1, i = n, j = m;
            char[] ans = new char[l];
            while (k >= 0) {
                char c;
                if (i == 0) {
                    c = t[--j];
                } else if (j == 0) {
                    c = s[--i];
                } else {
                    if (s[i - 1] == t[j - 1]) {
                        c = s[i - 1];
                        --i;
                        --j;
                    } else {
                        if (f[i][j - 1] < f[i - 1][j]) {
                            c = t[--j];
                        } else {
                            c = s[--i];
                        }
                    }
                }
                ans[k--] = c;
            }
            return new String(ans);
        }
    }
}
