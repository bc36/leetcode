package src;

import java.util.*;

public class Lc1000_1099 {
    // 1000. Minimum Cost to Merge Stones - HARD
    class Solution1000a {
        private int k, pre[], memo[][][];

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
