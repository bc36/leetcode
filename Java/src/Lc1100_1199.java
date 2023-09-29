package src;

import java.util.*;
// import javafx.util.Pair;

public class Lc1100_1199 {
    // 1123. Lowest Common Ancestor of Deepest Leaves - MEDIUM
    class Solution1123 {
        // public TreeNode lcaDeepestLeaves(TreeNode root) {
        //     return dfs(root).getValue();
        // }

        // private Pair<Integer, TreeNode> dfs(TreeNode node) {
        //     if (node == null)
        //         return new Pair<>(0, null);
        //     var left = dfs(node.left);
        //     var right = dfs(node.right);
        //     if (left.getKey() > right.getKey())
        //         return new Pair<>(left.getKey() + 1, left.getValue());
        //     if (left.getKey() < right.getKey())
        //         return new Pair<>(right.getKey() + 1, right.getValue());
        //     return new Pair<>(left.getKey() + 1, node);
        // }
    }

    // 1130. Minimum Cost Tree From Leaf Values - MEDIUM
    class Solution1130a {
        public int mctFromLeafValues(int[] arr) {
            // 1. 如果栈顶元素比当前元素小, 弹出栈顶元素, 
            //    栈顶元素与当前元素和栈顶下一个元素中得最小值组合
            // 2. 如果栈顶元素比当前元素大, 入栈
            Stack<Integer> st = new Stack<>();
            st.push(Integer.MAX_VALUE); // 哨兵
            int ans = 0;
            for (int i = 0; i < arr.length; i++) {
                while (st.peek() < arr[i])
                    ans += st.pop() * Math.min(arr[i], st.peek());
                st.push(arr[i]);
            }

            while (st.size() > 2)
                ans += st.pop() * st.peek();

            return ans;
        }
    }

    // 1147. Longest Chunked Palindrome Decomposition - HARD
    class Solution1147 { // 1ms
        private long[] h;
        private long[] p;

        private long get(int i, int j) {
            return h[j] - h[i - 1] * p[j - i + 1];
        }

        public int longestDecomposition(String text) {
            int n = text.length();
            int base = 131;
            h = new long[n + 10];
            p = new long[n + 10];
            p[0] = 1;
            for (int i = 0; i < n; ++i) {
                int t = text.charAt(i) - 'a' + 1;
                h[i + 1] = h[i] * base + t;
                p[i + 1] = p[i] * base;
            }
            int ans = 0;
            for (int i = 0, j = n - 1; i <= j;) {
                boolean ok = false;
                for (int k = 1; i + k - 1 < j - k + 1; ++k) {
                    if (get(i + 1, i + k) == get(j - k + 2, j + 1)) {
                        ans += 2;
                        i += k;
                        j -= k;
                        ok = true;
                        break;
                    }
                }
                if (!ok) {
                    ++ans;
                    break;
                }
            }
            return ans;
        }
    }

    class Solution1147b { // 0ms
        public int longestDecomposition(String text) {
            int ans = 0;
            for (int i = 0, j = text.length() - 1; i <= j;) {
                boolean ok = false;
                for (int k = 1; i + k - 1 < j - k + 1; ++k) {
                    if (check(text, i, j - k + 1, k)) {
                        ans += 2;
                        i += k;
                        j -= k;
                        ok = true;
                        break;
                    }
                }
                if (!ok) {
                    ++ans;
                    break;
                }
            }
            return ans;
        }

        private boolean check(String s, int i, int j, int k) {
            while (k-- > 0) {
                if (s.charAt(i++) != s.charAt(j++)) {
                    return false;
                }
            }
            return true;
        }
    }

    class Solution1147c { // 0ms
        public int longestDecomposition(String text) {
            int ans = 0, n = text.length(), i = 0, j = n - 1, end = n;
            while (j - i >= end - j) {
                if (check(text, i, j, end)) {
                    i += end - j;
                    end = j;
                    ans += 2;
                }
                j--;
            }
            if (i <= j)
                ans++;
            return ans;
        }

        private boolean check(String text, int i, int j, int end) {
            while (j < end) {
                if (text.charAt(i++) != text.charAt(j++)) {
                    return false;
                }
            }
            return true;
        }
    }

    // 1156. Swap For Longest Repeated Character Substring - MEDIUM
    class Solution1156a {
        public int maxRepOpt1(String text) {
            int[] cnt = new int[26];
            int n = text.length();
            for (int i = 0; i < n; ++i) {
                ++cnt[text.charAt(i) - 'a'];
            }
            int ans = 0, i = 0;
            while (i < n) {
                int j = i;
                while (j < n && text.charAt(j) == text.charAt(i)) {
                    ++j;
                }
                int l = j - i;
                int k = j + 1;
                while (k < n && text.charAt(k) == text.charAt(i)) {
                    ++k;
                }
                int r = k - j - 1;
                ans = Math.max(ans, Math.min(l + r + 1, cnt[text.charAt(i) - 'a']));
                i = j;
            }
            return ans;
        }
    }

    // 1171. Remove Zero Sum Consecutive Nodes from Linked List - MEDIUM
    class Solution1171a { // 2ms
        public ListNode removeZeroSumSublists(ListNode head) {
            ListNode dummy = new ListNode(0, head);
            int pre = 0;
            Map<Integer, ListNode> vis = new HashMap<>();
            for (ListNode node = dummy; node != null; node = node.next) {
                pre += node.val;
                vis.put(pre, node);
            }
            pre = 0;
            for (ListNode node = dummy; node != null; node = node.next) {
                pre += node.val;
                node.next = vis.get(pre).next;
            }
            return dummy.next;
        }
    }

    class Solution { // 1ms
        public ListNode removeZeroSumSublists(ListNode head) {
            if (head == null)
                return null;
            int sum = head.val;
            if (sum == 0)
                return removeZeroSumSublists(head.next);
            var hn = head.next;
            for (var cur = hn; cur != null; cur = cur.next)
                if ((sum += cur.val) == 0)
                    return removeZeroSumSublists(cur.next);
            head.next = removeZeroSumSublists(hn);
            return head;
        }
    }
}
