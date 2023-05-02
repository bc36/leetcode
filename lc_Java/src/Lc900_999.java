package src;

import java.util.*;

public class Lc900_999 {
    // 970. Powerful Integers - MEDIUM
    class Solution970a {
        // 1ms
        public List<Integer> powerfulIntegers(int x, int y, int bound) {
            Set<Integer> ans = new HashSet<>();
            int ix = 1;
            while (ix <= bound) {
                int jy = 1;
                while (ix + jy <= bound) {
                    ans.add(ix + jy);
                    jy *= y;
                    if (y == 1)
                        break;
                }
                if (x == 1)
                    break;
                ix *= x;
            }
            return List.copyOf(ans);
            // return new ArrayList<>(ans);
        }
    }

    class Solution970b {
        public List<Integer> powerfulIntegers(int x, int y, int bound) {
            Set<Integer> set = new HashSet<>();
            int m = x == 1 ? 0 : (int) (Math.log10(bound) / Math.log10(x));
            int n = y == 1 ? 0 : (int) (Math.log10(bound) / Math.log10(y));
            for (int i = 0; i <= m; i++) {
                for (int j = 0; j <= n; j++) {
                    int cur = (int) Math.pow(x, i) + (int) Math.pow(y, j);
                    if (cur <= bound)
                        set.add(cur);
                }
            }
            return new ArrayList<>(set);
        }
    }

    // 989. Add to Array-Form of Integer - EASY
    class Solution989a {
        // 3ms
        public List<Integer> addToArrayForm(int[] num, int k) {
            List<Integer> ans = new ArrayList<>();
            for (int i = num.length - 1, carry = 0; i >= 0 || k > 0 || carry > 0; --i) {
                int tmp = carry;
                if (k > 0) {
                    tmp += k % 10;
                    k /= 10;
                }
                if (i >= 0) {
                    tmp += num[i];
                }
                ans.add(tmp % 10);
                carry = tmp / 10;
            }
            Collections.reverse(ans);
            return ans;
        }
    }

    class Solution989b {
        // 2ms
        public List<Integer> addToArrayForm(int[] num, int k) {
            LinkedList<Integer> ans = new LinkedList<>();
            for (int i = num.length - 1, carry = 0; i >= 0 || k > 0 || carry > 0; --i) {
                int tmp = carry;
                if (k > 0) {
                    tmp += k % 10;
                    k /= 10;
                }
                if (i >= 0) {
                    tmp += num[i];
                }
                ans.addFirst(tmp % 10);
                carry = tmp / 10;
            }
            return ans;
        }
    }

    class Solution989c {
        // 2ms
        public List<Integer> addToArrayForm(int[] num, int k) {
            LinkedList<Integer> ans = new LinkedList<>();
            // List<Integer> ans = new ArrayList<>();
            for (int i = num.length - 1; i >= 0; --i) {
                int tmp = num[i] + k % 10;
                k /= 10;
                if (tmp >= 10) {
                    ++k;
                    tmp -= 10;
                }
                ans.addFirst(tmp);
                // ans.add(tmp);
            }
            while (k > 0) {
                ans.addFirst(k % 10);
                // ans.add(k % 10);
                k /= 10;
            }
            // Collections.reverse(ans); // 3ms
            return ans;
        }
    }

    // 990. Satisfiability of Equality Equations - MEDIUM
    class Solution990a {
        public boolean equationsPossible(String[] equations) {
            int[] p = new int[26];
            for (int i = 0; i < 26; i++) {
                p[i] = i;
            }
            for (String e : equations) {
                if (e.charAt(1) == '=') {
                    int x = e.charAt(0) - 'a';
                    int y = e.charAt(3) - 'a';
                    p[find(p, y)] = find(p, x);
                }
            }
            for (String e : equations) {
                if (e.charAt(1) == '!') {
                    int x = e.charAt(0) - 'a';
                    int y = e.charAt(3) - 'a';
                    if (find(p, x) == find(p, y)) {
                        return false;
                    }
                }
            }
            return true;
        }

        public int find(int[] p, int x) {
            // 1ms
            // if (p[x] != x) {
            //     p[x] = find(p, p[x]);
            // }
            // return p[x];

            // 0ms
            while (p[x] != x) {
                p[x] = p[p[x]];
                x = p[x];
            }
            return x;
        }
    }
}
