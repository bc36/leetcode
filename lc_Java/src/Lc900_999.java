package src;

import java.util.*;

public class Lc900_999 {
    // 989. Add to Array-Form of Integer - E
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

    // 2ms
    public List<Integer> addToArrayForm3(int[] num, int k) {
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

    // 2ms
    public List<Integer> addToArrayForm2(int[] num, int k) {
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

    // 990. Satisfiability of Equality Equations - M
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
