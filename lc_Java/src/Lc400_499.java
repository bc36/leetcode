package src;

import java.util.*;

public class Lc400_499 {
    // 409. Longest Palindrome - E
    class Solution409a {
        public int longestPalindrome(String s) {
            Map<Character, Integer> cnt = new HashMap<Character, Integer>();
            for (int i = 0; i < s.length(); i++) {
                cnt.put(s.charAt(i), cnt.getOrDefault(s.charAt(i), 0) + 1);
            }
            int ans = 0;
            boolean odd = false;
            for (Map.Entry<Character, Integer> e : cnt.entrySet()) {
                if ((e.getValue() & 1) == 1) {
                    odd = true;
                }
                ans += e.getValue() / 2 * 2;
            }
            return odd ? ans + 1 : ans;
        }
    }

    class Solution409b {
        public int longestPalindrome(String s) {
            int[] arr = new int[128];
            for (char c : s.toCharArray()) {
                arr[c]++;
            }
            int odd = 0;
            for (int i : arr) {
                odd += (i % 2);
            }
            return odd == 0 ? s.length() : (s.length() - odd + 1);
        }
    }

    // 415. Add Strings - E
    public String addStrings(String num1, String num2) {
        return "";
    }

    // 438. Find All Anagrams in a String - M
    class Solution438a {
        // O(C * n + m), O(C)
        public List<Integer> findAnagrams(String s, String p) {
            int sl = s.length();
            int pl = p.length();
            if (sl < pl)
                return new ArrayList<Integer>();
            List<Integer> ans = new ArrayList<Integer>();
            int[] ss = new int[26];
            int[] pp = new int[26];
            for (int i = 0; i < pl; i++) {
                ss[s.charAt(i) - 'a']++;
                pp[p.charAt(i) - 'a']++;
            }
            if (Arrays.equals(ss, pp)) {
                ans.add(0);
            }
            for (int i = pl; i < sl; i++) {
                ss[s.charAt(i) - 'a']++;
                ss[s.charAt(i - pl) - 'a']--;
                if (Arrays.equals(ss, pp)) {
                    ans.add(i - pl + 1);
                }
            }
            return ans;
        }
    }

    class Solution438b {
        // O(m + C + n) / O(C)
        public List<Integer> findAnagrams(String s, String p) {
            int sl = s.length();
            int pl = p.length();
            if (sl < pl)
                return new ArrayList<Integer>();
            List<Integer> ans = new ArrayList<Integer>();
            int[] cnt = new int[26];
            for (int i = 0; i < pl; i++) {
                cnt[s.charAt(i) - 'a']++;
                cnt[p.charAt(i) - 'a']--;
            }
            int diff = 0;
            for (int i = 0; i < 26; i++) {
                if (cnt[i] != 0) {
                    diff++;
                }
            }
            if (diff == 0) {
                ans.add(0);
            }
            for (int i = 0; i < sl - pl; i++) {
                if (--cnt[s.charAt(i) - 'a'] == 0) {
                    diff--;
                } else if (cnt[s.charAt(i) - 'a'] == -1) {
                    diff++;
                }
                if (++cnt[s.charAt(i + pl) - 'a'] == 1) {
                    diff++;
                } else if (cnt[s.charAt(i + pl) - 'a'] == 0) {
                    diff--;
                }
                if (diff == 0) {
                    ans.add(i + 1);
                }
            }
            return ans;
        }
    }
}
