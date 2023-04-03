package src;

import java.util.*;

public class Lc800_899 {
    // 831. Masking Personal Information - MEDIUM
    class Solution831a {
        public String maskPII(String s) {
            int at = s.indexOf("@");
            if (at > 0) {
                s = s.toLowerCase();
                return (s.charAt(0) + "*****" + s.substring(at - 1)).toLowerCase();
            }
            s = s.replaceAll("[^0-9]", "");
            return new String[] { "", "+*-", "+**-", "+***-" }[s.length() - 10] + "***-***-"
                    + s.substring(s.length() - 4);
        }
    }

    // 884. Uncommon Words from Two Sentences - EASY
    class Solution884a {
        public String[] uncommonFromSentences(String s1, String s2) {
            HashMap<String, Integer> m = new HashMap<String, Integer>();
            String[] arr1 = s1.split(" ");
            for (String w : arr1) {
                m.put(w, m.getOrDefault(w, 0) + 1);
            }
            String[] arr2 = s2.split(" ");
            for (String w : arr2) {
                m.put(w, m.getOrDefault(w, 0) + 1);
            }
            ArrayList<String> ans = new ArrayList<String>();
            for (Map.Entry<String, Integer> e : m.entrySet()) {
                if (e.getValue() == 1) {
                    ans.add(e.getKey());
                }
            }
            return ans.toArray(new String[0]);
        }
    }
}
