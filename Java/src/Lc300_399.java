package src;

import java.util.*;

public class Lc300_399 {
    // 344. Reverse String - EASY
    class Solution344a {
        public void reverseString(char[] s) {
            for (int i = 0; i < s.length / 2; i++) {
                char tmp = s[i];
                s[i] = s[s.length - 1 - i];
                s[s.length - 1 - i] = tmp;
            }
            return;
        }
    }
}
