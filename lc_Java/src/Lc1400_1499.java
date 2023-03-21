package src;

import java.util.*;

public class Lc1400_1499 {
    // 1414. Find the Minimum Number of Fibonacci Numbers Whose Sum Is K - M
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

    public int findMinFibonacciNumbers2(int k) {
        if (k == 0) {
            return 0;
        }
        int f1 = 1, f2 = 1;
        while (f1 + f2 <= k) {
            f2 = f1 + f2;
            f1 = f2 - f1;
        }
        return findMinFibonacciNumbers2(k - f2) + 1;
    }

    // 1487. Making File Names Unique - M
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
