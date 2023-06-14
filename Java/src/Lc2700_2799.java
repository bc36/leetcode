package src;

import java.util.*;

public class Lc2700_2799 {
    // 2706. Buy Two Chocolates - EASY
    class Solution2706a {
        public int buyChoco(int[] prices, int money) {
            Arrays.sort(prices);
            return prices[0] + prices[1] > money ? money : money - prices[0] - prices[1];
        }
    }

    // 2707. Extra Characters in a String - MEDIUM
    class Solution2707a {
        public int minExtraChar(String s, String[] dictionary) {
            Set<String> set = Set.of(dictionary);
            int[] dp = new int[s.length() + 1];
            for (int i = 0; i < s.length(); i++) {
                dp[i + 1] = Integer.MAX_VALUE;
                for (int j = 0; j <= i; j++) {
                    dp[i + 1] = Math.min(dp[i + 1], dp[j] + (set.contains(s.substring(j, i + 1)) ? 0 : i - j + 1));
                }
            }
            return dp[s.length()];
        }
    }

    // 2708. Maximum Strength of a Group - MEDIUM
    class Solution2708a {
        public long maxStrength(int[] nums) {
            int n = nums.length;
            Arrays.sort(nums);
            long prod = 1;
            for (int i = 0; i < n; i++) {
                prod *= Math.max(1, nums[i]) * (i % 2 > 0 && nums[i] < 0 ? nums[i] * nums[i - 1] : 1);
            }
            return n > 1 ? (nums[n - 1] == 0 && nums[1] == 0 ? 0 : prod) : nums[0];
        }
    }

    // 2709. Greatest Common Divisor Traversal - HARD
    class Solution2709a {
        public boolean canTraverseAllPairs(int[] nums) {
            if (nums.length == 1) {
                return true;
            }
            HashMap<Integer, HashSet<Integer>> map = new HashMap<>(), visited = new HashMap<>();
            for (int num : nums) {
                if (num == 1) {
                    return false;
                }
                ArrayList<Integer> list = new ArrayList<>();
                for (int i = 2; i * i <= num; i++) {
                    if (num % i == 0) {
                        for (list.add(i); num % i == 0; num /= i) {
                        }
                    }
                }
                if (num > 1) {
                    list.add(num);
                }
                map.putIfAbsent(list.get(0), new HashSet<>());
                for (int i = 1; i < list.size(); i++) {
                    map.computeIfAbsent(list.get(i - 1), t -> new HashSet<>()).add(list.get(i));
                    map.computeIfAbsent(list.get(i), t -> new HashSet<>()).add(list.get(i - 1));
                }
            }
            canTraverseAllPairs(map.keySet().iterator().next(), map, visited);
            return map.size() == visited.size();
        }

        private void canTraverseAllPairs(int i, HashMap<Integer, HashSet<Integer>> map,
                HashMap<Integer, HashSet<Integer>> visited) {
            if (visited.put(i, new HashSet<>()) == null) {
                for (int j : map.get(i)) {
                    canTraverseAllPairs(j, map, visited);
                }
            }
        }
    }
}
