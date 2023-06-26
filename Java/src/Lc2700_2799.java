package src;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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

    // 2729. Check if The Number is Fascinating - EASY
    class Solution2729a {
        public boolean isFascinating(int n) {
            String s = "" + n + n * 2 + n * 3;
            return s.length() == 9 && s.chars().distinct().count() == 9 && s.chars().min().getAsInt() > '0';
        }
    }

    // 2730. Find the Longest Semi-Repetitive Substring - MEDIUM
    class Solution2730a {
        public int longestSemiRepetitiveSubstring(String s) {
            int ans = 1;
            for (int i = 1, j = 0, k = 0; i < s.length(); ans = Math.max(ans, ++i - j)) {
                for (k += s.charAt(i) == s.charAt(i - 1) ? 1 : 0; k > 1; j++) {
                    k -= s.charAt(j) == s.charAt(j + 1) ? 1 : 0;
                }
            }
            return ans;
        }
    }

    // 2731. Movement of Robots - MEDIUM
    class Solution2731a {
        public int sumDistance(int[] nums, String s, int d) {
            long after[] = new long[nums.length], ans = 0, k = 0;
            for (int i = 0; i < nums.length; i++) {
                after[i] = nums[i] + (s.charAt(i) == 'R' ? 1L : -1L) * d;
            }
            Arrays.sort(after);
            for (int i = 0; i < nums.length; k += after[i++]) {
                ans = (ans + i * after[i] - k) % 1000000007;
            }
            return (int) ans;
        }
    }

    // 2732. Find a Good Subset of the Matrix - HARD
    class Solution2732a {
        public List<Integer> goodSubsetofBinaryMatrix(int[][] grid) {
            HashMap<Integer, Integer> mp = new HashMap<>();
            for (int i = 0; i < grid.length; i++) {
                mp.put(Integer.parseInt(IntStream.of(grid[i]).mapToObj(t -> "" + t).collect(Collectors.joining()), 2),
                        i);
            }
            if (mp.containsKey(0)) {
                return List.of(mp.get(0));
            }
            for (int i = 1; i < 1 << grid[0].length; i++) {
                for (int j = 1; j < 1 << grid[0].length; j++) {
                    if ((i & j) == 0 && mp.containsKey(i) && mp.containsKey(j)) {
                        return List.of(mp.get(i), mp.get(j)).stream().sorted().toList();
                    }
                }
            }
            return List.of();
        }
    }

    // 2733. Neither Minimum nor Maximum - EASY
    class Solution2733a {
        public int findNonMinOrMax(int[] nums) {
            Arrays.sort(nums);
            return nums.length > 2 ? nums[1] : -1;
        }
    }

    // 2734. Lexicographically Smallest String After Substring Operation - MEDIUM
    class Solution2734a {
        public String smallestString(String s) {
            char[] c = s.toCharArray();
            for (int i = 0; i < s.length(); i++) {
                if (c[i] > 'a') {
                    for (; i < c.length && c[i] > 'a'; i++) {
                        c[i]--;
                    }
                    return new String(c);
                }
            }
            c[c.length - 1] = 'z';
            return new String(c);
        }
    }

    // 2735. Collecting Chocolates - MEDIUM
    class Solution2735a {
        public long minCost(int[] nums, int x) {
            long mi = IntStream.of(nums).mapToLong(v -> v).sum();
            for (long i = 1; i <= nums.length; i++) {
                int[] next = new int[nums.length];
                for (int j = 0; j < next.length; j++) {
                    next[j] = Math.min(nums[j], nums[(j + 1) % nums.length]);
                }
                mi = Math.min(mi, i * x + IntStream.of(nums = next).mapToLong(v -> v).sum());
            }
            return mi;
        }
    }

    // 2736. Maximum Sum Queries - HARD
    class Solution2736a {
        public int[] maximumSumQueries(int[] nums1, int[] nums2, int[][] queries) {
            int[] arr[] = new int[nums1.length][], ans = new int[queries.length];
            for (int i = 0; i < arr.length; i++) {
                arr[i] = new int[] { nums1[i], nums2[i] };
            }
            Arrays.sort(arr, (o, p) -> p[0] - o[0]);
            Integer[] index = new Integer[queries.length];
            for (int i = 0; i < queries.length; i++) {
                index[i] = i;
            }
            Arrays.sort(index, (o, p) -> queries[p][0] - queries[o][0]);
            TreeMap<Integer, Integer> mp = new TreeMap<>(Map.of(0, Integer.MAX_VALUE, Integer.MAX_VALUE, -1));
            for (int i = 0, j = 0; i < queries.length; i++) {
                for (; j < arr.length && arr[j][0] >= queries[index[i]][0]; j++) {
                    if (mp.ceilingEntry(arr[j][1]).getValue() < arr[j][0] + arr[j][1]) {
                        while (mp.floorEntry(arr[j][1]).getValue() <= arr[j][0] + arr[j][1]) {
                            mp.remove(mp.floorKey(arr[j][1]));
                        }
                        mp.put(arr[j][1], arr[j][0] + arr[j][1]);
                    }
                }
                ans[index[i]] = mp.ceilingEntry(queries[index[i]][1]).getValue();
            }
            return ans;
        }
    }

    // 2739. Total Distance Traveled - EASY
    class Solution2739a {
        public int distanceTraveled(int mainTank, int additionalTank) {
            return (Math.min(additionalTank, (mainTank - 1) / 4) + mainTank) * 10;
        }
    }

    // 2740. Find the Value of the Partition - MEDIUM
    class Solution2740a {
        public int findValueOfPartition(int[] nums) {
            Arrays.sort(nums);
            int ans = Integer.MAX_VALUE;
            for (int i = 1; i < nums.length; i++) {
                ans = Math.min(ans, nums[i] - nums[i - 1]);
            }
            return ans;
        }
    }

    // 2741. Special Permutations - MEDIUM
    class Solution2741a {
        public int specialPerm(int[] nums) {
            int dp[][] = new int[1 << nums.length][nums.length], sum = 0;
            for (int i = 0; i < nums.length; i++) {
                dp[1 << i][i] = 1;
            }
            for (int i = 1; i < 1 << nums.length; i++) {
                for (int j = 0; j < nums.length; j++) {
                    for (int k = 0; (i & 1 << j) == 0 && k < nums.length; k++) {
                        if ((i & 1 << k) > 0 && (nums[j] % nums[k] == 0 || nums[k] % nums[j] == 0)) {
                            dp[i | 1 << j][j] = (dp[i | 1 << j][j] + dp[i][k]) % 1000000007;
                        }
                    }
                }
            }
            for (int i = 0; i < nums.length; i++) {
                sum = (sum + dp[(1 << nums.length) - 1][i]) % 1000000007;
            }
            return sum;
        }
    }

    // 2742. Painting the Walls - HARD
    class Solution2742a {
        public int paintWalls(int[] cost, int[] time) {
            int[] dp = new int[cost.length + 1];
            for (int i = 1; i <= cost.length; i++) {
                dp[i] = 1000000000;
            }
            for (int i = 0; i < cost.length; i++) {
                for (int j = cost.length - 1; j >= 0; j--) {
                    dp[j + 1] = Math.min(dp[j + 1], dp[Math.max(0, j - time[i])] + cost[i]);
                }
            }
            return dp[cost.length];
        }
    }

    // 2744. Find Maximum Number of String Pairs - EASY
    class Solution2744a {
        public int maximumNumberOfStringPairs(String[] words) {
            int ans = 0;
            for (int i = 0; i < words.length; i++) {
                for (int j = i + 1; j < words.length; j++) {
                    ans += words[i].equals("" + new StringBuilder(words[j]).reverse()) ? 1 : 0;
                }
            }
            return ans;
        }
    }

    // 2745. Construct the Longest New String - MEDIUM
    class Solution2745a {
        public int longestString(int x, int y, int z) {
            return Math.min(x, y) * 4 + (x == y ? 0 : 2) + z * 2;
        }
    }

    // 2746. Decremental String Concatenation - MEDIUM
    class Solution2746a {
        // 32ms
        public int minimizeConcatenatedLength(String[] words) {
            return words[0].length() + dfs(1, words[0].charAt(0) - 'a',
                    words[0].charAt(words[0].length() - 1) - 'a', words, new int[words.length][26][26]);
        }

        // index, start, end, words, f
        private int dfs(int i, int s, int e, String[] w, int[][][] f) {
            if (i == w.length) {
                return 0;
            }
            if (f[i][s][e] == 0) {
                f[i][s][e] = Math.min(
                        dfs(i + 1, s, w[i].charAt(w[i].length() - 1) - 'a', w, f) - (w[i].charAt(0) - 'a' == e ? 1 : 0),
                        dfs(i + 1, w[i].charAt(0) - 'a', e, w, f) - (s == w[i].charAt(w[i].length() - 1) - 'a' ? 1 : 0))
                        + w[i].length();
            }
            return f[i][s][e];
        }
    }

    // 2747. Count Zero Request Servers - HARD
    class Solution2747a {
        public int[] countServers(int n, int[][] logs, int x, int[] queries) {
            Arrays.sort(logs, (o, p) -> o[1] - p[1]);
            Integer[] index = new Integer[queries.length];
            for (int i = 0; i < queries.length; i++) {
                index[i] = i;
            }
            Arrays.sort(index, (o, p) -> queries[o] - queries[p]);
            int result[] = new int[queries.length], mp[] = new int[n + 1], count = n;
            for (int i = 0, j = 0, k = 0; i < queries.length; i++) {
                for (; k < logs.length && logs[k][1] <= queries[index[i]]; k++) {
                    count -= mp[logs[k][0]]++ == 0 ? 1 : 0;
                }
                for (; j < logs.length && logs[j][1] < queries[index[i]] - x; j++) {
                    count += --mp[logs[j][0]] == 0 ? 1 : 0;
                }
                result[index[i]] = count;
            }
            return result;
        }
    }

}
