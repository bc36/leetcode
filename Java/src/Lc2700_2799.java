package src;

import java.math.BigInteger;
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

    // 2748. Number of Beautiful Pairs - EASY
    class Solution2748a {
        public int countBeautifulPairs(int[] nums) {
            int count = 0;
            for (int i = 0; i < nums.length; i++) {
                for (int j = i + 1; j < nums.length; j++) {
                    count += new BigInteger(("" + nums[i]).substring(0, 1)).gcd(BigInteger.valueOf(nums[j] % 10))
                            .equals(BigInteger.ONE) ? 1 : 0;
                }
            }
            return count;
        }
    }

    // 2749. Minimum Operations to Make the Integer Zero - MEDIUM
    class Solution2749a {
        public int makeTheIntegerZero(int num1, long num2) {
            for (int i = 1; i < 40; i++) {
                if (num1 - i * num2 >= i && Long.bitCount(num1 - i * num2) <= i) {
                    return i;
                }
            }
            return -1;
        }
    }

    class Solution2749b {
        public int makeTheIntegerZero(int num1, int num2) {
            for (long k = 1; k <= num1 - num2 * k; k++)
                if (k >= Long.bitCount(num1 - num2 * k))
                    return (int) k;
            return -1;
        }
    }

    // 2750. Ways to Split Array Into Good Subarrays - MEDIUM
    class Solution2750a {
        public int numberOfGoodSubarraySplits(int[] nums) {
            ArrayList<Integer> list = new ArrayList<>();
            for (int i = 0; i < nums.length; i++) {
                if (nums[i] == 1) {
                    list.add(i);
                }
            }
            long prod = 1;
            for (int i = 1; i < list.size(); i++) {
                prod = (prod * (list.get(i) - list.get(i - 1))) % 1000000007;
            }
            return list.isEmpty() ? 0 : (int) prod;
        }
    }

    // 2751. Robot Collisions - HARD
    class Solution2751a {
        public List<Integer> survivedRobotsHealths(int[] positions, int[] healths, String directions) {
            Integer[] index = new Integer[positions.length];
            for (int i = 0; i < positions.length; i++) {
                index[i] = i;
            }
            Arrays.sort(index, (o, p) -> positions[o] - positions[p]);
            ArrayDeque<Integer> deque = new ArrayDeque<>();
            for (int i = 0; i < positions.length; i++) {
                if (directions.charAt(index[i]) == 'L') {
                    while (!deque.isEmpty() && healths[index[i]] > 0) {
                        if (healths[deque.peek()] > healths[index[i]]) {
                            healths[index[i]] = 0;
                            healths[deque.peek()]--;
                        } else if (healths[deque.peek()] < healths[index[i]]) {
                            healths[index[i]]--;
                            healths[deque.pop()] = 0;
                        } else {
                            healths[index[i]] = healths[deque.pop()] = 0;
                        }
                    }
                } else {
                    deque.push(index[i]);
                }
            }
            ArrayList<Integer> list = new ArrayList<>();
            for (int health : healths) {
                if (health > 0) {
                    list.add(health);
                }
            }
            return list;
        }
    }

    // 2760. Longest Even Odd Subarray With Threshold - EASY
    class Solution2760a {
        public int longestAlternatingSubarray(int[] nums, int threshold) {
            int ans = 0;
            for (int i = 0; i < nums.length; i++) {
                for (int j = i; nums[i] % 2 == 0 && j < nums.length && nums[j] <= threshold
                        && (j == i || nums[j] % 2 != nums[j - 1] % 2); j++) {
                    ans = Math.max(ans, j - i + 1);
                }
            }
            return ans;
        }
    }

    // 2761. Prime Pairs With Target Sum - MEDIUM
    class Solution2761a {
        // 331ms
        private static HashSet<Integer> primes = new HashSet<>() {
            {
                boolean[] flag = new boolean[1000000];
                for (int i = 2; i < flag.length; i++) {
                    if (!flag[i]) {
                        add(i);
                        for (int j = i; j < flag.length; j += i) {
                            flag[j] = true;
                        }
                    }
                }
            }
        };

        public List<List<Integer>> findPrimePairs(int n) {
            ArrayList<List<Integer>> list = new ArrayList<>();
            for (int i = 2; i <= n / 2; i++) { // 可以只枚举质数加速
                if (primes.contains(i) && primes.contains(n - i)) {
                    list.add(List.of(i, n - i));
                }
            }
            return list;
        }
    }

    class Solution2761b {
        // 30ms
        private final static int MX = (int) 1e6;
        private final static int[] primes = new int[78498];
        private final static boolean[] np = new boolean[MX + 1];

        static {
            var pi = 0;
            for (var i = 2; i <= MX; ++i) {
                if (!np[i]) {
                    primes[pi++] = i;
                    for (var j = i; j <= MX / i; ++j) // 避免溢出的写法
                        np[i * j] = true;
                }
            }
        }

        public List<List<Integer>> findPrimePairs(int n) {
            if (n % 2 > 0)
                return n > 4 && !np[n - 2] ? List.of(List.of(2, n - 2)) : List.of();
            var ans = new ArrayList<List<Integer>>();
            for (int x : primes) {
                int y = n - x;
                if (y < x)
                    break;
                if (!np[y])
                    ans.add(List.of(x, y));
            }
            return ans;
        }
    }

    class Solution2761c {
        // 11ms
        private static int primeLen = 0;
        private static final int N = 1000001;
        private static final boolean[] PRIMES = new boolean[N];
        private static final int[] NP = new int[78498];

        static {
            boolean[] isVisited = new boolean[N];
            for (int i = 2; i < N; i++) {
                if (isVisited[i])
                    continue;
                PRIMES[i] = true;
                NP[primeLen++] = i;
                if (i > 1000)
                    continue;
                for (int j = i * i; j < N; j += i) {
                    isVisited[j] = true;
                }
            }
        }

        public List<List<Integer>> findPrimePairs(int n) {
            List<List<Integer>> ans = new ArrayList<>();
            if ((n & 1) == 1) {
                if (n > 1 && PRIMES[n - NP[0]])
                    ans.add(List.of(2, n - 2));
                return ans;
            }
            int half = n >> 1;
            for (int i = 0; i < primeLen && NP[i] <= half; i++) {
                if (PRIMES[n - NP[i]])
                    ans.add(List.of(NP[i], n - NP[i]));
            }
            return ans;
        }
    }

    // 2762. Continuous Subarrays - MEDIUM
    class Solution2762a {
        public long continuousSubarrays(int[] nums) {
            long count = 0;
            TreeMap<Integer, Integer> map = new TreeMap<>();
            for (int i = 0, j = 0; j < nums.length; j++) {
                map.put(nums[j], map.getOrDefault(nums[j], 0) + 1);
                for (; map.lastKey() - map.firstKey() > 2; i++) {
                    map.put(nums[i], map.get(nums[i]) - 1);
                    if (map.get(nums[i]) == 0) {
                        map.remove(nums[i]);
                    }
                }
                count += j - i + 1;
            }
            return count;
        }
    }

    // 2763. Sum of Imbalance Numbers of All Subarrays - HARD
    class Solution2763a {
        public int sumImbalanceNumbers(int[] nums) {
            int sum = 0;
            for (int i = 0; i < nums.length; i++) {
                HashSet<Integer> set = new HashSet<>(Set.of(nums[i]));
                for (int j = i + 1, count = 0; j < nums.length; j++) {
                    sum += count += set.add(nums[j]) ? set.contains(nums[j] - 1) && set.contains(nums[j] + 1) ? -1
                            : !set.contains(nums[j] - 1) && !set.contains(nums[j] + 1) ? 1 : 0 : 0;
                }
            }
            return sum;
        }
    }

    // 2769. Find the Maximum Achievable Number - EASY
    class Solution2769a {
        public int theMaximumAchievableX(int num, int t) {
            return num + t * 2;
        }
    }

    // 2770. Maximum Number of Jumps to Reach the Last Index - MEDIUM
    class Solution2770a {
        public int maximumJumps(int[] nums, int target) {
            int[] f = new int[nums.length];
            for (int i = 1; i < nums.length; i++) {
                f[i] = -1;
                for (int j = 0; j < i; j++) {
                    f[i] = Math.max(f[i], f[j] < 0 || Math.abs(nums[i] - nums[j]) > target ? -1 : f[j] + 1);
                }
            }
            return f[nums.length - 1];
        }
    }

    // 2771. Longest Non-decreasing Subarray From Two Arrays - MEDIUM
    class Solution2771a {
        // 4ms
        public int maxNonDecreasingLength(int[] nums1, int[] nums2) {
            int f1[] = new int[nums1.length], f2[] = new int[nums1.length], mx = 1;
            f1[0] = f2[0] = 1;
            for (int i = 1; i < nums1.length; i++) {
                mx = Math.max(mx,
                        Math.max(
                                f1[i] = Math.max(nums1[i] < nums1[i - 1] ? 1 : f1[i - 1] + 1,
                                        nums1[i] < nums2[i - 1] ? 1 : f2[i - 1] + 1),
                                f2[i] = Math.max(nums2[i] < nums1[i - 1] ? 1 : f1[i - 1] + 1,
                                        nums2[i] < nums2[i - 1] ? 1 : f2[i - 1] + 1)));
            }
            return mx;
        }
    }

    // 2772. Apply Operations to Make All Array Elements Equal to Zero - MEDIUM
    class Solution2772a {
        // 2ms
        public boolean checkArray(int[] nums, int k) {
            int curr = 0, count[] = new int[nums.length + k];
            for (int i = 0; i < nums.length; i++) {
                if (nums[i] < (curr -= count[i]) || nums[i] > curr && i + k > nums.length) {
                    return false;
                }
                // a += b = c
                // 1. b = c
                // 2. a += b (which is the same as c)
                curr += count[i + k] = nums[i] - curr;
            }
            return true;
        }
    }

    // 2778. Sum of Squares of Special Elements - EASY
    class Solution2778a {
        public int sumOfSquares(int[] nums) {
            int ans = 0;
            for (int i = 0; i < nums.length; i++) {
                ans += nums.length % (i + 1) > 0 ? 0 : nums[i] * nums[i];
            }
            return ans;
        }
    }

    // 2779. Maximum Beauty of an Array After Applying Operation - MEDIUM
    class Solution2779a {
        public int maximumBeauty(int[] nums, int k) {
            Arrays.sort(nums);
            int ans = 0;
            for (int i = 0, j = 0; i < nums.length; i++) {
                for (; j < nums.length && nums[j] - nums[i] <= 2 * k; j++) {
                }
                ans = Math.max(ans, j - i);
            }
            return ans;
        }
    }

    // 2780. Minimum Index of a Valid Split - MEDIUM
    class Solution2780a {
        public int minimumIndex(List<Integer> nums) {
            HashMap<Integer, Integer> mp = new HashMap<>();
            for (int num : nums) {
                mp.put(num, mp.getOrDefault(num, 0) + 1);
            }
            int mx = nums.get(0), count = 0;
            for (Map.Entry<Integer, Integer> e : mp.entrySet()) {
                if (e.getValue() > mp.get(mx)) {
                    mx = e.getKey();
                }
            }
            for (int i = 0; i < nums.size() - 1; i++) {
                count += nums.get(i) == mx ? 1 : 0;
                if (count * 2 > i + 1 && (mp.get(mx) - count) * 2 >= nums.size() - i) {
                    return i;
                }
            }
            return -1;
        }
    }

    // 2781. Length of the Longest Valid Substring - HARD
    class Solution2781a {
        public int longestValidSubstring(String word, List<String> forbidden) {
            int ans = 0;
            HashSet<String> s = new HashSet<>(forbidden);
            for (int i = 0, j = 0, k; i < word.length(); ans = Math.max(ans, j - i++)) {
                for (; j < word.length(); j++) {
                    for (k = Math.max(i, j - 9); k <= j && !s.contains(word.substring(k, j + 1)); k++) {
                    }
                    if (k <= j) {
                        break;
                    }
                }
            }
            return ans;
        }
    }

    // 2798. Number of Employees Who Met the Target - EAST
    class Solution2798a {
        public int numberOfEmployeesWhoMetTarget(int[] hours, int target) {
            return (int) IntStream.of(hours).filter(v -> v >= target).count();
        }
    }

    // 2799. Count Complete Subarrays in an Array - MEDIUM
    class Solution2799a {
        public int countCompleteSubarrays(int[] nums) {
            HashSet<Integer> s = new HashSet<>();
            for (int num : nums) {
                s.add(num);
            }
            int ans = 0;
            for (int i = 0; i < nums.length; i++) {
                HashSet<Integer> curr = new HashSet<>();
                for (int j = i; j < nums.length; j++) {
                    curr.add(nums[j]);
                    ans += curr.size() / s.size(); // wow!
                }
            }
            return ans;
        }
    }
}
