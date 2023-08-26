package src;

import java.math.BigInteger;
import java.util.*;

public class Lc2800_2899 {
    // 2800. Shortest String That Contains Three Strings - MEDIUM
    class Solution2800a {
        public String minimumString(String a, String b, String c) {
            return minimumString("", 0, 0, List.of(a, b, c));
        }

        private String minimumString(String curr, int index, int mask, List<String> list) {
            if (index == 3) {
                return curr;
            }
            String s = null;
            for (int i = 0; i < 3; i++) {
                for (int j = list.get(i).length(); (mask & 1 << i) == 0 && j >= 0; j--) {
                    if (curr.endsWith(list.get(i).substring(0, j)) || curr.contains(list.get(i))) {
                        String n = minimumString(curr + list.get(i).substring(j), index + 1, mask | 1 << i, list);
                        s = s == null || s.length() > n.length() || s.length() == n.length() && s.compareTo(n) > 0 ? n
                                : s;
                        break;
                    }
                }
            }
            return s;
        }
    }

    // 2801. Count Stepping Numbers in Range - HARD
    class Solution2801a {
        private static ArrayList<Long> list = new ArrayList<>() {
            {
                addAll(List.of(0L, 9L, 17L, 32L, 61L, 116L));
                for (int i = 6; i < 100; i++) {
                    add(((get(i - 1) + 4 * get(i - 2) - 3 * get(i - 3) - 3 * get(i - 4) + get(i - 5)) % 1000000007
                            + 1000000007) % 1000000007);
                }
            }
        };

        public int countSteppingNumbers(String low, String high) {
            return (countSteppingNumbers(high) - countSteppingNumbers("" + new BigInteger(low).subtract(BigInteger.ONE))
                    + 1000000007) % 1000000007;
        }

        private int countSteppingNumbers(String s) {
            Integer count = 0, dp[][][] = new Integer[s.length()][10][2];
            for (int i = 1; i < s.length(); i++) {
                count = (count + list.get(i).intValue()) % 1000000007;
            }
            for (int i = 1; i <= s.charAt(0) - '0'; i++) {
                count = (count + countSteppingNumbers(1, i, i < s.charAt(0) - '0', s, dp)) % 1000000007;
            }
            return count;
        }

        private int countSteppingNumbers(int index, int prev, boolean flag, String s, Integer[][][] dp) {
            if (prev < 0 || prev > 9) {
                return 0;
            }
            if (index == s.length()) {
                return 1;
            }
            if (dp[index][prev][flag ? 1 : 0] == null) {
                dp[index][prev][flag ? 1 : 0] = ((!flag && prev >= s.charAt(index) - '0' ? 0
                        : countSteppingNumbers(index + 1, prev + 1, flag || prev + 1 < s.charAt(index) - '0', s, dp))
                        + (!flag && prev - 1 > s.charAt(index) - '0' ? 0
                                : countSteppingNumbers(index + 1, prev - 1, flag || prev <= s.charAt(index) - '0', s,
                                        dp)))
                        % 1000000007;
            }
            return dp[index][prev][flag ? 1 : 0];
        }
    }

    // 2815. Max Pair Sum in an Array - EASY
    class Solution2815a {
        public int maxSum(int[] nums) {
            int ans = -1;
            for (int i = 0; i < nums.length; i++) {
                for (int j = i + 1; j < nums.length; j++) {
                    if (("" + nums[i]).chars().max().getAsInt() == ("" + nums[j]).chars().max().getAsInt()) {
                        ans = Math.max(ans, nums[i] + nums[j]);
                    }
                }
            }
            return ans;
        }
    }

    // 2816. Double a Number Represented as a Linked List - MEDIUM
    class Solution2816a {
        public ListNode doubleIt(ListNode head) {
            int result = doubleIt2(head);
            return result > 0 ? new ListNode(1, head) : head;
        }

        private int doubleIt2(ListNode head) {
            if (head == null) {
                return 0;
            } else {
                int result = head.val = 2 * head.val + doubleIt2(head.next);
                head.val %= 10;
                return result / 10;
            }
        }
    }

    // 2817. Minimum Absolute Difference Between Elements With Constraint - MEDIUM
    class Solution2817a {
        public int minAbsoluteDifference(List<Integer> nums, int x) {
            TreeSet<Integer> treeSet = new TreeSet<>(Set.of(-1000000000, Integer.MAX_VALUE));
            int ans = Integer.MAX_VALUE;
            for (int i = x; i < nums.size(); i++) {
                treeSet.add(nums.get(i - x));
                ans = Math.min(ans,
                        Math.min(nums.get(i) - treeSet.floor(nums.get(i)), treeSet.ceiling(nums.get(i)) - nums.get(i)));
            }
            return ans;
        }
    }

    // 2818. Apply Operations to Maximize Score - HARD
    class Solution2818a {
        public int maximumScore(List<Integer> nums, int k) {
            ArrayDeque<Integer> deque = new ArrayDeque<>(List.of(-1));
            TreeMap<Integer, Integer> treeMap = new TreeMap<>();
            long score[] = new long[nums.size()], prod = 1;
            for (int i = 0; i <= nums.size(); deque.push(i++)) {
                if (i < nums.size()) {
                    int num = nums.get(i);
                    for (int j = 2; j * j < num; j++) {
                        for (score[i] += num % j > 0 ? 0 : 1; num % j == 0; num /= j) {
                        }
                    }
                    score[i] += num > 1 ? 1 : 0;
                }
                while (deque.size() > 1 && (i == nums.size() || score[deque.peek()] < score[i])) {
                    int pop = deque.pop();
                    treeMap.put(-nums.get(pop),
                            treeMap.getOrDefault(-nums.get(pop), 0) + (pop - deque.peek()) * (i - pop));
                }
            }
            for (Map.Entry<Integer, Integer> entry : treeMap.entrySet()) {
                prod = prod * BigInteger.valueOf(-entry.getKey())
                        .modPow(BigInteger.valueOf(Math.min(k, entry.getValue())), BigInteger.valueOf(1000000007))
                        .intValue() % 1000000007;
                if ((k -= entry.getValue()) <= 0) {
                    break;
                }
            }
            return (int) prod;
        }
    }

    // 2824. Count Pairs Whose Sum is Less than Target - EASY
    class Solution2824a {
        public int countPairs(List<Integer> nums, int target) {
            int count = 0;
            for (int i = 0; i < nums.size(); i++) {
                for (int j = i + 1; j < nums.size(); j++) {
                    count += nums.get(i) + nums.get(j) < target ? 1 : 0;
                }
            }
            return count;
        }
    }

    // 2825. Make String a Subsequence Using Cyclic Increments - MEDIUM
    class Solution2825a {
        public boolean canMakeSubsequence(String str1, String str2) {
            int j = 0;
            for (int i = 0; i < str1.length() && j < str2.length(); i++) {
                j += str1.charAt(i) == str2.charAt(j) || (str1.charAt(i) - 'a' + 1) % 26 == str2.charAt(j) - 'a' ? 1
                        : 0;
            }
            return j == str2.length();
        }
    }

    // 2826. Sorting Three Groups - MEDIUM
    class Solution2826a {
        public int minimumOperations(List<Integer> nums) {
            int[] dp = { Integer.MAX_VALUE, 0, 0, 0 };
            for (int num : nums) {
                for (int i = 1; i <= 3; i++) {
                    dp[i] = Math.min(dp[i - 1], dp[i] + (num == i ? 0 : 1));
                }
            }
            return dp[3];
        }
    }

    // 2827. Number of Beautiful Integers in the Range - HARD
    class Solution2827a {
        public int numberOfBeautifulIntegers(int low, int high, int k) {
            return calc(0, 0, 0, 0, 1, 1, high + "", k, new Integer[10][10][10][k][2][2])
                    - calc(0, 0, 0, 0, 1, 1, low - 1 + "", k, new Integer[10][10][10][k][2][2]);
        }

        private int calc(int index, int odd, int even, int mod, int start, int flag, String s, int k,
                Integer[][][][][][] dp) {
            if (index == s.length()) {
                return odd == even && mod == 0 ? 1 : 0;
            } else if (dp[index][odd][even][mod][start][flag] == null) {
                dp[index][odd][even][mod][start][flag] = start == 0 ? 0 : calc(index + 1, 0, 0, 0, 1, 0, s, k, dp);
                for (int i = start; i <= (flag == 1 ? s.charAt(index) - '0' : 9); i++) {
                    dp[index][odd][even][mod][start][flag] += calc(index + 1, odd + i % 2, even + 1 - i % 2,
                            (mod * 10 + i) % k, 0, i == (flag == 1 ? s.charAt(index) - '0' : 9) ? flag : 0, s, k, dp);
                }
            }
            return dp[index][odd][even][mod][start][flag];
        }
    }

    // 2828. Check if a String Is an Acronym of Words - EASY
    class Solution2828a {
        public boolean isAcronym(List<String> words, String s) {
            String t = "";
            for (String word : words) {
                t += word.charAt(0);
            }
            return t.equals(s);
        }
    }

    // 2829. Determine the Minimum Sum of a k-avoiding Array - MEDIUM
    class Solution2829a {
        public int minimumSum(int n, int k) {
            HashSet<Integer> set = new HashSet<>();
            int sum = 0;
            for (int i = 1; set.size() < n; i++) {
                if (!set.contains(k - i)) {
                    set.add(i);
                    sum += i;
                }
            }
            return sum;
        }
    }

    // 2830. Maximize the Profit as the Salesman - MEDIUM
    class Solution2830a {
        public int maximizeTheProfit(int n, List<List<Integer>> offers) {
            offers.sort((o, p) -> o.get(1) - p.get(1));
            int[] dp = new int[n + 1];
            for (int i = 0, j = 0; i < n; i++) {
                dp[i + 1] = dp[i];
                for (; j < offers.size() && offers.get(j).get(1) == i; j++) {
                    dp[i + 1] = Math.max(dp[i + 1], offers.get(j).get(2) + dp[offers.get(j).get(0)]);
                }
            }
            return dp[n];
        }
    }

    // 2831. Find the Longest Equal Subarray - MEDIUM
    class Solution2831a {
        public int longestEqualSubarray(List<Integer> nums, int k) {
            HashMap<Integer, ArrayList<Integer>> map = new HashMap<>();
            for (int i = 0; i < nums.size(); i++) {
                map.computeIfAbsent(nums.get(i), t -> new ArrayList<>()).add(i);
            }
            int ans = 0;
            for (ArrayList<Integer> list : map.values()) {
                for (int i = 0, j = 0; i < list.size(); i++) {
                    for (; list.get(i) - list.get(j) - i + j > k; j++) {
                    }
                    ans = Math.max(ans, i - j + 1);
                }
            }
            return ans;
        }
    }
}
