package src;

import java.util.*;
import java.util.stream.Collectors;

public class Lc2300_2399 {
    // 2325. Decode the Message - EASY
    class Solution2325a {
        public String decodeMessage(String key, String message) {
            char[] d = new char[128];
            d[' '] = 32; // chr(' ') = 32
            for (int i = 0, j = 0; i < key.length(); i++) {
                char c = key.charAt(i);
                if (d[c] == 0) {
                    d[c] = (char) ('a' + j++);
                }
            }
            StringBuilder ans = new StringBuilder();
            for (int i = 0; i < message.length(); i++) {
                ans.append(d[message.charAt(i)]);
            }
            return ans.toString();
        }
    }

    // 2337. Move Pieces to Obtain a String - MEDIUM
    class Solution2337a { // 47ms
        public boolean canChange(String start, String target) {
            if (!start.replace("_", "").equals(target.replace("_", ""))) {
                return false;
            }
            int j = 0;
            for (int i = 0; i < target.length(); i++) {
                if (target.charAt(i) == 'R') {
                    while (j < i && start.charAt(j) != 'R') {
                        j++;
                    }
                    if (start.charAt(j++) != 'R') {
                        return false;
                    }
                }
            }
            j = start.length() - 1;
            for (int i = target.length() - 1; i >= 0; i--) {
                if (target.charAt(i) == 'L') {
                    while (j > i && start.charAt(j) != 'L') {
                        j--;
                    }
                    if (start.charAt(j--) != 'L') {
                        return false;
                    }
                }
            }
            return true;
        }
    }

    class Solution2337b { // 17ms
        public boolean canChange(String start, String target) {
            for (int i = 0, j = 0;; i++, j++) {
                for (; i < start.length() && start.charAt(i) == '_'; i++) {
                }
                for (; j < target.length() && target.charAt(j) == '_'; j++) {
                }
                if (i == start.length() && j == target.length()) {
                    return true;
                } else if (i == start.length() || j == target.length() || start.charAt(i) != target.charAt(j)
                        || (start.charAt(i) == 'L' ? i < j : i > j)) {
                    return false;
                }
            }
        }
    }

    class Solution2337c { // 12ms
        public boolean canChange(String start, String target) {
            char[] sc = start.toCharArray(), tc = target.toCharArray();
            int n = sc.length, i = 0, j = 0;
            while (i < n || j < n) {
                while (i < n && sc[i] == '_')
                    ++i;
                while (j < n && tc[j] == '_')
                    ++j;
                if (i == n || j == n) {
                    return i == n && j == n;
                }
                if (sc[i] != tc[j] || (sc[i] == 'L' && i < j)
                        || (sc[i] == 'R' && i > j))
                    return false;
                ++i;
                ++j;
            }
            return true;
        }
    }

    // 2357. Make Array Zero by Subtracting Equal Amounts - EASY
    class Solution2357a {
        public int minimumOperations(int[] nums) {
            Set<Integer> s = new HashSet<>();
            for (int v : nums) {
                s.add(v);
            }
            return s.contains(0) ? s.size() - 1 : s.size();
        }
    }

    class Solution2357b {
        public int minimumOperations(int[] nums) {
            Set<Integer> s = new HashSet<>();
            for (int v : nums) {
                if (v > 0) {
                    s.add(v);
                }
            }
            return s.size();
        }
    }

    class Solution2357c {
        public int minimumOperations(int[] nums) {
            boolean[] arr = new boolean[101];
            arr[0] = true;
            int ans = 0;
            for (int x : nums) {
                if (!arr[x]) {
                    ++ans;
                    arr[x] = true;
                }
            }
            return ans;
        }
    }

    class Solution2357d {
        public int minimumOperations(int[] nums) {
            // 3ms
            return (int) Arrays.stream(nums).filter(v -> 0 != v).distinct().count();
        }
    }

    class Solution2357e {
        public int minimumOperations(int[] nums) {
            // 3ms
            return Arrays.stream(nums).filter(r -> r > 0).boxed().collect(Collectors.toSet()).size();
        }
    }

    // 2363. Merge Similar Items - EASY
    class Solution2363a {
        public List<List<Integer>> mergeSimilarItems(int[][] items1, int[][] items2) {
            List<List<Integer>> ans = new ArrayList<>();
            int[] cnt = new int[1001];
            for (int[] v : items1) {
                cnt[v[0]] += v[1];
            }
            for (var v : items2) {
                cnt[v[0]] += v[1];
            }
            for (int i = 0; i < cnt.length; i++) {
                if (cnt[i] > 0) {
                    ans.add(List.of(i, cnt[i])); // 2ms
                    // ans.add(Arrays.asList(i, cnt[i])); // 3ms
                }
            }
            return ans;
        }
    }

    class Solution2363b { // 8ms
        public List<List<Integer>> mergeSimilarItems(int[][] items1, int[][] items2) {
            List<List<Integer>> ans = new ArrayList<List<Integer>>();
            Map<Integer, Integer> m = new HashMap<Integer, Integer>();
            for (int v[] : items1) {
                m.put(v[0], v[1]);
            }
            for (int v[] : items2) {
                m.put(v[0], m.getOrDefault(v[0], 0) + v[1]);
            }
            for (Map.Entry<Integer, Integer> e : m.entrySet()) {
                List<Integer> tmp = new ArrayList<Integer>();
                tmp.add(e.getKey());
                tmp.add(e.getValue());
                ans.add(tmp);
            }
            ans.sort(new Comparator<List<Integer>>() {
                @Override
                public int compare(List<Integer> o1, List<Integer> o2) {
                    return o1.get(0) - o2.get(0);
                }
            });
            return ans;
        }
    }

    class Solution2363c { // 9ms
        public List<List<Integer>> mergeSimilarItems(int[][] items1, int[][] items2) {
            List<List<Integer>> ans = new ArrayList<>();
            HashMap<Integer, Integer> m = new HashMap<>();
            for (int i = 0; i < items1.length; i++) {
                m.put(items1[i][0], m.getOrDefault(items1[i][0], 0) + items1[i][1]);
            }
            for (int i = 0; i < items2.length; i++) {
                m.put(items2[i][0], m.getOrDefault(items2[i][0], 0) + items2[i][1]);
            }
            List<Map.Entry<Integer, Integer>> ori = new ArrayList<>(m.entrySet());
            Collections.sort(ori, (o1, o2) -> o1.getKey() - o2.getKey());
            for (Map.Entry<Integer, Integer> e : ori) {
                ans.add(Arrays.asList(e.getKey(), e.getValue()));
            }
            return ans;
        }
    }

    class Solution2363d {
        public List<List<Integer>> mergeSimilarItems(int[][] items1, int[][] items2) {
            List<List<Integer>> ans = new ArrayList<List<Integer>>();
            Map<Integer, Integer> m = new HashMap<Integer, Integer>();
            for (int[] v : items1) {
                m.put(v[0], m.getOrDefault(v[0], 0) + v[1]);
            }
            for (int[] v : items2) {
                m.put(v[0], m.getOrDefault(v[0], 0) + v[1]);
            }
            for (Map.Entry<Integer, Integer> e : m.entrySet()) {
                // 10ms
                int k = e.getKey(), v = e.getValue();
                List<Integer> pair = new ArrayList<Integer>();
                pair.add(k);
                pair.add(v);
                ans.add(pair);
                // 9ms
                // ans.add(Arrays.asList(e.getKey(), e.getValue()));
            }
            Collections.sort(ans, (a, b) -> a.get(0) - b.get(0));
            return ans;
        }
    }

    class Solution2363e {
        public List<List<Integer>> mergeSimilarItems(int[][] items1, int[][] items2) {
            List<List<Integer>> ans = new ArrayList<>();
            TreeMap<Integer, Integer> m = new TreeMap<>();
            for (int[] v : items1) {
                m.put(v[0], m.getOrDefault(v[0], 0) + v[1]);
            }
            for (int[] v : items2) {
                m.put(v[0], m.getOrDefault(v[0], 0) + v[1]);
            }
            for (Map.Entry<Integer, Integer> e : m.entrySet()) {
                // ans.add(new ArrayList<>(Arrays.asList(e.getKey(), e.getValue()))); // 13 ms
                ans.add(Arrays.asList(e.getKey(), e.getValue())); // 11ms

                // ans.add(List.of(e.getKey(), e.getValue())); // 11ms
            }
            return ans;
        }
    }

    // 2367. Number of Arithmetic Triplets - EASY
    class Solution2367a {
        public int arithmeticTriplets(int[] nums, int diff) {
            int ans = 0;
            int n = nums.length;
            for (int i = 0; i < n - 2; i++) {
                int j = i + 1;
                for (; j < n - 1; j++) {
                    if (nums[j] - nums[i] == diff) {
                        for (int k = j + 1; k < n; k++) {
                            if (nums[k] - nums[j] == diff) {
                                ans++;
                                break;
                            }
                        }
                        break;
                    }
                }
            }
            return ans;
        }
    }

    class Solution2367b {
        public int arithmeticTriplets(int[] nums, int diff) {
            int ans = 0;
            Set<Integer> s = new HashSet<>();
            for (int v : nums) {
                s.add(v);
            }
            for (int v : nums) {
                if (s.contains(v - diff) && s.contains(v + diff)) {
                    ans++;
                }
            }
            return ans;
        }
    }

    class Solution2367c {
        public int arithmeticTriplets(int[] nums, int diff) {
            int ans = 0;
            int[] arr = new int[251];
            for (int v : nums) {
                arr[v]++;
            }
            for (int v : nums) {
                if (arr[v + diff] != 0 && arr[v + diff * 2] != 0) {
                    ans++;
                }
            }
            return ans;
        }
    }

    // 2373. Largest Local Values in a Matrix - EASY
    class Solution2373a {
        public int[][] largestLocal(int[][] grid) {
            int n = grid.length;
            int[][] ans = new int[n - 2][n - 2];
            for (int i = 0; i < n - 2; i++) {
                for (int j = 0; j < n - 2; j++) {
                    for (int x = 0; x < 3; x++) {
                        for (int y = 0; y < 3; y++) {
                            ans[i][j] = Math.max(ans[i][j], grid[i + x][j + y]);
                        }
                    }
                }
            }
            return ans;
        }
    }

    // 2383. Minimum Hours of Training to Win a Competition - EASY
    class Solution2383a {
        public int minNumberOfHours(int initialEnergy, int initialExperience, int[] energy, int[] experience) {
            int s = 0;
            for (int v : energy)
                s += v;
            int ans = Math.max(0, s - initialEnergy + 1);
            for (int v : experience) {
                if (initialExperience <= v) {
                    ans += v - initialExperience + 1;
                    initialExperience = v + 1;
                }
                initialExperience += v;
            }
            return ans;
        }
    }

    // 2389. Longest Subsequence With Limited Sum - EASY
    class Solution2389a {
        public int[] answerQueries(int[] nums, int[] queries) {
            Arrays.sort(nums);
            for (int i = 1; i < nums.length; i++) {
                nums[i] += nums[i - 1];
            }
            int[] ans = new int[queries.length];
            for (int i = 0; i < queries.length; i++) {
                ans[i] = Sort.upperBound(nums, queries[i]);
            }
            return ans;
        }
    }

    // 2395. Find Subarrays With Equal Sum - EASY
    class Solution2395a {
        public boolean findSubarrays(int[] nums) {
            Set<Integer> s = new HashSet<>();
            for (int i = 0; i < nums.length - 1; ++i) {
                int sum = nums[i] + nums[i + 1];
                if (!s.add(sum)) {
                    return true;
                }
            }
            return false;
        }
    }
}
