package src;

import java.util.*;
import java.math.*;
import java.util.function.*;

import javax.lang.model.util.Elements;

@SuppressWarnings("unchecked")
public class Lc3000_3099 {
    // 3014. Minimum Number of Pushes to Type Word I - EASY
    class Solution3014a {
        public int minimumPushes(String word) {
            int map[] = new int[26], count = 0;
            for (char c : word.toCharArray()) {
                map[c - 'a']--;
            }
            Arrays.sort(map);
            for (int i = 0; i < 26; i++) {
                count -= map[i] * (i / 8 + 1);
            }
            return count;
        }
    }

    // 3015. Count the Number of Houses at a Certain Distance I - MEDIUM
    class Solution3015a {
        public int[] countOfPairs(int n, int x, int y) {
            if (x > y) {
                return countOfPairs(n, y, x);
            }
            int[] count = new int[n];
            for (int i = 1; i < n; i++) {
                count[Math.min(i, Math.abs(y - i - 1) + x) - Math.min(i, x)] += 2;
                count[Math.min(i, Math.abs(y - i - 1) + x)] -= 2;
                if (x < i) {
                    count[0] += 2;
                    count[i - Math.max(x, (x + i - Math.abs(y - i - 1) - 1) / 2)] -= 2;
                    if (x + i > Math.abs(y - i - 1) + 2) {
                        count[Math.abs(y - i - 1) + 1] += 2;
                        count[Math.abs(y - i - 1) + 1 + Math.max(0, (x + i - Math.abs(y - i - 1) - 1) / 2 - x)] -= 2;
                    }
                }
            }
            for (int i = 1; i < n; i++) {
                count[i] += count[i - 1];
            }
            return count;
        }
    }

    // 3016. Minimum Number of Pushes to Type Word II - MEDIUM
    class Solution3016a {
        public int minimumPushes(String word) {
            int map[] = new int[26], count = 0;
            for (char c : word.toCharArray()) {
                map[c - 'a']--;
            }
            Arrays.sort(map);
            for (int i = 0; i < 26; i++) {
                count -= map[i] * (i / 8 + 1);
            }
            return count;
        }
    }

    // 3017. Count the Number of Houses at a Certain Distance II - HARD
    class Solution3017a {
        public long[] countOfPairs(int n, int x, int y) {
            if (x > y) {
                return countOfPairs(n, y, x);
            }
            long[] count = new long[n];
            for (int i = 1; i < n; i++) {
                count[Math.min(i, Math.abs(y - i - 1) + x) - Math.min(i, x)] += 2;
                count[Math.min(i, Math.abs(y - i - 1) + x)] -= 2;
                if (x < i) {
                    count[0] += 2;
                    count[i - Math.max(x, (x + i - Math.abs(y - i - 1) - 1) / 2)] -= 2;
                    if (x + i > Math.abs(y - i - 1) + 2) {
                        count[Math.abs(y - i - 1) + 1] += 2;
                        count[Math.abs(y - i - 1) + 1 + Math.max(0, (x + i - Math.abs(y - i - 1) - 1) / 2 - x)] -= 2;
                    }
                }
            }
            for (int i = 1; i < n; i++) {
                count[i] += count[i - 1];
            }
            return count;
        }
    }

    // 3019. Number of Changing Keys - EASY
    class Solution3019a {
        public int countKeyChanges(String s) {
            int count = 0;
            for (int i = 1; i < s.length(); i++) {
                count += Character.toLowerCase(s.charAt(i)) == Character.toLowerCase(s.charAt(i - 1)) ? 0 : 1;
            }
            return count;
        }
    }

    // 3020. Find the Maximum Number of Elements in Subset - MEDIUM
    class Solution3020a {
        public int maximumLength(int[] nums) {
            HashMap<Long, Integer> map = new HashMap<>();
            for (long num : nums) {
                map.put(num, map.getOrDefault(num, 0) + 1);
            }
            int max = 1;
            for (long num : nums) {
                int curr = 0;
                for (; num > 1 && map.getOrDefault(num, 0) > 1; num *= num) {
                    curr += 2;
                }
                max = Math.max(max, num > 1 ? curr + map.getOrDefault(num, -1) : 0);
            }
            return Math.max(max, (map.getOrDefault(1L, 0) - 1) / 2 * 2 + 1);
        }
    }

    // 3021. Alice and Bob Playing Flower Game - MEDIUM
    class Solution3021a {
        public long flowerGame(long n, long m) {
            return (n / 2 + n % 2) * (m / 2) + (m / 2 + m % 2) * (n / 2);
        }
    }

    // 3022. Minimize OR of Remaining Elements Using Operations - HARD
    class Solution3022a {
        public int minOrAfterOperations(int[] nums, int k) {
            int mask = 0;
            for (int t = 29; t >= 0; t--) {
                int p = 1 << t;
                mask |= p;
                int s = nums.length;
                for (int i = 0; i < nums.length && s > k; s--) {
                    int c = nums[i++];
                    for (; i < nums.length && (c & mask) > 0; c &= nums[i++]) {
                    }
                    if ((c & mask) > 0) {
                        break;
                    }
                }
                if (s > k) {
                    mask ^= p;
                }
            }
            return (1 << 30) - 1 ^ mask;
        }
    }

    // 3065. Minimum Operations to Exceed Threshold Value I - EASY
    class Solution3065a {
        public int minOperations(int[] nums, int k) {
            int ans = 0;
            for (int num : nums) {
                if (num < k)
                    ans++;
            }
            return ans;
        }
    }

    // 3066. Minimum Operations to Exceed Threshold Value II - MEDIUM
    class Solution3066a {
        public int minOperations(int[] nums, int k) {
            PriorityQueue<Long> heap = new PriorityQueue<>();
            for (int num : nums) {
                heap.add(num + 0L);
            }
            int ans = 0;
            while (heap.peek() < k) {
                ans++;
                long a = heap.poll(), b = heap.poll();
                heap.add(a * 2 + b);
            }
            return ans;
        }
    }

    // 3067. Count Pairs of Connectable Servers in a Weighted Tree Network - MEDIUM
    class Solution3067a {
        List<int[]>[] list;
        int signalSpeed;

        public int[] countPairsOfConnectableServers(int[][] edges, int signalSpeed) {
            int n = edges.length + 1;
            this.signalSpeed = signalSpeed;
            this.list = new ArrayList[n];
            Arrays.setAll(list, x -> new ArrayList<>());
            for (int[] edge : edges) {
                int a = edge[0], b = edge[1], w = edge[2];
                list[a].add(new int[] { b, w });
                list[b].add(new int[] { a, w });
            }
            int[] ans = new int[n];
            for (int x = 0; x < n; x++) {
                int cur = 0;
                for (int[] l : list[x]) {
                    int y = l[0], w = l[1];
                    int cnt = dfs(y, x, w);
                    ans[x] += cur * cnt;
                    cur += cnt;
                }
            }
            return ans;
        }

        private int dfs(int x, int fa, int weight) {
            int res = 0;
            if (weight % signalSpeed == 0)
                res++;
            for (int[] l : list[x]) {
                int y = l[0], w = l[1];
                if (y != fa)
                    res += dfs(y, x, weight + w);
            }
            return res;
        }
    }

    // 3068. Find the Maximum Sum of Node Values - HARD
    class Solution3068a {
        public long maximumValueSum(int[] nums, int k, int[][] edges) {
            long ans = 0;
            PriorityQueue<Integer> heap = new PriorityQueue<>((a, b) -> b - a);
            for (int num : nums) {
                ans += num;
                heap.add((num ^ k) - num);
            }
            while (heap.size() > 1) {
                int a = heap.poll(), b = heap.poll();
                if (a + b > 0)
                    ans += a + b;
                else
                    break;
            }
            return ans;
        }
    }

    // 3069. Distribute Elements Into Two Arrays I - EASY
    class Solution3069a {
        public int[] resultArray(int[] nums) {
            ArrayList<Integer> arr1 = new ArrayList<>(), arr2 = new ArrayList<>();
            arr1.add(nums[0]);
            arr2.add(nums[1]);
            for (int i = 2; i < nums.length; i++) {
                if (arr1.get(arr1.size() - 1) > arr2.get(arr2.size() - 1)) {
                    arr1.add(nums[i]);
                } else {
                    arr2.add(nums[i]);
                }
            }
            for (int i = 0; i < arr1.size(); i++) {
                nums[i] = arr1.get(i);
            }
            for (int i = 0; i < arr2.size(); i++) {
                nums[i + arr1.size()] = arr2.get(i);
            }
            return nums;
        }
    }

    // 3070. Count Submatrices with Top-Left Element and Sum Less Than k - MEDIUM
    class Solution3070a {
        public int countSubmatrices(int[][] grid, int k) {
            int count = 0, sum[] = new int[grid[0].length];
            for (int i = 0; i < grid.length; i++) {
                int curr = 0;
                for (int j = 0; j < grid[0].length; j++) {
                    count += (curr += sum[j] += grid[i][j]) > k ? 0 : 1;
                }
            }
            return count;
        }
    }

    // 3071. Minimum Operations to Write the Letter Y on a Grid - MEDIUM
    class Solution3071a {
        public int minimumOperationsToWriteY(int[][] grid) {
            int other[] = new int[3], ys[] = new int[3], min = Integer.MAX_VALUE, n = grid.length;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    other[grid[i][j]]++;
                }
            }
            for (int i = 0; i < n / 2; i++) {
                other[grid[i][i]]--;
                other[grid[i][n - i - 1]]--;
                ys[grid[i][i]]++;
                ys[grid[i][n - i - 1]]++;
            }
            for (int i = n / 2; i < n; i++) {
                other[grid[i][n / 2]]--;
                ys[grid[i][n / 2]]++;
            }
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    min = Math.min(min, i == j ? Integer.MAX_VALUE : n * n - other[i] - ys[j]);
                }
            }
            return min;
        }
    }

    // 3072. Distribute Elements Into Two Arrays II - HARD
    class Solution3072a {
        public int[] resultArray(int[] nums) {
            int[] clone = nums.clone(), tree1 = new int[100005], tree2 = new int[100005];
            Arrays.sort(clone);
            HashMap<Integer, Integer> map = new HashMap<>();
            for (int i = 0; i < nums.length; i++) {
                map.put(clone[i], i + 1);
            }
            ArrayList<Integer> arr1 = new ArrayList<>(), arr2 = new ArrayList<>();
            arr1.add(nums[0]);
            arr2.add(nums[1]);
            update(map.get(nums[0]), tree1);
            update(map.get(nums[1]), tree2);
            for (int i = 2; i < nums.length; i++) {
                if (arr1.size() - get(map.get(nums[i]), tree1) > arr2.size() - get(map.get(nums[i]), tree2)
                        || arr1.size() - get(map.get(nums[i]), tree1) == arr2.size() - get(map.get(nums[i]), tree2)
                                && arr1.size() <= arr2.size()) {
                    arr1.add(nums[i]);
                    update(map.get(nums[i]), tree1);
                } else {
                    arr2.add(nums[i]);
                    update(map.get(nums[i]), tree2);
                }
            }
            for (int i = 0; i < arr1.size(); i++) {
                nums[i] = arr1.get(i);
            }
            for (int i = 0; i < arr2.size(); i++) {
                nums[i + arr1.size()] = arr2.get(i);
            }
            return nums;
        }

        private int get(int i, int[] tree) {
            int num = 0;
            while (i > 0) {
                num += tree[i];
                i -= i & (-i);
            }
            return num;
        }

        private void update(int i, int[] tree) {
            while (i < tree.length) {
                tree[i]++;
                i += i & (-i);
            }
        }
    }
}