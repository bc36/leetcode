package src;

import static java.math.BigInteger.valueOf;
import java.math.BigInteger;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Stream;

public class Lc2600_2699 {
    // 2600. K Items With the Maximum Sum - EASY
    class Solution2600a {
        public int kItemsWithMaximumSum(int numOnes, int numZeros, int numNegOnes, int k) {
            return k >= (numOnes + numZeros) ? numOnes - (k - (numOnes + numZeros)) : Math.min(k, numOnes);
        }
    }

    class Solution2600b {
        public int kItemsWithMaximumSum(int numOnes, int numZeros, int numNegOnes, int k) {
            return Math.min(k, numOnes) - Math.max(0, k - numOnes - numZeros);
        }
    }

    // 2601. Prime Subtraction Operation - MEDIUM
    class Solution2601a {
        // 11ms
        public boolean primeSubOperation(int[] nums) {
            boolean[] composite = new boolean[1001];
            for (int i = 2; i < 1001; ++i) {
                for (int j = i + i; j < 1001; j += i) {
                    composite[j] = true;
                }
            }
            for (int i = 0; i < nums.length; ++i) {
                for (int j = nums[i] - 1; j > 1; --j) {
                    if (!composite[j] && (i == 0 || nums[i] - j > nums[i - 1])) {
                        nums[i] -= j;
                        break;
                    }
                }
                if (i > 0 && nums[i] <= nums[i - 1]) {
                    return false;
                }
            }
            return true;
        }
    }

    class Solution2601b {
        // 17ms
        public boolean primeSubOperation(int[] nums) {
            boolean[] primes = new boolean[1001];
            for (int i = 2; i <= 1000; ++i) {
                primes[i] = true;
                for (int j = 2; j * j <= i; ++j) {
                    if (i % j == 0) {
                        primes[i] = false;
                        break;
                    }
                }
            }
            for (int i = 0; i < nums.length; ++i) {
                for (int j = nums[i] - 1; j > 1; --j) {
                    if (primes[j] && nums[i] > j && (i == 0 || nums[i - 1] < nums[i] - j)) {
                        nums[i] -= j;
                        break;
                    }
                }
                if (i > 0 && nums[i] <= nums[i - 1]) {
                    return false;
                }
            }
            return true;
        }
    }

    // 2602. Minimum Operations to Make All Array Elements Equal - MEDIUM
    class Solution2602a {
        // 122ms
        public List<Long> minOperations(int[] nums, int[] queries) {
            Arrays.sort(nums);
            long sum = 0;
            TreeMap<Long, long[]> map = new TreeMap<>(Map.of(0L, new long[2]));
            System.out.println(map.get(0L)[1]);
            for (int i = 0; i < nums.length; i++) {
                map.put((long) nums[i], new long[] { i + 1, sum += nums[i] });
            }
            ArrayList<Long> ans = new ArrayList<>();
            for (long q : queries) {
                long[] val = map.floorEntry(q).getValue();
                // ans.add((q * val[0] - val[1]) + (sum - val[1] - q * (nums.length - val[0])));
                ans.add(sum + 2 * q * val[0] - 2 * val[1] - q * nums.length);
            }
            return ans;
        }
    }

    class Solution2602b {
        // 41ms
        public List<Long> minOperations(int[] nums, int[] queries) {
            Arrays.sort(nums);
            int n = nums.length;
            long[] pre = new long[n + 1];
            for (int i = 0; i < n; ++i)
                pre[i + 1] = pre[i] + nums[i];
            ArrayList<Long> ans = new ArrayList<Long>(queries.length);
            for (int q : queries) {
                int i = Sort.lowerBound(nums, q);
                long left = (long) q * i - pre[i];
                long right = pre[n] - pre[i] - (long) q * (n - i);
                ans.add(left + right);
            }
            return ans;
        }
    }

    // 2603. Collect Coins in a Tree - HARD
    class Solution2603a {
        // 93ms
        @SuppressWarnings("unchecked")
        public int collectTheCoins(int[] coins, int[][] edges) {
            HashSet<Integer> sets[] = new HashSet[coins.length], set = new HashSet<>();
            for (int i = 0; i < coins.length; i++) {
                sets[i] = new HashSet<>();
            }
            for (int[] e : edges) {
                sets[e[0]].add(e[1]);
                sets[e[1]].add(e[0]);
            }
            int ans = coins.length * 2 - 2;
            for (int i = 0, j; i < coins.length; i++) {
                for (j = i; sets[j].size() == 1 && coins[j] == 0; ans -= 2) {
                    sets[sets[j].iterator().next()].remove(j);
                    sets[j].remove(j = sets[j].iterator().next());
                }
                if (sets[j].size() == 1) {
                    set.add(j);
                }
            }
            for (int i = 0; i < 2; i++) {
                HashSet<Integer> next = new HashSet<>();
                for (int j : set) {
                    if (sets[j].size() == 1) {
                        sets[sets[j].iterator().next()].remove(j);
                        if (sets[sets[j].iterator().next()].size() == 1) {
                            next.add(sets[j].iterator().next());
                        }
                        sets[j].remove(sets[j].iterator().next());
                        ans -= 2;
                    }
                }
                set = next;
            }
            return ans;
        }
    }

    class Solution2603b {
        List<Integer>[] g;
        int[] height;
        int[] coins;
        int[] depth;

        public void dfsHeight(int root, int p) {
            height[root] = -1;
            depth[root] = p == -1 ? 0 : depth[p] + 1;
            if (coins[root] == 1) {
                height[root] = 0;
            }
            for (int node : g[root]) {
                if (node == p) {
                    continue;
                }
                dfsHeight(node, root);
                if (height[node] >= 0) {
                    height[root] = Math.max(height[root], height[node] + 1);
                }
            }
        }

        public boolean findPath(int root, int p, int target, List<Integer> path) {
            path.add(root);
            if (root == target) {
                return true;
            }
            for (int node : g[root]) {
                if (node == p) {
                    continue;
                }
                if (findPath(node, root, target, path)) {
                    return true;
                }
            }
            path.remove(path.size() - 1);
            return false;
        }

        @SuppressWarnings("unchecked")
        public int collectTheCoins(int[] coins, int[][] edges) {
            int n = coins.length;
            g = new List[n];
            height = new int[n];
            depth = new int[n];
            this.coins = coins;
            for (int i = 0; i < n; i++) {
                g[i] = new ArrayList<Integer>();
            }
            for (int[] e : edges) {
                g[e[0]].add(e[1]);
                g[e[1]].add(e[0]);
            }
            int root = -1;
            for (int i = 0; i < n; i++) {
                if (coins[i] == 1) {
                    root = i;
                    break;
                }
            }
            if (root == -1) {
                return 0;
            }
            dfsHeight(root, -1);
            int end = root;
            for (int i = 0; i < n; i++) {
                if (depth[i] > depth[end] && coins[i] > 0) {
                    end = i;
                }
            }
            root = end;
            dfsHeight(root, -1);
            end = root;
            for (int i = 0; i < n; i++) {
                if (depth[i] > depth[end] && coins[i] > 0) {
                    end = i;
                }
            }
            List<Integer> path = new ArrayList<Integer>(n);
            findPath(root, -1, end, path);
            int middle = path.get(path.size() / 2);
            dfsHeight(middle, -1);
            int cnt = 0;
            for (int i = 0; i < n; i++) {
                if (height[i] >= 2) {
                    cnt++;
                }
            }
            return Math.max(0, cnt - 1) * 2;
        }
    }

    class Solution2603c {
        // 35ms
        @SuppressWarnings("unchecked")
        public int collectTheCoins(int[] coins, int[][] edges) {
            int n = coins.length, ind[] = new int[n];
            List<Integer> g[] = new ArrayList[n];
            Arrays.setAll(g, e -> new ArrayList<>());
            for (var e : edges) {
                int x = e[0], y = e[1];
                g[x].add(y);
                g[y].add(x);
                ++ind[x];
                ++ind[y];
            }
            ArrayDeque<Integer> q = new ArrayDeque<Integer>();
            for (int i = 0; i < n; ++i)
                if (ind[i] == 1 && coins[i] == 0)
                    q.add(i);
            while (!q.isEmpty()) {
                int x = q.poll();
                for (int y : g[x])
                    if (--ind[y] == 1 && coins[y] == 0)
                        q.add(y);
            }
            for (int i = 0; i < n; ++i)
                if (ind[i] == 1 && coins[i] == 1)
                    q.add(i);
            if (q.size() <= 1)
                return 0;
            int[] time = new int[n];
            while (!q.isEmpty()) {
                int x = q.peek();
                q.pop();
                for (int y : g[x])
                    if (--ind[y] == 1) {
                        time[y] = time[x] + 1;
                        q.add(y);
                    }
            }
            int ans = 0;
            for (var e : edges)
                if (time[e[0]] >= 2 && time[e[1]] >= 2)
                    ans += 2;
            return ans;
        }
    }

    // 2605. Form Smallest Number From Two Digit Arrays - EASY
    class Solution2605a {
        public int minNumber(int[] nums1, int[] nums2) {
            int[] count = new int[10];
            for (int v : nums1) {
                count[v] = 1;
            }
            for (int v : nums2) {
                count[v] += 2;
            }
            for (int i = 1; i < 10; i++) {
                if (count[i] == 3) {
                    return i;
                }
            }
            for (int i = 1;; i++) {
                for (int j = i + 1; j < 10; j++) {
                    if (count[i] * count[j] == 2) {
                        return i * 10 + j;
                    }
                }
            }
        }
    }

    // 2606. Find the Substring With Maximum Cost - MEDIUM
    class Solution2606a {
        // 14ms
        public int maximumCostSubstring(String s, String chars, int[] vals) {
            HashMap<Character, Integer> m = new HashMap<>();
            for (int i = 0; i < chars.length(); i++) {
                m.put(chars.charAt(i), vals[i]);
            }
            int ans = 0, cur = 0;
            for (char c : s.toCharArray()) {
                ans = Math.max(ans, cur = Math.max(0, cur + m.getOrDefault(c, c - 'a' + 1)));
            }
            return ans;
        }
    }

    class Solution2606b {
        // 3ms
        public int maximumCostSubstring(String s, String chars, int[] vals) {
            int ans = 0, cur = 0;
            int[] m = new int[26];
            for (int i = 0; i < 26; ++i)
                m[i] = i + 1;
            for (int i = 0; i < chars.length(); ++i) {
                m[chars.charAt(i) - 'a'] = vals[i];
            }
            for (char c : s.toCharArray()) {
                ans = Math.max(ans, cur = Math.max(0, cur + m[c - 'a']));
            }
            return ans;
        }
    }

    // 2607. Make K-Subarray Sums Equal - MEDIUM
    class Solution2607a {
        public long makeSubKSumEqual(int[] arr, int k) {
            long gcd = BigInteger.valueOf(arr.length).gcd(BigInteger.valueOf(k)).intValue(), sum = 0;
            for (int i = 0; i < gcd; i++) {
                ArrayList<Integer> list = new ArrayList<>();
                for (int j = i; j < arr.length; j += gcd) {
                    list.add(arr[j]);
                }
                Collections.sort(list);
                for (int j = 0; j < list.size(); j++) {
                    sum += Math.abs(list.get(j) - list.get(list.size() / 2));
                }
            }
            return sum;
        }
    }

    // 2608. Shortest Cycle in a Graph - HARD
    class Solution2608a {
        // 351ms
        public int findShortestCycle(int n, int[][] edges) {
            HashMap<Integer, ArrayList<Integer>> g = new HashMap<>();
            for (int[] e : edges) {
                g.computeIfAbsent(e[0], t -> new ArrayList<>()).add(e[1]);
                g.computeIfAbsent(e[1], t -> new ArrayList<>()).add(e[0]);
            }
            int ans = Integer.MAX_VALUE;
            for (int i = 0; i < n; i++) {
                int[] dist = new int[n];
                dist[i] = 1;
                for (ArrayDeque<Integer> deque = new ArrayDeque<>(List.of(i)); !deque.isEmpty();) {
                    int x = deque.poll();
                    for (int y : g.getOrDefault(x, new ArrayList<>())) {
                        if (dist[y] == 0) {
                            dist[y] = dist[x] + 1;
                            deque.offer(y);
                        } else if (dist[y] != dist[x] - 1) {
                            ans = Math.min(ans, dist[x] + dist[y] - 1);
                        }
                    }
                }
            }
            return ans == Integer.MAX_VALUE ? -1 : ans;
        }
    }

    class Solution2608b {
        // 226ms
        public int findShortestCycle(int n, int[][] edges) {
            List<List<Integer>> g = new ArrayList<>();
            for (int i = 0; i < n; ++i) {
                g.add(new ArrayList<>());
            }
            for (int[] e : edges) {
                g.get(e[0]).add(e[1]);
                g.get(e[1]).add(e[0]);
            }
            int inf = 10000, ans = inf;
            Function<Integer, Integer> root = i -> {
                List<Integer> dist = new ArrayList<>(Collections.nCopies(n, inf));
                dist.set(i, 0);
                Queue<Integer> q = new LinkedList<>(Arrays.asList(i));
                while (!q.isEmpty()) {
                    i = q.poll();
                    for (int j : g.get(i)) {
                        if (dist.get(j) == inf) {
                            dist.set(j, 1 + dist.get(i));
                            q.offer(j);
                        } else if (dist.get(i) <= dist.get(j)) {
                            return dist.get(i) + dist.get(j) + 1;
                        }
                    }
                }
                return inf;
            };
            for (int i = 0; i < n; ++i)
                ans = Math.min(ans, root.apply(i));
            return ans < inf ? ans : -1;
        }
    }

    // 2609. Find the Longest Balanced Substring of a Binary String - EASY
    class Solution2609a {
        // 36ms
        public int findTheLongestBalancedSubstring(String s) {
            for (int i = s.length() / 2; i > 0; i--) {
                if (s.matches(".*0{" + i + "}1{" + i + "}.*")) {
                    return 2 * i;
                }
            }
            return 0;
        }
    }

    class Solution2609b {
        // 1ms
        public int findTheLongestBalancedSubstring(String s) {
            int ans = 0, pre = 0, cur = 0, n = s.length();
            char[] cs = s.toCharArray();
            for (int i = 0; i < n; ++i) {
                ++cur;
                if (i == n - 1 || cs[i] != cs[i + 1]) {
                    if (cs[i] == '1') {
                        ans = Math.max(ans, Math.min(pre, cur) * 2);
                    }
                    pre = cur;
                    cur = 0;
                }
            }
            return ans;
        }
    }

    // 2610. Convert an Array Into a 2D Array With Conditions - MEDIUM
    class Solution2610a {
        // 1ms
        public List<List<Integer>> findMatrix(int[] nums) {
            ArrayList<List<Integer>> ans = new ArrayList<>();
            int[] count = new int[nums.length + 1];
            for (int v : nums) {
                if (ans.size() <= count[v]) {
                    ans.add(new ArrayList<>());
                }
                ans.get(count[v]++).add(v);
            }
            return ans;
        }
    }

    class Solution2610b {
        // 4ms
        public List<List<Integer>> findMatrix(int[] nums) {
            HashMap<Integer, Integer> cnt = new HashMap<>();
            for (int k : nums)
                cnt.merge(k, 1, Integer::sum);
            List<List<Integer>> ans = new ArrayList<>();
            while (!cnt.isEmpty()) {
                List<Integer> row = new ArrayList<>();
                for (Iterator<Map.Entry<Integer, Integer>> it = cnt.entrySet().iterator(); it.hasNext();) {
                    Map.Entry<Integer, Integer> e = it.next();
                    row.add(e.getKey());
                    e.setValue(e.getValue() - 1);
                    if (e.getValue() == 0)
                        it.remove();
                }
                ans.add(row);
            }
            return ans;
        }
    }

    // 2611. Mice and Cheese - MEDIUM
    class Solution2611a {
        // 11ms
        public int miceAndCheese(int[] reward1, int[] reward2, int k) {
            int ans = 0, n = reward1.length, diff[] = new int[n];
            for (int i = 0; i < n; ++i) {
                diff[i] = reward2[i] - reward1[i];
                ans += reward2[i];
            }
            Arrays.sort(diff);
            for (int i = 0; i < k; ++i) {
                ans -= diff[i];
            }
            return ans;
        }
    }

    class Solution2611b {
        // 11ms
        public int miceAndCheese(int[] reward1, int[] reward2, int k) {
            int ans = 0, n = reward1.length;
            for (int i = 0; i < n; ++i) {
                ans += reward2[i]; // 全部给第二只老鼠
                reward1[i] -= reward2[i];
            }
            Arrays.sort(reward1);
            for (int i = n - k; i < n; ++i)
                ans += reward1[i];
            return ans;
        }
    }

    // 2612. Minimum Reverse Operations - HARD
    class Solution2612a {
        // 420ms
        @SuppressWarnings("unchecked")
        public int[] minReverseOperations(int n, int p, int[] banned, int k) {
            int[] result = new int[n];
            TreeSet<Integer>[] set = new TreeSet[] { new TreeSet<>(), new TreeSet<>() };
            for (int i = 0; i < n; ++i) {
                set[i % 2].add(i);
                result[i] = i == p ? 0 : -1;
            }
            set[p % 2].remove(p);
            for (int i : banned) {
                set[i % 2].remove(i);
            }
            for (ArrayDeque<Integer> deque = new ArrayDeque<>(List.of(p)); !deque.isEmpty();) {
                for (Integer poll = deque.poll(), i = Math.abs(poll - k + 1), j = set[i % 2].ceiling(i); j != null
                        && j < n - Math.abs(n - poll - k); j = set[i % 2].higher(j)) {
                    deque.offer(j);
                    result[j] = result[poll] + 1;
                    set[i % 2].remove(j);
                }
            }
            return result;
        }
    }

    class Solution2612b {
        // 170ms
        @SuppressWarnings("unchecked")
        public int[] minReverseOperations(int n, int p, int[] banned, int k) {
            var ban = new boolean[n];
            ban[p] = true;
            for (int i : banned)
                ban[i] = true;
            TreeSet<Integer>[] sets = new TreeSet[2];
            sets[0] = new TreeSet<>();
            sets[1] = new TreeSet<>();
            for (int i = 0; i < n; i++)
                if (!ban[i])
                    sets[i % 2].add(i);
            sets[0].add(n);
            sets[1].add(n); // 哨兵

            var ans = new int[n];
            Arrays.fill(ans, -1);
            var q = new ArrayList<Integer>();
            q.add(p);
            for (int step = 0; !q.isEmpty(); ++step) {
                var tmp = q;
                q = new ArrayList<>();
                for (int i : tmp) {
                    ans[i] = step;
                    // 从 mn 到 mx 的所有位置都可以翻转到
                    int mn = Math.max(i - k + 1, k - i - 1);
                    int mx = Math.min(i + k - 1, n * 2 - k - i - 1);
                    var s = sets[mn % 2];
                    for (var j = s.ceiling(mn); j <= mx; j = s.ceiling(mn)) {
                        q.add(j);
                        s.remove(j);
                    }
                }
            }
            return ans;
        }
    }

    // 2614. Prime In Diagonal - EASY
    class Solution2614a {
        public int diagonalPrime(int[][] nums) {
            int n = nums.length, ans = 0;
            for (int i = 0; i < n; ++i) {
                int x = nums[i][i];
                if (x > ans && isPrime(x))
                    ans = x;
                x = nums[i][n - 1 - i];
                if (x > ans && isPrime(x))
                    ans = x;
            }
            return ans;
        }

        private boolean isPrime(int n) {
            for (int i = 2; i * i <= n; i++) {
                if (n % i == 0) {
                    return false;
                }
            }
            return n > 1;
        }
    }

    // 2615. Sum of Distances - MEDIUM
    class Solution2615a {
        // 18ms
        public long[] distance(int[] nums) {
            HashMap<Integer, ArrayList<Integer>> group = new HashMap<>();
            for (int i = 0; i < nums.length; i++) {
                group.computeIfAbsent(nums[i], k -> new ArrayList<>()).add(i);
            }
            long[] ans = new long[nums.length];
            for (ArrayList<Integer> list : group.values()) {
                int n = list.size();
                long[] pre = new long[n + 1];
                for (int i = 0; i < n; i++) {
                    pre[i + 1] = pre[i] + list.get(i);
                }
                for (int i = 0; i < n; i++) {
                    // int t = list.get(i);
                    // long left = (long) t * i - pre[i]; // without (long), overflow -> ERROR
                    // long right = pre[n] - pre[i] - 1L * t * (n - i);
                    // ans[t] = left + right;
                    ans[list.get(i)] = pre[n] - 2 * pre[i] - list.get(i) * (n - 2L * i);
                }
            }
            return ans;
        }
    }

    // 2616. Minimize the Maximum Difference of Pairs - MEDIUM
    class Solution2616a {
        public int minimizeMax(int[] nums, int p) {
            // int l = 0, r = 1000000000; // 20ms
            // for (Arrays.sort(nums); l < r;) {

            Arrays.sort(nums); // 17ms
            int l = 0, r = nums[nums.length - 1] - nums[0];
            while (l < r) {

                int m = l + r >> 1, count = 0;
                for (int i = 1; i < nums.length; i++) {
                    if (nums[i] - nums[i - 1] <= m) {
                        count++;
                        i++;
                    }
                }
                if (count < p) {
                    l = m + 1;
                } else {
                    r = m;
                }
            }
            return l;
        }
    }

    // 2617. Minimum Number of Visited Cells in a Grid - HARD
    class Solution2617a {
        @SuppressWarnings("unchecked")
        public int minimumVisitedCells(int[][] grid) {
            int n = grid.length;
            int m = grid[0].length;
            TreeSet<Integer>[] rows = new TreeSet[n];
            TreeSet<Integer>[] cols = new TreeSet[m];
            for (int i = 0; i < n; i++) {
                rows[i] = new TreeSet<>();
                for (int j = 0; j < m; j++) {
                    rows[i].add(j);
                }
            }
            for (int i = 0; i < m; i++) {
                cols[i] = new TreeSet<>();
                for (int j = 0; j < n; j++) {
                    cols[i].add(j);
                }
            }
            int inf = (int) 1e8;
            int[][] dist = new int[n][m];
            for (int[] row : dist) {
                Arrays.fill(row, inf);
            }
            dist[0][0] = 1;
            Deque<int[]> dq = new ArrayDeque<>();
            dq.addLast(new int[] { 0, 0 });
            rows[0].remove(0);
            cols[0].remove(0);
            while (!dq.isEmpty()) {
                int[] head = dq.removeFirst();
                int x = head[0];
                int y = head[1];
                int r = grid[x][y] + y;
                int b = grid[x][y] + x;

                while (true) {
                    var next = rows[x].higher(y);
                    if (next == null || next > r) {
                        break;
                    }
                    dist[x][next] = dist[x][y] + 1;
                    rows[x].remove(next);
                    cols[next].remove(x);
                    dq.addLast(new int[] { x, next });
                }

                while (true) {
                    var next = cols[y].higher(x);
                    if (next == null || next > b) {
                        break;
                    }
                    dist[next][y] = dist[x][y] + 1;
                    rows[next].remove(y);
                    cols[y].remove(next);
                    dq.addLast(new int[] { next, y });
                }
            }

            int ans = dist[n - 1][m - 1];
            if (ans == inf) {
                return -1;
            }
            return ans;
        }
    }

    // 2639. Find the Width of Columns of a Grid - EASY
    // 2640. Find the Score of All Prefixes of an Array - MEDIUM
    class Solution2640a {
        // 3ms
        public long[] findPrefixScore(int[] nums) {
            long ans[] = new long[nums.length], sum = 0, mx = 0;
            for (int i = 0; i < nums.length; i++) {
                ans[i] = sum += (mx = Math.max(mx, nums[i])) + nums[i];
            }
            return ans;
        }
    }

    // 2641. Cousins in Binary Tree II - MEDIUM
    class Solution2641a {
        // 92ms
        public TreeNode replaceValueInTree(TreeNode root) {
            HashMap<TreeNode, Integer> map = new HashMap<>();
            HashMap<Integer, Integer> map2 = new HashMap<>();
            dfs(root, null, 0, map, map2);
            dfs2(root, null, 0, map, map2);
            return root;
        }

        private void dfs(TreeNode root, TreeNode parent, int depth, HashMap<TreeNode, Integer> map,
                HashMap<Integer, Integer> map2) {
            if (root != null) {
                if (parent != null) {
                    map.put(parent, map.getOrDefault(parent, 0) + root.val);
                }
                map2.put(depth, map2.getOrDefault(depth, 0) + root.val);
                dfs(root.left, root, depth + 1, map, map2);
                dfs(root.right, root, depth + 1, map, map2);
            }
        }

        private void dfs2(TreeNode root, TreeNode parent, int depth, HashMap<TreeNode, Integer> map,
                HashMap<Integer, Integer> map2) {
            if (root != null) {
                root.val = parent == null ? 0 : map2.get(depth) - map.get(parent);
                dfs2(root.left, root, depth + 1, map, map2);
                dfs2(root.right, root, depth + 1, map, map2);
            }
        }
    }

    // 2642. Design Graph With Shortest Path Calculator - HARD
    class Graph extends HashMap<Integer, ArrayList<int[]>> {
        public Graph(int n, int[][] edges) {
            for (int[] edge : edges) {
                addEdge(edge);
            }
        }

        public void addEdge(int[] edge) {
            computeIfAbsent(edge[0], t -> new ArrayList<>()).add(edge);
        }

        public int shortestPath(int node1, int node2) {
            PriorityQueue<int[]> queue = new PriorityQueue<>((o, p) -> o[0] - p[0]);
            queue.offer(new int[] { 0, node1 });
            for (HashSet<Integer> set = new HashSet<>(); !queue.isEmpty();) {
                int[] poll = queue.poll();
                if (poll[1] == node2) {
                    return poll[0];
                } else if (!set.contains(poll[1])) {
                    set.add(poll[1]);
                    for (int[] i : getOrDefault(poll[1], new ArrayList<>())) {
                        queue.offer(new int[] { poll[0] + i[2], i[1] });
                    }
                }
            }
            return -1;
        }
    }

    // 2643. Row With Maximum Ones - EASY
    // 2644. Find the Maximum Divisibility Score - EASY
    // 2645. Minimum Additions to Make Valid String - MEDIUM
    class Solution2645a {
        // 1ms
        public int addMinimum(String word) {
            int count = 0, i = 0;
            for (int j = 0; j < word.length(); i++) {
                if (word.charAt(j) == i % 3 + 'a') {
                    j++;
                } else {
                    count++;
                }
            }
            return count + (3 - i % 3) % 3;
        }
    }

    // 2646. Minimize the Total Price of the Trips - HARD
    class Solution2646a {
        // 11ms
        public int minimumTotalPrice(int n, int[][] edges, int[] price, int[][] trips) {
            HashMap<Integer, ArrayList<Integer>> g = new HashMap<>();
            for (int[] edge : edges) {
                g.computeIfAbsent(edge[0], t -> new ArrayList<>()).add(edge[1]);
                g.computeIfAbsent(edge[1], t -> new ArrayList<>()).add(edge[0]);
            }
            int[] count = new int[n];
            for (int[] trip : trips) {
                dfs(trip[0], -1, trip[1], count, g);
            }
            int[] result = dfs2(0, -1, price, count, g);
            return Math.min(result[0], result[1]);
        }

        private boolean dfs(int u, int p, int end, int[] count, HashMap<Integer, ArrayList<Integer>> g) {
            if (u == end) {
                count[u]++;
                return true;
            }
            for (int v : g.getOrDefault(u, new ArrayList<>())) {
                if (v != p && dfs(v, u, end, count, g)) {
                    count[u]++;
                    return true;
                }
            }
            return false;
        }

        private int[] dfs2(int u, int p, int[] count, int[] price, HashMap<Integer, ArrayList<Integer>> g) {
            int[] result = { price[u] * count[u], price[u] * count[u] / 2 };
            for (int v : g.getOrDefault(u, new ArrayList<>())) {
                if (v != p) {
                    int[] next = dfs2(v, u, count, price, g);
                    result = new int[] { result[0] + Math.min(next[0], next[1]), result[1] + next[0] };
                }
            }
            return result;
        }
    }

    class Solution2646b {
        // 13ms
        List<Integer>[] g;
        int[] count;
        int[] prices;

        @SuppressWarnings("unchecked")
        public int minimumTotalPrice(int n, int[][] edges, int[] price, int[][] trips) {
            g = new List[n];
            count = new int[n];
            this.prices = price;
            for (int i = 0; i < n; i++) {
                g[i] = new ArrayList<>();
            }
            for (int[] e : edges) {
                int a = e[0];
                int b = e[1];
                g[a].add(b);
                g[b].add(a);
            }
            for (int[] t : trips) {
                int a = t[0];
                int b = t[1];
                List<Integer> trace = new ArrayList<>();
                dfs(a, -1, b, trace);
                for (int x : trace) {
                    count[x]++;
                }
            }
            int[] ans = dp(0, -1);
            return Math.min(ans[0], ans[1]);
        }

        private boolean dfs(int x, int p, int end, List<Integer> trace) {
            trace.add(x);
            if (x == end) {
                return true;
            }
            for (int y : g[x]) {
                if (y == p) {
                    continue;
                }
                if (dfs(y, x, end, trace)) {
                    return true;
                }
            }
            trace.remove(trace.size() - 1);
            return false;
        }

        private int[] dp(int x, int p) {
            int[] ans = new int[] { count[x] * prices[x], count[x] * (prices[x] / 2) };
            for (int y : g[x]) {
                if (y == p) {
                    continue;
                }
                int[] sub = dp(y, x);
                ans = new int[] { ans[0] + Math.min(sub[0], sub[1]), ans[1] + sub[0] };
            }
            return ans;
        }
    }

    // 2651. Calculate Delayed Arrival Time - EASY
    // 2652. Sum Multiples - EASY
    // 2653. Sliding Subarray Beauty - MEDIUM
    class Solution2653a {
        public int[] getSubarrayBeauty(int[] nums, int k, int x) {
            int[] result = new int[nums.length - k + 1], count = new int[101];
            for (int i = 0; i < nums.length; i++) {
                count[nums[i] + 50]++;
                if (i >= k - 1) {
                    for (int j = 0, c = 0; j < 50 && c < x; j++) {
                        if ((c += count[j]) >= x) {
                            result[i - k + 1] = j - 50;
                        }
                    }
                    count[nums[i - k + 1] + 50]--;
                }
            }
            return result;
        }
    }

    // 2654. Minimum Number of Operations to Make All Array Elements Equal to 1 - MEDIUM
    class Solution2654a {
        public int minOperations(int[] nums) {
            int count = 0;
            for (int num : nums) {
                count += num == 1 ? 1 : 0;
            }
            int min = count > 0 ? nums.length - count : Integer.MAX_VALUE;
            for (int i = 0; i < nums.length; i++) {
                for (int j = i + 1, gcd = nums[i]; j < nums.length; j++) {
                    min = Math.min(min, (gcd = valueOf(gcd).gcd(valueOf(nums[j])).intValue()) > 1 ? Integer.MAX_VALUE
                            : j - i + nums.length - 1);
                }
            }
            return min == Integer.MAX_VALUE ? -1 : min;
        }
    }

    // 2670. Find the Distinct Difference Array - EASY
    // 2671. Frequency Tracker - MEDIUM
    class FrequencyTracker {
        private int[] count = new int[100001], freq = new int[200001];

        public void add(int number) {
            freq[count[number]]--;
            freq[++count[number]]++;
        }

        public void deleteOne(int number) {
            freq[count[number]]--;
            freq[count[number] -= count[number] > 0 ? 1 : 0]++;
        }

        public boolean hasFrequency(int frequency) {
            return freq[frequency] > 0;
        }
    }

    // 2672. Number of Adjacent Elements With the Same Color - MEDIUM
    class Solution2672a {
        public int[] colorTheArray(int n, int[][] queries) {
            int[] result = new int[queries.length], A = new int[n + 2];
            for (int i = 0, count = 0; i < queries.length; i++) {
                result[i] = count -= (A[queries[i][0] + 1] > 0 ? (A[queries[i][0]] == A[queries[i][0] + 1] ? 1 : 0)
                        + (A[queries[i][0] + 1] == A[queries[i][0] + 2] ? 1 : 0) : 0)
                        - (A[queries[i][0]] == (A[queries[i][0] + 1] = queries[i][1]) ? 1 : 0)
                        - (A[queries[i][0] + 1] == A[queries[i][0] + 2] ? 1 : 0);
            }
            return result;
        }
    }

    // 2673. Make Costs of Paths Equal in a Binary Tree - MEDIUM
    class Solution2673a {
        public int minIncrements(int n, int[] cost) {
            return dfs(cost, 0)[0];
        }

        private int[] dfs(int[] cost, int n) {
            if (n >= cost.length) {
                return new int[2];
            }
            int[] left = dfs(cost, 2 * n + 1), right = dfs(cost, 2 * n + 2);
            return new int[] { left[0] + right[0] + Math.abs(left[1] - right[1]),
                    cost[n] + Math.max(left[1], right[1]) };
        }
    }

    // 2678. Number of Senior Citizens - EASY
    class Solution2678a {
        public int countSeniors(String[] details) {
            return (int) Stream.of(details).filter(t -> t.matches(".*(6[1-9]|[789]\\d)..")).count();
        }
    }

    // 2679. Sum in a Matrix - MEDIUM
    class Solution2679a {
        public int matrixSum(int[][] nums) {
            int ans = 0;
            for (int[] v : nums) {
                Arrays.sort(v);
            }
            for (int i = 0; i < nums[0].length; i++) {
                int mx = 0;
                for (int j = 0; j < nums.length; j++) {
                    mx = Math.max(mx, nums[j][i]);
                }
                ans += mx;
            }
            return ans;
        }
    }

    // 2680. Maximum OR - MEDIUM
    class Solution2680a {
        public long maximumOr(int[] nums, int k) {
            long right[] = new long[nums.length], left = 0, ans = 0;
            for (int i = nums.length - 2; i >= 0; i--) {
                right[i] = right[i + 1] | nums[i + 1];
            }
            for (int i = 0; i < nums.length; left |= nums[i++]) {
                ans = Math.max(ans, left | right[i] | (long) nums[i] << k);
            }
            return ans;
        }
    }

    // 2681. Power of Heroes - HARD
    class Solution2681a {
        public int sumOfPower(int[] nums) {
            Arrays.sort(nums);
            long result = 0, sum = 0;
            for (long v : nums) {
                result = (result + v * v % 1000000007 * (sum + v)) % 1000000007;
                sum = (sum * 2 + v) % 1000000007;
            }
            return (int) result;
        }
    }

    // 2682. Find the Losers of the Circular Game - EASY
    class Solution2682a {
        public int[] circularGameLosers(int n, int k) {
            TreeSet<Integer> set = new TreeSet<>();
            for (int i = 1; i <= n; i++) {
                set.add(i);
            }
            for (int i = 0, j = 1; set.remove(i + 1); i = (i + j++ * k) % n) {
            }
            return set.stream().mapToInt(v -> v).toArray();
        }
    }

    // 2683. Neighboring Bitwise XOR - MEDIUM
    class Solution2683a {
        public boolean doesValidArrayExist(int[] derived) {
            int xor = 0;
            for (int i : derived) {
                xor ^= i;
            }
            return xor == 0;
        }
    }

    // 2684. Maximum Number of Moves in a Grid - MEDIUM
    class Solution2684a {
        public int maxMoves(int[][] grid) {
            int m = grid.length, n = grid[0].length, f[][] = new int[m][n], ans = 0;
            for (int j = 1; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    ans = (f[i][j] = Math.max(i > 0 && grid[i][j] > grid[i - 1][j - 1] ? f[i - 1][j - 1] + 1 : 0,
                            Math.max(i < m - 1 && grid[i][j] > grid[i + 1][j - 1] ? f[i + 1][j - 1] + 1 : 0,
                                    grid[i][j] > grid[i][j - 1] ? f[i][j - 1] + 1 : 0))) == j ? j : ans;
                }
            }
            return ans;
        }
    }

    // 2685. Count the Number of Complete Components - MEDIUM
    class Solution2685a {
        @SuppressWarnings("unchecked")
        public int countCompleteComponents(int n, int[][] edges) {
            ArrayList<Integer>[] g = new ArrayList[n];
            for (int i = 0; i < n; i++) {
                g[i] = new ArrayList<>();
            }
            for (int[] e : edges) {
                g[e[0]].add(e[1]);
                g[e[1]].add(e[0]);
            }
            int count = 0, vis[] = new int[n];
            for (int i = 0; i < n; i++) {
                if (vis[i] == 0) {
                    int[] next = dfs(i, vis, g);
                    count += next[0] * (next[0] - 1) == next[1] ? 1 : 0;
                }
            }
            return count;
        }

        private int[] dfs(int n, int[] vis, ArrayList<Integer>[] g) {
            if (vis[n] == 1) {
                return new int[2];
            }
            vis[n] = 1;
            int[] res = { 1, g[n].size() };
            for (int i : g[n]) {
                int[] next = dfs(i, vis, g);
                res = new int[] { res[0] + next[0], res[1] + next[1] };
            }
            return res;
        }
    }

    // 2696. Minimum String Length After Removing Substrings - EASY
    class Solution2696a {
        public int minLength(String s) {
            for (int i = 0; i < 50; i++) {
                s = s.replaceAll("AB|CD", "");
            }
            return s.length();
        }
    }

    // 2697. Lexicographically Smallest Palindrome - EASY
    class Solution2697a {
        public String makeSmallestPalindrome(String s) {
            char[] c = new char[s.length()];
            for (int i = 0; i < s.length(); i++) {
                c[i] = (char) Math.min(s.charAt(i), s.charAt(s.length() - 1 - i));
            }
            return new String(c);
        }
    }

    // 2698. Find the Punishment Number of an Integer - MEDIUM
    class Solution2698a {
        public int punishmentNumber(int n) {
            int ans = 0;
            for (int i = 1; i <= n; i++) {
                ans += punishmentNumber(0, "" + i * i, i) ? i * i : 0;
            }
            return ans;
        }

        private boolean punishmentNumber(int i, String s, int n) {
            if (i == s.length()) {
                return n == 0;
            }
            for (int j = i + 1; j <= s.length(); j++) {
                if (punishmentNumber(j, s, n - Integer.parseInt(s.substring(i, j)))) {
                    return true;
                }
            }
            return false;
        }
    }

    // 2699. Modify Graph Edge Weights - HARD
    class Solution2699a {
        public int[][] modifiedGraphEdges(int n, int[][] edges, int source, int destination, int target) {
            HashMap<Integer, ArrayList<int[]>> map = new HashMap<>();
            for (int[] edge : edges) {
                map.computeIfAbsent(edge[0], t -> new ArrayList<>()).add(edge);
                map.computeIfAbsent(edge[1], t -> new ArrayList<>()).add(edge);
            }
            if (modifiedGraphEdges(n, source, destination, target, map, false) < target
                    || modifiedGraphEdges(n, source, destination, target, map, true) > target) {
                return new int[0][];
            }
            for (int[] edge : edges) {
                edge[2] = edge[2] > 0 ? edge[2] : 2000000000;
            }
            return edges;
        }

        private int modifiedGraphEdges(int n, int source, int destination, int target,
                HashMap<Integer, ArrayList<int[]>> map, boolean flag) {
            int[] start = { 0, source }, visited[] = new int[n * n][n];
            HashMap<int[], ArrayList<int[]>> edges = new HashMap<>(Map.of(start, new ArrayList<>()));
            PriorityQueue<int[]> queue = new PriorityQueue<>(
                    (o, p) -> edges.get(o).size() == edges.get(p).size() ? o[0] - p[0]
                            : edges.get(o).size() - edges.get(p).size());
            for (queue.offer(start); !queue.isEmpty();) {
                int[] poll = queue.poll();
                if (poll[0] <= target) {
                    if (poll[1] == destination) {
                        for (int i = 0; i < edges.get(poll).size(); i++) {
                            edges.get(poll).get(i)[2] = i > 0 ? 1 : target - poll[0] + 1;
                        }
                        return poll[0];
                    } else if (visited[edges.get(poll).size()][poll[1]] == 0) {
                        visited[edges.get(poll).size()][poll[1]] = 1;
                        for (int[] edge : map.getOrDefault(poll[1], new ArrayList<>())) {
                            if (edge[2] > 0 || flag) {
                                int[] next = { poll[0] + Math.max(1, edge[2]), edge[0] == poll[1] ? edge[1] : edge[0] };
                                edges.put(next, new ArrayList<>(edges.get(poll)));
                                if (edge[2] < 0) {
                                    edges.get(next).add(edge);
                                }
                                queue.offer(next);
                            }
                        }
                    }
                }
            }
            return Integer.MAX_VALUE;
        }
    }
}
