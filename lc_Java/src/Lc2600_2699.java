package src;

import java.util.*;
import java.math.BigInteger;
import java.util.function.Function;

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

    // 2607. Make K-Subarray Sums Equal - M
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

    // 2608. Shortest Cycle in a Graph - H
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
}
