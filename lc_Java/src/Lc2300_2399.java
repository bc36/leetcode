package src;

import java.util.*;

public class Lc2300_2399 {
    // 2325. Decode the Message - E
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

    // 2363. Merge Similar Items - E
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

    public List<List<Integer>> mergeSimilarItems3(int[][] items1, int[][] items2) {
        // 8ms
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

    public List<List<Integer>> mergeSimilarItems4(int[][] items1, int[][] items2) {
        // 9ms
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

    public List<List<Integer>> mergeSimilarItems5(int[][] items1, int[][] items2) {
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

    public List<List<Integer>> mergeSimilarItems2(int[][] items1, int[][] items2) {
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
