package src;

import java.util.*;

public class Lc800_899 {
	// 884. Uncommon Words from Two Sentences - E
	public String[] uncommonFromSentences(String s1, String s2) {
		HashMap<String, Integer> m = new HashMap<String, Integer>();
		String[] arr1 = s1.split(" ");
		for (String w : arr1) {
			m.put(w, m.getOrDefault(w, 0) + 1);
		}
		String[] arr2 = s2.split(" ");
		for (String w : arr2) {
			m.put(w, m.getOrDefault(w, 0) + 1);
		}
		ArrayList<String> ans = new ArrayList<String>();
		for (Map.Entry<String, Integer> e : m.entrySet()) {
			if (e.getValue() == 1) {
				ans.add(e.getKey());
			}
		}
		return ans.toArray(new String[0]);
	}
}
