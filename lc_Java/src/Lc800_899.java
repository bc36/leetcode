package src;

import java.util.*;

public class Lc800_899 {
	public class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode() {
		}

		TreeNode(int val) {
			this.val = val;
		}

		TreeNode(int val, TreeNode left, TreeNode right) {
			this.val = val;
			this.left = left;
			this.right = right;
		}
	}

	/**
	 * all e in a[:i] have e < x, and all e in a[i:] have e >= x.
	 * @param nums
	 * @param x
	 * @return position <code> i <code>
	 */
	private int lowerBound(int[] nums, int x) {
		int l = 0, r = nums.length;
		while (l < r) {
			int m = (l + r) >> 1;
			if (nums[m] < x)
				l = m + 1;
			else
				r = m;
		}
		return l;
	}

	/**
	 * all e in a[:i] have e <= x, and all e in a[i:] have e > x.
	 * @param nums
	 * @param x
	 * @return position <code> i <code>
	 */
	private int upperBound(int[] nums, int x) {
		int l = 0, r = nums.length;
		while (l < r) {
			int m = (l + r) >> 1;
			if (nums[m] > x)
				r = m;
			else
				l = m + 1;
		}
		return l;
	}

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
