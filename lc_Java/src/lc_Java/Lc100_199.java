package lc_Java;

// List / Array
import java.util.Arrays;
import java.util.ArrayList;
import java.util.LinkedList;
// Queue
//import java.util.Stack;
import java.util.Deque;
import java.util.Queue;
import java.util.ArrayDeque;
import java.util.PriorityQueue;
// Map / Set
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeMap;
import java.util.LinkedHashMap;
import java.util.Map;

public class Lc100_199 {
	// 136. Single Number - E
	public int singleNumber(int[] nums) {
		for (int i = 0; i < nums.length - 1; i++) {
			nums[i + 1] ^= nums[i];
		}
		return nums[nums.length - 1];
	}

	public int singleNumber2(int[] nums) {
		int x = 0;
		for (int n : nums) {
			x ^= n;
		}
		return x;
	}

	// 169. Majority Element - E
	// O(n) / O(1)
	public int majorityElement(int[] nums) {
		int ans = 0;
		int cnt = 0;
		for (int n : nums) {
			if (cnt == 0) {
				ans = n;
			}
//			if (ans == n) {
//				cnt++;
//			} else {
//				cnt--;
//			}
			cnt = ans == n ? cnt + 1 : cnt - 1;
		}
		return ans;
	}

	public int majorityElement2(int[] nums) {
		Map<Integer, Integer> cnt = new HashMap<Integer, Integer>();
		for (int n : nums) {
			if (!cnt.containsKey(n)) {
				cnt.put(n, 1);
			} else {
				cnt.put(n, cnt.get(n) + 1);
			}
		}
//		int ans = 0;
//		int c = 0;
//		for (Map.Entry<Integer, Integer> e : cnt.entrySet()) {
//			if (e.getValue() > c) {
//				ans = e.getKey();
//				c = e.getValue();
//			}
//		}
		for (int n : nums) {
			if (cnt.get(n) > nums.length / 2)
				return n;
		}

		return -1;
	}

	// O(n * logn) / O(logn)
	public int majorityElement3(int[] nums) {
		Arrays.sort(nums);
		return nums[nums.length / 2];
	}
}
