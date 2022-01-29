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
//import java.util.Map;

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

	// O(n * logn) / O(logn)
	public int majorityElement2(int[] nums) {
		Arrays.sort(nums);
		return nums[nums.length / 2];
	}
}
