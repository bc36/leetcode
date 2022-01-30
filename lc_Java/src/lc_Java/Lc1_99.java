package lc_Java;

//List / Array
import java.util.Arrays;
import java.util.ArrayList;
//import java.util.LinkedList;
import java.util.List;
//Queue
//import java.util.Stack;
//import java.util.Deque;
//import java.util.Queue;
//import java.util.ArrayDeque;
//import java.util.PriorityQueue;
//Map / Set
//import java.util.HashMap;
//import java.util.HashSet;
//import java.util.TreeMap;
//import java.util.LinkedHashMap;
//import java.util.Map;

public class Lc1_99 {
	// 15. 3Sum - M
	public List<List<Integer>> threeSum(int[] nums) {
		List<List<Integer>> ans = new ArrayList<>();
		if (nums == null || nums.length < 3) {
			return ans;
		}
		Arrays.sort(nums);
		for (int i = 0; i < nums.length - 2; i++) {
			if (nums[i] > 0)
				break;
			if (i > 0 && nums[i] == nums[i - 1])
				continue;
			int target = -nums[i];
			int left = i + 1, right = nums.length - 1;
			while (left < right) {
				if (nums[left] + nums[right] == target) {
					ans.add(Arrays.asList(nums[i], nums[left], nums[right]));
					left++;
					right--;
					while (left < right && nums[left] == nums[left - 1]) {
						left++;
					}
					while (left < right && nums[right] == nums[right + 1]) {
						right--;
					}
				} else if (nums[left] + nums[right] < target) {
					left++;
				} else {
					right--;
				}
			}
		}
		return ans;
	}
}
