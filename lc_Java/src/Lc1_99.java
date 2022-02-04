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

	// 48. Rotate Image - E
	public void rotate(int[][] matrix) {
		int n = matrix.length;
		for (int i = 0; i < n / 2; i++) {
			for (int j = 0; j < n; j++) {
				int tmp = matrix[i][j];
				matrix[i][j] = matrix[n - 1 - i][j];
				matrix[n - 1 - i][j] = tmp;
			}
		}
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < i; j++) {
				int tmp = matrix[i][j];
				matrix[i][j] = matrix[j][i];
				matrix[j][i] = tmp;
			}
		}
		return;
	}

	// 56. Merge Intervals - M
	public int[][] merge(int[][] intervals) {
		List<int[]> ans = new ArrayList<>();
		Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
//		int left = intervals[0][0];
//		int right = intervals[0][1];
//		for (int i = 1; i < intervals.length; i++) {
//			if (intervals[i][0] <= right) {
//				right = Math.max(right, intervals[i][1]);
//			} else {
//				ans.add(new int[] { left, right });
//				left = intervals[i][0];
//				right = intervals[i][1];
//			}
//		}
//		ans.add(new int[] { left, right });

		int[] newInterval = intervals[0];
		ans.add(newInterval);
		for (int[] interval : intervals) {
			if (interval[0] <= newInterval[1]) {
				newInterval[1] = Math.max(newInterval[1], interval[1]);
			} else {
				newInterval = interval;
				ans.add(newInterval);
			}
		}
		return ans.toArray(new int[0][]);
	}
}
