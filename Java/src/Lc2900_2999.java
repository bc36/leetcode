package src;

public class Lc2900_2999 {
    // 2917. Find the K-or of an Array - EASY
    class Solution {
        public int findKOr(int[] nums, int k) {
            int result = 0;
            for (int i = 0; i < 32; i++) {
                int count = 0;
                for (int num : nums) {
                    count += (num >> i) % 2 > 0 ? 1 : 0;
                }
                if (count >= k) {
                    result |= 1 << i;
                }
            }
            return result;
        }
    }
}
