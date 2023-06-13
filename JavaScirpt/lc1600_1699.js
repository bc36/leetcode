// 1696. Jump Game VI - medium
/**
 * @param {number[]} nums
 * @param {number} k
 * @return {number}
 */
var maxResult = function (nums, k) {
    let q = [[nums[0], 0]];
    let ans = nums[0];
    for (let i = 1; i < nums.length; i++) {
        while (i - q[0][1] > k) {
            q.shift();
        }
        ans = q[0][0] + nums[i];
        while (q.length > 0 && ans >= q[q.length - 1][0]) {
            q.pop();
        }
        q.push([ans, i]);
    }
    return ans;
};