/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
function TreeNode(val, left, right) {
    this.val = (val === undefined ? 0 : val)
    this.left = (left === undefined ? null : left)
    this.right = (right === undefined ? null : right)
}

// 102. Binary Tree Level Order Traversal - medium
/**
 * @param {TreeNode} root
 * @return {number[][]}
 */
var levelOrder = function (root) {
    let ans = [];
    if (!root) {
        return ans;
    }
    let q = [];
    q.push(root);
    while (q.length !== 0) {
        let arr = [];
        let size = q.length;
        for (let i = 0; i < size; i++) {
            let n = q.shift();
            arr.push(n.val)
            if (n.left)
                q.push(n.left);
            if (n.right)
                q.push(n.right);
        }
        ans.push(arr);
    }
    return ans;
};