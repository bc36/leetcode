package src;

public class Lc600_699 {
    // 617. Merge Two Binary Trees - EASY
    class Solution617a {
        public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
            if (root1 != null && root2 != null) {
                root1.val += root2.val;
                root1.left = mergeTrees(root1.left, root2.left);
                root1.right = mergeTrees(root1.right, root2.right);
            }
            return root1 != null ? root1 : root2;
        }
    }
}
