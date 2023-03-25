package template_trick;

import java.util.*;

// 二维转一维:
// 初始化 r * c
// (x, y) -> x * c + y

public class UnionFind1 {
    class UnionFind {
        // 记录每个节点的父节点
        private Map<Integer, Integer> p;

        public UnionFind(int[] nums) {
            p = new HashMap<>();
            for (int v : nums) {
                p.put(v, v);
            }
        }

        // 寻找x的父节点, 实际上也就是x的最远连续右边界, 这点类似于方法2
        public Integer find(int x) {
            // nums不包含x
            if (!p.containsKey(x)) {
                return null;
            }
            // 遍历找到x的父节点
            while (x != p.get(x)) {
                // 进行路径压缩, 不写下面这行也可以, 但是时间会慢些
                p.put(x, p.get(p.get(x)));
                x = p.get(x);
            }
            return x;
        }

        // 合并两个连通分量, 在本题中只用来将num并入到num+1的连续区间中
        public void union(int x, int y) {
            int fx = find(x);
            int fy = find(y);
            if (fx == fy) {
                return;
            }
            p.put(fx, fy);
        }
    }
}