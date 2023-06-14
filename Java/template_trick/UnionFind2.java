package template_trick;

import java.util.*;

public class UnionFind2 {
    class UnionFind {
        private int part;
        private int[] p;
        private int[] sz;

        public UnionFind(int n) {
            this.part = n;
            p = new int[n];
            sz = new int[n];
            for (int i = 0; i < n; i++) {
                p[i] = i;
                sz[i] = 1;
            }
            Arrays.fill(sz, 1);
        }

        public void union(int x, int y) {
            int fx = find(x);
            int fy = find(y);
            if (fx == fy)
                return;

            // 平衡性优化
            if (sz[fx] < sz[fy]) {
                p[fx] = fy;
                sz[fy] += sz[fx];
            } else {
                p[fy] = fx;
                sz[fx] += sz[fy];
            }

            this.part--;
        }

        public boolean isConnected(int x, int y) {
            return find(x) == find(y);
        }

        public int part() {
            return this.part;
        }

        public int find(int x) {
            // 路径压缩
            if (p[x] != x) {
                p[x] = find(p[x]);
            }
            return p[x];
        }
    }
}
