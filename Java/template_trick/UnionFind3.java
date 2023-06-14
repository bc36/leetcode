package template_trick;

public class UnionFind3 {
    class UnionFind {
        private int[] p;

        public UnionFind(int n) {
            p = new int[n];
            for (int i = 0; i < n; i++) {
                p[i] = i;
            }
        }

        public void union(int x, int y) {
            int fx = find(x);
            int fy = find(y);
            if (fx != fy) {
                p[fx] = fy; // x's root = y
            }
        }

        public int find(int x) {
            // 没有压缩
            while (x != p[x]) {
                p[x] = p[p[x]];
                x = p[x];
            }
            return x;

            // slow?
            // if (p[x] != x) {
            //     p[x] = find(p[x]);
            // }
            // return p[x];

        }

        public boolean isConnected(int x, int y) {
            return find(x) == find(y);
        }
    }
}
