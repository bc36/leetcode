package template_trick;

// 区间更新, 区间求和 LC 1109 https://leetcode.cn/problems/corporate-flight-bookings/description/
class Solution {
    public int[] corpFlightBookings(int[][] bookings, int n) {
        SegmentTree2 st = new SegmentTree2(n);
        st.build(1, 1, n); // 只初始化点, 不赋值
        for (int[] b : bookings) {
            st.update(1, b[0], b[1], b[2]);
        }
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            ans[i] = st.query(1, i + 1, i + 1);
        }
        return ans;
    }
}

/**
 * 区间更新, 区间求和, 下标 1 开始, 维护 [1, n] 之间信息
 */
public class SegmentTree2 {
    // int N = 20009;
    // Node[] t = new Node[N * 4];

    // 对比直接开满空间, 性能区别不大
    Node[] t = null;

    SegmentTree2(int n) {
        t = new Node[4 * n];
    }

    class Node {
        int l, r, v, lazy;

        Node(int _l, int _r) {
            l = _l;
            r = _r;
        }
    }

    void build(int o, int L, int R) {
        t[o] = new Node(L, R);
        if (L != R) {
            int m = L + R >> 1;
            build(o << 1, L, m);
            build(o << 1 | 1, m + 1, R);
        }
    }

    void pushup(int o) {
        t[o].v = t[o << 1].v + t[o << 1 | 1].v;
    }

    /**
     * 注意与 {@link SegmentTree1#pushdown(int)} 的区别
     */
    void pushdown(int o, int cnt) {
        if (t[o].lazy != 0) {
            t[o << 1].v += t[o].lazy * (cnt - cnt / 2);
            t[o << 1 | 1].v += t[o].lazy * (cnt / 2);
            t[o << 1].lazy += t[o].lazy;
            t[o << 1 | 1].lazy += t[o].lazy;
            t[o].lazy = 0;
        }
    }

    void update(int o, int L, int R, int v) {
        if (L <= t[o].l && t[o].r <= R) {
            t[o].v += v * (t[o].r - t[o].l + 1);
            t[o].lazy += v;
            return;
        }
        pushdown(o, t[o].r - t[o].l + 1);
        int m = t[o].l + t[o].r >> 1;
        if (L <= m)
            update(o << 1, L, R, v);
        if (R > m)
            update(o << 1 | 1, L, R, v);
        pushup(o);
        return;
    }

    int query(int o, int L, int R) {
        if (L <= t[o].l && t[o].r <= R) {
            return t[o].v;
        }
        pushdown(o, t[o].r - t[o].l + 1);
        int m = t[o].l + t[o].r >> 1;
        int res = 0;
        if (L <= m)
            res += query(o << 1, L, R);
        if (R > m)
            res += query(o << 1 | 1, L, R);
        return res;
    }
}
