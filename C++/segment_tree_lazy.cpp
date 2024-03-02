class Solution {
public:
  struct Node {
    long long sum, lazy;
  };
  vector<Node> t;
  vector<long long> A;
  void build(int id, int l, int r, int root) {
    t[root].lazy = 0;
    if (l == r) {
      t[root].sum = ((A[l] >> id) & 1);
      return;
    }
    build(id, l, ((l + r) >> 1), root << 1);
    build(id, ((l + r) >> 1) + 1, r, root << 1 | 1);
    t[root].sum = t[root << 1].sum + t[root << 1 | 1].sum;
    return;
  }
  void pushdown(int id, int l, int r, int root) {
    if (t[root].lazy == 1) {
      t[root].lazy = 0;
      t[root << 1].lazy ^= 1;
      t[root << 1 | 1].lazy ^= 1;
      t[root << 1].sum = (((l + r) >> 1) - l + 1) - t[root << 1].sum;
      t[root << 1 | 1].sum = (r - ((l + r) >> 1)) - t[root << 1 | 1].sum;
    }
    return;
  }
  long long query(int id, int l, int r, int L, int R, int root) {
    if (l >= L && r <= R) {
      return t[root].sum;
    }
    pushdown(id, l, r, root);
    long long ans = 0;
    if (L <= ((l + r) >> 1))
      ans += query(id, l, ((l + r) >> 1), L, R, root << 1);
    if (R > ((l + r) >> 1))
      ans += query(id, ((l + r) >> 1) + 1, r, L, R, root << 1 | 1);
    return ans;
  }
  void modify(int id, int l, int r, int L, int R, int root) {
    if (l >= L && r <= R) {
      t[root].sum = (r - l + 1) - t[root].sum;
      t[root].lazy ^= 1;
      return;
    }
    pushdown(id, l, r, root);
    if (L <= ((l + r) >> 1))
      modify(id, l, ((l + r) >> 1), L, R, root << 1);
    if (R > ((l + r) >> 1))
      modify(id, ((l + r) >> 1) + 1, r, L, R, root << 1 | 1);
    t[root].sum = t[root << 1].sum + t[root << 1 | 1].sum;
    return;
  }
  vector<long long> handleQuery(vector<int> &nums1, vector<int> &nums2,
                                vector<vector<int>> &Q) {
    vector<long long> res;
    long long c = 0;
    for (int i : nums2)
      c += i;
    A.clear();
    A.resize(nums1.size() + 1);
    t.clear();
    t.resize(nums1.size() * 4 + 100);
    for (int i = 1; i <= (int)nums1.size(); i++)
      A[i] = nums1[i - 1];
    build(0, 1, nums1.size(), 1);
    for (auto i : Q) {
      if (i[0] == 1)
        modify(0, 1, nums1.size(), i[1] + 1, i[2] + 1, 1);
      if (i[0] == 2)
        c += 1ll * query(0, 1, nums1.size(), 1, nums1.size(), 1) * i[1];
      if (i[0] == 3)
        res.push_back(c);
    }
    return res;
  }
};