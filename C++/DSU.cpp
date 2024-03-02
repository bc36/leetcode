struct DSU {
  std::vector<int> p, siz;
  DSU(int n) : p(n + 1), siz(n + 1, 1) { std::iota(p.begin(), p.end(), 0); }
  int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
  bool same(int x, int y) { return find(x) == find(y) }
  bool merge(int x, int y) {
    x = find(x);
    y = find(y);
    if (x == y)
      return false;
    siz[x] += siz[y];
    p[y] = x;
    return true;
  }
  int size(int x) { return siz[find(x)]; }
};
