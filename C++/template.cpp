// From 2426. Number of Pairs Satisfying Inequality
// 树状数组模版
class TreeArray {
private:
    int minv, maxv, N;
    long long *tr;
    long long all;
    
public:
    // 给定定义域 [minv, maxv], 初始化一个树状数组
    TreeArray(int minv, int maxv) :
        minv(minv), maxv(maxv), N(maxv - minv + 1), all(0) {
        if(minv > maxv) {
            cout << "TreeArray Init: Value Range Invalid!" << endl;
        } else {
            this->maxv = maxv, this->
            N = maxv - minv + 2;
            tr = new long long[N]();
        }
    }
    
    ~TreeArray() {
        delete[] tr;
    }
    
    void add(int x, long long v) {
        if(x < minv || x > maxv) {
            std::cout << "Add: Index Invalid!" << std::endl;
        } else {
            all += v;
            x = x - minv + 1;
            while(x < N) {
                tr[x] += v;
                x += x & -x;
            }
        }
    }
    
    // 求小于等于 x 的值的和
    // 如果求小于，那么用 query_LE(x-1)
    long long query_LE(int x) {
        x = max(0, min(x - minv + 1, N-1));
        long long ret = 0;
        while(x) {
            ret += tr[x];
            x -= x & -x;
        }
        return ret;
    }

    // 求大于等于 x 的值的和
    // 如果求大于，那么用 query_BE(x+1)
    long long query_BE(int x) {
        x = max(0, min(x - minv + 1, N-1));
        return all - query_LE(x-1);
    }
};

class Solution {
public:
    long long numberOfPairs(vector<int>& nums1, vector<int>& nums2, int diff) {    
        TreeArray tr(-20000, 20000);
        
        long long res = 0;
        for(int i = 0; i < nums1.size(); ++i) {
            res += tr.query_LE(nums1[i] - nums2[i] + diff);
            tr.add(nums1[i] - nums2[i], 1);
        }
        
        return res;
    }
};
