class Solution {
public:
    bool primeSubOperation(vector<int>& nums) {
        int n = nums.size();
        int mx = 0;

        // 筛法求质数
        bool flag[mx + 1];
        memset(flag, 0, sizeof(flag));
        for (int i = 2; i <= mx; i++) if (!flag[i]) for (int j = i * 2; j <= mx; j += i) flag[j] = true;
        vector<int> prime;
        for (int i = 2; i <= mx; i++) if (!flag[i]) prime.push_back(i);

        return true;
    }
};
