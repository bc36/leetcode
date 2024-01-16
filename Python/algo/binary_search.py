"""binary search

https://codeforces.com/blog/entry/75879

Suppose you have a predicate P(n) which goes from being false to being true as 'n' increases, and you want to find the least 'n' for which it is true.
There are two things to remember so you never get a binary search wrong:

1) Remember the invariant you are maintaining!
At the end, you'll have l = r, P(l-1) false and P(l) true,
so a good invariant is to say that P(l-1) should always be false and P(r) should always be true.
With this, you can initialize the variables appropriately.

Now let's look at the iteration steps:

    while (l < r) {
        int mid = (l+r)/2;
        if (P(mid))
            r = mid; // Note that P(r) = P(mid) is true, so the invariant is maintained.
        else
            l = mid+1; // Note that P(l-1) = P(mid+1-1) is false, so the invariant is maintained.
    }

2) Both updates must decrease the length of the interval [l,r], and we must round up or down to ensure that.
Let's check the above code is correct:
Since l < r, we have that (as real numbers) l < (l+r)/2 < r, and therefore l <= (l+r)/2 < r after rounding down.
Therefore, r = mid decreases 'r' and l = mid + 1 increases 'l'.

Let's do the same for a predicate P(n) that goes from being true to being false as 'n' increases.
Suppose we want to find the largest 'n' for which P(n) is true.
Then at the end, we will have l = r, P(l) true and P(l+1) false.
Therefore, the invariant we will maintain is that P(l) should always be true and P(r+1) should always be false.

How does the code look like in this case?

    while (l < r) {
        int mid = ????; # (l + r + 1) / 2
        if (P(mid))
            l = mid; // Note that P(l) = P(mid) is true, so the invariant is maintained.
        else
            r = mid-1; // Note that P(r+1) = P(mid-1+1) is false, so the invariant is maintained.
    }

Now, it is still true that (as real numbers) l < (l+r)/2 < r.
But if we want l = mid to increase 'l', then we cannot round the division down.
Rounding it up (by doing (l+r+1)/2) is fine, because then l < (l+r+1)/2 <= r, and therefore r = mid - 1 decreases 'r' and l = mid increases 'l'.


See another way of maximizing the index
LC 3007, https://leetcode.com/problems/maximum-number-that-sum-of-the-prices-is-less-than-or-equal-to-k/
https://www.bilibili.com/video/BV1zt4y1R7Tc

    while l + 1 < r:
        if P(mid):
            l = m
        else:
            r = m
    return l


try it on
lc 1552 https://leetcode.cn/problems/magnetic-force-between-two-balls/
lc 1802 https://leetcode.cn/problems/maximum-value-at-a-given-index-in-a-bounded-array/
lc 2517 https://leetcode.cn/problems/maximum-tastiness-of-candy-basket/

another type:
lc 2560 https://leetcode.cn/problems/house-robber-iv/

which type?
lc 875  https://leetcode.cn/problems/koko-eating-bananas/
lc 1870 https://leetcode.cn/problems/minimum-speed-to-arrive-on-time/
lc 2187 https://leetcode.cn/problems/minimum-time-to-complete-trips/
lc 2226 https://leetcode.cn/problems/maximum-candies-allocated-to-k-children/
"""

check = lambda x: x


def a():
    """最小化最大值 / 左侧第一个满足条件的值"""
    while l < r:
        m = l + r >> 1
        if check(m):
            r = m
        else:
            l = m + 1
    return l


def b():
    """最小值最大化 / 左侧最后一个满足条件的值"""
    while l < r:
        m = l + r + 1 >> 1
        if check(m):
            l = m
        else:
            r = m - 1
    return l
