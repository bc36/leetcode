"""euler - sieve of Euler - 欧拉筛/线性筛

数论, 筛法 sieve method

埃氏筛法仍有优化空间, 它会将一个合数重复多次标记, 比如 12 被 2, 3 同时划掉

每一个合数都是被最小的质因子筛掉 -> 时间复杂度 O(n)

2 划掉 4(乘2)
3 划掉 6(乘2), 9(乘3)
4 划掉 8(乘2), 不能划掉 12, 因为 3 已经超过了 4 的最小质因子(2)
5 划掉 10(乘2), 15(乘3), 25(乘5)

每个数 x, 乘以 <= lpf[x] 的质数, lpf[x] 为 x 的最小的质因子

因为取模操作是算术操作中最慢的, 数据范围小时, 不如埃氏筛快
"""


from typing import List


def euler(n: int) -> List[int]:
    """[2, x] 内的质数"""
    primes = []
    isPrime = [True] * (n + 1)
    for i in range(2, n + 1):
        if isPrime[i]:
            primes.append(i)
        for p in primes:
            if i * p >= n:
                break
            isPrime[i * p] = False
            if i % p == 0:  # p 是 lpf[i]
                break
    return primes


primes = euler(10**6)
