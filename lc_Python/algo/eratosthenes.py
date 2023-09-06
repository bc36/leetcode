"""eratosthenes - sieve of Eratosthenes - 埃式筛

数论, 筛法 sieve method

枚举质数, 划掉质数的倍数

时间复杂度: O(n * loglogn), n * (1/2 + 1/3 + 1/5 + ...) -> n * 素数倒数之和 -> n * O(loglogn)
空间复杂度: O(n / logn), [2, n] 范围内素数个数 
"""

from typing import List


def eratosthenes(n: int) -> List[int]:
    """[2, x] 内的质数"""
    primes = []
    isPrime = [True] * (n + 1)
    for i in range(2, n + 1):
        if isPrime[i]:
            primes.append(i)
            for j in range(i * i, n + 1, i):  # 注意是 *, 不是 +, 比 i 小的 i 的倍数已经被枚举过了
                isPrime[j] = False
    return primes


primes = eratosthenes(10**6)


def eratosthenes(n: int) -> List[int]:
    """[0, x] 内的每个数的 不同质因数个数/质因数种类, 特例: 1 没有因子, 返回 0"""
    diffPrime = [0] * (n + 1)
    for i in range(2, n + 1):
        if diffPrime[i] == 0:
            for j in range(i, n + 1, i):
                diffPrime[j] += 1
    return diffPrime
