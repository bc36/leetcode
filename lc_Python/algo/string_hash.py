"""string hash
字符串哈希, 定义一个把字符串映射到整数的函数 f 这个 f 称为是 Hash 函数
希望这个函数 f 可以方便地帮我们判断两个字符串是否相等
Hash 的核心思想在于, 将输入映射到一个值域较小、可以方便比较的范围
通常采用的多项式 Hash 的方法,  MOD 需要选择一个素数(至少要比最大的字符要大), base 可以任意选择

py 切片较快, 大部分情况可以直接比较切片
"""

from typing import List


def string_hash(arr: List[int]) -> None:
    n = len(arr)
    base = 131  # 哈希指数, 是一个经验值, 可以取 1331 等等
    mod = 998244353
    p = [0] * 4001
    h = [0] * 4001
    p[0] = 1
    for i in range(1, n + 1):
        p[i] = (p[i - 1] * base) % mod
        h[i] = (h[i - 1] * base + ord(arr[i - 1])) % mod

    def getHash(l: int, r: int) -> int:
        return (h[r] - h[l - 1] * p[r - l + 1]) % mod

    return
