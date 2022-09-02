# https://leetcode.cn/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/
# 剑指 Offer 09. 用两个栈实现队列 - EASY
class CQueue:
    def __init__(self):
        self.inn = []
        self.out = []

    def appendTail(self, value: int) -> None:
        self.inn.append(value)
        return

    def deleteHead(self) -> int:
        if not self.out:
            if not self.inn:
                return -1
            while self.inn:
                self.out.append(self.inn.pop())
        return self.out.pop()


# https://leetcode.cn/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/
# 剑指 Offer 10- II. 青蛙跳台阶问题 - EASY
class Solution:
    def numWays(self, n: int) -> int:
        a = b = 1
        mod = 10**9 + 7
        for _ in range(n):
            a, b = b, (a + b) % mod
        return a


# 计算放在 global 加速
f = [0] * 101
f[0] = 1
f[1] = 1
mod = 10**9 + 7
for i in range(2, 101):
    f[i] = (f[i - 1] + f[i - 2]) % mod


class Solution:
    def numWays(self, n: int) -> int:
        return f[n]
