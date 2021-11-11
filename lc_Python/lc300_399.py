# 367 - Valid Perfect Square - EASY
# binary search
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        left, right = 0, num
        while left <= right:
            mid = (left + right) // 2
            if mid * mid > num:
                right = mid - 1
            elif mid * mid < num:
                left = mid + 1
            else:
                return True
        return False


# math: sum of odd -> 1+3+5+7+... = n^2
#       (n+1)^2 - n^2 = 2n+1
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        odd = 1
        while num > 0:
            num -= odd
            odd += 2
        if num == 0:
            return True
        return False


# 374 - Guess Number Higher or Lower - EASY
# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num: int) -> int:


class Solution:
    def guessNumber(self, n: int) -> int:
        left, right, mid = 1, n, 1
        while guess(mid) != 0:
            if guess(mid) > 0:
                left = mid + 1
            else:
                right = mid - 1
            mid = (right + left) // 2
        return mid


def guess(self, n: int) -> int:
    pick = 1  # specify internally
    if n > pick:
        return 1
    elif n < pick:
        return -1
    else:
        return 0


class Solution:
    def guessNumber(self, n: int) -> int:
        left, right = 1, n
        while left < right:
            mid = (left + right) // 2
            if guess(mid) <= 0:
                right = mid  # in [left, mid]
            else:
                left = mid + 1  # in [mid+1, right]

        # left == right
        return left