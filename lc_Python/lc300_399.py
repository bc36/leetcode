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
            odd +=2
        if num == 0:
            return True
        return False