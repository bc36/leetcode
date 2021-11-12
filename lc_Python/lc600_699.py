# 680 - Valid Palindrome II - EASY
class Solution:
    def validPalindrome(self, s: str) -> bool:
        left, right = 0, len(s) - 1
        while left < right:
            if s[left] != s[right]:
                deleteI, deleteJ = s[left:right], s[left + 1:right + 1]
                return deleteI == deleteI[::-1] or deleteJ == deleteJ[::-1]
            left += 1
            right -= 1
        return True