from typing import List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 617 - Merge Two Binary Trees - EASY
class Solution:
    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:

        return


# 670 - Maximum Swap - MEDIUM
# Greedy, O(n)
# find the last occurrence of each number (guarantee that the rightmost number)
# enumerate each number from left to right,
# swap the number when a larger number is found
class Solution:
    def maximumSwap(self, num: int) -> int:
        s = list(str(num))
        later = {int(x): i for i, x in enumerate(s)}
        for i, x in enumerate(s):
            for d in range(9, int(x), -1):
                if later.get(d, -1) > i:
                    s[i], s[later[d]] = s[later[d]], s[i]
                    return "".join(s)
        return num


# 676 - Implement Magic Dictionary - MEDIUM
class MagicDictionary:
    def __init__(self):
        self.dic = {}

    def buildDict(self, dictionary: List[str]) -> None:
        for i in dictionary:
            self.dic[len(i)] = self.dic.get(len(i), []) + [i]

    def search(self, searchWord: str) -> bool:
        for candi in self.dic.get(len(searchWord), []):
            diff = 0
            for j in range(len(searchWord)):
                if candi[j] != searchWord[j]:
                    diff += 1
                if diff > 2:
                    break
            if diff == 1:
                return True
        return False


class MagicDictionary:
    def __init__(self):
        self.dic = {}

    def buildDict(self, dictionary: List[str]) -> None:
        for d in dictionary:
            self.dic.setdefault(len(d), []).append(d)

    def search(self, searchWord: str) -> bool:
        l = len(searchWord)
        for candidate in self.dct.get(l, []):
            isDifferent = False
            for idx in range(l):
                if candidate[idx] != searchWord[idx]:
                    if isDifferent:
                        break
                    else:
                        isDifferent = True
            else:
                if isDifferent:
                    return True
        return False


# 677 - Map Sum Pairs - MEDIUM
class MapSum:
    def __init__(self):
        self.d = {}

    def insert(self, key: str, val: int) -> None:
        self.d[key] = val

    def sum(self, prefix: str) -> int:
        return sum(self.d[i] for i in self.d if i.startswith(prefix))


# 680 - Valid Palindrome II - EASY
class Solution:
    def validPalindrome(self, s: str) -> bool:
        i, j = 0, len(s) - 1
        while i < j:
            if s[i] != s[j]:
                deleteJ = s[i:j]
                deleteI = s[i + 1:j + 1]
                return deleteI == deleteI[::-1] or deleteJ == deleteJ[::-1]
            i += 1
            j -= 1
        return True


class Solution:
    def validPalindrome(self, s):
        i = 0
        while i < len(s) / 2 and s[i] == s[-(i + 1)]:
            i += 1
        s = s[i:len(s) - i]
        return s[1:] == s[1:][::-1] or s[:-1] == s[:-1][::-1]


# 689 - Maximum Sum of 3 Non-Overlapping Subarrays - HARD
class Solution:
    def maxSumOfThreeSubarrays(self, nums: List[int], k: int) -> List[int]:
        # Best single, double, and triple sequence found so far
        bestSeq = 0
        bestTwoSeq = [0, k]
        bestThreeSeq = [0, k, k*2]

        # Sums of each window
        seqSum = sum(nums[0:k])
        seqTwoSum = sum(nums[k:k*2])
        seqThreeSum = sum(nums[k*2:k*3])

        # Sums of combined best windows
        bestSeqSum = seqSum
        bestTwoSum = seqSum + seqTwoSum
        bestThreeSum = seqSum + seqTwoSum + seqThreeSum

        # Current window positions
        seqIndex = 1
        twoSeqIndex = k + 1
        threeSeqIndex = k*2 + 1
        while threeSeqIndex <= len(nums) - k:
            # Update the three sliding windows
            seqSum = seqSum - nums[seqIndex - 1] + nums[seqIndex + k - 1]
            seqTwoSum = seqTwoSum - nums[twoSeqIndex - 1] + nums[twoSeqIndex + k - 1]
            seqThreeSum = seqThreeSum - nums[threeSeqIndex - 1] + nums[threeSeqIndex + k - 1]
            
            # Update best single window
            if seqSum > bestSeqSum:
                bestSeq = seqIndex
                bestSeqSum = seqSum

            # Update best two windows
            if seqTwoSum + bestSeqSum > bestTwoSum:
                bestTwoSeq = [bestSeq, twoSeqIndex]
                bestTwoSum = seqTwoSum + bestSeqSum

            # Update best three windows
            if seqThreeSum + bestTwoSum > bestThreeSum:
                bestThreeSeq = bestTwoSeq + [threeSeqIndex]
                bestThreeSum = seqThreeSum + bestTwoSum

            # Update the current positions
            seqIndex += 1
            twoSeqIndex += 1
            threeSeqIndex += 1

        return bestThreeSeq


# 696 - Count Binary Substrings - EASY
class Solution:
    def countBinarySubstrings(self, s: str) -> int:
        ans, prev, cur = 0, 0, 1
        for i in range(1, len(s)):
            if s[i] != s[i - 1]:
                ans += min(prev, cur)
                prev = cur
                cur = 1
            else:
                cur += 1
        ans += min(prev, cur)
        return ans