from typing import List, Optional
import collections, random, heapq


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 203 - Remove Linked List Elements - EASY
class Solution:
    def removeElements(self, head: Optional[ListNode],
                       val: int) -> Optional[ListNode]:
        dummyHead = ListNode(-1)
        dummyHead.next = head
        cur = dummyHead
        while cur.next != None:
            if cur.next.val == val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return dummyHead.next


# recursive
class Solution:
    def removeElements(self, head: Optional[ListNode],
                       val: int) -> Optional[ListNode]:
        if head == None:
            return None
        # head.next = self.removeElements(head.next, val)
        # return head.next if head.val == val else head
        next = self.removeElements(head.next, val)
        if head.val == val:
            return next
        head.next = next
        return head


# v1 two pointers
class Solution:
    def removeElements(self, head: Optional[ListNode],
                       val: int) -> Optional[ListNode]:
        pre = None
        cur = head
        while cur:
            if cur.val == val:
                if not pre:
                    head = head.next
                else:
                    pre.next = cur.next
            else:
                pre = cur
            cur = cur.next
        return head


# v2 two pointers
class Solution:
    def removeElements(self, head: Optional[ListNode],
                       val: int) -> Optional[ListNode]:
        dummyHead = ListNode(-1)
        dummyHead.next = head
        cur = head
        pre = dummyHead
        while cur:
            if cur.val == val:
                pre.next = cur.next
            else:
                pre = pre.next
            cur = cur.next
        return dummyHead.next


# v3 NOT WORK!! / input: [7,7,7,7] 7
# head did not change
class Solution:
    def removeElements(self, head: Optional[ListNode],
                       val: int) -> Optional[ListNode]:
        pre = ListNode(-1)
        pre.next = head
        cur = head
        '''
        # is also wrong assigning way, it will create two new objects
        dummyHead, pre = ListNode(-1), ListNode(-1)
        dummyHead.next, pre.next = head, head
        return dummyHead.next
        '''
        while cur:
            if cur.val == val:
                pre.next = cur.next
            else:
                pre = pre.next
            cur = cur.next
        return head


# 206 - Reverse Linked List - EASY
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre = None
        while head:
            tmp = head.next
            head.next = pre
            pre = head
            head = tmp
        return pre


# 215 - Kth Largest Element in an Array - MEDIUM
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums.sort()
        return nums[-k]


# two stacks, almost TLE(Time Limit Exceeded)
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        ans, tmp = [], []
        for num in nums:
            if ans:
                while ans and num > ans[-1]:
                    # need put to the right position in stack
                    tmp.append(ans.pop())
                if len(ans) == k:
                    tmp = []
                    continue
                # descending order, put into stack directly
                else:
                    ans.append(num)
                    while tmp and len(ans) < k:
                        ans.append(tmp.pop())
                    tmp = []
            else:
                ans.append(num)

        return ans[-1]


# partition: based on quick sort
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        n = len(nums)
        left, right = 0, n - 1
        while True:
            i, j = left, right
            # idx = random.choice(nums[left, right])
            idx = random.randint(left, right)
            nums[left], nums[idx] = nums[idx], nums[left]
            while i < j:
                while i < j and nums[j] >= nums[left]:
                    j -= 1
                while i < j and nums[i] <= nums[left]:
                    i += 1
                nums[i], nums[j] = nums[j], nums[i]
            nums[i], nums[left] = nums[left], nums[i]
            if i == n - k:
                return nums[i]
            elif i > n - k:
                right = i - 1
            else:
                left = i + 1


# quick select
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def partition(left: int, right: int, pivot_idx: int):
            pivot = nums[pivot_idx]
            # 1. move pivot to end
            nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
            # 2. move all smaller elements to the left
            store_idx = left
            for i in range(left, right):
                if nums[i] < pivot:
                    nums[store_idx], nums[i] = nums[i], nums[store_idx]
                    store_idx += 1
            # 3. move pivot to its final place, the correct order position
            nums[right], nums[store_idx] = nums[store_idx], nums[right]
            return store_idx

        def select(left: int, right: int, k: int):
            if left == right:
                return nums[left]
            pivotIndex = partition(left, right, random.randint(left, right))
            if k == pivotIndex:
                return nums[k]
            elif k < pivotIndex:
                return select(left, pivotIndex - 1, k)
            else:
                return select(pivotIndex + 1, right, k)

        return select(0, len(nums) - 1, len(nums) - k)


class Solution:
    def findKthLargest(self, nums, k):
        if not nums:
            return
        pivot = random.choice(nums)
        left = [x for x in nums if x < pivot]
        mid = [x for x in nums if x == pivot]
        right = [x for x in nums if x > pivot]
        if k <= len(right):
            return self.findKthLargest(right, k)
        elif k <= len(right) + len(mid):
            return pivot
        else:
            return self.findKthLargest(left, k - len(right) - len(mid))


class Solution:
    def findKthLargest(self, nums, k):
        if not nums: return
        pivot = random.choice(nums)
        left = [x for x in nums if x > pivot]  # different from above
        mid = [x for x in nums if x == pivot]
        right = [x for x in nums if x < pivot]

        less, more = len(left), len(mid)

        if k <= less:
            return self.findKthLargest(left, k)
        elif k > less + more:
            return self.findKthLargest(right, k - less - more)
        else:
            return mid[0]


# 11.14 mock
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def helper(nums: List[int], k: int):
            pivot = random.choice(nums)
            left = [x for x in nums if x < pivot]
            mid = [x for x in nums if x == pivot]
            right = [x for x in nums if x > pivot]
            if k <= len(right):
                return helper(right, k)
            elif k > len(right) + len(mid):
                return helper(left, k - len(right) - len(mid))
            else:
                return mid[0]

        return helper(nums, k)


# heap (a.k.a: priority queue)
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return heapq.nlargest(k, nums)[-1]


# heap
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        for num in nums:
            heapq.heappush(heap, num)
        for _ in range(len(nums) - k):
            heapq.heappop(heap)
        return heapq.heappop(heap)


# O(nk): selections sort idea, almost TLE(Time Limit Exceeded)
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        for i in range(len(nums), len(nums) - k, -1):
            tmp_max = 0
            for j in range(i):
                if nums[j] > nums[tmp_max]:
                    tmp_max = j
            nums[tmp_max], nums[i - 1] = nums[i - 1], nums[tmp_max]
        return nums[len(nums) - k]


# O(nk): bubble sort idea, TLE(Time Limit Exceeded)
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        for i in range(k):
            for j in range(len(nums) - i - 1):
                if nums[j] > nums[j + 1]:
                    # exchange elements, time consuming
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
        return nums[len(nums) - k]


# 226 - Invert Binary Tree - EASY
# breadth-first search
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        stack = []
        if root:
            stack = [root]
        while stack:
            node = stack.pop()
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
            node.left, node.right = node.right, node.left

        return root


# depth-first search
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(root: TreeNode):
            if not root:
                return
            dfs(root.left)
            dfs(root.right)
            root.left, root.right = root.right, root.left
            return

        dfs(root)
        return root


# recursively
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root:
            root.left, root.right = self.invertTree(
                root.right), self.intertTree(root.left)
        return root


# 227 - Basic Calculator II - MEDIUM
class Solution:
    def calculate(self, s: str) -> int:
        s = s.replace(" ", "")
        newS = []
        num = ""
        for ch in s:
            if ch == '+' or ch == '-' or ch == '*' or ch == '/':
                if num != "":
                    newS.append(num)
                    num = ""
                newS.append(ch)
            else:
                num = num + ch
        if num != "":
            newS.append(num)

        addAndSubs = [int(s[0])]
        for i in range(1, len(s) - 1):
            if s[i] != '*' and s[i] != "/":
                addAndSubs.append(s[i])
            elif s[i] == '*':
                addAndSubs[-1] = addAndSubs[-1] * s[i + 1]
                i += 1
            else:
                return
        return


# stack
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        s += '$'
        pre_flag = '+'
        pre_num = 0

        for ch in s:
            if ch.isdigit():
                pre_num = pre_num * 10 + int(ch)
            elif ch == ' ':
                continue
            else:
                if pre_flag == '+':
                    stack.append(pre_num)
                elif pre_flag == '-':
                    stack.append(-pre_num)
                elif pre_flag == '*':
                    stack.append(stack.pop() * pre_num)
                elif pre_flag == '/':
                    stack.append(stack.pop() // pre_num)
                pre_flag = ch
                pre_num = 0

        return sum(stack)


# 228 - Summary Ranges - EASY
class Solution:
    def summaryRanges(self, nums: List[int]) -> List[str]:
        if len(nums) == 0:
            return []
        elif len(nums) == 1:
            return [str(nums[0])]
        ans, length = [], 1
        nums.append(float('inf'))  # help to process the last element
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1] + 1:
                if length == 1:
                    ans.append("".join([str(nums[i - 1])]))
                else:
                    ans.append("->".join(
                        [str(nums[i - length]),
                         str(nums[i - 1])]))
                length = 1
            else:
                length += 1
        return ans


# 229 - Majority Element II - MEDIUM
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        cnt = collections.Counter(nums)
        ans, n = [], len(nums)
        for i in cnt:
            if cnt[i] > n // 3:
                ans.append(i)
        return ans


# up to two potential number appear more then n/3 times
# when the first 'num1' appears too many times,
# the second 'num2' may not get enough votes
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:

        time1, time2, num1, num2 = 0, 0, 0, 0
        for num in nums:
            if time1 > 0 and num == num1:
                time1 += 1
            elif time2 > 0 and num == num2:
                time2 += 1
            elif time1 == 0:
                num1 = num
                time1 = 1
            elif time2 == 0:
                num2 = num
                time2 = 1
            else:
                time1 -= 1
                time2 -= 1

        vote1, vote2 = 0, 0
        for num in nums:
            if num == num1:
                vote1 += 1
            elif num == num2:
                vote2 += 1

        ans = []
        if vote1 > len(nums) // 3:
            ans.append(num1)
        if vote2 > len(nums) // 3:
            ans.append(num2)
        return ans


# 235 - Lowest Common Ancestor of a Binary Search Tree - EASY
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


# recursive solution
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        return root


# Non-recursive solution
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        while (root.val - p.val) * (root.val - q.val) > 0:
            if root.val > p.val and root.val > q.val:
                root = root.left
            else:
                root = root.right
        return root


# 236 - Lowest Common Ancestor of a Binary Tree - MEDIUM
# need to know the status of left and right subtrees
# then we can proceed to the next step, so we use postorder traversal
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        if not root or p.val == root.val or q.val == root.val:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if not left:
            return right
        if not right:
            return left
        return root


# three cases:
# 1. root == p || root == q
# 2. p, q are subtree in two sides (p in left, q in right and vice versa)
# 3. p, q on the same side of subtree, recursive


class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        if root.val == p.val: return root
        if root.val == q.val: return root
        leftNode = self.lowestCommonAncestor(root.left, p, q)
        rightNode = self.lowestCommonAncestor(root.right, p, q)
        if leftNode and rightNode: return root
        return leftNode if leftNode else rightNode


# iterative solution


class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        stack = [root]
        parent = {root: None}
        # find p's and q's parents
        while p not in parent or q not in parent:
            node = stack.pop()
            if node.left:
                parent[node.left] = node
                stack.append(node.left)
            if node.right:
                parent[node.right] = node
                stack.append(node.right)
        ancestors = set()
        # Backtracking p's all ancestors, until to the root
        while p:
            ancestors.add(p)
            p = parent[p]
            print(p)
        print(ancestors)
        # find q's ancestor, if not, until to the root
        while q not in ancestors:
            q = parent[q]
        return q


# 237 - Delete Node in a Linked List - EASY
class Solution:
    def deleteNode(self, node: ListNode):
        node.val = node.next.val
        node.next = node.next.next


# 260 - Single Number III - MEDUIM
# Hash / O(n) + O(n)
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        cnt = collections.Counter(nums)
        ans = [num for num, times in cnt.items() if times == 1]
        return ans


# "lsb" is the last 1 of its binary representation, means that two numbers are different in that bit
# split nums[] into two lists, one with that bit as 0 and the other with that bit as 1.
# separately perform XOR operation, find the number that appears once in each list.
# O(n) + O(1)
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        xorSum = 0
        for i in nums:
            xorSum ^= i
        lsb = xorSum & -xorSum
        # mask = 1
        # while(xorSum & mask == 0):
        #     mask = mask << 1
        ans1, ans2 = 0, 0
        for i in nums:
            if i & lsb > 0:
                ans1 ^= i
            else:
                ans2 ^= i
        return [ans1, ans2]


# 268 - Missing Number - EASY
# sort
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        nums.sort()
        for i in range(len(nums)):
            if nums[i] != i:
                return i
        return len(nums)


# XOR
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        ans = len(nums)
        for i in range(len(nums)):
            ans = ans ^ i ^ nums[i]
        return ans


# math
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        # (0 + n) * (n + 1) // 2
        n = len(nums)
        total = (0 + n) * (n + 1) // 2
        sum = 0
        for v in nums:
            sum += v
        return total - sum


# 299 - Bulls and Cows - MEDIUM
class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        bull = 0
        numS, numG = [0] * 10, [0] * 10
        for k, _ in enumerate(secret):
            if secret[k] == guess[k]:
                bull += 1
            else:
                numS[int(secret[k])] += 1
                numG[int(guess[k])] += 1
        cow = sum([min(numS[k], numG[k]) for k, _ in enumerate(numS)])

        # base = set(guess)
        # for i in base:
        #     if i in secret:
        #         cow += min(guess.count(i),secret.count(i))
        # cow = cow-bull

        # str(bull) + "A" + str(cow) + "B"
        # "{}A{}B".format(bull, cow)
        return f'{bull}A{cow}B'