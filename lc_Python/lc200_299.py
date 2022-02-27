from typing import List, Optional
import collections, random, heapq, math, bisect


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 200 - Number of Islands - MEDIUM
class Solution:
    # dfs
    def numIslands(self, grid: List[List[str]]) -> int:
        row = len(grid)
        col = len(grid[0])

        def move(x: int, y: int) -> tuple(int, int):
            for i, j in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= x + i < row and 0 <= y + j < col:
                    yield x + i, y + j

        def dfs(x: int, y: int, index: int):
            grid[x][y] = index
            for i, j in move(x, y):
                if grid[i][j] == "1":
                    dfs(i, j, index)
            return

        count = 0
        for x in range(row):
            for y in range(col):
                if grid[x][y] == "1":
                    dfs(x, y, count)
                    count += 1
        return count

    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])

        def dfs(i: int, j: int) -> None:
            if 0 <= i < m and 0 <= j < n and grid[i][j] == '1':
                grid[i][j] = '#'
                dfs(i, j + 1)
                dfs(i, j - 1)
                dfs(i + 1, j)
                dfs(i - 1, j)
            return

        count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    count += 1
                    dfs(i, j)

        return count

    # bfs
    def numIslands(self, grid: List[List[str]]):
        def helper(grid: List[List[str]], queue: collections.deque()) -> None:
            while queue:
                x, y = queue.popleft()
                for i, j in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if 0 <= i < len(grid) and 0 <= j < len(
                            grid[0]) and grid[i][j] == '1':
                        queue.append((i, j))
                        grid[i][j] = 0
            return

        count = 0
        queue = collections.deque([])
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    grid[i][j] = 0
                    queue.append((i, j))
                    helper(grid, queue)  # turn the adjancent '1' to '0'
                    count += 1
        return count


# 201 - Bitwise AND of Numbers Range - MEDIUM


# 202 - Happy Number - EASY
class Solution:
    def isHappy(self, n: int) -> bool:
        while n >= 10:
            new = 0
            while n:
                d = n % 10
                new += d**2
                n //= 10
            if new == 1:
                return True
            n = new
        return n == 1 or n == 7

    def isHappy(self, n: int) -> bool:
        seen = set()
        while n not in seen:
            seen.add(n)
            n = sum([int(x)**2 for x in str(n)])
        return n == 1


# 203 - Remove Linked List Elements - EASY
class Solution:
    def removeElements(self, head: Optional[ListNode],
                       val: int) -> Optional[ListNode]:
        dummyHead = ListNode(-1, head)
        cur = dummyHead
        while cur.next:
            if cur.next.val == val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return dummyHead.next

    # two pointers
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        dummy = pre = ListNode(-1, head)
        while head:
            if head.val == val:
                pre.next = head.next
            else:
                pre = head
            head = head.next
        return dummy.next

    # NOT WORK!! / input: [7,7,7,7] 7
    # head did not change
    def removeElements(self, head: Optional[ListNode],
                       val: int) -> Optional[ListNode]:
        pre = ListNode(-1)
        pre.next = head
        cur = head
        '''
        # It is a wrong assigning way, it will create two new objects: 'dummy' and 'pre'
        dummy, pre = ListNode(-1), ListNode(-1)
        dummy.next, pre.next = head, head
        return dummy.next
        '''
        while cur:
            if cur.val == val:
                pre.next = cur.next
            else:
                pre = pre.next
            cur = cur.next
        return head

    # recursive
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


# 206 - Reverse Linked List - EASY
class Solution:
    # iterative
    def reverseList(self, head: ListNode) -> ListNode:
        pre = None
        while head:
            tmp = head.next
            head.next = pre
            pre = head
            head = tmp
            # one line:
            # head.next, head, pre = pre, head.next, head
        return pre

    # recursive
    def reverseList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        new = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return new


# 207 - Course Schedule I - MEDIUM
# [0, 1] you have to first take course 1
class Solution:
    # bfs, adjacency list, indegree, save successor
    def canFinish(self, numCourses: int,
                  prerequisites: List[List[int]]) -> bool:
        graph = collections.defaultdict(list)  # or {i: set()}
        in_d = [0] * numCourses  # or {i: 0}
        for p in (prerequisites):
            graph[p[1]].append(p[0])  # save successor, no dependence
            in_d[p[0]] += 1
        dq, count = collections.deque([]), 0
        for i in range(numCourses):
            if in_d[i] == 0:
                dq.append(i)
        while dq:
            node = dq.popleft()
            # seen.add(node)
            count += 1
            for successor in graph[node]:
                in_d[successor] -= 1
                if in_d[successor] == 0:
                    dq.append(successor)
        # return len(seen) == numCourses
        # return not sum(in_d) # not use count or in_d
        return count == numCourses

    # dfs, whether there is a cycle
    def canFinish(self, numCourses: int,
                  prerequisites: List[List[int]]) -> bool:
        def hasCycle(v: int) -> bool:  # 0 default
            if flags[v] == -1: return True  # is being processing
            if flags[v] == 1: return False  # is processed
            flags[v] = -1  # is being processing
            for i in adj[v]:
                if hasCycle(i): return True
            flags[v] = 1  # process finished
            return False

        adj = [[] for _ in range(numCourses)]
        flags = [0] * numCourses
        for cur, pre in prerequisites:
            adj[pre].append(cur)  # save successor
        for i in range(numCourses):
            if hasCycle(i): return False
        return True


# 208 - Implement Trie (Prefix Tree) - MEDIUM
class TrieNode:
    def __init__(self):
        # self.children = collections.defaultdict(TrieNode)
        self.children = {}
        self.is_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        cur = self.root
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = TrieNode()
            cur = cur.children[ch]
        cur.is_word = True

    def search(self, word: str) -> bool:
        cur = self.root
        for ch in word:
            if ch not in cur.children:
                return False
            cur = cur.children[ch]
        return cur.is_word

    def startsWith(self, prefix: str) -> bool:
        cur = self.root
        for ch in prefix:
            if ch not in cur.children:
                return False
            cur = cur.children[ch]
        return True


class Trie:
    def __init__(self):
        self.root = {}

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node:
                node[ch] = {}
            node = node[ch]
        node['END'] = True
        return

    def search(self, word: str) -> bool:
        node = self.root
        for ch in word:
            if ch not in node: return False
            node = node[ch]
        return 'END' in node

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for ch in prefix:
            if ch not in node: return False
            node = node[ch]
        return True


# 209 - Minimum Size Subarray Sum - MEDIUM
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        s = l = r = 0
        n = len(nums)
        ans = float('inf')
        while r < n:
            s += nums[r]
            while s >= target:
                ans = min(ans, r - l + 1)
                s -= nums[l]
                l += 1
            r += 1
        return ans if ans != float('inf') else 0


# 210 - Course Schedule II - MEDIUM
class Solution:
    def findOrder(self, numCourses: int,
                  prerequisites: List[List[int]]) -> List[int]:
        graph, in_d = collections.defaultdict(list), [0] * numCourses
        for p in prerequisites:
            graph[p[1]].append(p[0])  # save successor
            in_d[p[0]] += 1
        dq, ans = collections.deque([]), []
        for i in range(len(in_d)):
            if not in_d[i]:
                dq.append(i)
        while dq:
            node = dq.popleft()
            ans.append(node)
            for n in graph[node]:
                in_d[n] -= 1
                if in_d[n] == 0:
                    dq.append(n)
        return ans if len(ans) == numCourses else []

    def findOrder(self, numCourses: int,
                  prerequisites: List[List[int]]) -> List[int]:
        def dfs(v: int) -> bool:
            if visited[v] == -1: return False  # cycle detected
            if visited[v] == 1: return True  # finished, need added
            visited[v] = -1  # mark as visited
            for x in graph[v]:
                if not dfs(x): return False
            visited[v] = 1  # mark as finished
            ans.append(v)
            return True

        graph = collections.defaultdict(list)
        visited, ans = [0] * numCourses, []
        for pair in prerequisites:
            graph[pair[0]].append(pair[1])  # save predecessor
        for vertex in range(numCourses):
            if not dfs(vertex): return []
        # if build graph with successor, return ans[::-1]
        # the first vertex will be in the bottom of stack in recursive process
        return ans


# 211 - Design Add and Search Words Data Structure - MEDIUM
class WordDictionary:
    def __init__(self):
        self.root = {}

    def addWord(self, word: str) -> None:
        r = self.root
        for ch in word:
            r = r.setdefault(ch, {})
        r['END'] = True
        return

    def search(self, word: str) -> bool:
        def dfs(word, r):
            for i, ch in enumerate(word):
                if ch in r:
                    r = r[ch]
                elif ch == '.':
                    for k in r:
                        if k == 'END':  # not having the 'isWord' property
                            continue
                        if dfs(word[i + 1:], r[k]):
                            return True
                    return False
                else:
                    return False
            return 'END' in r

        return dfs(word, self.root)


class WordDictionary:
    def __init__(self):
        self.root = {}

    def addWord(self, word):
        r = self.root
        for ch in word:
            r = r.setdefault(ch, {})
        r[None] = None

    def search(self, word):
        def find(word, r):
            if not word:
                return None in r
            ch, word = word[0], word[1:]
            if ch != '.':
                return ch in r and find(word, r[ch])
            return any(find(word, kid) for kid in r.values() if kid)

        return find(word, self.root)


class WordDictionary:
    def __init__(self):
        self.d = collections.defaultdict(list)

    def addWord(self, word: str) -> None:
        self.d[len(word)] += [word]

    def search(self, word: str) -> bool:
        if '.' not in word:
            return word in self.d[len(word)]
        for x in self.d[len(word)]:
            for i in range(len(word)):
                if word[i] != x[i] and word[i] != '.':
                    break
            else:
                return True
        return False


# 213 - House Robber II - MEDIUM
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) < 3:
            return max(nums)
        # dp0: start from index 0, dp1: start from index 1
        dp0 = [0] * (len(nums) - 1)
        dp0[0], dp0[1] = nums[0], max(nums[0], nums[1])
        dp1 = [0] * (len(nums) - 1)
        dp1[0], dp1[1] = nums[1], max(nums[1], nums[2])
        for i in range(2, len(nums) - 1):
            dp0[i] = max(dp0[i - 2] + nums[i], dp0[i - 1])
        for i in range(3, len(nums)):
            dp1[i - 1] = max(dp1[i - 3] + nums[i], dp1[i - 2])
        return max(dp0[-1], dp1[-1])

    def rob(self, nums: List[int]) -> int:
        def my_rob(nums):
            cur, pre = 0, 0
            for num in nums:
                '''
                Correct:
                cur, pre = max(pre + num, cur), cur 
                
                pre, cur = cur, max(pre + num, cur)
                
                tmp = pre
                pre = cur
                cur = max(tmp + num, cur)

                tmp = cur
                cur = max(pre + num, cur)
                pre = tmp

                Wrong:
                pre = cur
                cur = max(tmp + num, cur)

                cur = max(tmp + num, cur)
                pre = cur
                '''
                pre, cur = cur, max(pre + num, cur)
            return cur

        return max(my_rob(nums[:-1]), my_rob(
            nums[1:])) if len(nums) != 1 else nums[0]


# 215 - Kth Largest Element in an Array - MEDIUM - REVIEW
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums.sort()
        return nums[-k]

    # partition: based on quick sort
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
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return heapq.nlargest(k, nums)[-1]

    # heap
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        for num in nums:
            heapq.heappush(heap, num)
        for _ in range(len(nums) - k):
            heapq.heappop(heap)
        return heapq.heappop(heap)


# 217 - Contains Duplicate - EASY
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(nums) != len(set(nums))

    def containsDuplicate(self, nums: List[int]) -> bool:
        s = set()
        for num in nums:
            if num in s: return True
            s.add(num)
        return False


# 219 - Contains Duplicate II - EASY
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        dic = {}
        for i in range(len(nums)):
            if i - dic.get(nums[i], float('-inf')) <= k:
                return True
            dic[nums[i]] = i
            # or
            # if nums[i] in dic and i - dic[nums[i]] <= k:
            #     return True
            # dic[nums[i]] = i
        return False
        # slow, > 8000ms
        # i = j = 0
        # while j < len(nums):
        #     if j - i == k + 1:
        #         i += 1
        #     if nums[j] in nums[i:j]:
        #         return True
        #     j += 1
        # return False


# 221 - Maximal Square - MEDIUM
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        rows, cols = len(matrix), len(matrix[0])
        dp, maxSide = [[0] * cols for _ in range(rows)], 0
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j],
                                       dp[i][j - 1]) + 1
                maxSide = max(maxSide, dp[i][j])
        return maxSide**2

    def maximalSquare(self, matrix: List[List[str]]) -> int:
        rows, cols = len(matrix), len(matrix[0])
        dp = [0] * (cols + 1)
        max_side = 0
        for r in range(1, rows + 1):
            nxt_dp = [0] * (cols + 1)
            for c in range(1, cols + 1):
                if matrix[r - 1][c - 1] == '1':
                    nxt_dp[c] = 1 + min(dp[c], dp[c - 1], nxt_dp[c - 1])
                max_side = max(max_side, nxt_dp[c])
            dp = nxt_dp
        return max_side**2


# 226 - Invert Binary Tree - EASY
class Solution:
    # breadth-first search
    def invertTree(self, root: TreeNode) -> TreeNode:
        stack = [root]
        while stack:
            node = stack.pop()
            if node:
                node.left, node.right = node.right, node.left
                stack += node.left, node.right
        return root

    # depth-first search
    def invertTree(self, root: TreeNode) -> TreeNode:
        def dfs(root: TreeNode):
            if not root:
                return
            root.left, root.right = root.right, root.left
            dfs(root.left)
            dfs(root.right)
            return

        dfs(root)
        return root

    # recursively
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root:
            root.left, root.right = self.invertTree(
                root.right), self.intertTree(root.left)
        return root


# 227 - Basic Calculator II - MEDIUM
class Solution:
    def calculate(self, s: str) -> int:
        pre, cur, result, sign = 0, 0, 0, '+'
        for ch in s + '#':
            if ch == ' ': continue
            if ch.isdigit():
                pre = 10 * pre + int(ch)
                continue
            if sign == '+':
                result += cur
                cur = pre
            elif sign == '-':
                result += cur
                cur = -pre
            elif sign == '*':
                cur = cur * pre
            elif sign == '/':
                cur = int(cur / pre)
            pre, sign = 0, ch
        return result + cur

    # stack
    def calculate(self, s: str) -> int:
        stack, sign, pre = [], '+', 0
        s += '#'

        for ch in s:
            if ch.isdigit():
                pre = pre * 10 + int(ch)
            elif ch == ' ':
                continue
            else:
                if sign == '+':
                    stack.append(pre)
                elif sign == '-':
                    stack.append(-pre)
                elif sign == '*':
                    stack.append(stack.pop() * pre)
                elif sign == '/':
                    stack.append(stack.pop() // pre)
                sign = ch
                pre = 0

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


# 230 - Kth Smallest Element in a BST - MEDIUM
class Solution:
    # O(h + k) / O(h), h is the height of this tree
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def inorder(root):
            if not root:
                return
            inorder(root.left)
            self.k -= 1
            if self.k == 0:
                self.ans = root.val
                return
            inorder(root.right)
            return

        self.k = k
        self.ans = 0
        inorder(root)
        return self.ans

    # O(n) / O(n)
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def inorder(root):
            if not root:
                return
            inorder(root.left)
            self.arr.append(root.val)
            inorder(root.right)
            return
        
        self.arr = []
        inorder(root)
        return self.arr[k-1]

# 231 - Power of Two - EASY
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n <= 0:
            return False
        while n > 1:
            if n & 1:
                return False
            n >>= 1
        return True

    def isPowerOfTwo(self, n: int) -> bool:
        # return n > 0 and 2**30 % n == 0
        return n and n & (n - 1) == 0


# 232 - Implement Queue using Stacks - EASY
class MyQueue:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, x: int) -> None:
        self.stack1.append(x)

    def pop(self) -> int:
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()

    def peek(self) -> int:
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2[-1]

    def empty(self) -> bool:
        return not self.stack1 and not self.stack2


# 235 - Lowest Common Ancestor of a Binary Search Tree - EASY
class Solution:
    # recursive solution
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        return root

    # Non-recursive solution
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        while (root.val - p.val) * (root.val - q.val) > 0:
            if root.val > p.val and root.val > q.val:
                root = root.left
            else:
                root = root.right
        return root


# 236 - Lowest Common Ancestor of a Binary Tree - MEDIUM
class Solution:
    # need to know the status of left and right subtrees
    # then we can proceed to the next step, so we use postorder traversal
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        if not root or p.val == root.val or q.val == root.val:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left if left else right

    # three cases:
    # 1. root == p || root == q
    # 2. p, q are subtree in two sides (p in left, q in right and vice versa)
    # 3. p, q on the same side of subtree, recursive
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        if root.val == p.val: return root
        if root.val == q.val: return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right: return root
        return left if left else right

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        if p == root:
            return p
        if q == root:
            return q
        l = self.lowestCommonAncestor(root.left, p, q)
        r = self.lowestCommonAncestor(root.right, p, q)
        return root if l and r else l or r

    # iterative solution
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode',
                             q: 'TreeNode') -> 'TreeNode':
        stack = [root]
        # stack = collections.deque([root])
        parent = {root: None}  # {child: father}
        while p not in parent or q not in parent:
            node = stack.pop()
            # node = stack.popleft()
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
        # find q's ancestor, if not, until to the root
        while q not in ancestors:
            q = parent[q]
        return q


# 237 - Delete Node in a Linked List - EASY
class Solution:
    def deleteNode(self, node: ListNode):
        node.val = node.next.val
        node.next = node.next.next


# 238 - Product of Array Except Self - MEDIUM
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        pro, ans = 1, []
        for i in range(len(nums)):
            ans.append(pro)
            pro *= nums[i]
        pro = 1
        for i in range(len(nums) - 1, -1, -1):
            ans[i] *= pro
            pro *= nums[i]
        return ans


# 240 - Search a 2D Matrix II - MEDIUM
class Solution:
    # O(m + n) / O(1)
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        i, j = 0, n - 1
        while i < m and j > -1:
            if matrix[i][j] < target:
                i += 1
            elif matrix[i][j] > target:
                j -= 1
            else:
                return True
        return False

    # O(m * logn) / O(1)
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        for i in range(m):
            idx = bisect.bisect_left(matrix[i], target)
            if idx != n and matrix[i][idx] == target:
                return True
        return False


# 242 - Valid Anagram - EASY
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return collections.Counter(s) == collections.Counter(t)


# 249 - Group Shifted Strings - MEDIUM
# tuple + tuple: (1,) + (2,)
class Solution:
    def groupStrings(self, strings: List[str]) -> List[List[str]]:
        dic = {}
        for s in strings:
            key = ()
            for i in range(len(s) - 1):
                circular_difference = 26 + ord(s[i + 1]) - ord(s[i])
                key += (circular_difference % 26, )
            dic[key] = dic.get(key, []) + [s]
        return list(dic.values())


# 252 - Meeting Rooms - EASY - PREMIUM
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                return False
        return True


# 258 - Add Digits - EASY
class Solution:
    def addDigits(self, num: int) -> int:
        while num > 9:
            mod = 0
            while num:
                mod += num % 10
                num = num // 10
            num = mod
        return num

    def addDigits(self, num: int) -> int:
        if num == 0:
            return 0
        if num % 9 == 0:
            return 9
        return num % 9


# 260 - Single Number III - MEDIUM
class Solution:
    # Hash / O(n) + O(n)
    def singleNumber(self, nums: List[int]) -> List[int]:
        cnt = collections.Counter(nums)
        ans = [num for num, times in cnt.items() if times == 1]
        return ans

    # "lsb" is the last 1 of its binary representation, means that two numbers are different in that bit
    # split nums[] into two lists, one with that bit as 0 and the other with that bit as 1.
    # separately perform XOR operation, find the number that appears once in each list.
    # O(n) + O(1)
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


# 264 - Ugly Number II - MEDIUM
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        factors = [2, 3, 5]
        seen, heap = {1}, [1]
        for _ in range(n - 1):
            cur = heapq.heappop(heap)
            for factor in factors:
                if (nxt := cur * factor) not in seen:
                    seen.add(nxt)
                    heapq.heappush(heap, nxt)
        return heapq.heappop(heap)

    def nthUglyNumber(self, n: int) -> int:
        dp = [0] * n
        dp[0] = 1
        p2 = p3 = p5 = 0
        for i in range(1, n):
            num2, num3, num5 = dp[p2] * 2, dp[p3] * 3, dp[p5] * 5
            dp[i] = min(num2, num3, num5)
            if dp[i] == num2:
                p2 += 1
            if dp[i] == num3:
                p3 += 1
            if dp[i] == num5:
                p5 += 1
        return dp[-1]

    # much faster 40ms, it does not compute time spent before entering objective function
    ugly = sorted(2**a * 3**b * 5**c for a in range(32) for b in range(20)
                  for c in range(14))

    def nthUglyNumber(self, n):
        return self.ugly[n - 1]
        # quite slow, 4000ms
        ugly = sorted(2**a * 3**b * 5**c for a in range(32) for b in range(20)
                      for c in range(14))
        return ugly


# 268 - Missing Number - EASY
class Solution:
    # sort
    def missingNumber(self, nums: List[int]) -> int:
        nums.sort()
        for i in range(len(nums)):
            if nums[i] != i:
                return i
        return len(nums)

    # XOR
    def missingNumber(self, nums: List[int]) -> int:
        ans = len(nums)
        for i in range(len(nums)):
            ans = ans ^ i ^ nums[i]
        return ans

    # math
    def missingNumber(self, nums: List[int]) -> int:
        # (0 + n) * (n + 1) // 2
        n = len(nums)
        total = (0 + n) * (n + 1) // 2
        sum = 0
        for v in nums:
            sum += v
        return total - sum


# 278 - First Bad Version - EASY
def isBadVersion(n: int) -> bool:
    pass


class Solution:
    def firstBadVersion(self, n: int) -> int:
        left, right = 1, n
        while left < right:
            '''
            precedence of '>>' is lower than '+'
            '''
            mid = ((right - left) >> 1) + left  # wrong way
            # mid = left + (right - left) >> 1 # right way
            if isBadVersion(mid):
                right = mid
            else:
                left = mid + 1
        return left


# 279 - Perfect Squares - MEDIUM
class Solution:
    # Lagrange's four-square theorem
    def numSquares(self, n: int) -> int:
        while n % 4 == 0:
            n /= 4
        if n % 8 == 7:
            return 4
        a = 0
        while a**2 <= n:
            b = int((n - a**2)**0.5)
            if a**2 + b**2 == n:
                return (not not a) + (not not b)
                # or
                if a != 0 and b != 0:
                    return 2
                else:
                    return 1
            a += 1
        return 3

    def numSquares(self, n: int) -> int:
        def divisible(n, count):
            if count == 1:
                return n in ps
            for p in ps:
                if divisible(n - p, count - 1):
                    return True
            return False

        ps = set([i * i for i in range(1, int(n**0.5) + 1)])
        for count in range(1, n + 1):
            if divisible(n, count):
                return count


# 283 - Move Zeroes - EASY
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != 0 and nums[slow] == 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
            if nums[slow] != 0:
                slow += 1
        return

    def moveZeroes(self, nums: List[int]) -> None:
        nums[:] = [i for i in nums if i != 0] + nums.count(0) * [0]
        return


# 290 - Word Pattern - EASY
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        s = s.split()
        if len(pattern) != len(s):
            return False
        p2s = dict()
        s2p = dict()
        for i, ch in enumerate(pattern):
            if ch not in p2s:
                p2s[ch] = s[i]
            if s[i] not in s2p:
                s2p[s[i]] = ch
            if p2s[ch] != s[i] or s2p[s[i]] != ch:
                return False
        return True

    def wordPattern(self, pattern: str, s: str) -> bool:
        s = s.split()
        if len(pattern) != len(s):
            return False
        if len(set(pattern)) != len(set(s)):
            return False  # for the case `words=['dog', 'cat']` and  `p='aa'`, or use two dictionaries.
        dic = {}
        for i in range(len(pattern)):
            if pattern[i] not in dic:
                dic[pattern[i]] = s[i]
            elif dic[pattern[i]] != s[i]:
                return False
        return True

    # better, save index, rather element
    def wordPattern(self, pattern: str, s: str) -> bool:
        s = s.split()
        if len(pattern) != len(s):
            return False
        pt, st = {}, {}
        for i in range(len(pattern)):
            if pt.get(pattern[i], 0) != st.get(s[i], 0):
                return False
            pt[pattern[i]] = i + 1
            st[s[i]] = i + 1
        return True

    def wordPattern(self, pattern: str, s: str) -> bool:
        s = s.split()
        return len(set(zip(pattern, s))) == len(set(pattern)) == len(set(s)) \
            and len(pattern) == len(s)


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
        '''
        str(bull) + "A" + str(cow) + "B"
        "{}A{}B".format(bull, cow)
        '''
        return f'{bull}A{cow}B'