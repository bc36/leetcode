import bisect, collections, functools, heapq, itertools, math, operator, random, re, string
from typing import List, Optional, Tuple
import sortedcontainers


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Node:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next


# 700 - Search in a Binary Search Tree - EASY
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        while root:
            if root.val < val:
                root = root.right
            elif root.val > val:
                root = root.left
            else:
                return root
        return None

    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            return None
        if root.val == val:
            return root
        return self.searchBST(root.left if root.val > val else root.right, val)


# 701 - Insert into a Binary Search Tree - MEDIUM
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            return TreeNode(val)
        r = root
        while (r.val > val and r.left) or r.val < val and r.right:
            r = r.left if r.val > val else r.right
        if r.val > val:
            r.left = TreeNode(val)
        else:
            r.right = TreeNode(val)
        return root

    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            return TreeNode(val)
        r = root
        while True:
            if val < r.val:
                if r.left:
                    r = r.left
                else:
                    r.left = TreeNode(val)
                    break
            else:
                if r.right:
                    r = r.right
                else:
                    r.right = TreeNode(val)
                    break
        return root

    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            return TreeNode(val)
        if root.val > val:
            root.left = self.insertIntoBST(root.left, val)
        else:
            root.right = self.insertIntoBST(root.right, val)
        return root


# 704 - Binary Search - EASY
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (right + left) // 2
            if nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                return mid
        return -1

    def search(self, nums: List[int], target: int) -> int:
        l = 0
        r = len(nums) - 1
        while l < r:
            m = (l + r) // 2
            if nums[m] >= target:
                r = m
            else:
                l = m + 1
        return l if nums[l] == target else -1


# 707 - Design Linked List - MEDIUM
class MyLinkedList:
    def __init__(self):
        self.arr = []

    def get(self, index: int) -> int:
        if not self.arr or not 0 <= index < len(self.arr):
            return -1
        return self.arr[index]

    def addAtHead(self, val: int) -> None:
        self.arr.insert(0, val)
        return

    def addAtTail(self, val: int) -> None:
        self.arr.append(val)
        return

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0:
            self.arr.insert(0, val)
        elif 0 <= index < len(self.arr):
            self.arr.insert(index, val)
        elif index == len(self.arr):
            self.arr.append(val)
        return

    def deleteAtIndex(self, index: int) -> None:
        if 0 <= index < len(self.arr):
            self.arr.pop(index)
        return


# 708 - Insert Into a Sorted Circular Linked List - MEDIUM
class Solution:
    def insert(self, head: "Node", insertVal: int) -> "Node":
        if not head:
            n = Node(insertVal)
            n.next = n
            return n
        r = head
        while head.next != r:
            if head.val <= insertVal <= head.next.val:
                break
            if head.next.val < head.val and (
                head.val <= insertVal or insertVal <= head.next.val
            ):
                break
            head = head.next

        # n = Node(insertVal, head.next) # slow

        n = Node(insertVal)
        n.next = head.next
        head.next = n
        return r


# 709 -To Lower Case - EASY
class Solution:
    def toLowerCase(self, s: str) -> str:
        """
        upper, lower exchange: asc ^= 32;
        upper, lower to lower: asc |= 32;
        lower, upper to upper: asc &= -33
        """
        return "".join(
            chr(asc | 32) if 65 <= (asc := ord(ch)) <= 90 else ch for ch in s
        )
        return s.lower()


# 710 - Random Pick with Blacklist - HARD
class Solution:
    def __init__(self, n: int, blacklist: List[int]):
        m = len(blacklist)
        self.p = j = n - m
        self.d = {}
        b = set(blacklist)
        for x in blacklist:
            if x < n - m:
                while j in b:
                    j += 1
                self.d[x] = j
                j += 1

    def pick(self) -> int:
        p = random.randint(0, self.p - 1)
        return self.d.get(p, p)


# 713 - Subarray Product Less Than K - MEDIUM
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        r = l = ans = 0
        p = nums[0]
        while l < len(nums) and r < len(nums):
            if p < k:
                ans += r - l + 1
                r += 1
                if r < len(nums):
                    p *= nums[r]
            else:
                p //= nums[l]
                l += 1
        return ans


# 714 - Best Time to Buy and Sell Stock with Transaction Fee - MEDIUM
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        dp = [[0, -prices[0]]] + [[0, 0] for _ in range(len(prices) - 1)]
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee)
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        return dp[-1][0]

    def maxProfit(self, prices: List[int], fee: int) -> int:
        sell, buy = 0, -prices[0]
        for i in range(1, len(prices)):
            sell, buy = max(sell, buy + prices[i] - fee), max(buy, sell - prices[i])
        return sell


# 715 - Range Module - HARD
class RangeModule:
    def __init__(self):
        self.track = []

    def addRange(self, left, right):
        start = bisect.bisect_left(self.track, left)
        end = bisect.bisect_right(self.track, right)
        subtrack = []
        if start % 2 == 0:
            subtrack.append(left)
        if end % 2 == 0:
            subtrack.append(right)
        self.track[start:end] = subtrack

    def queryRange(self, left, right):
        start = bisect.bisect_right(self.track, left)
        end = bisect.bisect_left(self.track, right)
        return start == end and start % 2 == 1

    def removeRange(self, left, right):
        start = bisect.bisect_left(self.track, left)
        end = bisect.bisect_right(self.track, right)
        subtrack = []
        if start % 2 == 1:
            subtrack.append(left)
        if end % 2 == 1:
            subtrack.append(right)
        self.track[start:end] = subtrack


class SegmentTree:
    def __init__(self):
        self.t = collections.defaultdict(int)
        # 经验: 区间覆盖问题似乎都可以不用懒标记
        # 初始是0, 全覆盖是1, 全不覆盖是2, 部分覆盖又是改回0是吗

    def pushdown(self, o: int) -> None:
        if self.t[o]:
            self.t[o << 1] = self.t[o]
            self.t[o << 1 | 1] = self.t[o]
        return

    def range_update(self, o: int, l: int, r: int, L: int, R: int, val: int) -> None:
        if L <= l and r <= R:
            self.t[o] = val
            return
        self.pushdown(o)
        m = l + r >> 1
        if L <= m:
            self.range_update(o << 1, l, m, L, R, val)
        if m < R:
            self.range_update(o << 1 | 1, m + 1, r, L, R, val)
        # 注意这里不要用 and, 里面有 0, 1, 2, & 是按位运算符, 1 & 2 = 0, 1 and 2 = 2, 2 and 1 = 1
        self.t[o] = self.t[o << 1] & self.t[o << 1 | 1]  # push up
        return

    def query(self, o: int, l: int, r: int, L: int, R: int):
        if r < L or R < l:
            return True
        if L <= l and r <= R:
            return self.t[o] == 1
        self.pushdown(o)
        m = l + r >> 1
        return self.query(o << 1, l, m, L, R) and self.query(o << 1 | 1, m + 1, r, L, R)


class RangeModule:
    def __init__(self):
        self.st = SegmentTree()

    def addRange(self, left: int, right: int) -> None:
        self.st.range_update(1, 1, 10**9, left, right - 1, 1)

    def queryRange(self, left: int, right: int) -> bool:
        return self.st.query(1, 1, 10**9, left, right - 1)

    def removeRange(self, left: int, right: int) -> None:
        self.st.range_update(1, 1, 10**9, left, right - 1, 2)


# 717 - 1-bit and 2-bit Characters - EASY
class Solution:
    def isOneBitCharacter(self, bits: List[int]) -> bool:
        if bits[-1] == 1:
            return False
        one = 0
        for i in range(len(bits) - 2, -1, -1):
            if bits[i] == 1:
                one += 1
            else:
                break
        return not one & 1

    def isOneBitCharacter(self, bits: List[int]) -> bool:
        n = len(bits)
        i = n - 2
        while i >= 0 and bits[i]:
            i -= 1
        return (n - i) % 2 == 0

    def isOneBitCharacter(self, bits: List[int]) -> bool:
        i, n = 0, len(bits)
        while i < n - 1:
            i += bits[i] + 1
        return i == n - 1


# 719 - Find K-th Smallest Pair Distance - HARD
class Solution:
    # O(n * logD) / O(logD), D = max(nums) - min(nums)
    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        nums.sort()
        l = 0
        r = nums[-1] - nums[0]
        while l < r:
            m = (l + r) >> 1
            cnt = 0
            # Enumerate right endpoints
            i = 0
            for j in range(len(nums)):
                while nums[j] - nums[i] > m:
                    i += 1
                cnt += j - i

            # # Enumerate left endpoints
            # j = 0
            # for i in range(len(nums)):
            #     while j < len(nums) and nums[j] - nums[i] <= m:
            #         j += 1
            #     cnt += j - i - 1

            if cnt >= k:
                r = m
            else:
                l = m + 1
        return l

    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        def check(dist: int) -> bool:
            cnt = i = 0
            for j in range(len(nums)):
                while nums[j] - nums[i] > dist:
                    i += 1
                cnt += j - i
            return cnt >= k

        nums.sort()
        return bisect.bisect_left(range(nums[-1] - nums[0]), True, key=check)

    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        def check(dist: int) -> int:
            cnt = i = 0
            for j in range(len(nums)):
                while nums[j] - nums[i] > dist:
                    i += 1
                cnt += j - i
            return cnt

        nums.sort()
        return bisect.bisect_left(range(nums[-1] - nums[0]), k, key=check)

    # O(nlogn * logD) / O(logn + logD), D = max(nums) - min(nums)
    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        def count(dist: int) -> int:
            cnt = 0
            for j, v in enumerate(nums):
                i = bisect.bisect_left(nums, v - dist, 0, len(nums))
                # i = bisect.bisect_left(nums, v - dist, 0, j)
                # i = bisect_left(nums, v - dist) # cuz 'v - mid <= v' at insertion, so it won't happen 'i = len(nums)'
                cnt += j - i
            return cnt

        nums.sort()
        return bisect.bisect_left(range(nums[-1] - nums[0]), k, key=count)

    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        def check(dist: int) -> int:
            return sum(
                bisect.bisect_right(nums, v + dist) - 1 - i for i, v in enumerate(nums)
            )

        nums.sort()
        return bisect.bisect_left(
            range(nums[-1] - nums[0] + 1), True, key=lambda x: check(x) >= k
        )


# 720 - Longest Word in Dictionary - EASY
class Solution:
    # O(n + n * m * m), m = len(words[i])
    def longestWord(self, words: List[str]) -> str:
        if not words:
            return ""
        ans = ""
        s = set(words)
        for w in words:
            if len(w) > len(ans) or len(w) == len(ans) and w < ans:
                for i in range(1, len(w) + 1):
                    if w[:i] not in s:
                        break
                else:  # for-else statement
                    ans = w
        return ans

    def longestWord(self, words: List[str]) -> str:
        # words.sort(key=lambda x: (len(x), -x)) # 字符不好比大小, TypeError: bad operand type for unary -: 'str'
        words.sort(key=lambda x: (-len(x), x), reverse=True)  # 所以用这种方法
        ans = ""
        s = {""}
        for w in words:
            if w[:-1] in s:
                ans = w
                s.add(w)
        return ans

    # O(n * m * logn + m) / O(n), m = len(words[i])
    def longestWord(self, words: List[str]) -> str:
        # ordered by lexicographical, then ordered by length
        words.sort()
        s = {""}
        ans = ""
        for w in words:
            if w[:-1] in s:
                s.add(w)
                if len(w) > len(ans):
                    ans = w
        return ans

    # O(n * m), m = len(words[i]), Trie
    def longestWord(self, words: List[str]) -> str:
        trie = {}
        for w in words:
            r = trie
            for c in w:
                if c not in r:
                    r[c] = {}
                r = r[c]
            r["end"] = True
        ans = ""
        for w in words:
            if len(w) > len(ans) or len(w) == len(ans) and w < ans:
                r = trie
                f = True  # flag, can be replaced by for-else statement
                for c in w:
                    if c not in r or "end" not in r[c]:
                        f = False
                        break
                    r = r[c]
                if f:
                    ans = w
        return ans


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.isEnd = False
        self.word = ""


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        r = self.root
        for c in word:
            r = r.children[c]
        r.isEnd = True
        r.word = word
        return

    def bfs(self) -> str:
        dq = collections.deque([self.root])
        ans = ""
        while dq:
            node = dq.popleft()
            for ch in node.children.values():
                if ch.isEnd:
                    dq.append(ch)
                    if len(ch.word) > len(ans) or ch.word < ans:
                        ans = ch.word
        return ans


class Solution:
    def longestWord(self, words: List[str]) -> str:
        trie = Trie()
        for w in words:
            trie.insert(w)
        return trie.bfs()


# 721 - Accounts Merge - MEDIUM
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        name = set()
        dic = {}

        def ifexist(account: List[str]) -> bool:
            for acc in account[1:]:
                for i, person in enumerate(dic[account[0]]):
                    for p in person:
                        if acc == p:
                            # the same person
                            dic[account[0]][i] = dic[account[0]][i].union(
                                set(account[1:])
                            )
                            return True
            return False

        for account in accounts:
            if account[0] not in name:
                dic[account[0]] = [set(account[1:])]
                name.add(account[0])
            else:
                ex = ifexist(account)
                if not ex:
                    dic[account[0]].append(set(account[1:]))
        ans = []
        name = list(name)
        name.sort()
        for samename in name:
            for person in dic[samename]:
                tmp = [samename]
                tmp.extend(list(person))
                ans.append(tmp)
        return ans


# UnionFind
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def union(self, index1: int, index2: int):
        self.parent[self.find(index2)] = self.find(index1)

    def find(self, index: int) -> int:
        if self.parent[index] != index:
            self.parent[index] = self.find(self.parent[index])
        return self.parent[index]


class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        emailToIndex = dict()
        emailToName = dict()

        for account in accounts:
            name = account[0]
            for email in account[1:]:
                if email not in emailToIndex:
                    emailToIndex[email] = len(emailToIndex)
                    emailToName[email] = name

        uf = UnionFind(len(emailToIndex))
        for account in accounts:
            firstIndex = emailToIndex[account[1]]
            for email in account[2:]:
                uf.union(firstIndex, emailToIndex[email])

        indexToEmails = collections.defaultdict(list)
        for email, index in emailToIndex.items():
            index = uf.find(index)
            indexToEmails[index].append(email)

        ans = list()
        for emails in indexToEmails.values():
            ans.append([emailToName[emails[0]]] + sorted(emails))
        return ans


# 722 - Remove Comments - MEDIUM
class Solution:
    def removeComments(self, source: List[str]) -> List[str]:
        # 匹配所有 // 和 /* */, 后者用非贪婪模式, 将所有匹配结果替换成空串, 最后移除多余空行,
        return list(
            filter(
                None, re.sub("//.*|/\*(.|\n)*?\*/", "", "\n".join(source)).split("\n")
            )
        )


# 729 - My Calendar I - MEDIUM
class MyCalendar:
    def __init__(self):
        # 0: no booking
        # 1: all booked
        # 2: exist booking
        self.t = collections.defaultdict(int)

    def merge(self, l, r, s, e):
        if l == s and r == e:
            self.t[(l, r)] = 1
            return
        m = (l + r) >> 1
        # pushdown
        if self.t[(l, r)] == 1:
            self.t[(l, m)] = 1
            self.t[(m, r)] = 1

        if e <= m:
            self.merge(l, m, s, e)
        elif m <= s:
            self.merge(m, r, s, e)
        else:
            self.merge(l, m, s, m)
            self.merge(m, r, m, e)
        # pushup
        self.t[(l, r)] = self.t[(l, m)] & self.t[(m, r)]
        if self.t[(l, r)] == 0:
            self.t[(l, r)] = 2
        return

    def check(self, l, r, s, e):
        if l == s and r == e:
            return self.t[(l, r)] == 0
        m = (l + r) >> 1
        if self.t[(l, r)] == 1:
            return False
        if e <= m:
            return self.check(l, m, s, e)
        elif m <= s:
            return self.check(m, r, s, e)
        else:
            return self.check(l, m, s, m) and self.check(m, r, m, e)

    def book(self, start: int, end: int) -> bool:
        if self.check(0, 10**9, start, end):
            self.merge(0, 10**9, start, end)
            return True
        return False


# 731 - My Calendar II - MEDIUM
class MyCalendarTwo:
    def __init__(self):
        self.t = collections.defaultdict(int)
        self.l = collections.defaultdict(int)

    def query(self, k: int, l: int, r: int, s: int, e: int) -> int:
        if s > r or e < l:
            return False
        if s <= l and r <= e:
            return self.t[k] >= 2
        # push down
        if self.l[k]:
            self.l[k << 1] += self.l[k]
            self.t[k << 1] += self.l[k]
            self.l[k << 1 | 1] += self.l[k]
            self.t[k << 1 | 1] += self.l[k]
            self.l[k] = 0
        m = l + r >> 1
        return self.query(k << 1, l, m, s, e) or self.query(k << 1 | 1, m + 1, r, s, e)

    def update(self, k: int, l: int, r: int, s: int, e: int) -> None:
        if s > r or e < l:
            return
        if s <= l and r <= e:
            self.t[k] += 1
            self.l[k] += 1
            return
        # push down(optional)
        if self.l[k]:
            self.l[k << 1] += self.l[k]
            self.t[k << 1] += self.l[k]
            self.l[k << 1 | 1] += self.l[k]
            self.t[k << 1 | 1] += self.l[k]
            self.l[k] = 0
        m = l + r >> 1
        self.update(k << 1, l, m, s, e)
        self.update(k << 1 | 1, m + 1, r, s, e)
        # push up
        self.t[k] = self.l[k] + max(self.t[k << 1], self.t[k << 1 | 1])
        return

    def book(self, start: int, end: int) -> bool:
        if self.query(1, 0, 10**9, start, end - 1):  # whether >= 2
            return False
        self.update(1, 0, 10**9, start, end - 1)
        return True


class MyCalendarTwo:
    def __init__(self):
        self.t = collections.defaultdict(int)
        self.l = collections.defaultdict(int)

    def query(self, k: int, l: int, r: int, s: int, e: int) -> int:
        if s <= l and r <= e:
            return self.t[k]
        m = l + r >> 1
        if self.l[k] > 0:
            self.t[2 * k] += self.l[k]
            self.t[2 * k + 1] += self.l[k]
            self.l[2 * k] += self.l[k]
            self.l[2 * k + 1] += self.l[k]
            self.l[k] = 0
        ans = 0
        if s <= m:
            ans = max(ans, self.query(2 * k, l, m, s, e))
        if e > m:
            ans = max(ans, self.query(2 * k + 1, m + 1, r, s, e))
        return ans

    def update(self, k: int, l: int, r: int, s: int, e: int) -> None:
        if s > r or e < l:
            return
        if s <= l and r <= e:
            self.t[k] += 1
            self.l[k] += 1
            return
        m = l + r >> 1
        self.update(2 * k, l, m, s, e)
        self.update(2 * k + 1, m + 1, r, s, e)
        self.t[k] = self.l[k] + max(self.t[2 * k], self.t[2 * k + 1])
        return

    def book(self, start: int, end: int) -> bool:
        if self.query(1, 0, 10**9, start, end - 1) >= 2:
            return False
        self.update(1, 0, 10**9, start, end - 1)
        return True


class MyCalendarTwo:
    def __init__(self):
        self.tree = {}

    def update(self, start: int, end: int, val: int, l: int, r: int, idx: int) -> None:
        if r < start or end < l:
            return
        if start <= l and r <= end:
            p = self.tree.get(idx, [0, 0])
            p[0] += val
            p[1] += val
            self.tree[idx] = p
            return
        mid = (l + r) // 2
        self.update(start, end, val, l, mid, 2 * idx)
        self.update(start, end, val, mid + 1, r, 2 * idx + 1)
        p = self.tree.get(idx, [0, 0])
        p[0] = p[1] + max(
            self.tree.get(2 * idx, (0,))[0], self.tree.get(2 * idx + 1, (0,))[0]
        )
        self.tree[idx] = p

    def book(self, start: int, end: int) -> bool:
        self.update(start, end - 1, 1, 0, 10**9, 1)
        if self.tree[1][0] > 2:
            self.update(start, end - 1, -1, 0, 10**9, 1)
            return False
        return True


class MyCalendarTwo:
    def __init__(self):
        self.d = sortedcontainers.SortedDict()

    def book(self, start: int, end: int) -> bool:
        self.d[start] = self.d.get(start, 0) + 1
        self.d[end] = self.d.get(end, 0) - 1
        cnt = 0
        ans = True
        for v in self.d.values():
            cnt += v
            if cnt >= 3:
                ans = False
                break
        if not ans:
            self.d[start] -= 1
            self.d[end] += 1
        return ans


# 732 - My Calendar III - HARD
class MyCalendarThree:
    def __init__(self):
        self.t = collections.defaultdict(int)
        self.lazy = collections.defaultdict(int)

    def range_update(self, o: int, l: int, r: int, L: int, R: int):
        if L <= l and r <= R:
            self.t[o] += 1
            self.lazy[o] += 1
            return
        m = l + r >> 1
        if L <= m:
            self.range_update(o << 1, l, m, L, R)
        if m < R:
            self.range_update(o << 1 | 1, m + 1, r, L, R)
        self.t[o] = self.lazy[o] + max(self.t[o << 1], self.t[o << 1 | 1])
        return

    def book(self, startTime: int, endTime: int) -> int:
        self.range_update(1, 0, 10**9, startTime, endTime - 1)
        return self.t[1]


class Node:
    def __init__(self):
        self.left = self.right = None
        self.val = self.lazy = 0


class SegmentTree:
    def __init__(self):
        self.root = Node()

    def pushdown(self, o: Node) -> None:
        if not o.left:
            o.left = Node()
        if not o.right:
            o.right = Node()
        if o.lazy != 0:
            o.left.val += o.lazy
            o.right.val += o.lazy
            o.left.lazy += o.lazy
            o.right.lazy += o.lazy
            o.lazy = 0
        return

    def range_update(self, o: Node, l: int, r: int, L: int, R: int, val: int) -> None:
        if L <= l and r <= R:
            o.val += val
            o.lazy += val
            return
        self.pushdown(o)
        m = l + r >> 1
        if L <= m:
            self.range_update(o.left, l, m, L, R, val)
        if m < R:
            self.range_update(o.right, m + 1, r, L, R, val)
        o.val = max(o.left.val, o.right.val)
        return


class MyCalendarThree:
    def __init__(self):
        self.st = SegmentTree()

    def book(self, startTime: int, endTime: int) -> int:
        self.st.range_update(self.st.root, 0, 10**9, startTime, endTime - 1, 1)
        return self.st.root.val


# 733 - Flood Fill - EASY
class Solution:
    # bfs
    def floodFill(
        self, image: List[List[int]], sr: int, sc: int, newColor: int
    ) -> List[List[int]]:
        if image[sr][sc] == newColor:
            return image
        position, originalColor = [(sr, sc)], image[sr][sc]
        while position:
            pos = position.pop()
            image[pos[0]][pos[1]] = newColor
            for m in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if (
                    0 <= pos[0] + m[0] < len(image)
                    and 0 <= pos[1] + m[1] < len(image[0])
                    and image[pos[0] + m[0]][pos[1] + m[1]] == originalColor
                ):
                    position.append((pos[0] + m[0], pos[1] + m[1]))
        return image

    # dfs
    def floodFill(
        self, image: List[List[int]], sr: int, sc: int, newColor: int
    ) -> List[List[int]]:
        if image[sr][sc] == newColor:
            return image
        originalColor = image[sr][sc]

        def dfs(row: int, col: int):
            image[row][col] = newColor
            for m in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if (
                    0 <= row + m[0] < len(image)
                    and 0 <= col + m[1] < len(image[0])
                    and image[row + m[0]][col + m[1]] == originalColor
                ):
                    dfs(row + m[0], col + m[1])
            return

        dfs(sr, sc)
        return image

    # recursive
    def floodFill(
        self, image: List[List[int]], sr: int, sc: int, newColor: int
    ) -> List[List[int]]:
        if image[sr][sc] == newColor:
            return image
        originalColor = image[sr][sc]
        image[sr][sc] = newColor
        for x, y in [(sr + 1, sc), (sr - 1, sc), (sr, sc + 1), (sr, sc - 1)]:
            if (
                0 <= x < len(image)
                and 0 <= y < len(image[0])
                and image[x][y] == originalColor
            ):
                self.floodFill(image, x, y, newColor)
        return image


# 735 - Asteroid Collision - MEDIUM
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        r = []
        l = []
        for v in asteroids:
            if v > 0:
                r.append(v)
            while r and r[-1] < -v:
                r.pop()
            if r and r[-1] == -v:
                r.pop()
            elif not r:
                l.append(v)
        return l + r

    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        st = []
        for v in asteroids:
            f = True
            while f and v < 0 and st and st[-1] > 0:
                f = st[-1] < -v
                if st[-1] <= -v:
                    st.pop()
            if f:
                st.append(v)
        return st

    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        st = []
        for v in asteroids:
            if v > 0:
                st.append(v)
            else:
                while st and st[-1] > 0 and st[-1] < -v:
                    st.pop()
                if st and st[-1] > 0:
                    if st[-1] == -v:
                        st.pop()
                else:
                    st.append(v)
        return st


# 739 - Daily Temperatures - MEDIUM
class Solution:
    # O(n) / O(n)
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        st = [n - 1]  # 递减栈, 右边第一个比它大的元素
        for i in range(n - 2, -1, -1):
            while st and temperatures[st[-1]] <= temperatures[i]:
                st.pop()
            ans[i] = st[-1] - i if st else 0
            st.append(i)
        return ans

    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        st = []  # 递减栈, 右边第一个比它大的元素
        for i, v in enumerate(temperatures):
            while st and v > temperatures[st[-1]]:
                pre = st.pop()
                ans[pre] = i - pre
            st.append(i)
        return ans


# 740 - Delete and Earn - MEDIUM
class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        maxVal = max(nums)
        v = [0] * (maxVal + 1)
        for val in nums:
            v[val] += val
        pre, cur = 0, 0  # 198 - House Robber
        for i in range(1, len(v)):
            pre, cur = cur, max(pre + v[i], cur)
        return cur

    def deleteAndEarn(self, nums: List[int]) -> int:
        points = collections.defaultdict(int)
        size = 0
        for n in nums:
            points[n] += n
            size = max(size, n)
        dp = [0] * (size + 1)
        dp[1] = points[1]
        for i in range(2, len(dp)):
            dp[i] = max(dp[i - 1], dp[i - 2] + points[i])
        return dp[-1]


# 741 - Cherry Pickup - HARD
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        @functools.lru_cache(None)
        def dfs(x1, y1, x2, y2):
            # actually x1 + y1 == x2 + y2
            #       -> y2 = x1 + y1 - x2
            if not (0 <= x1 < m and 0 <= y1 < n and 0 <= x2 < m and 0 <= y2 < n):
                return -math.inf
            if grid[x1][y1] == -1 or grid[x2][y2] == -1:
                return -math.inf
            cur = 0
            if x1 != x2 or y1 != y2:
                cur += grid[x1][y1] + grid[x2][y2]
            elif grid[x1][y1] == 1:
                cur += 1
            if x1 == x2 == 0 and y1 == y2 == 0:  # if x1 == y1 == 0
                return cur
            a = dfs(x1 - 1, y1, x2 - 1, y2)
            b = dfs(x1 - 1, y1, x2, y2 - 1)
            c = dfs(x1, y1 - 1, x2 - 1, y2)
            d = dfs(x1, y1 - 1, x2, y2 - 1)
            cur += max(a, b, c, d)
            return cur

        m = len(grid)
        n = len(grid[0])
        return max(dfs(m - 1, n - 1, m - 1, n - 1), 0)

    def cherryPickup(self, grid: List[List[int]]) -> int:
        @functools.lru_cache(None)
        def dfs(r1, c1, c2):
            r2 = r1 + c1 - c2
            if r1 == m or r2 == m or c1 == m or c2 == m:
                return float("-inf")
            if grid[r1][c1] == -1 or grid[r2][c2] == -1:
                return float("-inf")
            if r1 == c1 == m - 1:
                return grid[r1][c1]
            else:
                cur = grid[r1][c1]
                if c1 != c2:
                    cur += grid[r2][c2]
                # TODO, why lru_cache still works
                cur += max(
                    dfs(r1, c1 + 1, c2 + 1),
                    dfs(r1, c1 + 1, c2),
                    dfs(r1 + 1, c1, c2),
                    dfs(r1 + 1, c1, c2 + 1),
                )
                return cur

        m = len(grid)
        return max(dfs(0, 0, 0), 0)


# 743 - Network Delay Time - MEDIUM
class Solution:
    # dijkstra, 边权为正的图
    # O(n * (n + m)) / O(n * m), m = len(times) -> Edge
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        g = [[math.inf] * n for _ in range(n)]
        for x, y, w in times:
            g[x - 1][y - 1] = w
        d = [math.inf] * n
        d[k - 1] = 0
        vis = [False] * n
        for _ in range(n):
            x = -1
            for y, v in enumerate(vis):
                if not v and (x == -1 or d[y] < d[x]):  # 每次选择一个已知到源节点距离最短的未访问节点
                    x = y
            vis[x] = True
            for y, w in enumerate(g[x]):
                d[y] = min(d[y], d[x] + w)
        ans = max(d)
        return ans if ans < math.inf else -1

    # O(n + mlogn) / O(n + m), m = len(times) -> Edge
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        g = [[] for _ in range(n)]
        for x, y, w in times:
            g[x - 1].append((y - 1, w))
        d = [math.inf] * n
        d[k - 1] = 0
        q = [(0, k - 1)]
        while q:
            w, x = heapq.heappop(q)  # 每次选择一个已知到源节点距离最短的未访问节点
            if d[x] < w:
                continue
            for y, w in g[x]:
                v = d[x] + w
                if v < d[y]:
                    d[y] = v
                    heapq.heappush(q, (v, y))
        ans = max(d)
        return ans if ans < math.inf else -1

    # 不是 dijkstra
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        g = collections.defaultdict(list)
        q = [(0, k)]
        vis = {}
        for x, y, w in times:
            g[x].append((y, w))
        while q:
            t, x = heapq.heappop(q)
            if x not in vis:
                vis[x] = t
                for v, w in g[x]:
                    heapq.heappush(q, (t + w, v))
        return max(vis.values()) if len(vis) == n else -1

    # SPFA(short path fast algorithm)
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        d = [0] + [math.inf] * n
        g = collections.defaultdict(list)
        q = collections.deque([(0, k)])
        for x, y, w in times:
            g[x].append((y, w))
        while q:
            t, x = q.popleft()
            if t < d[x]:
                d[x] = t
                for v, w in g[x]:
                    q.append((t + w, v))
        mx = max(d)
        return mx if mx < math.inf else -1


# 744 - Find Smallest Letter Greater Than Target - EASY
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        for ch in letters:
            if ch > target:
                return ch
        return letters[0]

    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        return next((ch for ch in letters if ch > target), letters[0])

    def nextGreatestLetter(self, l: List[str], target: str) -> str:
        return l[bisect.bisect_right(l, target)] if target < l[-1] else l[0]


# 745 - Prefix and Suffix Search - HARD
class WordFilter:
    def __init__(self, words: List[str]):
        self.d = dict()
        for i, w in enumerate(words):
            for j in range(0, len(w) + 1):
                p = w[:j]
                for k in range(0, len(w) + 1):
                    s = w[k:]
                    self.d[(p, s)] = i

    def f(self, pref: str, suff: str) -> int:
        return self.d.get((pref, suff), -1)


class WordFilter:
    def __init__(self, words: List[str]):
        self.p = {}
        self.s = {}
        for i, w in enumerate(words):
            r = self.p
            for c in w:
                if c not in r:
                    r[c] = {}
                r = r[c]
                if "MAX" in r:
                    r["MAX"].append(i)
                else:
                    r["MAX"] = [i]
        for i, w in enumerate(words):
            r = self.s
            for c in w[::-1]:
                if c not in r:
                    r[c] = {}
                r = r[c]
                if "MAX" in r:
                    r["MAX"].append(i)
                else:
                    r["MAX"] = [i]

    def f(self, pref: str, suff: str) -> int:
        r = self.p
        for c in pref:
            if c not in r:
                return -1
            r = r[c]
        p = r["MAX"]  # copy, incase p.pop() will change original arr
        r = self.s
        for c in suff[::-1]:
            if c not in r:
                return -1
            r = r[c]
        s = r["MAX"]
        i = len(p) - 1  # instead of using p.pop(), using two pointers
        j = len(s) - 1  # speed up, and reduce memory usage cuz no list copy
        while i > -1 and j > -1:
            if p[i] > s[j]:
                i -= 1
            elif p[i] < s[j]:
                j -= 1
            else:
                return p[i]
        return -1


class WordFilter:
    def __init__(self, words: List[str]):
        self.p = collections.defaultdict(set)
        self.s = collections.defaultdict(set)
        self.weights = {}
        for i, w in enumerate(words):
            p = s = ""
            for c in w:
                p += c
                self.p[p].add(w)
            for c in w[::-1]:
                s = c + s
                self.s[s].add(w)
            self.weights[w] = i

    def f(self, pref: str, suff: str) -> int:
        weight = -1
        for w in self.p[pref] & self.s[suff]:
            if self.weights[w] > weight:
                weight = self.weights[w]
        return weight


# 746 - Min Cost Climbing Stairs - EASY
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        for i in range(2, len(cost)):
            cost[i] = min(cost[i - 2], cost[i - 1]) + cost[i]
        return min(cost[-2], cost[-1])

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        prev = curr = 0
        for i in range(2, len(cost) + 1):
            nxt = min(curr + cost[i - 1], prev + cost[i - 2])
            prev, curr = curr, nxt
        return curr


# 747 - Largest Number At Least Twice of Others - EASY
class Solution:
    def dominantIndex(self, nums: List[int]) -> int:
        m1, m2, idx = -1, -1, 0
        for i, n in enumerate(nums):
            if n > m1:
                m1, m2, idx = n, m1, i
            elif n > m2:
                m2 = n
        return idx if m1 >= m2 * 2 else -1


# 748 - Shortest Completing Word - EASY
class Solution:
    def shortestCompletingWord(self, licensePlate: str, words: List[str]) -> str:
        licensePlate = [
            x for x in licensePlate.lower() if ord("a") <= ord(x) <= ord("z")
        ]
        s, cnt = set(licensePlate), collections.Counter(licensePlate)
        for word in sorted(words, key=lambda word: len(word)):
            word = [x for x in word.lower() if ord("a") <= ord(x) <= ord("z")]
            """
            Counter(word) - cnt: means that cnt(word) strictly larger than cnt
            not including the case: cnt(word) == cnt
            """
            if set(word).intersection(s) == s and not cnt - collections.Counter(word):
                return "".join(word)
        return ""

    # not use set, counter all character in each word, a little bit slow
    def shortestCompletingWord(self, licensePlate: str, words: List[str]) -> str:
        pc = collections.Counter(filter(str.isalpha, licensePlate.lower()))
        return min([w for w in words if collections.Counter(w) & pc == pc], key=len)


# 749 - Contain Virus - HARD
class Solution:
    def containVirus(self, isInfected: List[List[int]]) -> int:
        blank = set()
        virus = set()
        for i, row in enumerate(isInfected):
            for j, v in enumerate(row):
                if v:
                    virus.add((i, j))
                else:
                    blank.add((i, j))

        def dfs(i: int, j: int) -> int:
            if (i, j) in neib:
                return 0
            neib.add((i, j))
            wall = 0
            for x, y in (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1):
                if (x, y) in blank:
                    infect.add((x, y))
                    wall += 1
                elif (x, y) in virus:
                    wall += dfs(x, y)
            return wall

        ans = 0
        while virus and blank:
            seen = set()
            plan = (set(), set(), 0)
            area = []
            for i, j in virus:
                if (i, j) in seen:
                    continue
                infect = set()
                neib = set()
                walls = dfs(i, j)
                seen.update(neib)
                area.append((infect, neib, walls))
                if len(plan[0]) < len(infect):
                    plan = (infect, neib, walls)
            ans += plan[2]
            virus -= plan[1]
            for infect, neib, walls in area:
                if len(plan[0]) == len(infect):
                    continue
                if infect:
                    virus |= infect
                    blank -= infect
                else:
                    virus -= neib
        return ans

    def containVirus(self, isInfected: List[List[int]]) -> int:
        def dfs(i: int, j: int) -> None:
            vis[i][j] = True
            areas.append((i, j))
            for x, y in (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1):
                if 0 <= x < m and 0 <= y < n:
                    if isInfected[x][y] == 1 and not vis[x][y]:
                        dfs(x, y)
                    elif isInfected[x][y] == 0:
                        wall[0] += 1
                        infect.add((x, y))
            return

        m = len(isInfected)
        n = len(isInfected[0])
        ans = 0
        while 1:
            vis = [[False] * n for _ in range(m)]
            arr = []
            for i, row in enumerate(isInfected):
                for j, v in enumerate(row):
                    if v == 1 and not vis[i][j]:
                        areas = []
                        wall = [0]
                        infect = set()
                        dfs(i, j)
                        arr.append((infect, areas, wall[0]))
            if not arr:
                break
            arr.sort(key=lambda x: len(x[0]))
            _, area, w = arr.pop()
            ans += w
            for i, j in area:
                isInfected[i][j] = -1
            for _, area, _ in arr:
                for i, j in area:
                    for x, y in (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1):
                        if 0 <= x < m and 0 <= y < n and isInfected[x][y] == 0:
                            isInfected[x][y] = 1
        return ans


# 750


# 757 - Set Intersection Size At Least Two - HARD
class Solution:
    def intersectionSizeTwo(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: (x[1], -x[0]))
        # 1. no overlap
        # 2. only 1 number
        # 3. >= 2 number
        arr = [-1, -1]
        for x in intervals:
            if x[0] <= arr[-2]:
                continue
            if x[0] > arr[-1]:
                arr.append(x[1] - 1)
            arr.append(x[1])
        return len(arr) - 2


# 761 - Special Binary String - HARD
class Solution:
    def makeLargestSpecial(self, s: str) -> str:
        cnt = pre = 0
        arr = []
        for i in range(len(s)):
            cnt += 1 if s[i] == "1" else -1
            if cnt == 0:
                arr.append("1" + self.makeLargestSpecial(s[pre + 1 : i]) + "0")
                pre = i + 1
        return "".join(sorted(arr, reverse=True))


# 762 - Prime Number of Set Bits in Binary Representation - EASY
class Solution:
    # O((r - l) * sqrt(logr)) / O(1), how many '1': logr, isPrime:sqrt(x)
    def countPrimeSetBits(self, left: int, right: int) -> int:
        def count(n: int) -> int:
            cnt = 0
            while n:
                n = n & (n - 1)
                cnt += 1
            return cnt

        def isPrime(x: int) -> bool:
            if x < 2:
                return False
            i = 2
            while i * i <= x:
                if x % i == 0:
                    return False
                i += 1
            return True

        return sum(isPrime(count(x)) for x in range(left, right + 1))

    # O(r - l) / O(1)
    # right <= 10^6 < 2^20, so the number of 1's in binary will not exceed 19
    def countPrimeSetBits(self, left: int, right: int) -> int:
        primes = {2, 3, 5, 7, 11, 13, 17, 19}
        """
        'bit_count()' is a new feature in python3.10
        Return the number of ones in the binary representation of the absolute value of the integer
        
        def bit_count(self):
            return bin(self).count("1")
        """
        return sum(i.bit_count() in primes for i in range(left, right + 1))

    # mask = 665772 = 10100010100010101100
    # mask += 2 ^ x for x in [2 3 5 7 11 13 17 19]
    def countPrimeSetBits(self, left: int, right: int) -> int:
        return sum(((1 << x.bit_count()) & 665772) != 0 for x in range(left, right + 1))

    # for(int num=left;num<=right;++num){
    #     const int pos=__builtin_popcount(num);
    #     if(pos==2||pos==3||pos==5||pos==7||pos==11||pos==13||pos==17||pos==19){
    #         ans++;
    #     }
    # }


# 763 - Partition Labels - MEDIUM
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        last = [0] * 26
        for i, ch in enumerate(s):
            last[ord(ch) - ord("a")] = i
        partition = list()
        start = end = 0
        for i, ch in enumerate(s):
            end = max(end, last[ord(ch) - ord("a")])
            if i == end:
                partition.append(end - start + 1)
                start = end + 1
        return partition


# 764 - Largest Plus Sign - MEDIUM
class Solution:
    # O(n ** 3) / O(n), TLE
    def orderOfLargestPlusSign(self, n: int, mines: List[List[int]]) -> int:
        def calc(i: int, j: int) -> int:
            mx = 1
            q = ((i + 1, j), (i - 1, j), (i, j - 1), (i, j + 1))
            while all(0 <= x < n and 0 <= y < n and (x, y) not in s for x, y in q):
                q = (
                    (q[0][0] + 1, q[0][1]),
                    (q[1][0] - 1, q[1][1]),
                    (q[2][0], q[2][1] - 1),
                    (q[3][0], q[3][1] + 1),
                )
                mx += 1
            return mx

        s = set((x, y) for x, y in mines)
        ans = 0
        for i in range(n):
            for j in range(n):
                if (i, j) in s:
                    continue
                if min(i + 1, j + 1, n - i, n - j) <= ans:
                    continue
                u = d = l = r = 1
                while i - u >= 0 and (i - u, j) not in s:
                    u += 1
                while i + d < n and (i + d, j) not in s:
                    d += 1
                while j - l >= 0 and (i, j - l) not in s:
                    l += 1
                while j + r < n and (i, j + r) not in s:
                    r += 1
                ans = max(ans, min(u, d, l, r))
        return ans

    # O(n^2) / O(n)
    def orderOfLargestPlusSign(self, n: int, mines: List[List[int]]) -> int:
        grid = [[1] * n for _ in range(n)]
        for x, y in mines:
            grid[x][y] = 0
        up = [[0] * n for _ in range(n)]
        down = [[0] * n for _ in range(n)]
        left = [[0] * n for _ in range(n)]
        right = [[0] * n for _ in range(n)]
        for i in range(n):
            x = 0
            for j in range(n):
                x = 0 if grid[i][j] == 0 else x + 1
                left[i][j] = x
            x = 0
            for j in range(n)[::-1]:
                x = 0 if grid[i][j] == 0 else x + 1
                right[i][j] = x
        for j in range(n):
            x = 0
            for i in range(n):
                x = 0 if grid[i][j] == 0 else x + 1
                up[i][j] = x
            x = 0
            for i in range(n)[::-1]:
                x = 0 if grid[i][j] == 0 else x + 1
                down[i][j] = x
        return max(
            min(left[i][j], right[i][j], up[i][j], down[i][j])
            for i in range(n)
            for j in range(n)
        )


# 766 - Toeplitz Matrix - EASY
class Solution:
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        m = len(matrix)
        n = len(matrix[0])
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] != matrix[i - 1][j - 1]:
                    return False
        return True


# 767 - Reorganize String - MEDIUM
class Solution:
    # O(n * logk + n) / O(n)
    def reorganizeString(self, s: str) -> str:
        cnt = collections.Counter(s)
        if cnt.most_common(1)[0][1] > (len(s) + 1) // 2:
            return ""
        hp = [[-v, k] for k, v in cnt.items()]
        heapq.heapify(hp)
        ans = ""
        while hp:
            if len(hp) > 1:
                fst = heapq.heappop(hp)
                snd = heapq.heappop(hp)
                ans += fst[1] + snd[1]
                fst[0] += 1
                snd[0] += 1
                if fst[0] != 0:
                    heapq.heappush(hp, fst)
                if snd[0] != 0:
                    heapq.heappush(hp, snd)
            else:
                ans += hp[0][1]
                heapq.heappop(hp)
        return ans

    def reorganizeString(self, s: str) -> str:
        cnt = collections.Counter(s)
        hp = [[-v, k] for k, v in cnt.items()]
        heapq.heapify(hp)
        ans = ""
        pre = [1, 0]
        while hp:
            v, k = heapq.heappop(hp)
            ans += k
            if pre[0] < 0:
                heapq.heappush(hp, pre)
            pre = [v + 1, k]
        return "" if pre[0] < 0 else ans

    # O(n + C) / O(n)
    def reorganizeString(self, s: str) -> str:
        cnt = collections.Counter(s)
        bn = cnt.most_common(1)[0][1]
        if bn > (len(s) + 1) // 2:
            return ""
        buckets = [[] for _ in range(bn)]
        i = 0
        for c, v in cnt.most_common():
            while v:
                buckets[i].append(c)
                v -= 1
                i = (i + 1) % bn
        return "".join(c for b in buckets for c in b)

        return (
            "".join(c for b in buckets for c in b)
            if list(map(len, buckets)).count(1) <= 1
            else ""
        )


# 768 - Max Chunks To Make Sorted II - HARD
class Solution:
    # O(n) / O(n)
    def maxChunksToSorted(self, arr: List[int]) -> int:
        # leftMx[x] = max(arr[: x + 1])
        # rightMi[x] = min(arr[x:])
        leftMx = []
        cur = 0
        for v in arr:
            if v > cur:
                cur = v
            leftMx.append(cur)

        rightMi = []
        cur = 10**8
        for i in range(len(arr) - 1, -1, -1):
            if arr[i] < cur:
                cur = arr[i]
            rightMi.append(cur)
        rightMi.reverse()

        ans = 1
        for i in range(len(arr) - 1):
            if leftMx[i] <= rightMi[i + 1]:
                ans += 1
        return ans

    def maxChunksToSorted(self, arr: List[int]) -> int:
        n = len(arr)
        leftMx = [0] * n
        rightMi = [0] * n
        # arr[0] ~ arr[i] 最大值
        leftMx[0] = arr[0]
        for i in range(1, n):
            leftMx[i] = max(leftMx[i - 1], arr[i])
        # arr[i] ~ arr[n-1] 最小值
        rightMi[n - 1] = arr[n - 1]
        for i in range(n - 2, -1, -1):
            rightMi[i] = min(rightMi[i + 1], arr[i])
        # 可以在 i-1 和 i 之间切断
        ans = 1
        for i in range(1, n):
            ans += leftMx[i - 1] <= rightMi[i]
        return ans

    def maxChunksToSorted(self, arr: List[int]) -> int:
        st = []
        for v in arr:
            if not st or v >= st[-1]:
                st.append(v)
            else:
                mx = st.pop()
                while st and st[-1] > v:
                    st.pop()
                st.append(mx)
        return len(st)

    def maxChunksToSorted(self, arr: List[int]) -> int:
        st = [-1]
        for v in arr:
            mx = max(v, st[-1])
            while st[-1] > v:
                st.pop()
            st.append(mx)
        return len(st) - 1

    # O(nlogn) / O(n)
    def maxChunksToSorted(self, arr: List[int]) -> int:
        cnt = collections.Counter()
        ans = 0
        for x, y in zip(arr, sorted(arr)):
            cnt[x] += 1
            if cnt[x] == 0:
                del cnt[x]
            cnt[y] -= 1
            if cnt[y] == 0:
                del cnt[y]
            if len(cnt) == 0:
                ans += 1
        return ans


# 769 - Max Chunks To Make Sorted - MEDIUM
class Solution:
    # all solutions in 768 can solve this problem

    # other way: unique element + range(0, n-1) -> partition each chunk by index
    def maxChunksToSorted(self, arr: List[int]) -> int:
        ans = mx = 0
        for i, v in enumerate(arr):
            mx = max(v, mx)
            ans += mx == i
        return ans


# 771 - Jewels and Stones - EASY
class Solution:
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        cnt = collections.Counter(stones)
        return sum(cnt[j] for j in jewels)


#################
# 2022.03.30 VO #
#################
# 772 - Basic Calculator III - HARD - PREMIUM
# Write a calculator with support for +, -, *, / and parentheses.
# Assume the input expression is valid -> no /0, no 1+-2, no other operator
# and doesn't contain decimals. ->
# Also, assume all numbers are within the limit of double.
# input -> string
# output -> double
def calc(exp: str, start=0):
    stack = []
    ans = 0
    cur = 0
    f = "+"
    new = exp[start:]
    new += "#"  # as a stop label
    for i, ch in enumerate(new):
        if ch.isdigit():
            cur = cur * 10 + int(ch)
        elif ch == "(":
            if f == "+":
                stack.append(cur)
            elif f == "-":
                stack.append(-cur)
            elif f == "*":
                stack.append(stack.pop() * cur)
            elif f == "/":
                stack.append(stack.pop() / cur)
            cur = calc(exp, i + 1)
        elif ch == ")":
            return sum(stack) + cur
        else:
            if f == "+":
                stack.append(cur)
            elif f == "-":
                stack.append(-cur)
            elif f == "*":
                stack.append(stack.pop() * cur)
            elif f == "/":
                stack.append(stack.pop() / cur)
            f = ch
            cur = 0
    ans = sum(stack)
    return ans / 1.0


# print(calc("1+2"))  # => 3
# print(calc("1+2*3"))  # => 7
# print(calc("1+2*3/3"))  # => 3
# print(calc("1/3+2*3"))  # => 6.333333
# print(calc("1/3*6+2*3"))  # => 8


#################
# 2022.11.01 VO #
#################
# 773 - Sliding Puzzle - HARD
class Solution:
    # O((mn)! * mn * 4) / O((mn)! * mn), factorial complexity
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        adj = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4], 4: [1, 3, 5], 5: [2, 4]}

        # def toString(arr: List[int]) -> str:
        #     return "".join([str(v) for v in arr])

        toString = lambda arr: "".join([str(v) for v in arr])
        step = 0
        init = [v for row in board for v in row]
        dq = collections.deque([init])
        seen = set([toString(init)])
        while dq:
            k = len(dq)
            for _ in range(k):
                cur = dq.popleft()
                if toString(cur) == "123450":
                    return step
                i = cur.index(0)
                for j in adj[i]:
                    nxt = cur.copy()
                    nxt[i], nxt[j] = nxt[j], nxt[i]
                    if not toString(nxt) in seen:
                        dq.append(nxt)
                        seen.add(toString(nxt))
            step += 1
        return -1

    def slidingPuzzle(self, board: List[List[int]]) -> int:
        def neighbors(s: str) -> List[str]:
            d = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4], 4: [1, 3, 5], 5: [2, 4]}
            i = s.find("0")
            nei = []
            for j in d[i]:
                t = list(s)
                t[i], t[j] = t[j], t[i]
                nei.append("".join(t))
                # nei += ("".join(t),)
            return nei

        def neighbors(s: str) -> List[str]:
            i = s.find("0")
            nei = []
            x, y = i // c, i % c
            for nx, ny in (x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y):
                if 0 <= nx < r and 0 <= ny < c:
                    l = list(s)
                    l[i], l[nx * c + ny] = l[nx * c + ny], "0"
                    nei.append("".join(l))
            return nei

        r = len(board)
        c = len(board[0])
        start = "".join([str(v) for row in board for v in row])
        dq = collections.deque([(start, 0)])
        seen = {start}
        while dq:
            cur, step = dq.popleft()
            if cur == "123450":
                return step
            for v in neighbors(cur):
                if v not in seen:
                    seen.add(v)
                    dq += ((v, step + 1),)
        return -1

    def slidingPuzzle(self, board: List[List[int]]) -> int:
        s = "".join(str(v) for row in board for v in row)
        dq = collections.deque([(s, s.index("0"))])
        seen = {s}
        r = len(board)
        c = len(board[0])
        step = 0
        while dq:
            for _ in range(len(dq)):
                cur, i = dq.popleft()
                if cur == "123450":
                    return step
                x, y = i // c, i % c
                for nx, ny in (x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y):
                    if 0 <= nx < r and 0 <= ny < c:
                        l = [ch for ch in cur]
                        l[i], l[nx * c + ny] = l[nx * c + ny], "0"
                        s = "".join(l)
                        if s not in seen:
                            seen.add(s)
                            dq.append((s, nx * c + ny))
            step += 1
        return -1

    # more general way, r rows and c columns
    # convert a two-dimensional matrix to a one-dimensional matrix
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        r = len(board)
        c = len(board[0])
        arr = [None] * (r * c)
        for i in range(r):
            for j in range(c):
                arr[i * c + j] = board[i][j]

        target = [0] * (r * c)
        for i in range(1, r * c):
            target[i - 1] = i

        step = 0
        q = [arr]
        seen = set()
        while q:
            new = []
            for l in q:
                if l == target:
                    return step
                if tuple(l) not in seen:
                    seen.add(tuple(l))
                    i = l.index(0)
                    x, y = divmod(i, c)
                    # x, y = i // c, i % c
                    for nx, ny in (x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y):
                        if 0 <= nx < r and 0 <= ny < c:
                            nxt = l.copy()
                            nxt[nx * c + ny], nxt[i] = nxt[i], nxt[nx * c + ny]
                            new.append(nxt)
            step += 1
            q = new
        return -1


# 775 - Global and Local Inversions - MEDIUM
class Solution:
    # O(nlogn) / O(n), 暴力
    def isIdealPermutation(self, nums: List[int]) -> bool:
        loc = sum(nums[i] > nums[i + 1] for i in range(len(nums) - 1))
        glo = 0
        arr = sortedcontainers.SortedList()
        arr.add(nums[0])
        for i in range(1, len(nums)):
            glo += len(arr) - arr.bisect_right(nums[i])
            arr.add(nums[i])
        return loc == glo

    def isIdealPermutation(self, nums: List[int]) -> bool:
        loc = sum(nums[i] > nums[i + 1] for i in range(len(nums) - 1))
        glo = 0
        arr = [nums[0]]
        for i in range(1, len(nums)):
            glo += len(arr) - bisect.bisect_right(arr, nums[i])
            bisect.insort_right(arr, nums[i])  # insort, 搜索 O(logn), 插入 O(n), 1e5 还是能过的
        return loc == glo

    # 局部倒置一定是全局倒置
    # -> 检查有没有非局部倒置
    # -> nums[i] > nums[j], i < j - 1
    # -> 优化: 维护一个最小后缀, 检查间隔 > 1
    # O(n) / O(1)
    def isIdealPermutation(self, nums: List[int]) -> bool:
        minSuf = nums[-1]
        for i in range(len(nums) - 2, 0, -1):
            if nums[i - 1] > minSuf:
                return False
            minSuf = min(minSuf, nums[i])
        return True

    # nums 由 0 - n-1 组成, 每个数出现一次, 且下标差不能大于 1
    # O(n) / O(1)
    def isIdealPermutation(self, nums: List[int]) -> bool:
        return not any((abs(v - i) > 1 for i, v in enumerate(nums)))
        # return all((abs(v - i) <= 1 for i, v in enumerate(nums)))


# 777 - Swap Adjacent in LR String - MEDIUM
class Solution:
    def canTransform(self, start: str, end: str) -> bool:
        if "".join(start.replace("X", "")) != "".join(end.replace("X", "")):
            return False
        sl = [i for i, v in enumerate(start) if v == "L"]
        sr = [i for i, v in enumerate(start) if v == "R"]
        el = [i for i, v in enumerate(end) if v == "L"]
        er = [i for i, v in enumerate(end) if v == "R"]
        if any(a > b for a, b in zip(sr, er)):
            return False
        if any(a < b for a, b in zip(sl, el)):
            return False
        return True

    def canTransform(self, start: str, end: str) -> bool:
        l = r = 0
        for c1, c2 in zip(start, end):
            if (c1 == "L" and r > 0) or (c1 == "R" and l < 0):
                return False
            l += c1 == "L"
            r += c1 == "R"
            l -= c2 == "L"
            r -= c2 == "R"
            if (l != 0 and r != 0) or l > 0 or r < 0:
                return False
        return l == r == 0


# 780 - Reaching Points - HARD
class Solution:
    # O(log max(tx, ty)) / O(1)
    def reachingPoints(self, sx: int, sy: int, tx: int, ty: int) -> bool:
        while sx < tx and sy < ty:
            if tx > ty:
                tx %= ty
            else:
                ty %= tx
        if tx == sx and ty == sy:
            return True
        elif tx == sx:
            return ty > sy and (ty - sy) % tx == 0
        elif ty == sy:
            return tx > sx and (tx - sx) % ty == 0
        return False

    def reachingPoints(self, sx, sy, tx, ty):
        while sx < tx and sy < ty:
            tx, ty = tx % ty, ty % tx
        return (
            sx == tx
            and sy <= ty
            and (ty - sy) % sx == 0
            or sy == ty
            and sx <= tx
            and (tx - sx) % sy == 0
        )

    def reachingPoints(self, sx: int, sy: int, tx: int, ty: int) -> bool:
        if sx > tx or sy > ty:
            return False
        if (
            sx == tx
            and sy == ty
            or sx == tx
            and (ty - sy) % sx == 0
            or (tx - sx) % sy == 0
            and sy == ty
        ):
            return True
        if tx > ty:
            return self.reachingPoints(sx, sy, tx % ty, ty)
        return self.reachingPoints(sx, sy, tx, ty % tx)


# 782 - Transform to Chessboard - HARD
class Solution:
    def movesToChessboard(self, board: List[List[int]]) -> int:
        def check():
            if abs(n - sum(board[0]) - sum(board[0])) > 1:
                return False
            if abs(n - sum(list(zip(*board))[0]) - sum(list(zip(*board))[0])) > 1:
                return False
            f = 0
            for i, x in enumerate(board[0]):
                f |= x << i
            for b in board[1:]:
                tmp = 0
                for i, x in enumerate(b):
                    tmp |= x << i
                if not (f == tmp or f == ((1 << n) - 1) ^ tmp):
                    return False
            return True

        row = board[0]
        cnt = collections.Counter(row)
        if abs(cnt[0] - cnt[1]) > 1:
            return -1

        col = [x[0] for x in board]
        cnt = collections.Counter(col)
        if abs(cnt[0] - cnt[1]) > 1:
            return -1

        n = len(board)
        for i in range(n):
            for j in range(n):
                if board[i][j] ^ board[0][0] ^ board[i][0] ^ board[0][j] != 0:
                    return -1

        # if not check():
        #     return -1

        def fn(lst: List[int]) -> int:
            if len(lst) % 2:
                cnt = collections.Counter(lst)
                starter = 1 if cnt[1] > cnt[0] else 0
                ans = 0  # assume start with 0, 010101...
                for i, v in enumerate(lst):
                    ans += (i + v - starter) % 2
                return ans // 2
            ans = 0
            for i, v in enumerate(lst):
                ans += (i + v) % 2
            return min(ans, n - ans) // 2

        return fn(row) + fn(col)


# 784 - Letter Case Permutation - MEDIUM
class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
        ans = [""]
        for ch in s:
            if ch.isalpha():
                ans = [x + y for x in ans for y in [ch.upper(), ch.lower()]]
            else:
                ans = [x + ch for x in ans]
        return ans

    def letterCasePermutation(self, s: str) -> List[str]:
        ans = [""]
        for c in s.lower():
            ans = [x + c for x in ans] + (
                [x + c.upper() for x in ans] if c.isalpha() else []
            )
        return ans

        ## L = [['a', 'A'], '1', ['b', 'B'], '2']
        ## itertools.product(L) --> only 1 parameter [['a', 'A'], '1', ['b', 'B'], '2']
        ## itertools.product(*L) --> 4 parameter ['a', 'A'], '1', ['b', 'B'], '2'
        # L = [set([i.lower(), i.upper()]) for i in s]
        # return map(''.join, itertools.product(*L))

    def letterCasePermutation(self, s: str) -> List[str]:
        def backtrack(sub: str, i: int):
            if len(sub) == len(s):
                ans.append(sub)
            else:
                if s[i].isalpha():
                    # chr(ord(s[i]) ^ (1 << 5))
                    backtrack(sub + s[i].swapcase(), i + 1)
                    # backtrack(sub + s[i].lower(), i + 1)
                backtrack(sub + s[i], i + 1)
                # backtrack(sub + s[i].upper(), i + 1)

        ans = []
        backtrack("", 0)
        return ans


# 785
# 786 - K-th Smallest Prime Fraction - HARD
class Solution:
    def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
        div = []
        for i in range(len(arr) - 1):
            for j in range(i + 1, len(arr)):
                div.append((arr[i], arr[j]))
        div.sort(key=lambda x: x[0] / x[1])
        return div[k - 1]

    def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
        left, right = 0, 1
        while True:
            mid = (left + right) / 2
            i, count, x, y = -1, 0, 0, 1
            for j in range(1, len(arr)):
                while arr[i + 1] / arr[j] < mid:
                    i += 1
                    # a/b > c/d => a*d > b*c
                    # update the max fraction
                    if arr[i] * y > arr[j] * x:
                        x, y = arr[i], arr[j]
                count += i + 1

            if count > k:
                right = mid
            if count < k:
                left = mid
            else:
                return [x, y]


# 787


# 788 - Rotated Digits - MEDIUM
class Solution:
    # O(nlogn) / O(logn)
    def rotatedDigits(self, n: int) -> int:
        ans = 0
        for x in range(n + 1):
            ok = False
            while x:
                if x % 10 in [2, 5, 6, 9]:
                    ok = True
                if x % 10 in [3, 4, 7]:
                    ok = False
                    break
                x //= 10
            ans += ok
        return ans

    # O(logn) / O(logn)
    def rotatedDigits(self, n: int) -> int:
        can = [0, 0, 1, -1, -1, 1, 1, -1, 0, 1]
        s = str(n)

        @functools.lru_cache(None)
        def dfs(i: int, is_limit: bool, has2579: bool) -> int:
            if i == len(s):
                return int(has2579)
            ans = 0
            bound = int(s[i]) if is_limit else 9
            for d in range(0, bound + 1):
                if can[d] != -1:
                    ans += dfs(i + 1, is_limit and d == bound, has2579 or can[d] == 1)
            return ans

        return dfs(0, True, False)


# 789


# 790 - Domino and Tromino Tiling - MEDIUM
class Solution:
    # 需要考虑每列上下两个块分别的状态, f[i][s]: 表示平铺到第 i 列时, 各个状态 s 对应的平铺方法数量
    # f[i][0]: 都不覆盖, f[i][1]: 上块被覆盖, f[i][2]: 下块被覆盖, f[i][3]: 都被覆盖
    def numTilings(self, n: int) -> int:
        f = [[0] * 4 for _ in range(n + 1)]
        f[0][3] = 1
        mod = 10**9 + 7
        for i in range(1, n + 1):
            f[i][0] = f[i - 1][3]  # |
            f[i][1] = (f[i - 1][0] + f[i - 1][2]) % mod  # L -
            f[i][2] = (f[i - 1][0] + f[i - 1][1]) % mod  # L -
            f[i][3] = (
                f[i - 1][0] + f[i - 1][1] + f[i - 1][2] + f[i - 1][3]
            ) % mod  # = L L |
        return f[-1][3]


# 791 - Custom Sort String - MEDIUM
class Solution:
    def customSortString(self, order: str, s: str) -> str:
        cnt = collections.Counter(s)
        ans = ""
        for c in order:
            if c in cnt:
                ans += c * cnt[c]
                cnt.pop(c)
        return ans + "".join(c * cnt[c] for c in cnt)

    def customSortString(self, order: str, s: str) -> str:
        cnt = collections.Counter(s)
        ans = ""
        for c in order:
            ans += cnt[c] * c
            del cnt[c]
        for k, v in cnt.items():
            ans += k * v
        return ans


# 792 - Number of Matching Subsequences - MEDIUM
class Solution:
    # O(n + mw) / O(mw), m = len(words), w = len(words[i])
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        d = collections.defaultdict(list)
        for w in words:
            d[w[0]].append(w)
        t = len(words)
        for c in s:
            q = d[c]
            d[c] = []
            for w in q:
                if len(w) > 1:
                    d[w[1]].append(w[1:])

            # same = []  # in case repeated pops of the same letter
            # while d[c]:
            #     w = d[c].pop()
            #     if len(w) > 1:
            #         if w[1] == c:
            #             same.append(w[1:])
            #         else:
            #             d[w[1]].append(w[1:])
            # d[c] = same

        return t - sum(len(v) for v in d.values())

    # n 指针 / 多指针, 节省空间
    # O(n + mw) / O(m), m = len(words), w = len(words[i])
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        d = collections.defaultdict(list)
        for i, w in enumerate(words):
            d[w[0]].append((i, 0))
        ans = 0
        for c in s:
            q = d[c]
            d[c] = []
            for i, j in q:
                j += 1
                if j == len(words[i]):
                    ans += 1
                else:
                    d[words[i][j]].append((i, j))
        return ans

    # 二分, 不断查找 word[i] 的起始位置
    # O(mwlogn) / O(n), m = len(words), w = len(words[i])
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        pos = collections.defaultdict(list)
        for i, c in enumerate(s):
            pos[c].append(i)
        ans = len(words)
        for w in words:
            p = -1
            for c in w:
                j = bisect.bisect_right(pos[c], p)
                if j == len(pos[c]):
                    ans -= 1
                    break
                p = pos[c][j]
        return ans


# 793 - Preimage Size of Factorial Zeroes Function - HARD
class Solution:
    def preimageSizeFZF(self, k: int) -> int:
        summ = [1] * 12
        for i in range(1, 12):
            summ[i] = summ[i - 1] * 5 + 1
        for i in range(11, -1, -1):
            if k // summ[i] == 5:
                return 0
            k %= summ[i]
        return 5


# 794 - Valid Tic-Tac-Toe State - MEDIUM
class Solution:
    def validTicTacToe(self, board: List[str]) -> bool:
        return


# 795 - Number of Subarrays with Bounded Maximum - MEDIUM
class Solution:
    # 最大元素满足大于等于 L 小于等于 R 的子数组个数 = 最大元素小于等于 R 的子数组个数 - 最大元素小于 L 的子数组个数
    def numSubarrayBoundedMax(self, nums: List[int], left: int, right: int) -> int:
        def calc(x: int) -> int:
            ans = l = 0
            for r, v in enumerate(nums):
                if v > x:
                    l = r + 1
                ans += r - l + 1
            return ans

        return calc(right) - calc(left - 1)

    def numSubarrayBoundedMax(self, nums: List[int], left: int, right: int) -> int:
        ans = l = tmp = 0
        for r, v in enumerate(nums):
            if v > right:
                l = r + 1
            if v > left - 1:
                tmp = r - l + 1
            ans += tmp
        return ans


# 796 - Rotate String - EASY
class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        for _ in range(len(s)):
            if s == goal:
                return True
            s = "".join(list(s)[-len(s) + 1 :]) + s[0]
        return False

    def rotateString(self, s: str, goal: str) -> bool:
        for _ in range(len(s)):
            if s[_:] + s[:_] == goal:
                return True
        return False

    def rotateString(self, s: str, goal: str) -> bool:
        return len(s) == len(goal) and goal in s + s


# 797 - All Paths From Source to Target - MEDIUM
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        def dfs(cur: int, path: List[int]):
            if cur == len(graph) - 1:
                ret.append(path)
            else:
                for i in graph[cur]:
                    dfs(i, path + [i])
            return

        ret = []
        dfs(0, [0])
        return ret

    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        stack, ret = [(0, [0])], []
        while stack:
            cur, path = stack.pop()
            if cur == len(graph) - 1:
                ret.append(path)
            for nei in graph[cur]:
                stack.append((nei, path + [nei]))
        return ret


# 798

# 799
