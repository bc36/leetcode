import itertools, bisect, heapq, collections, math, functools
from typing import List, Optional, Tuple


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 605 - Can Place Flowers - EASY
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        f = [0] + flowerbed + [0]
        for i in range(1, len(f) - 1):
            if f[i - 1] == f[i] == f[i + 1] == 0:
                n -= 1
                f[i] = 1
        return n <= 0


# 611 - Valid Triangle Number - MEDIUM
class Solution:
    # O(n ** 2) / O(logn)
    def triangleNumber(self, nums: List[int]) -> int:
        nums.sort()
        ans = 0
        for i in range(len(nums) - 1, 1, -1):
            j = i - 1
            k = 0
            while k < j:
                if nums[k] + nums[j] > nums[i]:
                    ans += j - k
                    j -= 1
                else:
                    k += 1
        return ans


# 617 - Merge Two Binary Trees - EASY
class Solution:
    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
        if not root1:
            return root2
        if not root2:
            return root1
        root1.val += root2.val
        root1.left = self.mergeTrees(root1.left, root2.left)
        root1.right = self.mergeTrees(root1.right, root2.right)
        return root1

    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
        if not root1:
            return root2
        if not root2:
            return root1
        root = TreeNode(root1.val + root2.val)
        root.left = self.mergeTrees(root1.left, root2.left)
        root.right = self.mergeTrees(root1.right, root2.right)
        return root

    def mergeTrees(self, root1: TreeNode, root2: TreeNode) -> TreeNode:
        if root1 and root2:
            root1.val += root2.val
            root1.left = self.mergeTrees(root1.left, root2.left)
            root1.right = self.mergeTrees(root1.right, root2.right)
        return root1 or root2


# 622 - Design Circular Queue - MEDIUM
class MyCircularQueue:
    def __init__(self, k: int):
        self.dq = collections.deque(maxlen=k)
        self.k = k

    def enQueue(self, value: int) -> bool:
        if len(self.dq) == self.k:
            return False
        self.dq.append(value)
        return True

    def deQueue(self) -> bool:
        if self.dq:
            self.dq.popleft()
            return True
        return False

    def Front(self) -> int:
        if self.dq:
            return self.dq[0]
        return -1

    def Rear(self) -> int:
        if self.dq:
            return self.dq[-1]
        return -1

    def isEmpty(self) -> bool:
        return len(self.dq) == 0

    def isFull(self) -> bool:
        return len(self.dq) == self.k


class MyCircularQueue:
    def __init__(self, k: int):
        self.front = self.rear = 0
        self.arr = [0] * (k + 1)

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        self.arr[self.rear] = value
        self.rear = (self.rear + 1) % len(self.arr)
        return True

    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        self.front = (self.front + 1) % len(self.arr)
        return True

    def Front(self) -> int:
        return -1 if self.isEmpty() else self.arr[self.front]

    def Rear(self) -> int:
        return -1 if self.isEmpty() else self.arr[(self.rear - 1) % len(self.arr)]

    def isEmpty(self) -> bool:
        return self.rear == self.front

    def isFull(self) -> bool:
        return (self.rear + 1) % len(self.arr) == self.front


# 623 - Add One Row to Tree - MEDIUM
class Solution:
    def addOneRow(self, root: TreeNode, val: int, depth: int) -> TreeNode:
        if depth == 1:
            return TreeNode(val, left=root)
        q = [root]
        for _ in range(1, depth - 1):
            new = []
            for node in q:
                if node.left:
                    new.append(node.left)
                if node.right:
                    new.append(node.right)
            q = new
        for node in q:
            node.left = TreeNode(val, left=node.left)
            node.right = TreeNode(val, right=node.right)
        return root

    def addOneRow(self, root: TreeNode, val: int, depth: int) -> TreeNode:
        if depth == 1:
            return TreeNode(val, root)
        dq = collections.deque([root])
        lv = 1
        while dq:
            if lv + 1 == depth:
                for _ in range(len(dq)):
                    n = dq.popleft()
                    n.left = TreeNode(val, left=n.left)
                    n.right = TreeNode(val, right=n.right)
                return root
            for _ in range(len(dq)):
                n = dq.popleft()
                if n.left:
                    dq.append(n.left)
                if n.right:
                    dq.append(n.right)
            lv += 1
        return root


# 630 - Course Schedule III - HARD
class Solution:
    # O(nlogn) / O(n)
    def scheduleCourse(self, courses: List[List[int]]) -> int:
        courses.sort(key=lambda c: c[1])
        pq = []
        t = 0
        for d, ddl in courses:
            t += d
            if t > ddl:
                t += heapq.heappushpop(pq, -d)
                # t -= -heapq.heappushpop(pq, -d)
            else:
                heapq.heappush(pq, -d)

            # or
            # if t + d <= ddl:
            #     t += d
            #     heapq.heappush(pq, -d)
            # elif pq and -pq[0] > d:
            #     # t += heapq.heappushpop(pq, -d) + d
            #     t -= -pq[0] - d
            #     heapq.heappop(pq)
            #     heapq.heappush(pq, -d)
        return len(pq)

    def scheduleCourse(self, courses: List[List[int]]) -> int:
        pq = []
        t = 0
        for d, ddl in sorted(courses, key=lambda x: x[1]):
            if t + d > ddl and pq and -pq[0] > d:
                t += heapq.heappop(pq)
            if t + d <= ddl:
                heapq.heappush(pq, -d)
                t += d
        return len(pq)


# 633 - Sum of Square Numbers - MEDIUM
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        i = 0
        j = int(c**0.5)  # (int(c**0.5) + 1) ** 2 is larger than c
        while i <= j:
            p = i * i + j * j
            if p > c:
                j -= 1
            elif p < c:
                i += 1
            else:
                return True
        return False


# 636 - Exclusive Time of Functions - MEDIUM
class Solution:
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        ans = [0] * n
        total = 0
        st = []
        for l in logs:
            i, cmd, cur = l.split(":")
            i = int(i)
            cur = int(cur)
            if cmd == "start":
                st.append((i, cur, total))
            else:
                _, pre, other_task = st.pop()
                used = cur - pre + 1 - (total - other_task)
                ans[i] += used
                total += used
        return ans

    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        ans = [0] * n
        pre = -1
        st = []
        for l in logs:
            i, cmd, cur = l.split(":")
            i = int(i)
            cur = int(cur)
            if cmd == "start":
                if st:
                    ans[st[-1]] += cur - pre
                st.append(i)
                pre = cur
            else:
                idx = st.pop()
                ans[idx] += cur - pre + 1
                pre = cur + 1
        return ans


# 637 - Average of Levels in Binary Tree - EASY
class Solution:
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        ans = []
        q = [root]
        while q:
            t = summ = 0
            new = []
            for _ in range(len(q)):
                n = q.pop()
                if n.left:
                    new.append(n.left)
                if n.right:
                    new.append(n.right)
                t += 1
                summ += n.val
            ans.append(summ / t)
            q = new
        return ans


# 640 - Solve the Equation - MEDIUM
class Solution:
    def solveEquation(self, equation: str) -> str:
        def calc(s: str) -> Tuple[int, int]:
            x = val = n = 0
            f = 1
            near_f = True  # to handle "0x=0", "0x=1"
            num = set("0123456789")
            for c in s:
                if c == "=":
                    val += n * f
                    break
                if c == "-":
                    near_f = True
                    val += n * f
                    f = -1
                    n = 0
                if c == "+":
                    near_f = True
                    val += n * f
                    f = 1
                    n = 0
                if c in num:
                    near_f = False
                    n = 10 * n + int(c)
                if c == "x":
                    if near_f:
                        x += 1 * f
                    else:
                        x += n * f
                    n = 0
            return x, val

        idx = equation.index("=")
        x, left = calc(equation[: idx + 1])
        equation += "="
        xx, right = calc(equation[idx + 1 :])
        a = x - xx
        b = right - left
        if a == 0:
            return "No solution" if b else "Infinite solutions"
        return f"x={b // a}"


# 641 - Design Circular Deque - MEDIUM
class MyCircularDeque:
    def __init__(self, k: int):
        self.arr = [0] * (k + 1)
        self.l = k + 1
        self.f = 0  # front pointer
        self.r = 0  # rear pointer

    def insertFront(self, value: int) -> bool:
        if self.isFull():
            return False
        self.f = (self.f - 1) % self.l
        self.arr[self.f] = value
        return True

    def insertLast(self, value: int) -> bool:
        if self.isFull():
            return False
        self.arr[self.r] = value
        self.r = (self.r + 1) % self.l
        return True

    def deleteFront(self) -> bool:
        if self.isEmpty():
            return False
        self.f = (self.f + 1) % self.l
        return True

    def deleteLast(self) -> bool:
        if self.isEmpty():
            return False
        self.r = (self.r - 1) % self.l
        return True

    def getFront(self) -> int:
        return -1 if self.isEmpty() else self.arr[self.f]

    def getRear(self) -> int:
        return -1 if self.isEmpty() else self.arr[(self.r - 1) % self.l]

    def isEmpty(self) -> bool:
        return self.f == self.r

    def isFull(self) -> bool:
        return (self.r + 1) % self.l == self.f


# 643 - Maximum Average Subarray I - EASY
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        mx = w = sum(nums[:k])
        for i in range(k, len(nums)):
            w += nums[i] - nums[i - k]
            mx = max(mx, w)
        return mx / k


# 645 - Set Mismatch - EASY
class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        arr = [1] * len(nums)
        f = s = 0
        for n in nums:
            if arr[n - 1] == 0:
                f = n
            else:
                arr[n - 1] -= 1
        for n in range(1, len(nums) + 1):
            if arr[n - 1]:
                s = n
                break
        return [f, s]

    def findErrorNums(self, nums):
        cnt = collections.Counter(nums)
        f = s = -1
        for n in range(1, len(nums) + 1):
            v = cnt.get(n, 0)
            if v == 0:
                s = n
            elif v == 2:
                f = n
        return [f, s]

    def findErrorNums(self, nums):
        n = len(nums)
        summ = sum(set(nums))
        return [sum(nums) - summ, (1 + n) * n // 2 - summ]


# 648 - Replace Words - MEDIUM
class Solution:
    # O(sdw + dlogd) / O(s + logd), w = len(each word in d)
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        sentence = sentence.split()
        dictionary.sort()  # ori: catt, cat -> sorted: cat, catt
        for i, s in enumerate(sentence):
            for d in dictionary:
                if s.startswith(d):
                    sentence[i] = d
                    break
        return " ".join(sentence)

    # O(dw + sw) / O(dw + s), w = len(each word in s)
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        trie = dict()
        for w in dictionary:
            r = trie
            for c in w:
                if c not in r:
                    r[c] = {}
                r = r[c]
            r["#"] = 3  # ending flag
        sentence = sentence.split()
        for i, w in enumerate(sentence):
            r = trie
            for j, c in enumerate(w):
                if "#" in r:
                    sentence[i] = w[:j]
                    break
                if c not in r:
                    break
                r = r[c]
        return " ".join(sentence)


# 653 - Two Sum IV - Input is a BST - EASY
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        if not root:
            return False
        bfs, s = [root], set()
        for i in bfs:
            if k - i.val in s:
                return True
            s.add(i.val)
            if i.left:
                bfs.append(i.left)
            if i.right:
                bfs.append(i.right)
        return False


# 654 - Maximum Binary Tree - MEDIUM
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None
        mx = p = 0
        for i, v in enumerate(nums):
            if v > mx:
                p = i
                mx = v

        r = TreeNode(val=mx)
        r.left = self.constructMaximumBinaryTree(nums[:p])
        r.right = self.constructMaximumBinaryTree(nums[p + 1 :])
        return r


# 655 - Print Binary Tree - MEDIUM
class Solution:
    def printTree(self, root: Optional[TreeNode]) -> List[List[str]]:
        def depth(root: TreeNode) -> int:
            return max(
                depth(root.left) + 1 if root.left else 0,
                depth(root.right) + 1 if root.right else 0,
            )
            if root == None:
                return -1
            return max(depth(root.left), depth(root.right)) + 1

        h = depth(root)

        m = h + 1
        n = 2**m - 1
        ans = [[""] * n for _ in range(m)]

        def dfs(root: TreeNode, r: int, c: int) -> None:
            ans[r][c] = str(root.val)
            if root.left:
                dfs(root.left, r + 1, c - 2 ** (h - r - 1))
            if root.right:
                dfs(root.right, r + 1, c + 2 ** (h - r - 1))
            return

        dfs(root, 0, (n - 1) // 2)
        return ans


# 657 - Robot Return to Origin - EASY
class Solution:
    def judgeCircle(self, m: str) -> bool:
        return m.count("U") == m.count("D") and m.count("L") == m.count("R")


# 658 - Find K Closest Elements - MEDIUM
class Solution:
    # O(nlogn) / O(n)
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        d = [(abs(v - x), v) for v in arr]
        return sorted(v for _, v in sorted(d)[:k])

    # O(nlogn) / O(logn)
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        arr.sort(key=lambda v: abs(v - x))
        return sorted(arr[:k])

    # O(n) / O(1)
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        l = 0
        r = len(arr) - 1
        while l + k - 1 < r:
            if x - arr[l] > arr[r] - x:
                l += 1
            else:
                r -= 1
        return arr[l : r + 1]

    # O(logn + k) / O(1)
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        # [0, left] < x <= [right, n-1]
        r = bisect.bisect_left(arr, x)
        l = r - 1
        for _ in range(k):
            if l < 0:
                r += 1
            elif r > len(arr) - 1:
                l -= 1
            else:
                if x - arr[l] <= arr[r] - x:
                    l -= 1
                else:
                    r += 1
        return arr[l + 1 : r]

    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        l = 0
        r = len(arr) - k
        while l < r:
            m = l + r >> 1
            if x - arr[m] > arr[m + k] - x:
                l = m + 1
            else:
                r = m
        return arr[l : l + k]


# 661 - Image Smoother - EASY
class Solution:
    def imageSmoother(self, img: List[List[int]]) -> List[List[int]]:
        m = len(img)
        n = len(img[0])
        p = [[0] * (n + 1) for _ in range(m + 1)]
        a = [[0] * n for _ in range(m)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                p[i][j] = (
                    p[i - 1][j] + p[i][j - 1] - p[i - 1][j - 1] + img[i - 1][j - 1]
                )
        for i in range(m):
            for j in range(n):
                x1 = max(i - 1, 0)
                y1 = max(j - 1, 0)
                x2 = min(i + 2, m)
                y2 = min(j + 2, n)
                a[i][j] = (p[x2][y2] - p[x1][y2] - p[x2][y1] + p[x1][y1]) // (
                    (x2 - x1) * (y2 - y1)
                )
        return a

    def imageSmoother(self, img: List[List[int]]) -> List[List[int]]:
        m = len(img)
        n = len(img[0])
        ans = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                t, c = 0, 0
                for x in range(max(i - 1, 0), min(i + 2, m)):
                    for y in range(max(j - 1, 0), min(j + 2, n)):
                        t += img[x][y]
                        c += 1
                ans[i][j] = t // c
        return ans


# 662 - Maximum Width of Binary Tree - MEDIUM
class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        dq = collections.deque([(root, 0)])
        ans = 0
        while dq:
            ans = max(ans, dq[-1][1] - dq[0][1] + 1)
            for _ in range(len(dq)):
                n, p = dq.popleft()
                if n.left:
                    dq.append((n.left, 2 * p))
                if n.right:
                    dq.append((n.right, 2 * p + 1))
        return ans

    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        q = [(root, 0)]
        ans = 0
        while q:
            new = []
            ans = max(ans, q[-1][1] - q[0][1] + 1)
            for n, p in q:
                if n.left:
                    new.append([n.left, 2 * p])
                if n.right:
                    new.append([n.right, 2 * p + 1])
            q = new
        return ans


# 673 - Number of Longest Increasing Subsequence - MEDIUM
class Solution:
    # O(n^2)
    def findNumberOfLIS(self, nums: List[int]) -> int:
        dp = [1] * len(nums)
        cnt = [1] * len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        cnt[i] = cnt[j]
                    elif dp[j] + 1 == dp[i]:
                        cnt[i] += cnt[j]
        ans = 0
        for i in range(len(nums)):
            if dp[i] == max(dp):
                ans += cnt[i]
        return ans

    # O(nlogn)
    def findNumberOfLIS(self, nums: List[int]) -> int:
        dp = []
        cnt = collections.defaultdict(list)
        for num in nums:
            idx = bisect.bisect_left(dp, num)
            if idx == len(dp):
                dp.append(num)
            else:
                dp[idx] = num
            total = 0
            for count, last in cnt[idx]:
                if last < num:
                    total += count
            cnt[idx + 1].append((max(1, total), num))
        return sum([count for count, _ in cnt[len(dp)]])


# 670 - Maximum Swap - MEDIUM
class Solution:
    # O(n), Greedy
    # find the last occurrence of each number (guarantee that the rightmost number)
    # enumerate each number from left to right,
    # swap the number when a larger number is found
    def maximumSwap(self, num: int) -> int:
        s = list(str(num))
        later = {int(x): i for i, x in enumerate(s)}
        for i, x in enumerate(s):
            for d in range(9, int(x), -1):
                if later.get(d, -1) > i:
                    s[i], s[later[d]] = s[later[d]], s[i]
                    return "".join(s)
        return num


# 671 - Second Minimum Node In a Binary Tree - EASY
class Solution:
    def findSecondMinimumValue(self, root: Optional[TreeNode]) -> int:
        def dfs(root: TreeNode) -> None:
            if not root:
                return
            nonlocal f, s
            if root.val < f:
                f, s = root.val, f
            elif f < root.val < s:
                s = root.val
            dfs(root.left)
            dfs(root.right)
            return

        f = s = math.inf
        dfs(root)
        return s if s != math.inf else -1

    def findSecondMinimumValue(self, root: TreeNode) -> int:
        def dfs(root: TreeNode) -> None:
            if not root:
                return
            nonlocal ans
            if ans != -1 and root.val >= ans:
                return
            if root.val > rv:
                ans = root.val
            dfs(root.left)
            dfs(root.right)
            return

        ans = -1
        rv = root.val
        dfs(root)
        return ans


# 674 - Longest Continuous Increasing Subsequence - EASY
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        ans = l = 0
        pre = -1e9
        for v in nums:
            if pre >= v:
                l = 1
            else:
                l += 1
            pre = v
            ans = max(ans, l)
        return ans


# 676 - Implement Magic Dictionary - MEDIUM
class MagicDictionary:
    def __init__(self):
        self.w = list()

    def buildDict(self, dictionary: List[str]) -> None:
        self.w = dictionary

    def search(self, searchWord: str) -> bool:
        for w in self.w:
            if len(w) != len(searchWord):
                continue
            diff = 0
            for a, b in zip(w, searchWord):
                if a != b:
                    diff += 1
                if diff > 1:
                    break
            if diff == 1:
                return True
        return False


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
                if diff > 1:
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
        for candidate in self.dic.get(l, []):
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


# 678 - Valid Parenthesis String - MEDIUM
class Solution:
    # stack[index], be careful of '*(': '*' is at the left of '('
    def checkValidString(self, s: str) -> bool:
        left, star = [], []
        for i, ch in enumerate(list(s)):
            if ch == ")":
                if not left:
                    if not star:
                        return False
                    else:
                        star.pop()
                else:
                    left.pop()
            elif ch == "(":
                left.append(i)
            else:
                star.append(i)
        if len(star) < len(left):
            return False
        for i in range(len(left)):
            if left[-i - 1] > star[-i - 1]:
                return False
        return True

    # check two directions
    def checkValidString(self, s: str) -> bool:
        # left to right
        stack = []
        for x in s:
            if x == "(" or x == "*":
                stack.append(x)
            else:
                if len(stack) > 0:
                    stack.pop()
                else:
                    return False
        # right to left
        stack = []
        for x in s[::-1]:
            if x == ")" or x == "*":
                stack.append(x)
            else:
                if len(stack) > 0:
                    stack.pop()
                else:
                    return False
        return True

    # greedy, possible minimum and maximum values of '('
    def checkValidString(self, s: str) -> bool:
        cmin = cmax = 0
        for i in s:
            if i == "(":
                cmax += 1
                cmin += 1
            if i == ")":
                cmax -= 1
                cmin = max(cmin - 1, 0)
            if i == "*":
                cmax += 1
                cmin = max(cmin - 1, 0)
            if cmax < 0:
                return False
        return cmin == 0


# 680 - Valid Palindrome II - EASY
class Solution:
    def validPalindrome(self, s: str) -> bool:
        i, j = 0, len(s) - 1
        while i < j:
            if s[i] != s[j]:
                deleteJ = s[i:j]
                deleteI = s[i + 1 : j + 1]
                return deleteI == deleteI[::-1] or deleteJ == deleteJ[::-1]
            i += 1
            j -= 1
        return True

    def validPalindrome(self, s):
        i = 0
        while i < len(s) / 2 and s[i] == s[-(i + 1)]:
            i += 1
        s = s[i : len(s) - i]
        return s[1:] == s[1:][::-1] or s[:-1] == s[:-1][::-1]


# 682 - Baseball Game - EASY
class Solution:
    def calPoints(self, ops: List[str]) -> int:
        s = []
        for ch in ops:
            if ch == "D":
                s.append(2 * s[-1])
            elif ch == "C":
                s.pop()
            elif ch == "+":
                s.append(s[-1] + s[-2])
            else:
                s.append(int(ch))
        return sum(s)


# 686 - Repeated String Match - MEDIUM
class Solution:
    def repeatedStringMatch(self, a: str, b: str) -> int:
        i, cp_a = 1, a
        max_l = len(a) * 2 + len(b)
        while len(a) < max_l:
            if b in a:
                return i
            else:
                i += 1
                a += cp_a
        return -1


# 687 - Longest Univalue Path - MEDIUM
class Solution:
    def longestUnivaluePath(self, root: TreeNode) -> int:
        def postorder(root: TreeNode):  # return root.val and the longest path
            if not root:
                return (-1001, 0)
            left, ll = postorder(root.left)
            right, rl = postorder(root.right)
            l = 0
            if root.val == left == right:
                # two subtree may consist a longer path
                self.ans = max(self.ans, ll + rl + 2)
                l += max(ll, rl) + 1
            elif left == root.val:
                l += ll + 1
            elif right == root.val:
                l += rl + 1
            self.ans = max(self.ans, l)
            return (root.val, l)

        self.ans = 0
        postorder(root)
        return self.ans

    def longestUnivaluePath(self, root: TreeNode) -> int:
        def postorder(node: TreeNode, parent_val: int) -> int:
            if not node:
                return 0
            left = postorder(node.left, node.val)
            right = postorder(node.right, node.val)
            self.longest = max(self.longest, left + right)
            return 1 + max(left, right) if node.val == parent_val else 0

        self.longest = 0
        postorder(root, None)
        return self.longest


# 688 - Knight Probability in Chessboard - MEDIUM
class Solution:
    def knightProbability(self, n: int, k: int, row: int, column: int) -> float:
        dp = [[[0] * n for _ in range(n)] for _ in range(k + 1)]
        for step in range(k + 1):
            for i in range(n):
                for j in range(n):
                    if step == 0:
                        dp[step][i][j] = 1
                    else:
                        for di, dj in (
                            (-2, -1),
                            (-2, 1),
                            (2, -1),
                            (2, 1),
                            (-1, -2),
                            (-1, 2),
                            (1, -2),
                            (1, 2),
                        ):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < n and 0 <= nj < n:
                                dp[step][i][j] += dp[step - 1][ni][nj] / 8
        return dp[k][row][column]

    def knightProbability(self, N, K, r, c):
        memo = {}

        def dfs(i, j, p, k):
            if 0 <= i < N and 0 <= j < N and k < K:
                sm = 0
                for x, y in (
                    (-1, -2),
                    (-2, -1),
                    (-2, 1),
                    (-1, 2),
                    (1, 2),
                    (2, 1),
                    (2, -1),
                    (1, -2),
                ):
                    if (i + x, j + y, k) not in memo:
                        memo[(i + x, j + y, k)] = dfs(i + x, j + y, p / 8, k + 1)
                    sm += memo[(i + x, j + y, k)]
                return sm
            else:
                return 0 <= i < N and 0 <= j < N and p or 0

        return dfs(r, c, 1, 0)


# 689 - Maximum Sum of 3 Non-Overlapping Subarrays - HARD
class Solution:
    def maxSumOfThreeSubarrays(self, nums: List[int], k: int) -> List[int]:
        # Best single, double, and triple sequence found so far
        bestSeq = 0
        bestTwoSeq = [0, k]
        bestThreeSeq = [0, k, k * 2]

        # Sums of each window
        seqSum = sum(nums[0:k])
        seqTwoSum = sum(nums[k : k * 2])
        seqThreeSum = sum(nums[k * 2 : k * 3])

        # Sums of combined best windows
        bestSeqSum = seqSum
        bestTwoSum = seqSum + seqTwoSum
        bestThreeSum = seqSum + seqTwoSum + seqThreeSum

        # Current window positions
        seqIndex = 1
        twoSeqIndex = k + 1
        threeSeqIndex = k * 2 + 1
        while threeSeqIndex <= len(nums) - k:
            # Update the three sliding windows
            seqSum = seqSum - nums[seqIndex - 1] + nums[seqIndex + k - 1]
            seqTwoSum = seqTwoSum - nums[twoSeqIndex - 1] + nums[twoSeqIndex + k - 1]
            seqThreeSum = (
                seqThreeSum - nums[threeSeqIndex - 1] + nums[threeSeqIndex + k - 1]
            )

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


# 693 - Binary Number with Alternating Bits - EASY
class Solution:
    def hasAlternatingBits(self, n: int) -> bool:
        p = n & 1
        n >>= 1
        while n:
            if n & 1 == p:
                return False
            p = 1 - p
            n >>= 1
        return True

    def hasAlternatingBits(self, n: int) -> bool:
        a = n ^ (n >> 1)
        return a & (a + 1) == 0


# 695 - Max Area of Island - MEDIUM
class Solution:
    # dfs
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        self.area, ans, m, n = 0, 0, len(grid), len(grid[0])

        def dfs(x: int, y: int):
            self.area += 1
            grid[x][y] = -1  # visited
            for i, j in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if 0 <= i < m and 0 <= j < n and grid[i][j] == 1:
                    dfs(i, j)
            return

        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:  # search
                    dfs(i, j)
                    ans = max(ans, self.area)
                    self.area = 0
        return ans

    # dfs
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        def dfs(i: int, j: int) -> int:
            if 0 <= i < m and 0 <= j < n and grid[i][j]:
                grid[i][j] = 0
                return 1 + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i + 1, j) + dfs(i, j - 1)
            return 0

        areas = [dfs(i, j) for i in range(m) for j in range(n) if grid[i][j]]
        return max(areas) if areas else 0


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


# 697 - Degree of an Array - EASY
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        ans = 1e9
        cnt = collections.Counter(nums)
        mx = max(cnt.values())
        # if mx == 1:
        #     return 1
        degree = [k for k, v in cnt.items() if v == mx]
        for k in degree:
            ans = min(ans, len(nums) - nums[::-1].index(k) - nums.index(k))
        return ans


# 699 - Falling Squares - HARD
class Solution:
    # O(n ** 2) / O(1)
    def fallingSquares(self, positions: List[List[int]]) -> List[int]:
        n = len(positions)
        heights = [0] * n
        for i, (l, sz) in enumerate(positions):
            r = l + sz - 1
            heights[i] = sz
            for j in range(i):
                ll, rr = positions[j][0], positions[j][0] + positions[j][1] - 1
                if r >= ll and l <= rr:
                    heights[i] = max(heights[i], heights[j] + sz)
        for i in range(1, n):
            heights[i] = max(heights[i], heights[i - 1])
        return heights

    # segment tree: TODO
