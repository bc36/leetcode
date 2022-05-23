from typing import List
import collections, itertools, functools, bisect, math


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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


# 720 - Longest Word in Dictionary - EASY
class Solution:
    # O(n + n * 30), brute force
    def longestWord(self, words: List[str]) -> str:
        if not words:
            return ""
        ans = ""
        s = set(words)
        for w in words:
            if len(w) > len(ans) or (len(w) == len(ans) and w < ans):
                for i in range(1, len(w) + 1):
                    if w[:i] not in s:
                        break
                else:  # for-else statement
                    ans = w
        return ans

    # O(nlogn + n) / O(n)
    def longestWord(self, words: List[str]) -> str:
        # ordered by lexicographical, then ordered by length
        words.sort()
        s = set([""])
        ans = ""
        for w in words:
            if w[:-1] in s:
                s.add(w)
                if len(w) > len(ans):
                    ans = w
        return ans

    # Trie
    def longestWord(self, words: List[str]) -> str:
        trie = {}
        for w in words:
            r = trie
            for ch in w:
                if ch not in r:
                    r[ch] = {}
                r = r[ch]
            r["end"] = True
        ans = ""
        words.sort()
        for w in words:
            r = trie
            if len(w) > len(ans):
                f = True  # flag, can be replaced by for-else statement
                for ch in w:
                    if ch not in r or "end" not in r[ch]:
                        f = False
                        break
                    r = r[ch]
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

    def insert(self, word):
        r = self.root
        for ch in word:
            r = r.children[ch]
        r.isEnd = True
        r.word = word

    def bfs(self):
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


# 746 - Min Cost Climbing Stairs - EASY
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        dp = [0] * len(cost)
        dp[0], dp[1] = cost[0], cost[1]
        for i in range(2, len(cost)):
            dp[i] = min(dp[i - 2], dp[i - 1]) + cost[i]
        return min(dp[-2], dp[-1])


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


# 749


# 750
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


################
# 2022.3.30 VO #
################
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


print(calc("1+2"))  # => 3
print(calc("1+2*3"))  # => 7
print(calc("1+2*3/3"))  # => 3
print(calc("1/3+2*3"))  # => 6.333333
print(calc("1/3*6+2*3"))  # => 8


# 773 - Sliding Puzzle - HARD
class Solution:
    # O((mn)! * mn * 4) / O((mn)! * mn), factorial complexity
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        adj = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4], 4: [1, 3, 5], 5: [2, 4]}
        # # 'fn' equal to 'toString'
        # def toString(a: List[int]) -> str:
        #     return ''.join([str(x) for x in a])
        fn = lambda a: "".join([str(x) for x in a])
        depth = 0
        init = [n for row in board for n in row]
        q = collections.deque([init])
        seen = set([fn(init)])
        while q:
            k = len(q)
            for _ in range(k):
                cur = q.popleft()
                if fn(cur) == "123450":
                    return depth
                i = cur.index(0)
                for j in adj[i]:
                    nxt = cur.copy()
                    nxt[i], nxt[j] = nxt[j], nxt[i]
                    if not fn(nxt) in seen:
                        q.append(nxt)
                        seen.add(fn(nxt))
            depth += 1
        return -1

    def slidingPuzzle(self, board: List[List[int]]) -> int:
        d = {0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4], 4: [1, 3, 5], 5: [2, 4]}

        def neighbors(s: str):
            i = s.find("0")
            res = []
            for j in d[i]:
                t = list(s)
                t[i], t[j] = t[j], t[i]
                res += ("".join(t),)
            return res

        start = "".join([str(n) for row in board for n in row])
        target = "123450"
        dq = collections.deque([(start, 0)])
        seen = {start}
        while dq:
            cur, step = dq.popleft()
            if cur == target:
                return step
            for n in neighbors(cur):
                if n not in seen:
                    seen.add(n)
                    dq += ((n, step + 1),)
        return -1

    def slidingPuzzle(self, board: List[List[int]]) -> int:
        s = "".join(str(n) for row in board for n in row)
        dq = collections.deque([(s, s.index("0"))])
        seen = {s}
        r = len(board)
        c = len(board[0])
        steps = 0
        while dq:
            for _ in range(len(dq)):
                cur, i = dq.popleft()
                if cur == "123450":
                    return steps
                x, y = i // c, i % c
                for nx, ny in (x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y):
                    if 0 <= nx < r and 0 <= ny < c:
                        l = [ch for ch in cur]
                        l[i], l[nx * c + ny] = l[nx * c + ny], "0"
                        s = "".join(l)
                        if s not in seen:
                            seen.add(s)
                            dq.append((s, nx * c + ny))
            steps += 1
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
        dir = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        q = [arr]
        seen = set()
        while q:
            new = []
            for l in q:
                if l == target:
                    return step
                if tuple(l) not in seen:
                    seen.add(tuple(l))
                    z = l.index(0)
                    x, y = divmod(z, c)
                    for i, j in dir:
                        nx = x + i
                        ny = y + j
                        if 0 <= nx < r and 0 <= ny < c:
                            nxt = l.copy()
                            nxt[nx * c + ny], nxt[z] = nxt[z], nxt[nx * c + ny]
                            new.append(nxt)
            step += 1
            q = new
        return -1


# 780 - Reaching Points - HARD
class Solution:
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


# 784 - Letter Case Permutation - MEDIUM
class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
        ans = [""]
        for ch in s:
            if ch.isalpha():
                ans = [i + j for i in ans for j in [ch.upper(), ch.lower()]]
            else:
                ans = [i + ch for i in ans]
        return ans

        # ans = ['']
        # for c in s.lower():
        #     ans = [a + c
        #            for a in ans] + ([a + c.upper()
        #                              for a in ans] if c.isalpha() else [])
        # return ans

        ## L = ['a', 'A'], '1', ['b', 'B'], '2']
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
                    # chr(ord(s[i])^(1<<5))
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

# 788

# 789

# 790


# 791 - Custom Sort String - MEDIUM
class Solution:
    def customSortString(self, order: str, s: str) -> str:
        cnt = collections.Counter(s)
        ans = ""
        for ch in order:
            while cnt.get(ch, 0) and cnt[ch] > 0:
                ans += ch
                cnt[ch] -= 1
        """
        'ch' will have been assigned value and can be called,
        even if it in the last for loop and for loop ended
        print(ch)
        """
        for ch in cnt:
            while cnt[ch] != 0:
                ans += ch
                cnt[ch] -= 1
        return ans

    def customSortString(self, order: str, s: str) -> str:
        cnt, ans = collections.Counter(s), ""
        for ch in order:
            if ch in cnt:
                ans += ch * cnt[ch]
                cnt.pop(ch)

        return ans + "".join(ch * cnt[ch] for ch in cnt)


# 794 - Valid Tic-Tac-Toe State - MEDIUM
class Solution:
    def validTicTacToe(self, board: List[str]) -> bool:

        return


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
