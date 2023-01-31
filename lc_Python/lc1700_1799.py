import bisect, collections, functools, math, itertools, heapq, string, operator
from typing import List, Optional
import sortedcontainers


# 1700 - Number of Students Unable to Eat Lunch - EASY
class Solution:
    def countStudents(self, students: List[int], sandwiches: List[int]) -> int:
        cnt = collections.Counter(students)
        st = collections.deque(students)
        ans = len(students)
        for v in sandwiches:
            if cnt[v] == 0:
                break
            while st and st[0] != v:
                st.append(st.popleft())
            st.popleft()
            cnt[v] -= 1
            ans -= 1
        return ans

    def countStudents(self, students: List[int], sandwiches: List[int]) -> int:
        s1 = sum(students)
        s0 = len(students) - s1
        for v in sandwiches:
            if v == 0 and s0:
                s0 -= 1
            elif v == 1 and s1:
                s1 -= 1
            else:
                break
        return s0 + s1


# 1701 - Average Waiting Time - MEDIUM
class Solution:
    def averageWaitingTime(self, customers: List[List[int]]) -> float:
        finish = wait = 0
        for a, t in customers:
            wait += max(0, finish - a) + t
            finish = max(finish, a) + t
        return wait / len(customers)

    def averageWaitingTime(self, customers: List[List[int]]) -> float:
        wait = cur = 0
        for a, t in customers:
            cur = max(cur, a) + t
            wait += cur - a
        return wait / len(customers)


# 1702 - Maximum Binary String After Change - MEDIUM
class Solution:
    # 1. 开头的 1 不动, 找到第一个开头为 0 的子串
    # 2. 子串含至少两个 0 时, 可以把子串的 1 都挪到右边, 用左边的 0 生成 1, 直到耗尽 00
    # O(n) / O(1)
    def maximumBinaryString(self, binary: str) -> str:
        i = 0
        while i < len(binary) and binary[i] == "1":
            i += 1

        # 68 ms
        zero = binary[i:].count("0")
        one = len(binary) - i - zero

        # 352 ms
        # one = sum(map(int, binary[i:]))
        # zero = len(binary) - i - one

        # 660 ms
        # zero = sum(1 - int(binary[j]) for j in range(i, len(binary)))
        # one = len(binary) - i - zero

        if zero < 2:
            return binary
        return binary[:i] + "1" * (zero - 1) + "0" + "1" * one


# 1704 - Determine if String Halves Are Alike - EASY
class Solution:
    def halvesAreAlike(self, s: str, p: str = "aeiouAEIOU") -> bool:
        return sum(c in p for c in s[: len(s) // 2]) == sum(
            c in p for c in s[len(s) // 2 :]
        )


p = set("aeiouAEIOU")


class Solution:
    def halvesAreAlike(self, s: str) -> bool:
        return sum(c in p for c in s[: len(s) // 2]) == sum(
            c in p for c in s[len(s) // 2 :]
        )


# 1705 - Maximum Number of Eaten Apples - MEDIUM
class Solution:
    def eatenApples(self, apples: List[int], days: List[int]) -> int:
        pq, i, ans = [], 0, 0
        while i < len(apples) or pq:
            while pq and pq[0][0] <= i:
                heapq.heappop(pq)
            if i < len(apples) and apples[i]:
                heapq.heappush(pq, [i + days[i], apples[i]])
            if pq:
                pq[0][1] -= 1
                ans += 1
                if not pq[0][1]:
                    heapq.heappop(pq)
            i += 1
        return ans

    def eatenApples(self, apples: List[int], days: List[int]) -> int:
        pq, i, ans = [], 0, 0
        while i < len(apples):
            while pq and pq[0][0] <= i:
                heapq.heappop(pq)
            if apples[i]:
                heapq.heappush(pq, [i + days[i], apples[i]])
            if pq:
                pq[0][1] -= 1
                if not pq[0][1]:
                    heapq.heappop(pq)
                ans += 1
            i += 1
        while pq:
            cur = heapq.heappop(pq)
            d = min(cur[0] - i, cur[1])
            i += d
            ans += d
            while pq and pq[0][0] <= i:
                heapq.heappop(pq)
        return ans


# 1706 - Where Will the Ball Fall - MEDIUM
class Solution:
    def findBall(self, grid: List[List[int]]) -> List[int]:
        ans = []
        m, n = len(grid), len(grid[0])
        for b in range(n):
            i = 0
            j = b
            succ = True
            while i < m:
                if grid[i][j] == 1 and j + 1 < n and grid[i][j + 1] == 1:
                    j += 1
                    i += 1
                elif grid[i][j] == -1 and j - 1 >= 0 and grid[i][j - 1] == -1:
                    j -= 1
                    i += 1
                else:
                    succ = False
                    break
            if succ:
                ans.append(j)
            else:
                ans.append(-1)
        return ans

    # for-else
    def findBall(self, grid: List[List[int]]) -> List[int]:
        n = len(grid[0])
        ans = [-1] * n
        for j in range(n):
            col = j
            for row in grid:
                dir = row[col]
                col += dir
                if col < 0 or col == n or row[col] != dir:
                    break
            else:
                ans[j] = col
        return ans


# 1710 - Maximum Units on a Truck - EASY
class Solution:
    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        ans = 0
        for t, v in sorted(boxTypes, key=lambda x: x[1], reverse=True):
            if t > truckSize:
                ans += truckSize * v
                break
            truckSize -= t
            ans += t * v
        return ans


# 1711 - Count Good Meals - MEDIUM
class Solution:
    global arr
    arr = [2**i for i in range(22)]

    # O(nlogC) / O(n), 21n
    def countPairs(self, deliciousness: List[int]) -> int:
        mod = 10**9 + 7
        cnt = collections.defaultdict(int)
        ans = 0
        for v in deliciousness:
            for t in arr:
                ans += cnt[t - v]
            ans %= mod
            cnt[v] += 1
        return ans

    def countPairs(self, deliciousness: List[int]) -> int:
        mod = 10**9 + 7
        ans = 0
        cnt = collections.Counter(deliciousness)
        p = 1
        for k in sorted(cnt):
            if k == 0:
                continue
            while p < k:
                p *= 2
            ans += cnt[k] * cnt[p - k]
            if k == p:  # 自身是一个 2 的幂次
                ans += cnt[k] * (cnt[k] - 1) // 2
            ans %= mod
        return ans

    def countPairs(self, deliciousness: List[int]) -> int:
        # TODO
        def nextPower(x: int):
            x -= 1
            x |= x >> 1
            x |= x >> 2
            x |= x >> 4
            x |= x >> 8
            x |= x >> 16
            return x + 1

        mod = 10**9 + 7
        cnt = collections.Counter(deliciousness)
        ans = 0
        # 由于每个数只遍历了最接近自己的 2 的次幂的组合, 所以不可能重复
        # 比如说 1 + 3 = 4，只在 3 的时候计算, 3 + 5 = 8 只在 5 的时候计算
        for k in cnt:
            if k == 0:
                continue
            # k 的下一个 2 的幂次
            p = nextPower(k)
            # 自身是一个 2 的幂次
            if k == p:
                ans += cnt[k] * (cnt[k] - 1) // 2
            ans += cnt[k] * cnt[p - k]
        return ans % mod


# 1716 - Calculate Money in Leetcode Bank - EASY
class Solution:
    def totalMoney(self, n: int) -> int:
        div, mod = divmod(n, 7)
        ans = 0
        for i in range(mod):
            ans += i + 1 + div
        while div:
            ans += 28 + (div - 1) * 7
            div -= 1
        return ans

    def totalMoney(self, n: int) -> int:
        div, mod = divmod(n, 7)
        ans = 0
        for i in range(mod):
            ans += i + 1 + div
        ans += div * 28 + (div - 1) * div * 7 // 2 if div else 0
        return ans


# 1725 - Number Of Rectangles That Can Form The Largest Square - EASY
class Solution:
    def countGoodRectangles(self, rectangles: List[List[int]]) -> int:
        count = pre = 0
        for l, w in rectangles:
            sq = min(l, w)
            if sq > pre:
                count = 1
                pre = sq
            elif sq == pre:
                count += 1
        return count


# 1732 - Find the Highest Altitude - EASY
class Solution:
    # O(n) / O(n)
    def largestAltitude(self, gain: List[int]) -> int:
        h = [0]
        for v in gain:
            h.append(h[-1] + v)
        return max(h)

    # O(n) / O(1)
    def largestAltitude(self, gain: List[int]) -> int:
        ans = h = 0
        for v in gain:
            h += v
            ans = max(ans, h)
        return ans

    # O(n) / O(1)
    def largestAltitude(self, gain: List[int]) -> int:
        return max(itertools.accumulate(gain, initial=0))


# 1736 - Latest Time by Replacing Hidden Digits - EASY
class Solution:
    def maximumTime(self, time: str) -> str:
        ans = list(time)
        if time[0] == "?":
            ans[0] = "2" if time[1] == "?" or int(time[1]) < 4 else "1"
        if time[1] == "?":
            ans[1] = "3" if ans[0] == "2" else "9"
        if time[3] == "?":
            ans[3] = "5"
        if time[4] == "?":
            ans[4] = "9"
        return "".join(ans)


# 1739 - Building Boxes - HARD
class Solution:
    def minimumBoxes(self, n: int) -> int:
        s = 0
        k = 1
        # 放满, 1, 1 + 2, 1 + 2 + 3, 1 + 2 + ... + k
        while s + k * (k + 1) // 2 <= n:
            s += k * (k + 1) // 2
            k += 1
        # 还有剩余, 从最低一层继续摆放, 假设摆放 m 个, 那么累计可摆放的盒子个数为 1 + 2 + 3 + ... + i >= m
        ans = k * (k - 1) // 2
        k = 1
        while s < n:
            ans += 1
            s += k
            k += 1
        return ans


# 1742 - Maximum Number of Balls in a Box - EASY
class Solution:
    def countBalls(self, lowLimit: int, highLimit: int) -> int:
        d = collections.defaultdict(int)
        for i in range(lowLimit, highLimit + 1):
            x = 0
            while i:
                x += i % 10
                i //= 10
            d[x] += 1
        return max(d.values())


# 1748 - Sum of Unique Elements - EASY
class Solution:
    def sumOfUnique(self, nums: List[int]) -> int:
        freq = [0] * 100
        for n in nums:
            freq[n - 1] += 1
        ans = 0
        for i in range(len(freq)):
            ans += i + 1 if freq[i] == 1 else 0
        return ans

    def sumOfUnique(self, nums: List[int]) -> int:
        return sum(k for k, v in collections.Counter(nums).items() if v == 1)


# 1749 - Maximum Absolute Sum of Any Subarray - MEDIUM
class Solution:
    def maxAbsoluteSum(self, nums: List[int]) -> int:
        ans = p = n = 0
        for v in nums:
            if v >= 0:
                p += v
                ans = max(ans, p)
                n = min(0, n + v)
            if v <= 0:
                n += v
                ans = max(ans, -n)
                p = max(0, p + v)
        return ans

    def maxAbsoluteSum(self, nums: List[int]) -> int:
        ans = p = n = 0
        for v in nums:
            p = max(0, p + v)
            n = min(0, n + v)
            ans = max(ans, p, -n)
        return ans

    def maxAbsoluteSum(self, nums: List[int]) -> int:
        ans = p = n = 0
        for v in nums:
            p = p + v if p > 0 else v
            n = n + v if n < 0 else v
            ans = max(ans, abs(p), abs(n))
        return ans

    # 子数组和最大 -> 某两个位置(i, j)的前缀和的差值 最大 (i <= j)
    # 子数组和最小 -> 某两个位置(i, j)的前缀和的差值 最小 (i <= j)
    # 子数组和的 abs 最大 -> 某两个位置(i, j)的前缀和的差距 最大, i 和 j 相对位置随意
    def maxAbsoluteSum(self, nums: List[int]) -> int:
        mx = mi = p = 0
        for i in range(1, len(nums) + 1):
            p += nums[i - 1]
            if mx < p:
                mx = p
            if mi > p:
                mi = p
        return mx - mi


# 1750 - Minimum Length of String After Deleting Similar Ends - MEDIUM
class Solution:
    def minimumLength(self, s: str) -> int:
        dq = collections.deque()
        for c, g in itertools.groupby(s):
            dq.append((c, len(tuple(g))))
        while dq:
            if dq[0][0] == dq[-1][0]:
                if len(dq) == 1:
                    return 1 if dq.pop()[1] == 1 else 0
                else:
                    dq.pop()
                    dq.popleft()
            else:
                break
        return sum(v[1] for v in dq)

    def minimumLength(self, s: str) -> int:
        l = 0
        r = len(s) - 1
        while l < r and s[l] == s[r]:
            while l + 1 < r and s[l] == s[l + 1]:
                l += 1
            while l < r - 1 and s[r - 1] == s[r]:
                r -= 1
            l += 1
            r -= 1
        return max(0, r - l + 1)

    def minimumLength(self, s: str) -> int:
        l = 0
        r = len(s) - 1
        while l < r and s[l] == s[r]:
            c = s[l]
            while l <= r and s[l] == c:
                l += 1
            while l <= r and s[r] == c:
                r -= 1
        return r - l + 1


# 1752 - Check if Array Is Sorted and Rotated - EASY
class Solution:
    # O(n**2) / O(n)
    def check(self, nums: List[int]) -> bool:
        n = len(nums)
        a = sorted(nums)
        return any(all(nums[i] == a[(i + x) % n] for i in range(n)) for x in range(n))

    # O(n) / O(1)
    def check(self, nums: List[int]) -> bool:
        cnt = 0
        n = len(nums)
        for i in range(n):
            if nums[i] > nums[(i + 1) % n]:
                cnt += 1
            if cnt > 1:
                return False
        return True


# 1753 - Maximum Score From Removing Stones - MEDIUM
class Solution:
    # O(nlogn) / O(1)
    def maximumScore(self, a: int, b: int, c: int) -> int:
        h = sorted([-a, -b, -c])
        ans = 0
        while len(h) > 1:
            x = -heapq.heappop(h) - 1
            y = -heapq.heappop(h) - 1
            ans += 1
            if x:
                heapq.heappush(h, -x)
            if y:
                heapq.heappush(h, -y)
        return ans

    # O(1) / O(1)
    def maximumScore(self, a: int, b: int, c: int) -> int:
        arr = sorted([a, b, c])
        if arr[0] + arr[1] <= arr[2]:
            return arr[0] + arr[1]
        return sum(arr) // 2


# 1754 - Largest Merge Of Two Strings - MEDIUM
class Solution:
    # O(m**2 + n**2) / O(m**2 + n**2)
    def largestMerge(self, word1: str, word2: str) -> str:
        if word1 >= word2 and word2 > "":
            return word1[0] + self.largestMerge(word1[1:], word2)
        if word2 >= word1 and word1 > "":
            return word2[0] + self.largestMerge(word1, word2[1:])
        return word1 + word2

    # O(m**2 + n**2) / O(1)
    def largestMerge(self, word1: str, word2: str) -> str:
        ans = ""
        i = j = 0
        m = len(word1)
        n = len(word2)
        while i < m and j < n:
            # k = 0
            # while i + k < m and j + k < n and word1[i + k] == word2[j + k]:
            #     k += 1
            # if j + k == n or i + k < m and word1[i + k] > word2[j + k]:
            #     ans += word1[i]
            #     i += 1
            # else:
            #     ans += word2[j]
            #     j += 1

            if word1[i:] > word2[j:]:
                ans += word1[i]
                i += 1
            else:
                ans += word2[j]
                j += 1

        while i < m:
            ans += word1[i]
            i += 1
        while j < n:
            ans += word2[j]
            j += 1
        return ans


# 1758 - Minimum Changes To Make Alternating Binary String - EASY
class Solution:
    # O(n) / O(n)
    def minOperations(self, s: str) -> int:
        n = len(s)
        a = "01" * ((n + 1) // 2)
        b = "10" * ((n + 1) // 2)
        ans = 1e4
        ans = min(ans, sum(x != y for x, y in zip(a, s)))
        ans = min(ans, sum(x != y for x, y in zip(b, s)))
        return ans

    def minOperations(self, s: str) -> int:
        x = y = 0  # x: 1010... y: 0101...
        for i, c in enumerate(s):
            if c != str(i % 2):
                x += 1
            else:
                y += 1
        return min(x, y)

    # 变成 01010... 的步骤数是 x,
    # 变成 10101... 的步骤数是 y,
    # 一定有 x + y = n
    # 任意计算其中一种, 返回 min(x, n-x) 就行
    def minOperations(self, s: str) -> int:
        x = 0
        for i, c in enumerate(s):
            if c != str(i % 2):
                x += 1
        return min(x, len(s) - x)


# 1759 - Count Number of Homogenous Substrings - MEDIUM
class Solution:
    def countHomogenous(self, s: str) -> int:
        ans = 0
        for _, g in itertools.groupby(s):
            l = len(list(g))
            ans += (l + 1) * l // 2
        return ans % (10**9 + 7)

    def countHomogenous(self, s: str) -> int:
        ans = t = 1
        for i in range(1, len(s)):
            if s[i - 1] == s[i]:
                t += 1
            else:
                t = 1
            ans += t
        return ans % (10**9 + 7)


# 1760 - Minimum Limit of Balls in a Bag - MEDIUM
class Solution:
    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        def check(x: int) -> int:
            t = 0
            for v in nums:
                t += (v - 1) // x
            return t <= maxOperations

        l = 1
        r = max(nums)
        while l < r:
            m = (l + r) // 2
            if check(m):
                r = m
            else:
                l = m + 1
        return l

    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        return (
            bisect.bisect_left(
                range(1, max(nums) + 1),
                x=True,
                key=lambda m: sum((v - 1) // m for v in nums) <= maxOperations,
            )
            + 1
        )


# 1762 - Buildings With an Ocean View - MEDIUM
class Solution:
    def findBuildings(self, heights: List[int]) -> List[int]:
        ans, maxH = [], 0
        for i in range(len(heights) - 1, -1, -1):
            if heights[i] > maxH:
                ans.append(i)
                maxH = heights[i]
        ans.reverse()
        return ans


# 1763 - Longest Nice Substring - EASY
class Solution:
    def longestNiceSubstring(self, s: str) -> str:
        if not s:
            return ""
        ss = set(s)
        for i, c in enumerate(s):
            if c.swapcase() not in ss:
                s0 = self.longestNiceSubstring(s[:i])
                s1 = self.longestNiceSubstring(s[i + 1 :])
                return max(s0, s1, key=len)
        return s


# 1764 - Form Array by Concatenating Subarrays of Another Array - MEDIUM
class Solution:
    # O(nm) / O(1)
    def canChoose(self, groups: List[List[int]], nums: List[int]) -> bool:
        i = j = 0
        while i < len(groups) and j < len(nums):
            g = groups[i]
            if g == nums[j : j + len(g)]:
                i += 1
                j += len(g)
            else:
                j += 1
        return i == len(groups)


# 1765 - Map of Highest Peak - MEDIUM
class Solution:
    def highestPeak(self, isWater: List[List[int]]) -> List[List[int]]:
        dq = collections.deque([])
        m, n = len(isWater), len(isWater[0])
        for i in range(m):
            for j in range(n):
                if isWater[i][j] == 1:
                    dq.append((i, j))
                isWater[i][j] -= 1
        while dq:
            x, y = dq.popleft()
            for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if 0 <= nx < m and 0 <= ny < n and isWater[nx][ny] == -1:
                    dq.append((nx, ny))
                    isWater[nx][ny] = isWater[x][y] + 1
        return isWater


# 1768 - Merge Strings Alternately - EASY
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        return "".join(
            a + b for a, b in itertools.zip_longest(word1, word2, fillvalue="")
        )


# 1769 - Minimum Number of Operations to Move All Balls to Each Box - MEDIUM
class Solution:
    # O(n) / O(n)
    def minOperations(self, boxes: str) -> List[int]:
        n = len(boxes)
        l = [0] * n
        step = 0
        for i in range(1, len(boxes)):
            if boxes[i - 1] == "1":
                step += 1
            l[i] += l[i - 1] + step
        r = [0] * n
        step = 0
        for i in range(len(boxes) - 2, -1, -1):
            if boxes[i + 1] == "1":
                step += 1
            r[i] += r[i + 1] + step
        return [a + b for a, b in zip(l, r)]

    # O(n) / O(1)
    def minOperations(self, boxes: str) -> List[int]:
        l = int(boxes[0])
        r = operations = 0
        for i in range(1, len(boxes)):
            if boxes[i] == "1":
                r += 1
                operations += i
        ans = [operations]
        for i in range(1, len(boxes)):
            operations += l - r
            if boxes[i] == "1":
                l += 1
                r -= 1
            ans.append(operations)
        return ans

    def minOperations(self, boxes: str) -> List[int]:
        n = len(boxes)
        l = r = cost = 0
        for i, v in enumerate(map(int, boxes)):
            r += v
            cost += v * (i + 1)
        ans = [-1] * n
        for i, v in enumerate(map(int, boxes)):
            cost += l - r
            l += v
            r -= v
            ans[i] = cost
        return ans


# 1770 - Maximum Score from Performing Multiplication Operations - MEDIUM
class Solution:
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        n = len(nums)
        m = len(multipliers)
        f = [[0] * (m + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            f[i][0] = f[i - 1][0] + nums[i - 1] * multipliers[i - 1]
        for j in range(1, m + 1):
            f[0][j] = f[0][j - 1] + nums[n - j] * multipliers[j - 1]
        ans = f[0][m] if f[0][m] > f[m][0] else f[m][0]
        # i: 左边拿走数量, j: 右边拿走数量
        for i in range(1, m + 1):
            for j in range(1, m - i + 1):
                left = f[i - 1][j] + nums[i - 1] * multipliers[i + j - 1]
                right = f[i][j - 1] + nums[n - j] * multipliers[i + j - 1]
                f[i][j] = max(left, right)
            ans = max(ans, f[i][m - i])
        return ans

    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        n = len(nums)
        m = len(multipliers)
        # l, r 是左右下标, i 是 mul 里的下标
        f = [[0] * (m + 1) for _ in range(m + 1)]
        for l in range(m - 1, -1, -1):
            for i in range(m - 1, -1, -1):
                r = n - (i - l) - 1
                if r < 0 or r >= n:
                    break
                a = f[l + 1][i + 1] + nums[l] * multipliers[i]
                b = f[l][i + 1] + nums[r] * multipliers[i]
                f[l][i] = max(a, b)
        return f[0][0]

    # cache maxsize is set to None will TLE
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        @functools.lru_cache(len(multipliers))
        def dfs(l: int, r: int, k: int) -> int:
            if k == len(multipliers):
                return 0
            a = nums[l] * multipliers[k] + dfs(l + 1, r, k + 1)
            b = nums[r] * multipliers[k] + dfs(l, r - 1, k + 1)
            return max(a, b)
            dfs.cache_clear()

        return dfs(0, len(nums) - 1, 0)


# 1773 - Count Items Matching a Rule - EASY
class Solution:
    def countMatches(self, items: List[List[str]], ruleKey: str, ruleValue: str) -> int:
        if ruleKey == "type":
            return sum(x == ruleValue for x, _, _ in items)
        if ruleKey == "color":
            return sum(x == ruleValue for _, x, _ in items)
        if ruleKey == "name":
            return sum(x == ruleValue for _, _, x in items)

    def countMatches(self, items: List[List[str]], ruleKey: str, ruleValue: str) -> int:
        idx = {"type": 0, "color": 1, "name": 2}[ruleKey]
        return sum(v[idx] == ruleValue for v in items)


# 1774 - Closest Dessert Cost - MEDIUM
class Solution:
    # O(n * 3**m) / O(m)
    def closestCost(
        self, baseCosts: List[int], toppingCosts: List[int], target: int
    ) -> int:
        def dfs(i: int, cur: int) -> None:
            nonlocal ans
            if i == len(toppingCosts):
                # 放外面也可以, 表示后面的都不选
                if abs(ans - target) > abs(cur - target):
                    ans = cur
                elif abs(ans - target) == abs(cur - target):
                    ans = min(ans, cur)
                return
            dfs(i + 1, cur)
            if cur <= target:
                dfs(i + 1, cur + toppingCosts[i])
                dfs(i + 1, cur + 2 * toppingCosts[i])
            return

        ans = math.inf
        for v in baseCosts:
            dfs(0, v)
        return ans

    def closestCost(
        self, baseCosts: List[int], toppingCosts: List[int], target: int
    ) -> int:
        def dfs(i: int, cur: int) -> None:
            nonlocal ans
            a = abs(ans - target)
            b = abs(cur - target)
            if cur > target and a < b:
                return
            if a > b:
                ans = cur
            if a == b:
                ans = min(ans, cur)
            if i == len(toppingCosts):
                return
            dfs(i + 1, cur + toppingCosts[i] * 2)
            dfs(i + 1, cur + toppingCosts[i])
            dfs(i + 1, cur)
            return

        ans = math.inf
        for v in baseCosts:
            dfs(0, v)
        return ans


# 1775 - Equal Sum Arrays With Minimum Number of Operations - MEDIUM
class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        s1 = sum(nums1)
        s2 = sum(nums2)
        if s1 == s2:
            return 0
        if s1 < s2:
            return self.minOperations(nums2, nums1)
        diff = s1 - s2
        cnt = collections.Counter(v - 1 for v in nums1) + collections.Counter(
            6 - v for v in nums2
        )
        ans = 0
        for k in range(5, 0, -1):
            if diff >= cnt[k] * k:
                diff -= cnt[k] * k
                ans += cnt[k]
            else:
                ans += (diff + k - 1) // k
                diff = 0
                break

        # for i in range(5, 0, -1):
        #     if diff <= 0:
        #         break
        #     for _ in range(cnt[i]):
        #         if diff <= 0:
        #             break
        #         ans += 1
        #         diff -= i

        return ans if diff == 0 else -1


# 1779 - Find Nearest Point That Has the Same X or Y Coordinate - EASY
class Solution:
    def nearestValidPoint(self, x: int, y: int, points: List[List[int]]) -> int:
        ans = -1
        mi = math.inf
        for i, (a, b) in enumerate(points):
            if a == x or b == y:
                d = abs(a - x) + abs(b - y)
                if d < mi:
                    mi = d
                    ans = i
        return ans


# 1780 - Check if Number is a Sum of Powers of Three - MEDIUM
class Solution:
    def checkPowersOfThree(self, n: int) -> bool:
        p3 = [3**i for i in reversed(range(15))]
        for p in p3:
            if n >= 2 * p:
                return False
            elif n >= p:
                n -= p
        return True

    def checkPowersOfThree(self, n: int) -> bool:
        while n > 0:
            if n % 3 == 2:
                return False
            n //= 3
        return True


# 1781 - Sum of Beauty of All Substrings - MEDIUM
class Solution:
    # O(n * n * C) / O(C)
    def beautySum(self, s: str) -> int:
        ans = 0
        for i in range(len(s)):
            d = collections.defaultdict(int)
            for j in range(i, len(s)):
                d[s[j]] += 1
                ans += max(d.values()) - min(d.values())
        return ans


# 1784 - Check if Binary String Has at Most One Segment of Ones - EASY
class Solution:
    def checkOnesSegment(self, s: str) -> bool:
        return len(list(v for v in s.split("0") if v != "")) <= 1
        return "01" not in s


# 1785 - Minimum Elements to Add to Form a Given Sum - MEDIUM
class Solution:
    def minElements(self, nums: List[int], limit: int, goal: int) -> int:
        return math.ceil(abs(goal - sum(nums)) / limit)
        return (abs(goal - sum(nums)) - 1) // limit + 1


# 1790 - Check if One String Swap Can Make Strings Equal - EASY
class Solution:
    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        cnt1 = collections.Counter(s1)
        cnt2 = collections.Counter(s2)
        if cnt1 != cnt2:
            return False
        diff = 0
        for x, y in zip(s1, s2):
            if x != y:
                diff += 1
        if diff == 0 or diff == 2:
            return True
        return False

    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        px = py = ""
        for x, y in zip(s1, s2):
            if x != y:
                if px == "used":
                    return False
                if px == "":
                    px = x
                    py = y
                elif px != y or py != x:
                    return False
                else:
                    px = "used"
        return True if px in "used" else False

    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        px = py = ""
        for x, y in zip(s1, s2):
            if x != y:
                if px == y and py == x:
                    px = "used"
                elif px == "":
                    px = x
                    py = y
                else:
                    return False
        return True if px in "used" else False

    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        i = j = -1
        for idx, (x, y) in enumerate(zip(s1, s2)):
            if x != y:
                if i < 0:
                    i = idx
                elif j < 0:
                    j = idx
                else:
                    return False
        return i < 0 or j >= 0 and s1[i] == s2[j] and s1[j] == s2[i]

    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        arr = []
        for x, y in zip(s1, s2):
            if x != y:
                arr.append(x)
                arr.append(y)
        return len(arr) <= 4 and arr == arr[::-1]


# 1791 - Find Center of Star Graph - EASY
class Solution:
    def findCenter(self, edges: List[List[int]]) -> int:
        n = len(edges)
        d = [0] * (n + 1)
        for o, i in edges:
            d[o - 1] += 1
            d[i - 1] += 1
        for i in range(len(d)):
            if d[i] == n:
                return i + 1
        return -1

    def findCenter(self, edges: List[List[int]]) -> int:
        if edges[0][0] in edges[1]:
            return edges[0][0]
        else:
            return edges[0][1]


# 1796 - Second Largest Digit in a String - EASY
class Solution:
    def secondHighest(self, s: str) -> int:
        a = b = -1
        for c in s:
            if c.isdigit():
                if int(c) > a:
                    a, b = int(c), a
                elif a > int(c) > b:
                    b = int(c)
        return b


# 1799 - Maximize Score After N Operations - HARD
class Solution:
    # 2400 ms
    def maxScore(self, nums: List[int]) -> int:
        def gcd(x: int, y: int) -> int:
            while y:
                x, y = y, x % y
            return x

        # 1600 ms
        # d = collections.defaultdict(dict)
        # for i in range(len(nums)):
        #     for j in range(i + 1, len(nums)):
        #         g = gcd(nums[i], nums[j])
        #         d[nums[i]][nums[j]] = g
        #         d[nums[j]][nums[i]] = g

        @functools.lru_cache(None)
        def dfs(t: tuple) -> int:
            m = 0
            l = list(t)
            for i in range(len(l)):
                for j in range(i + 1, len(l)):
                    new = l[:]
                    new.pop(j)
                    new.pop(i)
                    m = max(m, len(l) // 2 * gcd(l[i], l[j]) + dfs(tuple(new)))
                    # m = max(m, len(l) // 2 * d[l[i]][l[j]] + dfs(tuple(new)))
            return m

        return dfs(tuple(nums))

    # O(2**n * n**2 + logU * n**2) / O(2**n + n**2), 1500 ms, U = max(nums)
    def maxScore(self, nums: List[int]) -> int:
        def gcd(x: int, y: int) -> int:
            while y:
                x, y = y, x % y
            return x

        n = len(nums)
        g = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                g[i][j] = gcd(nums[i], nums[j])
        f = [0] * (1 << n)
        for k in range(1 << n):
            cnt = bin(k).count("1")
            if cnt % 2 == 0:
                for i in range(n):
                    if (k >> i) & 1:
                        for j in range(i + 1, n):
                            if (k >> j) & 1:
                                f[k] = max(
                                    f[k],
                                    f[k ^ (1 << i) ^ (1 << j)] + cnt // 2 * g[i][j],
                                )
        return f[(1 << n) - 1]

    def maxScore(self, nums: List[int]) -> int:
        def gcd(x: int, y: int) -> int:
            while y:
                x, y = y, x % y
            return x

        @functools.lru_cache(None)
        def dfs(mask: int, k: int) -> int:
            if k == len(nums) // 2 + 1:
                return 0
            r = 0
            for i in range(len(nums)):
                if mask & 1 << i:
                    continue
                for j in range(i + 1, len(nums)):
                    if mask & 1 << j:
                        continue
                    r = max(
                        r,
                        k * gcd(nums[i], nums[j]) + dfs(mask | 1 << i | 1 << j, k + 1),
                    )
            return r

        return dfs(0, 1)

    def maxScore(self, nums: List[int]) -> int:
        def gcd(x: int, y: int) -> int:
            while y:
                x, y = y, x % y
            return x

        @functools.lru_cache(None)
        def dfs(mask: int, k: int) -> int:
            if k == len(nums) // 2 + 1:
                return 0
            r = 0
            for i in range(len(nums)):
                if mask & 1 << i:
                    continue
                for j in range(i + 1, len(nums)):
                    if mask & 1 << j:
                        continue
                    r = max(
                        r,
                        k * gcd(nums[i], nums[j]) + dfs(mask ^ 1 << i ^ 1 << j, k + 1),
                    )
            return r

        return dfs(0, 1)
