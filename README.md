# Leetcode diary 📅
* Forgetting Go 😕
* Training Python 🥱
* ~~Java 🤯~~

<br><br><br>

# 🔧
[OI Wiki](https://oi-wiki.org/)  

[周赛评分算法](https://leetcode.cn/circle/article/neTUV4/) 

[力扣竞赛 - 勋章及成就规则](https://leetcode.cn/circle/discuss/0fKGDu/) 

[排名分数计算脚本](https://leetcode.cn/circle/discuss/6gnvEj/view/WbN5TH/) 

[clist.by](https://clist.by/) 

[Markdown语法](https://markdown.com.cn)

[Algorithms for Competitive Programming (translate from http://e-maxx.ru/algo/)](https://cp-algorithms.com/index.html)

[GeeksforGeeks](https://www.geeksforgeeks.org/)

<br><br><br>

# 📚: 
isbn: 978-7-83009-313-6

<br><br><br>

# Python:

* `math.lcm(*integers)`, [New in version 3.9](https://docs.python.org/3/library/math.html#math.lcm), If any of the arguments is zero, then the returned value is 0. lcm() without arguments returns 1.

* `@functools.cache(user_function)`, [New in version 3.9](https://docs.python.org/3/library/functools.html#functools.cache), Returns the same as lru_cache(maxsize=None).

* `itertools.pairwise(iterable)`, [New in version 3.10](https://docs.python.org/3/library/itertools.html#itertools.pairwise), list(pairwise('a')) -> [], list(pairwise('abc')) -> [('ab', 'bc')].

* `collections.Counter(a) > collections.Counter(b)`, [New in version 3.10](https://docs.python.org/3/library/collections.html#collections.Counter), All of those tests treat missing elements as having zero counts so that Counter(a=1) == Counter(a=1, b=0) returns true.

* [Glossary](https://docs.python.org/3/glossary.html)

<br><br><br>

# Big O
The O is short for “Order of”. If we’re discussing an algorithm with O(n), we say its **order of**, or **rate of growth**, is n, or linear complexity.

| O            | Complexity  |
| ------------ | ----------- |
| O(1)         | constant    |
| O(log n)     | logarithmic |
| O(n)         | linear      |
| O(n * log n) | log linear  |
| O(n^2)       | quadratic   |
| O(n^3)       | cubic       |
| O(2^n)       | exponential |
| O(n!)        | factorial   |

![big(O)](/pic/big-o-cheatsheet.png)

<br><br><br>

# DP: top-down vs bottom-up

* Memoization and Tabulation
  - **Tabulation(Bottom Up)(刷表)**: If you are calculating the Fibonacci sequence `fib(100)`, you would just call this, and it would call `fib(100) = fib(99) + fib(98)`, which would call `fib(99) = fib(98) + fib(97)`, ...etc..., which would call `fib(2) = fib(1) + fib(0) = 1 + 0 = 1`.
  - **Memoization(Top Down)**:  If you are performing fibonacci, you might choose to calculate the numbers in this order: `fib(2), fib(3), fib(4)`... caching every value so you can compute the next ones more easily.
* Memoization is easier to code.
* Difference:
![bottom-up vs top-down](/pic/Tabulation-vs-Memoization.png)

<br><br><br>

# 坑 / 技巧
* MOD:
  * 不取余python超时
  * dp中有减法, 负数 x 取余, 防止变一个大数: `(x + MOD) % MOD`
  * 区别: 
    * 取余(rem): 采用fix(), 向 0 方向舍入, `rem(x, y) = x - y. * fix(x./y)`
    * 取模(mod): 采用floor(), 向无穷小方向舍入, `mod(x, y) = x - y. * floor(x./y)`
    * 取模和求余应该是同一种运算, 只是在被除数和除数符号不同时, 余数的符号是有歧义的, 可为正也可为负
      * C, Go, JavaScript, Rust, Java, Swift, PHP中结果与被除数同符号
      * Python 中结果与除数同符号


<br><br><br>


# 7788
* 数组
* 链表
* 哈希表
* 字符串
* 栈与队列
* 二叉树
  * Trie (前缀树)
    LC: 139 / 140 / 208
* 回溯
* 贪心
* 动态规划

* XOR (exclusive OR)
  * 半加运算，其运算法则相当于不带进位的二进制加法
    * 与0异或 = 本身
    * 与1异或 = 取反 -> 翻转特定位
      * 翻转10100001的第2位和第3位 -> 10100001 ^ 00000110 = 10100111
    * 异或自己 = 置0
      * a ^ b ^ a = b (^ caret)
  
* 位运算(整数在计算机中是以补码的形式储存的)
  * 求整数二进制的最低位1
    1. `n & (-n)`: 任何整数，其二进制表示的最后一个'1'，可由该整数与其相反数按位取与得到
    2. `n & (n-1)`: 二进制数字 n 最右边的 1 变成 0, 其余不变, 消去二进制数中的最低位'1', Why: 'n-1'会一直向前寻找可借的位，从而跳过低位连续的'0'，而向最低位的'1'借位，借位后最低位的'1'变为'0'，原先最低位'1'的下一位从'0'变为'1'，其余位都不变，相与之后其它位不变
    3. `n > 0 && n & (n - 1) == 0` 判断是否是'2'的幂
    4. ```c
        upper, lower exchange: asc ^= 32
        upper, lower to lower: asc |= 32
        lower, upper to upper: asc &= -33
        ```

* 高频：宽度优先搜索（BFS），深度优先搜索（DFS），二分法（Binary Search），双指针（2 Pointer），堆、栈、队列、哈希表（Heap，Stack，Heap，HashMap/HashSet），前缀和（Prefix Sum），链表（LinkedList），二叉树（Binary Tree），二叉搜索树（Binary Search Tree），快速排序与归并排序（Quick Sort/ Merge Sort)

* 中频：动态规划（DP），扫描线（Sweep Line），字典树（Trie），并查集（Union Find），单调栈与单调队列（Monotone Stack/ Queue），TreeMap等

```py
fn = sys.stdin.readline
l = int(fn())
for _ in range(l):
  n = int(fn())
  h = list(map(int, fn().split()))
```
