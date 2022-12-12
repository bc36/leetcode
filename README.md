# Leetcode diary 📅
![Go version](https://img.shields.io/badge/Go-1.18-white?labelColor=deepskyblue&color=white)
![Py version](https://img.shields.io/badge/Python-3.8-white?labelColor=blue&color=white)
* Forgetting Go 😕
* Training Python 🥱
* ~~Java 🙅~~
* When can I be a JavaScript master 🤔


<br><br>


# 🔧
[OI Wiki](https://oi-wiki.org/)  

[clist.by](https://clist.by/) 

[Algorithms for Competitive Programming (translate from http://e-maxx.ru/algo/)](https://cp-algorithms.com/index.html)

[Markdown](https://markdown.com.cn)

[GeeksforGeeks](https://www.geeksforgeeks.org/)

[力扣竞赛 - 勋章及成就规则](https://leetcode.cn/circle/discuss/0fKGDu/) 

[周赛评分算法](https://leetcode.cn/circle/article/neTUV4/) 

[排名分数计算脚本](https://leetcode.cn/circle/discuss/6gnvEj/view/WbN5TH/) 

[Guardian / Knight 历史门槛](https://leetcode.cn/circle/discuss/SKzRis/) 


<br><br>


# 📚: 
isbn: 978-7-83009-313-6


<br><br>


# Python3:

* `math.lcm(*integers)`, [New in version 3.9](https://docs.python.org/3/library/math.html#math.lcm), If any of the arguments is zero, then the returned value is 0. lcm() without arguments returns 1.

* `@functools.cache(user_function)`, [New in version 3.9](https://docs.python.org/3/library/functools.html#functools.cache), Returns the same as lru_cache(maxsize=None).

* `int.bit_count()`, [New in version 3.10](https://docs.python.org/3/library/stdtypes.html#int.bit_count), Equivalent to: `bin(self).count("1")`.

* `itertools.pairwise(iterable)`, [New in version 3.10](https://docs.python.org/3/library/itertools.html#itertools.pairwise), list(pairwise('a')) -> [], list(pairwise('abc')) -> [('ab', 'bc')].

* `collections.Counter(a) > collections.Counter(b)`, [New in version 3.10](https://docs.python.org/3/library/collections.html#collections.Counter), All of those tests treat missing elements as having zero counts so that Counter(a=1) == Counter(a=1, b=0) returns true.

* `x != y != z` 表现为 `x != y and y != z`, 而不是 `x != b and x != z and b != z`, 所以 `x < y > z` 是有可能的, [6.10. Comparisons](https://docs.python.org/3/reference/expressions.html#comparisons)

* [Glossary](https://docs.python.org/3/glossary.html)


<br><br>


# JavaScript
```js
// for loop:
for ... in // 循环出 key, 并且还会遍历到属性, 实际上遍历的是对象的属性名称, ❎
for ... of // 循环出 value, 不能遍历普通对象(ES6)
arr.forEach(function (v, i, array) {});
```

<br><br>


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

<details>
<summary>Show Picture</summary>
<img src="./pic/big-o-cheatsheet.png"/> 
</details>


<br><br>


# DP

* **Memoization** and **Tabulation**
  - Tabulation(Bottom Up)(刷表)
  - Memoization(Top Down)
* Difference:
![bottom-up vs top-down](/pic/Tabulation-vs-Memoization.png)


<br><br>


# 🆚
[大佬 0x3f 的b站](https://space.bilibili.com/206214)


<br><br>


# 🚩 
大佬们的经验之谈:

[🍭可乐可乐吗QAQ](https://leetcode.cn/circle/discuss/PDJMWs/)

[mumuxinfie](https://leetcode.cn/circle/discuss/nv1nBg/)


<br><br>

# 💡
[python3 模版 / 技巧](./lc_Python/template-trick.py)


<br><br>

# :seven::seven::eight::eight:
* C++: 一年精通, 三年熟悉, 五年了解, 十年用过
* MOD:
  * **不取余python超时**
  * dp中有减法, 负数 x 取余, 防止变一个大数: `(x + MOD) % MOD`
  * 区别: 
    * 取余(rem): 采用fix(), 向 0 方向舍入, `rem(x, y) = x - y. * fix(x./y)`
    * 取模(mod): 采用floor(), 向无穷小方向舍入, `mod(x, y) = x - y. * floor(x./y)`
    * 取模和求余应该是同一种运算, 只是在被除数和除数符号不同时, 余数的符号是有歧义的, 可为正也可为负
      * C, Go, JavaScript, Rust, Java, Swift, PHP中结果与被除数同符号
      * Python 中结果与除数同符号
  
* 回溯:
  * 两重for循环, 内层for break之后, 回溯"路径"被打断了, 不能复原到初始状态

* python堆 / heapq / 堆不保证直接list的顺序
  ```py
    # 不断push tuple 到堆中
    0 [(5, 0)]
    0 [(5, 0), (5, 0)]
    1 [(4, 1), (5, 0), (5, 0)]
    2 [(3, 2), (4, 1), (5, 0), (5, 0)]
    3 [(2, 3), (3, 2), (5, 0), (5, 0), (4, 1)]
    3 [(2, 3), (3, 2), (2, 3), (5, 0), (4, 1), (5, 0)]
    4 [(1, 4), (3, 2), (2, 3), (5, 0), (4, 1), (5, 0), (2, 3)]
    7 [(1, 4), (2, 7), (2, 3), (3, 2), (4, 1), (5, 0), (2, 3), (5, 0)]
    7 [(1, 4), (2, 7), (2, 3), (2, 7), (4, 1), (5, 0), (2, 3), (5, 0), (3, 2)]
    ```

* XOR (exclusive OR)
  * 半加运算，不带进位的二进制加法
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

* Trick
  ```py
  # cf 快读
  fn = sys.stdin.readline
  l = int(fn())
  for _ in range(l):
    n = int(fn())
    h = list(map(int, fn().split()))
  ```
