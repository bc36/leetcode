
### Big O
The O is short for “Order of”. If we’re discussing an algorithm with O(n), we say its **order of**, or **rate of growth**, is n, or linear complexity.

| O            | Complexity  |
| -            | -           |
| O(1)         | constant    |
| O(log n)     | logarithmic |
| O(n)         | linear      |
| O(n * log n) | log linear	 |
| O(n^2)       | quadratic   |
| O(n^3)       | cubic       |
| O(2^n)       | exponential |
| O(n!)        | factorial   |

![big(O)](/pic/big-o-cheatsheet.png)

### DP: top-down vs bottom-up

* Memoization and Tabulation
  - **Tabulation(Bottom Up)**: If you are calculating the Fibonacci sequence `fib(100)`, you would just call this, and it would call `fib(100) = fib(99) + fib(98)`, which would call `fib(99) = fib(98) + fib(97)`, ...etc..., which would call `fib(2) = fib(1) + fib(0) = 1 + 0 = 1`.
  - **Memoization(Top Down)**:  If you are performing fibonacci, you might choose to calculate the numbers in this order: `fib(2), fib(3), fib(4)`... caching every value so you can compute the next ones more easily.
* Memoization is easier to code.
* Difference:
![bottom-up vs top-down](/pic/Tabulation-vs-Memoization.png)