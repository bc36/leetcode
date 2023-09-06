"""dp - digit DP"""

import functools


def countSpecialNumbers(n: int) -> int:
    s = str(n)

    @functools.lru_cache(None)
    def dfs(i: int, mask: int, is_limit: bool, is_num: bool) -> int:
        if i == len(s):
            return int(is_num)
        ans = 0
        if not is_num:
            ans = dfs(i + 1, mask, False, False)
        bound = int(s[i]) if is_limit else 9
        for d in range(0 if is_num else 1, bound + 1):
            if mask >> d & 1 == 0:
                ans += dfs(i + 1, mask | (1 << d), is_limit and d == bound, True)
        return ans

    return dfs(0, 0, True, False)
