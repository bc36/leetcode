"""permutation"""

import itertools, math


def fn() -> None:
    # 1. itertools.permutations
    for lst in itertools.permutations(range(1, 4), 3):
        # do sth
        pass

    # 2. math.perm
    n = 3
    k = 2
    answer = math.perm(n, k)
    return
