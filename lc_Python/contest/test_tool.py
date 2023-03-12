from collections import deque
from typing import List, Optional, Union

from typing_extensions import TypeAlias

null: TypeAlias = None
# print(type(None), null == None, null)

import logging

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(filename)s: %(message)s")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(lineno)d - [%(levelname)s]: %(message)s",
)
logger = logging.getLogger(__name__)


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def parseInput(s: str) -> List[tuple]:
    """
    support parsing:
        str
        int
        List[str]
        List[int]
        List[List[str]]
        List[List[int]]
    """
    totalCases = []
    for line in s.split("\n"):
        if len(line) == 0:
            continue
        case = []
        for equation in line.split(", "):
            logger.debug(f"equation, {equation}")

            i = equation.index("= ")
            target = equation[i + 2 :]
            logger.debug(f"target, {target}")

            if target[0] == '"':
                logging.debug(f"string, {target}")
                case.append(target)
            elif target[0].isdigit():
                logging.debug(f"number, {target}")
                case.append(int(target))
            elif target[0] == "[":
                if target[1] == "[":
                    if target[2] == '"':
                        logging.debug(f"List[List[str]], {target}")
                        l = []
                        for v in target[2:-2].split("],["):
                            l.append(list(w[1:-1] for w in v.split(",")))
                        case.append(l)
                    else:
                        logging.debug(f"List[List[int]], {target}")
                        l = []
                        for v in target[2:-2].split("],["):
                            l.append(list(map(int, (num for num in v.split(",")))))
                        case.append(l)
                else:
                    if target[1] == '"':
                        logging.debug(f"List[str], {target[1]}")
                        case.append(list(w[1:-1] for w in target[1:-1].split(",")))
                    else:
                        logging.debug(f"List[int], {target[1]}")
                        case.append(
                            list(map(int, (num for num in target[1:-1].split(","))))
                        )
        totalCases.append(case)
    return totalCases


testcase = """
s = "1010", target = "0110"
words = [["a","a"],["a"],["a","a","a"]], left = 0, right = 2
words = [[1,2,3],[4,5,6],[7,8,9]], left = 1, right = 4
"""
for v in parseInput(testcase):
    logger.debug(v)

# logger.debug("\n")


def array2tree(arr: List[Union[int, null]]) -> Optional[TreeNode]:
    """
    https://support.leetcode.cn/hc/kb/article/1549360/
    [1, null, 2, 3] 是个串行化格式, 表达了一个水平顺序遍历的二叉树

    [3,9,20,null,null,15,7]
    [5, 4, 7, 3, null, 2, null, -1, null, 9]
    """
    if len(arr) == 0:
        return None
    root = TreeNode(arr[0])
    q = deque([root])
    f = 0
    for v in arr[1:]:
        if v is not None:
            node = TreeNode(v)
            q.append(node)
            if f % 2 == 0:
                q[0].left = node
            else:
                q[0].right = node

        f ^= 1
        if f % 2 == 0:
            q.popleft()
    return root


def tree2array(root: Optional[TreeNode]) -> List[Union[int, None]]:
    if not root:
        return []
    array = [root.val]
    q = [root]
    while q:
        new = []
        vals = []
        f = False
        for v in q:
            if v.left:
                new.append(v.left)
                f = True
            if v.right:
                new.append(v.right)
                f = True
            vals.append(v.left.val if v.left else None)
            vals.append(v.right.val if v.right else None)
        if not f:
            break
        q = new
        array.extend(vals)
    while array[-1] is None:
        array.pop()
    return array


def show_vals(root: TreeNode):
    q = [root]
    arr = [root.val]
    while q:
        new = []
        for v in q:
            if v.left:
                new.append(v.left)
                arr.append(v.left.val)
            if v.right:
                new.append(v.right)
                arr.append(v.right.val)
        q = new
    logger.debug(arr)
    return


# fake = array2tree([1, null, 2, 3])
# logger.debug(tree2array(fake))
# show_vals(fake)
# arr = tree2array(fake)
# show_vals(array2tree(arr))
