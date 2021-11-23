from typing import List


# tiktok oa
# Q:
# https://www.chegg.com/homework-help/questions-and-answers/3-tree-binary-tree-represented-sequence-parent-child-pairs-example-b-c-b-g-c-h-e-f-b-d-c-e-q42330174
# A:
# https://leetcode.com/discuss/general-discussion/1503088/tiktok-online-asssesment-is-this-a-tree-hackerrank-solution
def getSExpression(s: str):
    graph = [[0 for _ in range(26)] for _ in range(26)]
    nodes = set()
    E2 = False
    for i in range(1, len(s), 6):
        x = ord(s[i]) - ord("A")
        y = ord(s[i + 2]) - ord("A")
        if graph[x][y]:
            E2 = True
        graph[x][y] = True
        nodes.add(ord(s[i]))
        nodes.add(ord(s[i + 2]))
    E1 = False
    for i in range(26):
        count = 0
        for j in range(26):
            if graph[i][j]:
                count += 1
        if count > 2:
            return "E1"
    if E2:
        return "E2"

    def isCycle(node: int, graph: List[List[bool]],
                visited: List[bool]) -> bool:
        if visited[node - ord("A")]:
            return True
        visited[node - ord("A")] = True
        for i in range(26):
            if graph[node - ord("A")][i]:
                if isCycle(i + ord("A"), graph, visited):
                    return True
        return False

    numOfRoots, root = 0, 0
    while len(nodes) > 0:
        node = nodes.pop()
        for i in range(26):
            if graph[i][node - ord("A")]:
                break
            if i == 25:
                numOfRoots += 1
                root = node
                visited = [0 for _ in range(26)]
                if isCycle(node, graph, visited):
                    return "E3"

    def getExpressionHelper(root: int, graph: List[List[bool]]) -> str:
        left, right = "", ""
        for i in range(26):
            if graph[root - ord("A")][i]:
                left = getExpressionHelper(i + ord("A"), graph)
                for j in range(i + 1, 26):
                    if graph[root - ord("A")][j]:
                        right = getExpressionHelper(j + ord("A"), graph)
                        break
                break
        return "(" + chr(root) + left + right + ")"

    if numOfRoots == 0:
        return "E3"
    if numOfRoots > 1:
        return "E4"
    if root == 0:
        return "E5"
    return getExpressionHelper(root, graph)


# print(getSExpression("(B,D) (D,E) (A,B) (C,F) (E,G) (A,C)"))
# print(getSExpression("(A,B) (A,C) (B,G) (C,H) (E,F) (B,D) (C,E)"))
# print(getSExpression("(A,B) (A,C) (B,D) (D,C)"))