from typing import List
import collections


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
                                set(account[1:]))
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