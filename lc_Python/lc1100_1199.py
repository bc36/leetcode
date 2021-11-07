import bisect


# 1146 - Snapshot Array - MEDIUM
# only update the change of each element, rather than record the whole arr
class SnapshotArray:
    def __init__(self, length: int):
        self.arr = [{0: 0} for _ in range(length)]
        self.snap_id = 0

    def set(self, index: int, val: int) -> None:
        self.arr[index][self.snap_id] = val
        return

    def snap(self) -> int:
        self.snap_id += 1
        return self.snap_id - 1

    def get(self, index: int, snap_id: int) -> int:
        d = self.arr[index]
        if snap_id in d:
            return d[snap_id]
        k = list(d.keys())
        i = bisect.bisect_left(k, snap_id)
        return d[k[i - 1]]