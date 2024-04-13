# 快速排序使用分治法策略来把一个序列分为较小和较大的2个子序列, 然后递归地排序两个子序列。

# 步骤为:
# 1. 挑选基准值: 从数列中挑出一个元素, 称为"基准"(pivot),
# 2. 分割: 重新排序数列, 所有比基准值小的元素摆放在基准前面, 所有比基准值大的元素摆在基准后面(与基准值相等的数可以到任何一边)。在这个分割结束之后, 对基准值的排序就已经完成,
# 3. 递归排序子序列: 递归地将小于基准值元素的子序列和大于基准值元素的子序列排序。

# 平均时间复杂度 nlogn
# 最坏时间复杂度 n^2
# 最优时间复杂度 nlogn
# 空间复杂度	根据实现的方式不同而不同


# https://www.hello-algo.com/chapter_sorting/quick_sort/

from typing import List


def partition(arr: List[int], left: int, right: int) -> int:
    # pivot = left
    i = j = left + 1
    while i <= right:
        if arr[left] > arr[i]:
            arr[i], arr[j] = arr[j], arr[i]
            j += 1
        i += 1
    arr[left], arr[j - 1] = arr[j - 1], arr[left]
    return j - 1


def partition2(nums: list[int], left: int, right: int) -> int:
    """哨兵划分"""
    i, j = left, right  # 以 nums[left] 为基准数
    while i < j:
        while i < j and nums[j] >= nums[left]:
            j -= 1  # 从右向左找首个小于基准数的元素
        while i < j and nums[i] <= nums[left]:
            i += 1  # 从左向右找首个大于基准数的元素
        nums[i], nums[j] = nums[j], nums[i]
    nums[i], nums[left] = nums[left], nums[i]  # 将基准数交换至两子数组的分界线
    return i  # 返回基准数的索引


def quick_sort(arr: List[int], left: int = None, right: int = None) -> List[int]:
    # 可选判断
    # if left == None:
    #     left = 0
    # if right == None:
    #     right = len(arr) - 1
    if left < right:
        partitionIndex = partition(arr, left, right)
        quick_sort(arr, left, partitionIndex - 1)
        quick_sort(arr, partitionIndex + 1, right)
    return arr


def quick_sort2(nums: list[int], left: int, right: int) -> list[int]:
    if left >= right:
        return
    pivot = partition2(nums, left, right)
    quick_sort2(nums, left, pivot - 1)
    quick_sort2(nums, pivot + 1, right)
    return nums


arr = [5, 2, 3, 1]
arr2 = [5, 1, 1, 2, 0, 0]


print(quick_sort(arr, 0, len(arr) - 1))
print(quick_sort(arr2, 0, len(arr2) - 1))


print(quick_sort2(arr, 0, len(arr) - 1))
print(quick_sort2(arr2, 0, len(arr2) - 1))
