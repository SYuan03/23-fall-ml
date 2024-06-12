import random

nums = [2, 3, 4, 2, 1, 4, 9]


def swap(idx1, idx2):
    temp = nums[idx1]
    nums[idx1] = nums[idx2]
    nums[idx2] = temp


def partition(left, right):
    pivot = nums[right]
    lt = left
    gt = right
    fast = left
    while fast <= gt:
        if nums[fast] < pivot:
            swap(lt, fast)
            lt += 1
            fast += 1
        elif nums[fast] > pivot:
            swap(fast, gt)
            gt -= 1
        else:
            fast += 1
    return lt, gt


def qs(left, right):
    if left >= right: return
    bound = partition(left, right)
    qs(left, bound[0] - 1)
    qs(bound[1], right)


def quickSort():
    qs(0, len(nums) - 1)


quickSort()
print(nums)
