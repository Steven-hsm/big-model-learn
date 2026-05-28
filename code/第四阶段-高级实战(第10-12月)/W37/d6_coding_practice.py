"""
W37 Day 6 - 编程练习
====================
主题: 10道LeetCode AI方向题目实现, 数据结构回顾
"""

import os
from datetime import datetime
from collections import deque, Counter
import heapq
import math


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================
# 1. 数据结构回顾
# =============================================
def review_data_structures():
    """回顾核心数据结构"""
    print_section("核心数据结构回顾")

    ds = {
        "数组(Array)": {
            "特点": "连续内存, O(1)随机访问, O(n)插入/删除",
            "常见题型": "双指针、滑动窗口、前缀和",
            "Python": "list"
        },
        "链表(LinkedList)": {
            "特点": "非连续内存, O(1)插入/删除, O(n)查找",
            "常见题型": "反转链表、合并链表、环检测",
            "Python": "自定义Node类"
        },
        "栈(Stack)": {
            "特点": "LIFO, O(1)push/pop",
            "常见题型": "括号匹配、表达式求值、单调栈",
            "Python": "list(append/pop)"
        },
        "队列(Queue)": {
            "特点": "FIFO, O(1)入队/出队",
            "常见题型": "BFS、滑动窗口、优先队列",
            "Python": "collections.deque"
        },
        "哈希表(HashMap)": {
            "特点": "O(1)平均查找/插入",
            "常见题型": "两数之和、频率统计、LRU缓存",
            "Python": "dict, set"
        },
        "堆(Heap)": {
            "特点": "O(1)获取最值, O(log n)插入/删除",
            "常见题型": "Top-K、中位数、合并有序列表",
            "Python": "heapq(小顶堆)"
        },
        "树(Tree)": {
            "特点": "层次结构, 常用二叉树、BST",
            "常见题型": "遍历、深度、路径、BST操作",
            "Python": "自定义TreeNode类"
        },
        "图(Graph)": {
            "特点": "节点+边, 有向/无向, 加权/无权",
            "常见题型": "BFS/DFS、最短路径、拓扑排序",
            "Python": "dict + list邻接表"
        }
    }

    for name, info in ds.items():
        print(f"\n  {name}")
        for key, value in info.items():
            print(f"    {key}: {value}")


# =============================================
# 2. LeetCode题目实现
# =============================================
def implement_leetcode_solutions():
    """实现10道AI方向相关LeetCode题"""

    # ===== 题目1: 两数之和 =====
    print_section("题目1: 两数之和 (LeetCode #1)")
    print("  给定数组和目标值, 返回两个数的下标")

    def two_sum(nums, target):
        """哈希表解法, O(n)时间"""
        lookup = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in lookup:
                return [lookup[complement], i]
            lookup[num] = i
        return []

    result = two_sum([2, 7, 11, 15], 9)
    print(f"  输入: nums=[2,7,11,15], target=9")
    print(f"  输出: {result}")

    # ===== 题目2: 前K个高频元素 =====
    print_section("题目2: 前K个高频元素 (LeetCode #347)")
    print("  返回数组中出现频率前k高的元素")

    def top_k_frequent(nums, k):
        """堆解法, O(n log k)时间"""
        counter = Counter(nums)
        return heapq.nlargest(k, counter.keys(), key=counter.get)

    result = top_k_frequent([1, 1, 1, 2, 2, 3], 2)
    print(f"  输入: nums=[1,1,1,2,2,3], k=2")
    print(f"  输出: {result}")

    # ===== 题目3: 无重复字符的最长子串 =====
    print_section("题目3: 无重复字符最长子串 (LeetCode #3)")
    print("  找出不含重复字符的最长子串长度")

    def length_of_longest_substring(s):
        """滑动窗口, O(n)时间"""
        char_set = set()
        left = 0
        max_len = 0
        for right in range(len(s)):
            while s[right] in char_set:
                char_set.remove(s[left])
                left += 1
            char_set.add(s[right])
            max_len = max(max_len, right - left + 1)
        return max_len

    result = length_of_longest_substring("abcabcbb")
    print(f"  输入: 'abcabcbb'")
    print(f"  输出: {result}")

    # ===== 题目4: 合并K个升序链表 =====
    print_section("题目4: 合并K个升序链表 (LeetCode #23)")
    print("  使用最小堆合并K个有序列表")

    def merge_k_sorted_lists(lists):
        """堆解法合并K个有序列表"""
        result = []
        heap = []
        for i, lst in enumerate(lists):
            if lst:
                heapq.heappush(heap, (lst[0], i, 0))

        while heap:
            val, list_idx, elem_idx = heapq.heappop(heap)
            result.append(val)
            if elem_idx + 1 < len(lists[list_idx]):
                next_val = lists[list_idx][elem_idx + 1]
                heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))

        return result

    lists = [[1, 4, 5], [1, 3, 4], [2, 6]]
    result = merge_k_sorted_lists(lists)
    print(f"  输入: [[1,4,5], [1,3,4], [2,6]]")
    print(f"  输出: {result}")

    # ===== 题目5: 单词搜索 =====
    print_section("题目5: 单词搜索 (LeetCode #79)")
    print("  在二维网格中搜索单词(DFS回溯)")

    def exist(board, word):
        """DFS + 回溯"""
        rows, cols = len(board), len(board[0])

        def dfs(r, c, idx):
            if idx == len(word):
                return True
            if r < 0 or r >= rows or c < 0 or c >= cols:
                return False
            if board[r][c] != word[idx]:
                return False

            temp = board[r][c]
            board[r][c] = '#'
            found = (dfs(r+1, c, idx+1) or dfs(r-1, c, idx+1) or
                     dfs(r, c+1, idx+1) or dfs(r, c-1, idx+1))
            board[r][c] = temp
            return found

        for r in range(rows):
            for c in range(cols):
                if dfs(r, c, 0):
                    return True
        return False

    board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
    result = exist(board, "ABCCED")
    print(f"  输入: board=[['A','B','C','E'],...], word='ABCCED'")
    print(f"  输出: {result}")

    # ===== 题目6: 编辑距离 =====
    print_section("题目6: 编辑距离 (LeetCode #72)")
    print("  计算两个字符串的最小编辑操作数(NLP核心算法)")

    def min_distance(word1, word2):
        """动态规划, O(m*n)时间"""
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[m][n]

    result = min_distance("horse", "ros")
    print(f"  输入: word1='horse', word2='ros'")
    print(f"  输出: {result}")

    # ===== 题目7: 最长递增子序列 =====
    print_section("题目7: 最长递增子序列 (LeetCode #300)")
    print("  找出最长严格递增子序列的长度")

    def length_of_lis(nums):
        """二分查找解法, O(n log n)"""
        tails = []
        for num in nums:
            left, right = 0, len(tails)
            while left < right:
                mid = (left + right) // 2
                if tails[mid] < num:
                    left = mid + 1
                else:
                    right = mid
            if left == len(tails):
                tails.append(num)
            else:
                tails[left] = num
        return len(tails)

    result = length_of_lis([10, 9, 2, 5, 3, 7, 101, 18])
    print(f"  输入: [10,9,2,5,3,7,101,18]")
    print(f"  输出: {result}")

    # ===== 题目8: LRU缓存 =====
    print_section("题目8: LRU缓存 (LeetCode #146)")
    print("  设计LRU(最近最少使用)缓存机制")

    class LRUCache:
        """LRU缓存实现"""
        def __init__(self, capacity):
            self.capacity = capacity
            self.cache = {}
            self.order = deque()

        def get(self, key):
            if key in self.cache:
                self.order.remove(key)
                self.order.append(key)
                return self.cache[key]
            return -1

        def put(self, key, value):
            if key in self.cache:
                self.order.remove(key)
            elif len(self.cache) >= self.capacity:
                oldest = self.order.popleft()
                del self.cache[oldest]
            self.cache[key] = value
            self.order.append(key)

    lru = LRUCache(2)
    lru.put(1, 1)
    lru.put(2, 2)
    print(f"  get(1): {lru.get(1)}")
    lru.put(3, 3)
    print(f"  get(2): {lru.get(2)} (应为-1, 已被淘汰)")

    # ===== 题目9: 岛屿数量 =====
    print_section("题目9: 岛屿数量 (LeetCode #200)")
    print("  计算二维网格中岛屿的数量(BFS/DFS)")

    def num_islands(grid):
        """DFS解法"""
        if not grid:
            return 0

        rows, cols = len(grid), len(grid[0])
        count = 0

        def dfs(r, c):
            if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
                return
            grid[r][c] = '0'
            dfs(r+1, c); dfs(r-1, c); dfs(r, c+1); dfs(r, c-1)

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1':
                    dfs(r, c)
                    count += 1
        return count

    grid = [["1","1","0","0","0"],
            ["1","1","0","0","0"],
            ["0","0","1","0","0"],
            ["0","0","0","1","1"]]
    result = num_islands(grid)
    print(f"  输入: 4x5网格")
    print(f"  输出: {result}")

    # ===== 题目10: 字符串解码 =====
    print_section("题目10: 字符串解码 (LeetCode #394)")
    print("  解码编码后的字符串(栈应用)")

    def decode_string(s):
        """栈解法"""
        stack = []
        curr_str = ""
        curr_num = 0

        for char in s:
            if char.isdigit():
                curr_num = curr_num * 10 + int(char)
            elif char == '[':
                stack.append((curr_str, curr_num))
                curr_str = ""
                curr_num = 0
            elif char == ']':
                prev_str, num = stack.pop()
                curr_str = prev_str + curr_str * num
            else:
                curr_str += char

        return curr_str

    result = decode_string("3[a2[c]]")
    print(f"  输入: '3[a2[c]]'")
    print(f"  输出: {result}")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("=" * 60)
    print("  W37 Day 6 - 编程练习")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    review_data_structures()
    implement_leetcode_solutions()

    print("\n" + "=" * 60)
    print("  编程题是面试的基础, 每天练习1-2题保持手感!")
    print("  建议: 先理解算法思路, 再手写代码, 最后优化")
    print("=" * 60)
