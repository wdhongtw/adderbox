import random
import time
import unittest

from adderbox import folder


class TestFolder(unittest.TestCase):

    def test_segment_tree(self) -> None:
        values = [1, -3, 5, 7, -9, 11]
        better = lambda a, b: a if a > b else b
        tree = folder.SegmentTree(better, values)
        self.assertEqual(tree.query(folder.Range(0, 1)), 1)
        self.assertEqual(tree.query(folder.Range(0, 6)), 11)
        self.assertEqual(tree.query(folder.Range(2, 5)), 7)
        tree.update(5, 3)
        tree.update(0, -1)
        self.assertEqual(tree.query(folder.Range(0, 1)), -1)
        self.assertEqual(tree.query(folder.Range(0, 3)), 5)
        self.assertEqual(tree.query(folder.Range(3, 6)), 7)

        characters = ["a", "b", "cd", "ef"]
        concat = lambda a, b: a + b
        word_tree = folder.SegmentTree(concat, characters)
        self.assertEqual(word_tree.query(folder.Range(1, 3)), "bcd")
        word_tree.update(0, "h")
        word_tree.update(1, "el")
        word_tree.update(2, "lo")
        self.assertEqual(word_tree.query(folder.Range(0, 3)), "hello")

    def test_union_find(self) -> None:
        find_set = folder.DisjointSet(range(4))
        self.assertEqual(find_set.find(3), 3)
        find_set.union(2, 3)
        self.assertEqual(find_set.find(3), 2)
        find_set.union(0, 1)
        find_set.union(1, 3)
        find_set.union(1, 3)
        self.assertEqual(find_set.find(3), 0)
        self.assertEqual(find_set.find(2), 0)
        self.assertEqual(len(find_set), 4)

        leader, group = next(find_set.as_sets())
        self.assertEqual(leader, 0)
        self.assertEqual(list(group), list(range(4)))

    def test_rb_tree(self) -> None:
        tree = folder.RbTree(v for v in range(8))

        for pair in range(8):
            self.assertTrue(pair in tree)

        self.assertEqual(list(range(8)), list(tree))
        self.assertEqual(list(reversed(range(8))), list(reversed(tree)))

        for val in range(4):
            tree.add(val)
        self.assertEqual([0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7], list(tree))
        for val in range(2, 6):
            tree.remove(val)
        self.assertEqual([0, 0, 1, 1, 2, 3, 6, 7], list(tree))

        self.assertIsNone(tree.find(8))
        self.assertEqual(2, tree.find(2))


class TestSkipList(unittest.TestCase):

    def test_empty_construct(self) -> None:
        items = folder.SkipList[int, int]()

        self.assertEqual(0, len(items))
        self.assertEqual([], list(items))
        self.assertTrue(1 not in items)

    def test_membership_check(self) -> None:
        items = folder.SkipList(range(2))

        self.assertEqual(2, len(items))
        self.assertTrue(0 in items)
        self.assertTrue(2 not in items)

    def test_basic_operations(self) -> None:
        items = folder.SkipList[int, int]()

        for num in range(8):
            items.add(num)
        for num in range(4):
            items.remove(num * 2 + 1)
        for num in range(2):
            items.add(num)

        self.assertEqual(6, len(items))
        self.assertEqual([0, 0, 1, 2, 4, 6], list(items))

        with self.assertRaises(ValueError):
            items.remove(8)

    def test_sorted_order(self) -> None:
        items = folder.SkipList(reversed(range(8)))

        self.assertEqual(list(range(8)), list(items))

    def test_none_as_value(self) -> None:
        items = folder.SkipList[None, int](key=lambda _: 0)
        items.add(None)

        self.assertEqual(1, len(items))
        self.assertEqual([None], list(items))

    def test_inverse_order(self) -> None:
        items = folder.SkipList(range(8), key=lambda x: -x)

        self.assertEqual(8, len(items))
        self.assertEqual(list(reversed(range(8))), list(items))

    def test_heavy_random_insert(self) -> None:
        items = folder.SkipList[int, int]()
        pivot: list[int] = []

        for _ in range(0x400):
            val = random.randint(0, 0x400)
            items.add(val)
            pivot.append(val)

        self.assertEqual(len(pivot), len(items))
        self.assertEqual(sorted(pivot), list(items))

    def test_index_operations(self) -> None:
        values = range(8)
        guards = [-1, 8]
        items = folder.SkipList(values)

        for val in range(8):
            self.assertEqual(val, items.index(val))
        for val in guards:
            with self.assertRaises(ValueError):
                items.index(val)
        for idx in values:
            self.assertEqual(idx, items.at(idx))
        for idx in guards:
            with self.assertRaises(IndexError):
                items.at(idx)

    @unittest.skip("for-benchmark")
    def test_query_complexity(self) -> None:

        def track_time(results, func):
            start = time.monotonic()
            func()
            elapsed = time.monotonic() - start
            results.append(elapsed)

        for size in range(0x0, 0x10000, 0x1000):
            if size == 0:
                continue
            values = list(range(size))
            random.shuffle(values)

            results = []
            for _ in range(0x10):
                items = folder.SkipList(values)

                def run_once():
                    for _ in range(0x4000):
                        _ = items.find(random.randint(0, size - 1))

                track_time(results, run_once)

            # expect O(log n) complexity
            print(f"size: 0x{size:04x} records, total: {sum(results):.4f} sec")
