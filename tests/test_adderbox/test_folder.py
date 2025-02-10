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

    def test_tree_map(self) -> None:

        container = folder.TreeMap({1: "2", 3: "4"}.items())

        self.assertEqual("2", container[1])
        self.assertEqual([1, 3], list(iter(container)))

        del container[3]
        container[1] = "5"
        container[0] = "2"
        self.assertEqual([0, 1], list(container))

    def test_multi_set(self) -> None:

        container = folder.MultiSet(reversed(range(8)))

        self.assertTrue(2 in container)
        self.assertTrue(9 not in container)

        self.assertEqual(list(range(8)), list(container))

        for val in range(4):
            container.add(val)
        self.assertEqual(12, len(container))
        for val in range(2, 6):
            container.discard(val)
        self.assertEqual(8, len(container))
        self.assertEqual([0, 0, 1, 1, 2, 3, 6, 7], list(container))
