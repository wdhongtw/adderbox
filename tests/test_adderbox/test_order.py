import unittest

from adderbox import order


class TestSearch(unittest.TestCase):

    def test_lower_bound(self) -> None:
        items = [1, 2, 2, 3, 3, 3]
        self.assertEqual(1, order.lower(items, 2))
        self.assertEqual(6, order.lower(items, 4))

    def test_lower_bound_by(self) -> None:
        class Value:
            def __init__(self, val: int) -> None:
                self.val = val

        key_func = lambda x: abs(x.val)
        items = [Value(1), Value(2), Value(-2), Value(-3), Value(3), Value(-3)]
        self.assertEqual(1, order.lower_by(items, 2, key=key_func))
        self.assertEqual(6, order.lower_by(items, 4, key=key_func))

    def test_upper_bound(self) -> None:
        items = [1, 2, 2, 3, 3, 3]
        self.assertEqual(1, order.upper(items, 1))
        self.assertEqual(6, order.upper(items, 3))

    def test_upper_bound_by(self) -> None:
        class Value:
            def __init__(self, val: int) -> None:
                self.val = val

        key_func = lambda x: abs(x.val)
        items = [Value(1), Value(2), Value(-2), Value(-3), Value(3), Value(-3)]
        self.assertEqual(1, order.upper_by(items, 1, key=key_func))
        self.assertEqual(6, order.upper_by(items, 3, key=key_func))


class TestSort(unittest.TestCase):

    def test_sort(self) -> None:
        items = [2, 1, 3, 3, 2, 3]
        self.assertEqual([1, 2, 2, 3, 3, 3], order.sort(items))

    def test_sort_by(self) -> None:
        class Value:
            def __init__(self, val: int) -> None:
                self.val = val

        key_func = lambda x: abs(x.val)
        items = [Value(1), Value(2), Value(-2), Value(-3), Value(3), Value(-3)]
        by_order = order.sort_by(items, key=key_func)
        self.assertEqual([1, 2, 2, 3, 3, 3], [key_func(v) for v in by_order])
