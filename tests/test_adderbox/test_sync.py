import concurrent.futures
import unittest

from adderbox import sync


class TestSync(unittest.TestCase):

    def test_threading_environment(self) -> None:
        counter = 0

        @sync.only_once
        def inc() -> int:
            nonlocal counter
            counter += 1
            return counter

        executor = concurrent.futures.ThreadPoolExecutor()
        try:
            entry = lambda _: inc()  # make wrapper to accept a None argument
            results = executor.map(entry, (None for _ in range(4)))
            self.assertTrue(all(r == 1 for r in results))
            self.assertEqual(1, counter)
        finally:
            executor.shutdown()

    def test_raise_same_error(self) -> None:
        the_error = RuntimeError("some-error")

        @sync.only_once
        def job() -> None:
            raise the_error

        with self.assertRaises(RuntimeError) as cm_1:
            job()
        self.assertIs(cm_1.exception, the_error)

        with self.assertRaises(RuntimeError) as cm_2:
            job()
        self.assertIs(cm_2.exception, the_error)
