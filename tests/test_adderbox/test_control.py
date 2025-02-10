import contextlib
import unittest
from collections.abc import Iterator

from adderbox.control import build_cm, extract_cm


class TestControl(unittest.TestCase):

    def test_build_context(self) -> None:
        resource = "Hello"
        done = False

        def close() -> None:
            nonlocal done
            done = True

        with build_cm(resource, close) as cm:
            self.assertEqual("Hello", cm)
        self.assertEqual(True, done)

    def test_extract_context(self) -> None:
        done = False

        @contextlib.contextmanager
        def job() -> Iterator[str]:
            yield "Hello"
            nonlocal done
            done = True

        cm = job()
        resource, close = extract_cm(cm)
        self.assertEqual("Hello", resource)

        close()
        self.assertEqual(True, done)
