"""
sync is a module to provide synchronize primitives, e.g. do-once.
"""

from collections.abc import Callable
from threading import Lock


def only_once[T](func: Callable[[], T]) -> Callable[[], T]:
    """
    only_once is a decorator that ensure func to run exactly once.

    In multithreading environment, it's tedious to make sure some value to be
    build only once, if the value is evaluate lazily. So comes this decorator.
    Unlike functools.cache, this decorator provide guarantee exactly once.

    func can be function, method on some instance or lambda.
    func with no return value can also be used since no return is return None.
    If func raise exception, the exception is captured and raise again
    in all invocation of func.

    function with parameter is not supported since that it makes no sense
    to cache output of some particular input for other input.

    Performance is part of design, we do not acquire any lock on happy path.
    However the GIL may still slow down the decorated function before GIL
    is complete removed from CPython.

    Notice: the the evaluation of decorator on some function must
    happen in main thread, e.g. in top-level function or nested function in some
    function executed by single thread, otherwise this decorator won't work.

    Inspired by golang sync.Once interface.
    """

    lock = Lock()  # created when current once decorator is evaluated
    done = False
    error: Exception | None = None
    instance: T

    def wrapped() -> T:
        if not done:  # happy path should not acquire lock
            try_init()

        if error is not None:
            raise error
        return instance

    def try_init():
        nonlocal done, instance, error
        with lock:
            if done:
                return
            try:
                instance = func()
            except Exception as err:
                error = err
            finally:
                done = True

    return wrapped
