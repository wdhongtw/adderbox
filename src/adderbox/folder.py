"""
folder is a module for some containers.

membership-check and iteration(folding) is supported whenever possible.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    MutableMapping,
    MutableSet,
    Sequence,
)
from typing import assert_never
from typing import NamedTuple
from typing import Protocol


type BinOp[T] = Callable[[T, T], T]


class Range(NamedTuple):
    lo: int
    hi: int


def _must[T](item: T | None) -> T:
    assert item is not None
    return item


@dataclasses.dataclass
class _Segment[T]:
    """Node for segment tree."""

    inter: Range
    val: T
    left: _Segment[T] | None = None
    right: _Segment[T] | None = None


class SegmentTree[T]:
    """
    A segment tree implementation to support balance between update and query.
    """

    def __init__(self, aggr: BinOp[T], values: Sequence[T]) -> None:
        def build(inter: Range) -> _Segment[T]:
            if inter.hi - inter.lo == 1:
                return _Segment(inter, values[inter.lo])

            mid = (inter.lo + inter.hi) // 2
            left = build(Range(inter.lo, mid))
            right = build(Range(mid, inter.hi))
            val = aggr(left.val, right.val)
            return _Segment(inter, val, left, right)

        self._root = build(Range(0, len(values)))
        self._aggr = aggr

    def query(self, interval: Range) -> T:
        def query(node: _Segment, inter: Range) -> T:
            if node.inter.hi - node.inter.lo == 1:
                return node.val

            left, right = _must(node.left), _must(node.right)
            if inter.hi <= left.inter.hi:
                return query(left, inter)
            if inter.lo >= right.inter.lo:
                return query(right, inter)

            left_val = query(left, Range(inter.lo, left.inter.hi))
            right_val = query(right, Range(right.inter.lo, inter.hi))
            return self._aggr(left_val, right_val)

        return query(self._root, interval)

    def update(self, idx: int, value: T) -> None:
        def update(node: _Segment[T]) -> None:
            if node.inter.hi - node.inter.lo == 1:
                node.val = value
                return

            left, right = _must(node.left), _must(node.right)
            if idx < left.inter.hi:
                update(left)
            if idx >= right.inter.lo:
                update(right)
            node.val = self._aggr(left.val, right.val)

        update(self._root)


class DisjointSet[T]:
    """
    A minimal generic union find set.

    Satisfy "Collection" protocol (Container, Sized, Iterable).
    """

    def __init__(self, values: Iterable[T]) -> None:
        self._parent = {value: value for value in values}

    def find(self, val: T) -> T:
        """
        Return the leader of the set which contains this value.
        """
        store = self._parent
        if store[val] == val:
            return val
        store[val] = self.find(store[val])
        return store[val]

    def union(self, val_a: T, val_b: T, *, check: bool = False) -> None:
        """
        Union two sets by two values in corresponding set.

        When check=True, ValueError is raised if two value already in same set.
        """
        top_a, top_b = self.find(val_a), self.find(val_b)
        if top_a == top_b and check:
            raise ValueError("two value already in same set")

        # notice that it's nop if top_a equals top_b already
        self._parent[top_b] = top_a

    def __contains__(self, val: T) -> bool:
        return val in self._parent

    def __iter__(self) -> Iterator[T]:
        return iter(self._parent)

    def __len__(self) -> int:
        return len(self._parent)

    def __repr__(self) -> str:
        groups = (g for _, g in self.as_sets())
        return repr([set(group) for group in groups])

    def as_sets(self) -> Iterator[tuple[T, Iterator[T]]]:
        """
        Return a iterable of items in the disjoint set, grouped by set leader.
        """
        by_leader: dict[T, list[T]] = {}
        for item in self:
            by_leader.setdefault(self.find(item), list()).append(item)

        for leader, group in by_leader.items():
            yield (leader, iter(group))


class Cmp(Protocol):
    """Support less than comparison."""

    def __lt__(self, other, /) -> bool: ...


class C(enum.Enum):
    """Color enum for RB Tree"""

    R = 1
    B = 2


class _Nil(NamedTuple):
    """Nil node as the leaf node for RB Tree"""


# TODO: Use NamedTuple once mypy support pattern match on generic named tuple
@dataclasses.dataclass(frozen=True)
class T[K]:
    """Generic node type for RB Tree"""

    color: C
    left: T[K] | _Nil
    item: K
    right: T[K] | _Nil


type _RbTree[_K] = T[_K] | _Nil
"""
A Red Black Tree is a Node or Nil

Support adding repetitive item, for implementing multi-set / ordered-list.

With product element type which compares on partial field, it's possible to
implement ordered-mapping.

The implementation is based on FP style Red Black Tree by Abhiroop Sarkar,
which is based on the work of Chris Okasaki.

- https://github.com/Abhiroop/okasaki/blob/master/src/RedBlackTree.hs
- https://abhiroop.github.io/Haskell-Red-Black-Tree/

Other useful resource

- https://fstaals.net/RedBlackTree.html
- https://softwarefoundations.cis.upenn.edu/vfa-current/Redblack.html

Invariants:

- No red node has a red parent
- Every path from some node to an nil contains the same number of black nodes
"""


def _rb_traverse[K: Cmp](node: _RbTree[K], *, reverse: bool = False) -> Iterator[K]:
    match node:
        case _Nil():
            return
        case T(_, l, x, r):
            before, after = (l, r) if not reverse else (r, l)
            yield from _rb_traverse(before, reverse=reverse)
            yield x
            yield from _rb_traverse(after, reverse=reverse)
        case _ as unreachable:
            assert_never(unreachable)


def _rb_member[K: Cmp](node: _RbTree[K], x: K) -> _RbTree[K]:
    match node:
        case _Nil():
            return node
        case T(_, l, y, _) if x < y:
            return _rb_member(l, x)
        case T(_, _, y, r) if y < x:
            return _rb_member(r, x)
        case T():
            return node
        case _ as unreachable:
            assert_never(unreachable)


def _rb_ins_root[K: Cmp](root: _RbTree[K], x: K) -> _RbTree[K]:
    return _black(_rb_ins(root, x))


def _rb_ins[K: Cmp](node: _RbTree[K], x: K) -> _RbTree[K]:
    match node:
        case _Nil():
            return T(C.R, _Nil(), x, _Nil())
        case T(c, l, y, r) if x < y:
            return _rb_bal(T(c, _rb_ins(l, x), y, r))
        case T(c, l, y, r):  # prefer right for equal
            return _rb_bal(T(c, l, y, _rb_ins(r, x)))
        case _ as unreachable:
            assert_never(unreachable)


def _black[K: Cmp](node: _RbTree[K]) -> _RbTree[K]:
    """make the tree root as black"""
    match node:
        case _Nil():
            return node
        case T(C.R, l, y, r):
            return T(C.B, l, y, r)
        case T():
            return node
        case _ as unreachable:
            assert_never(unreachable)


def _rb_bal[K: Cmp](node: T[K]) -> T[K]:
    match node:
        case T(C.B, T(C.R, T(C.R, a, x, b), y, c), z, d):
            return T(C.R, T(C.B, a, x, b), y, T(C.B, c, z, d))
        case T(C.B, T(C.R, a, x, T(C.R, b, y, c)), z, d):
            return T(C.R, T(C.B, a, x, b), y, T(C.B, c, z, d))
        case T(C.B, a, x, T(C.R, T(C.R, b, y, c), z, d)):
            return T(C.R, T(C.B, a, x, b), y, T(C.B, c, z, d))
        case T(C.B, a, x, T(C.R, b, y, T(C.R, c, z, d))):
            return T(C.R, T(C.B, a, x, b), y, T(C.B, c, z, d))
        case _:
            return node


def _rb_del[K: Cmp](node: _RbTree[K], x: K) -> _RbTree[K]:
    match node:
        case _Nil():
            raise ValueError("remove from empty tree")
        case T(_, l, y, r) if x < y:
            return _rb_del_l(node, x)
        case T(_, l, y, r) if y < x:
            return _rb_del_r(node, x)
        case T(_, l, _, r):
            return _rb_fuse(l, r)
        case _ as unreachable:
            assert_never(unreachable)


def _rb_del_l[K: Cmp](node: T[K], x: K) -> T[K]:
    match node:
        case T(_, T(C.B, _, _, _) as t1, y, t2):
            return _rb_bal_l(T(C.B, _rb_del(t1, x), y, t2))
        case T(_, t1, y, t2):
            return T(C.R, _rb_del(t1, x), y, t2)
        case _ as unreachable:
            assert_never(unreachable)


def _rb_bal_l[K: Cmp](node: T[K]) -> T[K]:
    match node:
        case T(C.B, T(C.R, t1, x, t2), y, t3):
            return T(C.R, T(C.B, t1, x, t2), y, t3)
        case T(C.B, t1, y, T(C.B, t2, z, t3)):
            return _rb_bal(T(C.B, t1, y, T(C.R, t2, z, t3)))
        case T(C.B, t1, y, T(C.R, T(C.B, t2, u, t3), z, t4)):
            return T(C.R, T(C.B, t1, y, t2), u, _rb_bal(T(C.B, t3, z, _black(t4))))
        case _:
            raise AssertionError("unexpected rb-tree structure")


def _rb_del_r[K: Cmp](node: T[K], x: K) -> T[K]:
    match node:
        case T(_, t1, y, T(C.B, _, _, _) as t2):
            return _rb_bal_r(T(C.B, t1, y, _rb_del(t2, x)))
        case T(_, t1, y, t2):
            return T(C.R, t1, y, _rb_del(t2, x))
        case _ as unreachable:
            assert_never(unreachable)


def _rb_bal_r[K: Cmp](node: T[K]) -> T[K]:
    match node:
        case T(C.B, t1, y, T(C.R, t2, x, t3)):
            return T(C.R, t1, y, T(C.B, t2, x, t3))
        case T(C.B, T(C.B, t1, z, t2), y, t3):
            return _rb_bal(T(C.B, T(C.R, t1, z, t2), y, t3))
        case T(C.B, T(C.R, t1, z, T(C.B, t2, u, t3)), y, t4):
            return T(C.R, _rb_bal(T(C.B, _black(t1), z, t2)), u, T(C.B, t3, y, t4))
        case _:
            raise AssertionError("unexpected rb-tree structure")


def _rb_fuse[K: Cmp](left: _RbTree[K], right: _RbTree[K]) -> _RbTree[K]:
    match (left, right):
        case (_Nil(), t):
            return t
        case (t, _Nil()):
            return t
        case (T(C.B, _, _, _) as t1, T(C.R, t3, y, t4)):
            return T(C.R, _rb_fuse(t1, t3), y, t4)
        case (T(C.R, t1, x, t2), T(C.B, _, _, _) as t3):
            return T(C.R, t1, x, _rb_fuse(t2, t3))
        case (T(C.R, t1, x, t2), T(C.R, t3, y, t4)):
            mid = _rb_fuse(t2, t3)
            match mid:
                case T(C.R, m1, z, m2):
                    return T(C.R, T(C.R, t1, x, m1), z, T(C.R, m2, y, t4))
                case _:  # cover black node or Nil
                    return T(C.R, t1, x, T(C.R, mid, y, t4))
        case (T(C.B, t1, x, t2), T(C.B, t3, y, t4)):
            mid = _rb_fuse(t2, t3)
            match mid:
                case T(C.R, m1, z, m2):
                    return T(C.R, T(C.B, t1, x, m1), z, T(C.B, m2, y, t4))
                case T(C.B, _, _, _):
                    return _rb_bal_l(T(C.B, t1, x, T(C.B, mid, y, t4)))
                case _:  # cover black node or Nil
                    return _rb_bal_l(T(C.B, t1, x, T(C.B, mid, y, t4)))
        case _:
            raise AssertionError("unexpected rb-tree structure")


class RbTree[K: Cmp]:
    """
    A generic RB Tree which support Container, Sized and Iterable ...
    """

    def __init__(self, items: Iterable[K] = ()) -> None:
        self._root: _RbTree[K] = _Nil()
        self._size: int = 0

        for item in items:
            self.add(item)

    def __iter__(self) -> Iterator[K]:
        yield from (k for k in _rb_traverse(self._root))

    def __reversed__(self) -> Iterator[K]:
        yield from (k for k in _rb_traverse(self._root, reverse=True))

    def __contains__(self, item: K) -> bool:
        node = _rb_member(self._root, item)
        return False if isinstance(node, _Nil) else True

    def __len__(self) -> int:
        return self._size

    def add(self, item: K) -> None:
        """Insert a element into the tree"""
        self._root = _rb_ins_root(self._root, item)
        self._size += 1

    def remove(self, item: K) -> None:
        """Remove a element from the tree, throws if no such item."""
        if item not in self:
            raise ValueError("item not in container")
        self._root = _rb_del(self._root, item)
        self._size -= 1


class _None:
    """Customized None type to avoid type collision"""


class _ByKey[K: Cmp, V](NamedTuple):
    """Key value type that support comparison by key only"""

    key: K
    value: V | _None

    def __lt__(self, other) -> bool:
        if not isinstance(other, _ByKey):
            return NotImplemented
        return self.key < other.key

    @classmethod
    def only(cls, key: K) -> _ByKey:
        """Construct a pair that only contains key, useful for comparing."""
        return cls(key, _None())

    def val(self) -> V:
        """Convenient value getter to ensure existence"""
        if isinstance(self.value, _None):
            raise ValueError("expect value but got None")
        return self.value


class TreeMap[K: Cmp, V](MutableMapping[K, V]):
    """
    TreeMap container, supporting MutableMapping.

    key must support "<" (__lt__), and it assumed that
    if not (a < b) and not (b < a), then a == b.
    """

    def __init__(self, items: Iterable[tuple[K, V]] = ()) -> None:
        self._root: _RbTree[_ByKey[K, V]] = _Nil()
        self._size: int = 0

        for k, v in items:
            self[k] = v

    def __setitem__(self, key: K, value: V) -> None:
        node = _rb_member(self._root, _ByKey.only(key))
        if not isinstance(node, _Nil):
            self._del(key)

        self._root = _rb_ins_root(self._root, _ByKey(key, value))
        self._size += 1

    def __getitem__(self, key: K) -> V:
        node = _rb_member(self._root, _ByKey.only(key))
        if isinstance(node, _Nil):
            raise KeyError("no such key in the map")

        return node.item.val()

    def __delitem__(self, key: K) -> None:
        node = _rb_member(self._root, _ByKey.only(key))
        if isinstance(node, _Nil):
            raise KeyError("no such key in the map")

        self._del(key)

    def _del(self, key: K) -> None:
        self._root = _rb_del(self._root, _ByKey.only(key))
        self._size -= 1

    def __contains__(self, key: K) -> bool:  # type: ignore[override]
        node = _rb_member(self._root, _ByKey.only(key))
        return not isinstance(node, _Nil)

    def __iter__(self) -> Iterator[K]:
        yield from (p.key for p in _rb_traverse(self._root))

    def __reversed__(self) -> Iterator[K]:
        yield from (p.key for p in _rb_traverse(self._root, reverse=True))

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"TreeMap({dict((k, self[k]) for k in self)})"


class MultiSet[K: Cmp](MutableSet[K]):
    """
    MultiSet(OrderedList), support Container and insert, delete.

    item must support "<" (__lt__), and it assumed that
    if not (a < b) and not (b < a), then a == b.

    Use next(container) to get the smallest item, if any.
    """

    def __init__(self, items: Iterable[K] = ()) -> None:
        self._store: TreeMap[K, int] = TreeMap()
        self._size: int = 0

        for item in items:
            self.add(item)

    def __contains__(self, key: K) -> bool:
        return key in self._store

    def __iter__(self) -> Iterator[K]:
        yield from (k for k in self._store for _ in range(self._store[k]))

    def __reversed__(self) -> Iterator[K]:
        yield from (k for k in reversed(self._store) for _ in range(self._store[k]))

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"MultiSet({list(self)})"

    def add(self, item: K) -> None:
        self._size += 1
        self._store[item] = self._store.get(item, 0) + 1

    def discard(self, item: K) -> None:
        if item not in self._store:
            return
        self._size -= 1
        self._store[item] -= 1
        if self._store[item] == 0:
            del self._store[item]
