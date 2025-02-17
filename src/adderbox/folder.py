"""
folder is a module for some containers.

membership-check and iteration(folding) is supported whenever possible.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import (
    Callable,
    Collection,
    Container,
    Iterable,
    Iterator,
    Reversible,
    Sequence,
)
import random
from typing import Any
from typing import assert_never
from typing import cast
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


def _eq[T: Cmp](a: T, b: T) -> bool:
    """
    Check if two items are equal by less than operator.

    Useful since that all Python object has operator== implemented by default,
    but it's defined by object identity, not object value.

    It's only semantically correct if the relation is a total order.
    """
    return not a < b and not b < a


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


class RbTree[K: Cmp](Container[K], Reversible[K]):
    """
    A generic RB Tree which support Container, Sized and Iterable ...

    It's a sorted container with O(log n) access time.
    Useful for build a sorted container / mapping that need iteration from both ends.

    Relative order of repetitive elements is not guaranteed.
    Repetitive(equality) is defined by less than operator.
    """

    def __init__(self, items: Iterable[K] = ()) -> None:
        self._root: _RbTree[K] = _Nil()

        for item in items:
            self.add(item)

    def __iter__(self) -> Iterator[K]:
        return self.iter()

    def __reversed__(self) -> Iterator[K]:
        return self.iter(reverse=True)

    def __contains__(self, item: K) -> bool:
        return self.find(item) is not None

    def add(self, item: K) -> None:
        """Insert a element into the tree"""
        self._root = _rb_ins_root(self._root, item)

    def remove(self, item: K) -> None:
        """Remove a element from the tree, throws if no such item."""
        if item not in self:
            raise ValueError("item not in container")
        self._root = _rb_del(self._root, item)

    def find(self, item: K) -> K | None:
        """Return an item if there is at least one equal to given item."""
        node = _rb_member(self._root, item)
        return None if isinstance(node, _Nil) else node.item

    def iter(self, reverse: bool = False) -> Iterator[K]:
        """Return a iterator to traverse the tree."""
        return _rb_traverse(self._root, reverse=reverse)


class ByKey[K: Cmp, V](NamedTuple):
    """
    Key value type that support comparison by key only

    Useful for building a mapping upon a primitive container.

    It's not possible to use None as value type here.
    """

    # TODO: support type hint to ensure value type is not None

    key: K
    value: V | None

    def __lt__(self, other) -> bool:
        if not isinstance(other, ByKey):
            return NotImplemented
        return self.key < other.key

    @classmethod
    def only(cls, key: K) -> ByKey[K, V]:
        """Construct a pair that only contains key, useful for comparing."""
        return cls(key, cast(V, None))

    def val(self) -> V:
        """Convenient value getter to ensure existence"""
        if self.value is None:
            raise ValueError("expect value but got None")
        return self.value


class _Skip[T: Any]:
    """Node type in the skip list."""

    def __init__(
        self,
        val: T | None = None,
        *,
        down: _Skip[T] | None = None,
        size: int = 1,
    ) -> None:
        self.val: T | None = val
        """value of the node, not used in sentinel nodes"""

        self.do: _Skip[T] | None = down
        """down pointer, None for bottom nodes"""

        self.ri: _Skip[T] | None = None
        """right pointer, None for last node at each level"""

        self.size: int = size
        """size of the sub-tree rooted at this node"""

    def __repr__(self) -> str:
        val_repr = repr(self.val) if self.val is not None else "-inf"
        upper = repr(self.ri.val) if self.ri is not None else "+inf"
        return f"Node[key={val_repr}, right={upper}, size={self.size}]"


class SkipList[T: Any, C: Cmp](Collection[T]):
    """
    A generic Skip List container.

    It's a sorted container with O(log n) access time.
    Useful for build a sorted container / mapping that also need efficient n-th access.

    Relative order of repetitive elements is not guaranteed.
    Repetitive(equality) is defined by less than operator of projected value.

    Skip list is a probabilistic data structure by William Pugh.
    It allows average O(log n) read-write and use average O(n) space.
    Roughly equivalent to balanced BST, it ensures ordered iteration.
    """

    # Implementation notes:
    # For each level, we use a (singly) linked list to store the nodes.
    # Each node has a down-link to the node at the next level.
    # Singly linked list make insert/delete a little bit harder, but it reduce
    # the need for graph traversal during cleanup, either by __del__ or by GC.
    #
    # Sample structure for a height-3 skip list
    # L2: sentinel     > 3           (> None)
    # L1: sentinel     > 3       > 9 (> None)
    # L0: sentinel > 1 > 3  > 7  > 9 (> None)
    #
    # Sample structure for a empty (height-1) skip list
    # L0: sentinel (> None)
    #
    # It's easier to think the sentinel (with val None) as negative infinity,
    # and the None as positive infinity.
    #
    # Invariants:
    # - The head node is always the top left (sentinel) node
    # - Bottom level (L0) always exists, even when container is empty
    # - node.do (down-link) is None for bottom nodes, is not None otherwise
    # - node.val is None for sentinel nodes, is not None otherwise

    def __init__(
        self,
        items: Iterable[T] = [],
        *,
        key: Callable[[T], C] = lambda x: x,
    ) -> None:
        self._head: _Skip[T] = _Skip[T]()
        """head node at top left"""

        self._proj: Callable[[T], C] = key
        """projection function to compare values"""

        for item in items:
            self.add(item)

    @staticmethod
    def _random_level() -> int:
        __factor = 4  # roughly 1/4 chance to stop at each

        level = 0
        while random.choice(range(__factor)) == 0:
            level += 1
        else:
            return level

    def _size_in(self, down: _Skip[T], bound: T | None) -> int:
        """Return the size of some tree, by down link and key bound."""

        key = self._proj
        # -inf less than all values, and all values less than +inf
        is_good = lambda item: item is None or bound is None or key(item) < key(bound)

        size = 0
        cur: _Skip[T] | None = down
        while cur is not None and is_good(cur.val):
            size += cur.size
            cur = cur.ri

        return size

    def _height(self) -> int:
        """Return the height of the skip list."""

        height = 1
        row = self._head
        while row.do is not None:
            height += 1
            row = row.do

        return height

    def _fill_head(self, height: int) -> None:
        """Ensure there are at least height levels in the skip list."""

        assert height > 0

        row = self._head
        for _ in range(self._height(), height):
            node = _Skip[T](down=row, size=self._size_in(row, None))
            row = node
        self._head = row

    def _clean_head(self) -> None:
        """Remove empty levels from the top."""

        row = self._head
        # we always keep the bottom level, so check by down link here
        while row.do is not None and row.ri is None:
            row = row.do
        self._head = row

    def _traverse(self, item: T) -> tuple[
        tuple[_Skip[T], _Skip[T] | None],
        dict[int, tuple[_Skip[T], _Skip[T] | None]],
    ]:
        """
        Return target-pair at bottom and at each level.

        The target-pair consists of the last node with key < target key (if the "key" exists),
        and the first node with key >= target key (if the "node" exists).
        """

        by = self._proj
        tracked: dict[int, tuple[_Skip[T], _Skip[T] | None]] = {}
        level = self._height() - 1
        cur: _Skip[T] | None = self._head

        assert level >= 0
        assert cur is not None

        # Reach the node that have key < target key as much as possible at each level.
        # The sentinel nodes can be thought as node with negative infinity key.
        while cur is not None:
            nex: _Skip[T] | None = cur.ri
            while nex is not None and by(cast(T, nex.val)) < by(item):
                cur, nex = nex, nex.ri
            tracked[level] = (cur, cur.ri)

            cur, level = cur.do, level - 1

        assert level == -1
        assert cur is None

        return tracked[0], tracked

    def size(self) -> int:
        """Return the number of elements in the container."""

        # sentinel node is not counted
        return self._size_in(self._head, None) - 1

    def __len__(self) -> int:
        return self.size()

    def iter(self) -> Iterator[T]:
        """Return a iterator to traverse the container."""

        row = self._head
        while row.do is not None:
            row = row.do

        cur = row.ri  # now at bottom level
        while cur is not None:
            yield cast(T, cur.val)
            cur = cur.ri

    def __iter__(self) -> Iterator[T]:
        return self.iter()

    def add(self, item) -> None:
        """Insert an item into the container."""

        half = self._random_level() + 1
        self._fill_head(half)

        _, tracked = self._traverse(item)

        inserted: dict[int, _Skip[T]] = {}
        for idx in range(half):
            pre, nex = tracked[idx]

            down = inserted[idx - 1] if idx > 0 else None
            hi = nex.val if nex is not None else None
            size = self._size_in(down, hi) if down is not None else 1

            node = _Skip[T](item, down=down, size=size)
            pre.ri, node.ri = node, nex

            inserted[idx] = node

        for idx, (pre, _) in tracked.items():
            splitted = inserted[idx].size if idx in inserted else 0
            pre.size = pre.size + 1 - splitted

    def find(self, item: T) -> T | None:
        """Return an item from container if there is at least one."""
        (_, cur), _ = self._traverse(item)
        by = self._proj

        if cur is None or not _eq(by(cast(T, cur.val)), by(item)):
            return None

        # do not check against None so we support using None as value type V.
        return cast(T, cur.val)

    def __contains__(self, item: T) -> bool:
        return self.find(item) is not None

    def remove(self, item: T) -> None:
        """Remove an item, throws if no such item."""

        (_, cur), tracked = self._traverse(item)
        by = self._proj

        if cur is None or not _eq(by(cast(T, cur.val)), by(item)):
            raise ValueError("item not in container")

        joined: dict[int, int] = {}
        for idx, (pre, cur) in tracked.items():
            if cur is None or not _eq(by(cast(T, cur.val)), by(item)):
                joined[idx] = 0
                continue
            pre.ri = cur.ri
            joined[idx] = cur.size

        for idx, (pre, _) in tracked.items():
            pre.size = pre.size - 1 + joined[idx]

        self._clean_head()

    def __repr__(self) -> str:
        texts: list[str] = []
        texts.append(f"size: {self.__len__()}, height: {self._height()}")

        def format(row: _Skip[T]) -> str:
            results: list[str] = []
            cur = row.ri
            while cur is not None:
                results.append(repr(cur.val))
                cur = cur.ri
            return " ".join(results)

        row: _Skip[T] | None = self._head
        level = self._height() - 1
        while row is not None:
            texts.append(f"level {level}: {format(row)}")
            row, level = row.do, level - 1

        return "\n".join(texts)

    def at(self, idx: int) -> T:
        """Return the item at the given index."""

        def get(cur: _Skip[T] | None, idx: int) -> T:
            if cur is None:
                raise IndexError("index out of range")
            if idx < 0:
                raise IndexError("index out of range")
            if idx >= cur.size:
                return get(cur.ri, idx - cur.size)
            if idx == 0:
                return cast(T, cur.val)

            # now idx > 0 and idx < cur.size, so we can safely go down
            return get(cur.do, idx)

        # sentinel node is not counted
        if idx == -1:
            raise IndexError("index out of range")
        return get(self._head, idx + 1)

    def __getitem__(self, idx: int) -> T:
        idx = idx if idx >= 0 else idx + len(self)
        return self.at(idx)

    def index(self, item: T) -> int:
        """Return the first index of item equal to given item."""

        by = self._proj

        def find(cur: _Skip[T] | None) -> int:
            if cur is None:
                raise ValueError("item not in container")
            if cur.val is not None and _eq(by(cur.val), by(item)):
                return 0
            if cur.val is not None and by(item) < by(cur.val):
                raise ValueError("item not in container")
            if cur.ri is not None and not by(item) < by(cast(T, cur.ri.val)):
                return cur.size + find(cur.ri)

            # now item > cur.key and item < cur.ri.key, so we can go down
            return find(cur.do)

        # sentinel node is not counted
        return find(self._head) - 1
