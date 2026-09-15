"""Explicit, bounded caches for heavyweight runtime resources."""

from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping, Sequence
from threading import RLock
from typing import Any, Generic, TypeVar

T = TypeVar("T")


def freeze_cache_key(value: Any) -> Hashable:
    """Convert common nested option values into a deterministic hashable key."""
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (
                    (freeze_cache_key(key), freeze_cache_key(item))
                    for key, item in value.items()
                ),
                key=repr,
            )
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(freeze_cache_key(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


class RuntimeCache(Generic[T]):
    """A small LRU cache that owns and closes evicted resources.

    The cache never downloads or constructs anything by itself. Callers pass
    an explicit factory to :meth:`get_or_create`; this makes model/session
    reuse observable and keeps artifact lifecycle under application control.
    Values exposing ``close()`` are closed on eviction and on ``clear()``.
    """

    def __init__(self, capacity: int = 8) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self.capacity = capacity
        self._values: OrderedDict[Hashable, T] = OrderedDict()
        self._lock = RLock()

    def get(self, key: Any, default: T | None = None) -> T | None:
        """Return a cached value and mark it as recently used."""
        normalized = freeze_cache_key(key)
        with self._lock:
            if normalized not in self._values:
                return default
            value = self._values.pop(normalized)
            self._values[normalized] = value
            return value

    def get_or_create(self, key: Any, factory: Callable[[], T]) -> T:
        """Return a cached resource or create it exactly once under the lock."""
        if not callable(factory):
            raise TypeError("factory must be callable")
        normalized = freeze_cache_key(key)
        with self._lock:
            if normalized in self._values:
                value = self._values.pop(normalized)
                self._values[normalized] = value
                return value
            value = factory()
            if self.capacity == 0:
                return value
            self._values[normalized] = value
            while len(self._values) > self.capacity:
                _, evicted = self._values.popitem(last=False)
                self._close(evicted)
            return value

    def clear(self) -> None:
        """Close all owned resources and empty the cache."""
        with self._lock:
            values = tuple(self._values.values())
            self._values.clear()
        for value in values:
            self._close(value)

    def info(self) -> dict[str, int]:
        """Return bounded cache occupancy without exposing mutable internals."""
        with self._lock:
            return {"size": len(self._values), "capacity": self.capacity}

    def close(self) -> None:
        """Alias for :meth:`clear` for use in application shutdown hooks."""
        self.clear()

    def __enter__(self) -> "RuntimeCache[T]":
        return self

    def __exit__(self, *_: Any) -> None:
        self.clear()

    @staticmethod
    def _close(value: Any) -> None:
        close = getattr(value, "close", None)
        if callable(close):
            close()


__all__ = ["RuntimeCache", "freeze_cache_key"]
