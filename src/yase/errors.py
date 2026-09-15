"""Public exception types for applications and service boundaries."""


class YaseError(Exception):
    """Base class for errors raised by Yase itself."""


class InputError(YaseError, ValueError):
    """Input cannot be decoded or does not satisfy the API contract."""


class BackendError(YaseError, RuntimeError):
    """A named backend failed while processing an input."""

    def __init__(
        self,
        backend: str,
        message: str,
        *,
        original: BaseException | None = None,
        index: int | None = None,
    ) -> None:
        self.backend = backend
        self.index = index
        self.original = original
        prefix = f"backend '{backend}' failed"
        if index is not None:
            prefix += f" at index {index}"
        super().__init__(f"{prefix}: {message}")


class PipelineError(YaseError, RuntimeError):
    """A pipeline could not merge or execute its configured stages."""


class SchedulerError(PipelineError):
    """A stage graph could not be planned or respected its execution budget."""


class StageCancelled(PipelineError):
    """Execution was cancelled before the next stage could start."""


__all__ = [
    "BackendError",
    "InputError",
    "PipelineError",
    "SchedulerError",
    "StageCancelled",
    "YaseError",
]
