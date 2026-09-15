"""Small dependency-free ASGI surface for production hosts.

The adapter deliberately accepts encoded image bytes rather than filesystem
paths, avoiding path traversal and making the same endpoint usable from
containers, queues, and browser clients. A real ASGI server such as Uvicorn is
kept outside the base dependency set.
"""

import asyncio
import base64
import binascii
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from typing import Any

from .diagnostics import health_check
from .limits import InputLimits
from .serialization import result_to_dict


class _InferenceTimeout(TimeoutError):
    """Internal marker distinguishing service deadlines from backend errors."""


class YaseASGI:
    """Expose health, Prometheus metrics, and one-image extraction over ASGI."""

    def __init__(
        self,
        extractor: Any,
        *,
        max_body_bytes: int = 16 * 1024 * 1024,
        max_batch_size: int = 64,
        input_limits: InputLimits | None = None,
        max_concurrency: int = 4,
        timeout_seconds: float | None = None,
    ):
        if not hasattr(extractor, "extract"):
            raise TypeError("extractor must expose the Yase extract(image) API")
        if (
            isinstance(max_body_bytes, bool)
            or not isinstance(max_body_bytes, int)
            or max_body_bytes < 1
        ):
            raise ValueError("max_body_bytes must be a positive integer")
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, int)
            or max_batch_size < 1
        ):
            raise ValueError("max_batch_size must be a positive integer")
        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency < 1
        ):
            raise ValueError("max_concurrency must be a positive integer")
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive number or None")
        if input_limits is not None and not isinstance(input_limits, InputLimits):
            raise TypeError("input_limits must be an InputLimits instance")
        self.extractor = extractor
        self.max_body_bytes = max_body_bytes
        self.max_batch_size = max_batch_size
        self.input_limits = input_limits
        self.max_concurrency = max_concurrency
        self.timeout_seconds = float(timeout_seconds) if timeout_seconds else None
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="yase-asgi",
        )

    @staticmethod
    def _close_images(images: list[Any]) -> None:
        for image in images:
            close = getattr(image, "close", None)
            if callable(close):
                close()

    async def _run_blocking(self, function: Any, resources: list[Any]) -> Any:
        """Run extraction off-loop while keeping timed-out images alive safely."""
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, function)
        deferred_close = False
        try:
            if self.timeout_seconds is None:
                return await future
            try:
                return await asyncio.wait_for(
                    asyncio.shield(future), timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError as exc:
                deferred_close = True
                future.add_done_callback(lambda _future: self._close_images(resources))
                raise _InferenceTimeout(
                    f"inference exceeded timeout_seconds={self.timeout_seconds}"
                ) from exc
        except asyncio.CancelledError:
            deferred_close = True
            future.add_done_callback(lambda _future: self._close_images(resources))
            raise
        finally:
            if not deferred_close:
                self._close_images(resources)

    def close(self) -> None:
        """Stop the bounded worker pool during application shutdown."""
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _decode_image(self, encoded: Any) -> Any:
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("image_base64 must be a non-empty string")
        image_bytes = base64.b64decode(encoded, validate=True)
        if not image_bytes:
            raise ValueError("image_base64 decoded to empty bytes")
        from PIL import Image

        try:
            probe_stream = BytesIO(image_bytes)
            with Image.open(probe_stream) as probe:
                if self.input_limits is not None:
                    width, height = probe.size
                    self.input_limits.validate_shape(
                        width=width,
                        height=height,
                        channels=3,
                        byte_count=width * height * 3,
                    )
                probe.verify()
            return Image.open(BytesIO(image_bytes))
        except Image.DecompressionBombError as exc:
            raise ValueError(
                "image exceeds Pillow decompression safety limits"
            ) from exc
        except (Image.UnidentifiedImageError, OSError) as exc:
            raise ValueError("invalid image payload") from exc

    @staticmethod
    async def _read_body(receive: Any, limit: int) -> bytes:
        chunks = []
        total = 0
        while True:
            event = await receive()
            if event.get("type") == "http.disconnect":
                raise ConnectionError("client disconnected while sending request")
            if event.get("type") != "http.request":
                continue
            chunk = event.get("body", b"")
            total += len(chunk)
            if total > limit:
                raise OverflowError("request body exceeds max_body_bytes")
            chunks.append(chunk)
            if not event.get("more_body", False):
                return b"".join(chunks)

    @staticmethod
    async def _send(send: Any, status: int, payload: Any, content_type: str) -> None:
        body = (
            payload.encode("utf-8")
            if isinstance(payload, str)
            else json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", content_type.encode("ascii")),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            return
        method = scope.get("method", "GET").upper()
        path = scope.get("path", "/")
        if method == "GET" and path in ("/health", "/ready"):
            report = health_check(
                self.extractor if path == "/ready" else None,
            )
            await self._send(
                send,
                200 if report.ready else 503,
                report.to_dict(),
                "application/json",
            )
            return
        if method == "GET" and path == "/metrics":
            metrics = getattr(self.extractor, "metrics", None)
            text = metrics.prometheus_text() if metrics is not None else ""
            await self._send(send, 200, text, "text/plain; version=0.0.4")
            return
        if method != "POST" or path not in ("/extract", "/extract/batch"):
            await self._send(
                send,
                404,
                {"error": {"type": "not_found", "message": "route not found"}},
                "application/json",
            )
            return

        try:
            body = await self._read_body(receive, self.max_body_bytes)
        except OverflowError as exc:
            await self._send(
                send,
                413,
                {"error": {"type": "request_too_large", "message": str(exc)}},
                "application/json",
            )
            return
        except ConnectionError:
            return
        try:
            request = json.loads(body.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("request body must be a JSON object")
            if path == "/extract/batch":
                encoded_images = request.get("images_base64")
                if not isinstance(encoded_images, list) or not encoded_images:
                    raise ValueError("images_base64 must be a non-empty list")
                if len(encoded_images) > self.max_batch_size:
                    raise ValueError(
                        f"batch exceeds max_batch_size: {len(encoded_images)} > "
                        f"{self.max_batch_size}"
                    )
                timestamps = request.get("timestamps")
                if timestamps is not None:
                    if not isinstance(timestamps, list) or len(timestamps) != len(
                        encoded_images
                    ):
                        raise ValueError(
                            "timestamps must be a list with one value per image"
                        )
                error_policy = request.get("error_policy", "raise")
                if error_policy not in ("raise", "skip"):
                    raise ValueError("error_policy must be raise or skip")
                include_arrays = request.get("include_arrays", False)
                if not isinstance(include_arrays, bool):
                    raise ValueError("include_arrays must be boolean")
                images = []
                try:
                    images.extend(self._decode_image(item) for item in encoded_images)
                    extract_many = getattr(self.extractor, "extract_many", None)
                    if not callable(extract_many):
                        raise TypeError(
                            "batch extraction requires a Yase facade with "
                            "extract_many()"
                        )
                    results = await self._run_blocking(
                        partial(
                            extract_many,
                            images,
                            timestamps=timestamps,
                            error_policy=error_policy,
                        ),
                        images,
                    )
                except (_InferenceTimeout, asyncio.CancelledError):
                    raise
                except Exception:
                    self._close_images(images)
                    raise
                await self._send(
                    send,
                    200,
                    {
                        "results": [
                            None
                            if result is None
                            else result_to_dict(result, include_arrays=include_arrays)
                            for result in results
                        ]
                    },
                    "application/json",
                )
                return
            encoded = request.get("image_base64")
            timestamp = request.get("timestamp")
            include_arrays = request.get("include_arrays", False)
            if not isinstance(include_arrays, bool):
                raise ValueError("include_arrays must be boolean")
            image = self._decode_image(encoded)
            result = await self._run_blocking(
                partial(self.extractor.extract, image, timestamp=timestamp),
                [image],
            )
            await self._send(
                send,
                200,
                {"result": result_to_dict(result, include_arrays=include_arrays)},
                "application/json",
            )
        except _InferenceTimeout as exc:
            await self._send(
                send,
                504,
                {"error": {"type": "inference_timeout", "message": str(exc)}},
                "application/json",
            )
        except KeyError as exc:
            await self._send(
                send,
                400,
                {"error": {"type": "invalid_request", "message": str(exc)}},
                "application/json",
            )
        except (
            ValueError,
            TypeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            binascii.Error,
            OSError,
        ) as exc:
            await self._send(
                send,
                400,
                {"error": {"type": "invalid_request", "message": str(exc)}},
                "application/json",
            )
        except Exception as exc:
            await self._send(
                send,
                500,
                {"error": {"type": "inference_error", "message": str(exc)}},
                "application/json",
            )


def create_asgi_app(extractor: Any, **options: Any) -> YaseASGI:
    """Build a reusable ASGI application around a Yase facade or backend."""
    return YaseASGI(extractor, **options)


__all__ = ["YaseASGI", "create_asgi_app"]
