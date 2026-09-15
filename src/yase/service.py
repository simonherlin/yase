"""Small dependency-free ASGI surface for production hosts.

The adapter deliberately accepts encoded image bytes rather than filesystem
paths, avoiding path traversal and making the same endpoint usable from
containers, queues, and browser clients. A real ASGI server such as Uvicorn is
kept outside the base dependency set.
"""

import base64
import binascii
import json
from io import BytesIO
from typing import Any

from .diagnostics import health_check
from .serialization import result_to_dict


class YaseASGI:
    """Expose health, Prometheus metrics, and one-image extraction over ASGI."""

    def __init__(self, extractor: Any, *, max_body_bytes: int = 16 * 1024 * 1024):
        if not hasattr(extractor, "extract"):
            raise TypeError("extractor must expose the Yase extract(image) API")
        if (
            isinstance(max_body_bytes, bool)
            or not isinstance(max_body_bytes, int)
            or max_body_bytes < 1
        ):
            raise ValueError("max_body_bytes must be a positive integer")
        self.extractor = extractor
        self.max_body_bytes = max_body_bytes

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
            await self._send(send, 200, health_check().to_dict(), "application/json")
            return
        if method == "GET" and path == "/metrics":
            metrics = getattr(self.extractor, "metrics", None)
            text = metrics.prometheus_text() if metrics is not None else ""
            await self._send(send, 200, text, "text/plain; version=0.0.4")
            return
        if method != "POST" or path != "/extract":
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
            encoded = request.get("image_base64")
            if not isinstance(encoded, str) or not encoded:
                raise ValueError("image_base64 must be a non-empty string")
            image_bytes = base64.b64decode(encoded, validate=True)
            if not image_bytes:
                raise ValueError("image_base64 decoded to empty bytes")
            timestamp = request.get("timestamp")
            include_arrays = request.get("include_arrays", False)
            if not isinstance(include_arrays, bool):
                raise ValueError("include_arrays must be boolean")
            from PIL import Image

            with Image.open(BytesIO(image_bytes)) as image:
                result = self.extractor.extract(image, timestamp=timestamp)
            await self._send(
                send,
                200,
                {"result": result_to_dict(result, include_arrays=include_arrays)},
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
