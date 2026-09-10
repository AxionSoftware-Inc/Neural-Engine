"""Small process-local serving adapter for the native fused Neural Engine path.

The server deliberately owns one model and one :class:`NativeFusedShapeCache`
per process.  It is a correctness and integration entry point, not a claim
that the HTTP layer is production infrastructure.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch
from torch import nn

from .native_fused_serving import NativeFusedShapeCache


class NativeFusedService:
    """Validate requests and serve logits through one process-local cache."""

    def __init__(self, model: nn.Module, cache: NativeFusedShapeCache | None = None):
        self.model = model.eval()
        self.owner_pid = os.getpid()
        self.cache = cache or NativeFusedShapeCache(self.model)
        if self.cache.model is not self.model:
            raise ValueError("cache must wrap the service model")
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration as exc:
            raise ValueError("service model must have parameters") from exc
    def _check_owner(self) -> None:
        if os.getpid() != self.owner_pid:
            raise RuntimeError(
                "NativeFusedService is process-local; construct a model and cache per worker"
            )

    def _validate_rows(self, rows: Any) -> tuple[int, int]:
        if not isinstance(rows, list) or not rows:
            raise ValueError("inputs must be a non-empty list of token rows")
        if any(not isinstance(row, list) or not row for row in rows):
            raise ValueError("inputs must contain non-empty token rows")
        sequence_length = len(rows[0])
        if any(len(row) != sequence_length for row in rows):
            raise ValueError("all token rows must have the same sequence length")
        if any(
            isinstance(token, bool) or not isinstance(token, int)
            for row in rows for token in row
        ):
            raise ValueError("token ids must be integers")

        sequence_limit = getattr(self.model, "input_seq_len", None)
        if sequence_limit is None:
            position_embedding = getattr(self.model, "position_embedding", None)
            sequence_limit = None if position_embedding is None else position_embedding.shape[0]
        if sequence_limit is not None and sequence_length > sequence_limit:
            raise ValueError(
                f"sequence length {sequence_length} exceeds model limit "
                f"{sequence_limit}"
            )
        token_embedding = getattr(self.model, "token_embedding", None)
        vocab_size = getattr(self.model, "input_vocab_size", None)
        if vocab_size is None:
            vocab_size = getattr(token_embedding, "num_embeddings", None)
        if vocab_size is not None:
            token_min = min(token for row in rows for token in row)
            token_max = max(token for row in rows for token in row)
            if token_min < 0 or token_max >= vocab_size:
                raise ValueError(
                    f"token ids must be in [0, {vocab_size - 1}], got "
                    f"[{token_min}, {token_max}]"
                )
        return len(rows), sequence_length

    def infer(self, rows: Any, return_logits: bool = False) -> dict[str, Any]:
        self._check_owner()
        batch_size, sequence_length = self._validate_rows(rows)
        inputs = torch.tensor(rows, dtype=torch.long, device=self.device)
        # The cache serializes replay only when callers share a mutable
        # same-stream graph entry; no service-wide lock is needed.
        logits = self.cache(inputs)
        result: dict[str, Any] = {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "logit_shape": list(logits.shape),
            "predictions": logits.argmax(dim=-1).detach().cpu().tolist(),
            "cache": self.cache.stats(),
        }
        if return_logits:
            result["logits"] = logits.detach().cpu().tolist()
        return result

    def health(self) -> dict[str, Any]:
        self._check_owner()
        return {
            "status": "ok",
            "pid": self.owner_pid,
            "device": str(self.device),
            "training": bool(self.model.training),
            "dispatch_backend": getattr(self.model, "circuit_dispatch_backend", "unknown"),
            "cache": self.cache.stats(),
        }


class NativeFusedHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server carrying one process-local service instance."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], service: NativeFusedService):
        super().__init__(server_address, NativeFusedHTTPHandler)
        self.service = service


class NativeFusedHTTPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "NeuralEngineNativeFused/0.1"
    max_body_bytes = 64 * 1024 * 1024

    @property
    def service(self) -> NativeFusedService:
        return self.server.service  # type: ignore[attr-defined]

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length header is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer") from exc
        if length < 0 or length > self.max_body_bytes:
            raise ValueError(f"request body must be between 0 and {self.max_body_bytes} bytes")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlsplit(self.path).path
        try:
            if path == "/health":
                self._send_json(200, self.service.health())
            elif path == "/stats":
                self._send_json(200, {"cache": self.service.cache.stats()})
            else:
                self._send_json(404, {"error": "not found"})
        except RuntimeError as exc:
            self._send_json(503, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlsplit(self.path).path
        if path != "/infer":
            self._send_json(404, {"error": "not found"})
            return
        try:
            payload = self._read_json()
            result = self.service.infer(
                payload.get("inputs"),
                return_logits=bool(payload.get("return_logits", False)),
            )
            self._send_json(200, result)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
        except RuntimeError as exc:
            self._send_json(503, {"error": str(exc)})
        except Exception as exc:  # keep the process alive for a bad request/model error
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format: str, *args: object) -> None:
        # Keep benchmark and service stdout machine-readable unless callers
        # explicitly add their own access logging around the server.
        return


def load_native_fused_service(
    checkpoint: str | Path,
    device: str | torch.device | None = None,
    max_shapes: int = 8,
    warmup_iters: int = 5,
    capture_graphs: bool = True,
) -> NativeFusedService:
    """Load a checkpoint and create its native fused service in this process."""

    from train import make_model, seed_everything

    checkpoint_path = Path(checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "config" not in payload or "model_state" not in payload:
        raise ValueError("checkpoint must contain config and model_state")
    config = dict(payload["config"])
    config["circuit_dispatch_backend"] = "native_cuda_fused"
    seed_everything(int(config.get("seed", 0)))
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)
    model = make_model(config).to(target_device).eval()
    model.load_state_dict(payload["model_state"])
    # Keep raw request validation separate from the internal embedding table:
    # numeric-value models intentionally use a 16-row token table plus a value
    # encoder while their public token space remains config["vocab_size"].
    model.input_vocab_size = int(config["vocab_size"])  # type: ignore[attr-defined]
    model.input_seq_len = int(config["seq_len"])  # type: ignore[attr-defined]
    cache = NativeFusedShapeCache(
        model,
        max_shapes=max_shapes,
        warmup_iters=warmup_iters,
        capture_graphs=capture_graphs,
    )
    service = NativeFusedService(model, cache)
    service.checkpoint = str(checkpoint_path)  # type: ignore[attr-defined]
    return service


__all__ = [
    "NativeFusedHTTPHandler",
    "NativeFusedHTTPServer",
    "NativeFusedService",
    "load_native_fused_service",
]
