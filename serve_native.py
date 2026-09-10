"""Run the opt-in native fused Neural Engine serving entry point.

Example:
    python serve_native.py --checkpoint results/checkpoints/model.pt --port 8080
"""

from __future__ import annotations

import argparse
import json

from neural_engine.native_fused_server import (
    NativeFusedHTTPServer,
    load_native_fused_service,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-shapes", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument(
        "--batch-window-ms",
        type=float,
        default=0.0,
        help="opt-in shape-homogeneous admission window; 0 disables batching",
    )
    parser.add_argument("--no-graphs", action="store_true")
    args = parser.parse_args()

    service = load_native_fused_service(
        args.checkpoint,
        device=args.device,
        max_shapes=args.max_shapes,
        warmup_iters=args.warmup_iters,
        capture_graphs=not args.no_graphs,
        max_batch_size=args.max_batch_size,
        batch_window_ms=args.batch_window_ms,
    )
    server = NativeFusedHTTPServer((args.host, args.port), service)
    print(json.dumps({"listening": f"http://{args.host}:{args.port}", **service.health()}))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
