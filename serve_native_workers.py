"""Launch independent process-local native fused HTTP workers.

Each worker loads the checkpoint itself after Windows ``spawn`` and owns its
own model and CUDA-Graph shape cache. Workers listen on consecutive ports;
deployment infrastructure can place a load balancer in front of them.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import threading
import time
from typing import Any

import torch

from neural_engine.native_fused_server import (
    NativeFusedHTTPServer,
    load_native_fused_service,
)


def _worker_main(
    worker_index: int,
    checkpoint: str,
    host: str,
    port: int,
    device: str | None,
    max_shapes: int,
    warmup_iters: int,
    capture_graphs: bool,
    process_graph_lock: bool,
) -> None:
    """Worker target; model/cache creation intentionally happens in the child."""

    service = load_native_fused_service(
        checkpoint,
        device=device,
        max_shapes=max_shapes,
        warmup_iters=warmup_iters,
        capture_graphs=capture_graphs,
        process_graph_lock=process_graph_lock,
    )
    server = NativeFusedHTTPServer((host, port), service)
    startup: dict[str, Any] = {
        "worker_index": worker_index,
        "pid": service.owner_pid,
        "listening": f"http://{host}:{port}",
        **service.health(),
    }
    print(json.dumps(startup), flush=True)
    threading.Thread(target=_watch_parent, args=(server,), daemon=True).start()
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _watch_parent(server: NativeFusedHTTPServer) -> None:
    """Stop a worker if its multiprocessing launcher disappears unexpectedly."""

    parent = mp.parent_process()
    if parent is None:
        return
    while True:
        if not parent.is_alive():
            server.shutdown()
            return
        time.sleep(0.25)


def _stop_workers(processes: list[mp.Process]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def _prebuild_native_extension(device: str | None, skip: bool) -> bool:
    """Prebuild from the parent so spawned CUDA workers cannot race the linker."""

    if skip or (device is not None and torch.device(device).type == "cpu"):
        return False
    if not torch.cuda.is_available():
        return False
    from neural_engine.native_fused_dispatch import ensure_native_fused_extension

    ensure_native_fused_extension()
    return True


def run(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.port < 1 or args.port + args.workers - 1 > 65535:
        raise ValueError("port range must fit in [1, 65535]")

    prebuilt = _prebuild_native_extension(args.device, args.skip_native_prebuild)
    shared_device_graph_lock = bool(
        args.workers > 1 and not args.no_graphs and torch.cuda.is_available()
        and (args.device is None or torch.device(args.device).type == "cuda")
    )
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    for worker_index in range(args.workers):
        process = context.Process(
            target=_worker_main,
            args=(
                worker_index,
                args.checkpoint,
                args.host,
                args.port + worker_index,
                args.device,
                args.max_shapes,
                args.warmup_iters,
                not args.no_graphs,
                shared_device_graph_lock,
            ),
            name=f"neural-engine-native-worker-{worker_index}",
        )
        process.start()
        processes.append(process)

    print(json.dumps({
        "workers": args.workers,
        "ports": [args.port + index for index in range(args.workers)],
        "start_method": "spawn",
        "process_local_cache": True,
        "native_extension_prebuilt": prebuilt,
        "shared_device_graph_lock": shared_device_graph_lock,
    }), flush=True)
    try:
        while True:
            failed = [
                process for process in processes
                if process.exitcode is not None and process.exitcode != 0
            ]
            if failed:
                details = ", ".join(
                    f"{process.name}: exitcode={process.exitcode}" for process in failed)
                raise RuntimeError(f"native worker failed: {details}")
            if all(not process.is_alive() for process in processes):
                return
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        _stop_workers(processes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-shapes", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--skip-native-prebuild", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
