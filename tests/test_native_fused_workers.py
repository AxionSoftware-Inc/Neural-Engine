import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import torch

from neural_engine.model import NeuralEngineV0


def _free_consecutive_ports(count: int) -> int:
    for _ in range(20):
        sockets = [socket.socket() for _ in range(count)]
        try:
            for sock in sockets:
                sock.bind(("127.0.0.1", 0))
            ports = [sock.getsockname()[1] for sock in sockets]
            if ports == list(range(ports[0], ports[0] + count)):
                return ports[0]
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("could not reserve consecutive test ports")


def _get_json(url: str) -> dict:
    with urlopen(url, timeout=2) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=2) as response:
        return json.loads(response.read().decode("utf-8"))


def test_spawned_workers_have_independent_process_local_caches(tmp_path):
    config = {
        "model": "ne_worker_test",
        "vocab_size": 32,
        "num_classes": 16,
        "seq_len": 8,
        "d_model": 16,
        "state_dim": 16,
        "num_circuits": 8,
        "circuit_rank": 2,
        "router_branch": 2,
        "router_depth": 1,
        "candidate_pool": 4,
        "active_circuits": 2,
        "internal_steps": 1,
        "circuit_bank_mode": "factorized",
        "factor_count": 4,
        "ordered_factor_slots": True,
        "factor_address_layout": "stable_prefix",
        "legacy_factor_count": 2,
        "seed": 17,
    }
    model = NeuralEngineV0(
        vocab_size=config["vocab_size"], num_classes=config["num_classes"],
        seq_len=config["seq_len"], d_model=config["d_model"],
        state_dim=config["state_dim"], num_circuits=config["num_circuits"],
        circuit_rank=config["circuit_rank"], router_branch=config["router_branch"],
        router_depth=config["router_depth"], candidate_pool=config["candidate_pool"],
        active_circuits=config["active_circuits"], internal_steps=config["internal_steps"],
        circuit_bank_mode=config["circuit_bank_mode"], factor_count=config["factor_count"],
        ordered_factor_slots=config["ordered_factor_slots"],
        factor_address_layout=config["factor_address_layout"],
        legacy_factor_count=config["legacy_factor_count"],
    ).eval()
    checkpoint = tmp_path / "worker_test.pt"
    torch.save({"config": config, "model_state": model.state_dict()}, checkpoint)

    root = Path(__file__).resolve().parents[1]
    base_port = _free_consecutive_ports(2)
    process = subprocess.Popen(
        [
            sys.executable, "serve_native_workers.py",
            "--checkpoint", str(checkpoint), "--workers", "2",
            "--host", "127.0.0.1", "--port", str(base_port),
            "--device", "cpu", "--no-graphs",
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    health: list[dict] = []
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and len(health) < 2:
            health = []
            for index in range(2):
                try:
                    health.append(_get_json(f"http://127.0.0.1:{base_port + index}/health"))
                except Exception:
                    break
            if len(health) < 2:
                time.sleep(0.2)
        assert len(health) == 2, "workers did not start"
        assert len({item["pid"] for item in health}) == 2
        assert all(item["cache"]["owner_pid"] == item["pid"] for item in health)
        for index in range(2):
            result = _post_json(
                f"http://127.0.0.1:{base_port + index}/infer",
                {"inputs": [[1, 2, 3]]},
            )
            assert result["logit_shape"] == [1, 16]
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
