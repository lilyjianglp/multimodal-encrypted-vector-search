#!/usr/bin/env python3
"""Upload a client's CKKS rotation/relinearization keys to HeCompute."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import grpc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grpc", default="127.0.0.1:18082")
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--client-id", default="gw")
    parser.add_argument("--key-ver", default="v1")
    args = parser.parse_args()

    adapters = Path(__file__).resolve().parents[2] / "adapters"
    sys.path.insert(0, str(adapters))
    import hecompute_pb2 as pb
    import hecompute_pb2_grpc as pbg

    channel = grpc.insecure_channel(
        args.grpc,
        options=[
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ],
    )
    grpc.channel_ready_future(channel).result(timeout=10)
    reply = pbg.HeComputeServiceStub(channel).EvalKeys(
        pb.EvalKeysRequest(
            client_id=args.client_id,
            key_ver=args.key_ver,
            galois=(args.keys / "galois.bin").read_bytes(),
            relin=(args.keys / "relin.bin").read_bytes(),
        ),
        timeout=60,
        wait_for_ready=True,
    )
    if not reply.ok:
        raise SystemExit(f"HeCompute rejected evaluation keys: {reply.msg}")
    print(f"uploaded evaluation keys: client={args.client_id} version={args.key_ver}")


if __name__ == "__main__":
    main()
