#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
index_http_adapter.py  — 多维度 & 多 block 版（方案B）
用途：把 HTTP 请求转成对 IndexService(gRPC) 的调用，并在 /diag-blocks
按 dim 返回 mmap 对角素材路径，兼容服务端返回的 slot_ids/diag_offsets。
依赖：pip install flask grpcio grpcio-tools
放置：~/Desktop/adapters/index_http_adapter.py
"""

from flask import Flask, request, jsonify
import argparse
import grpc
import json
import traceback
import os
from math import ceil
from pathlib import Path

# 让 Python 能 import 同目录下生成的 *_pb2*.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
import index_service_pb2 as pb
import index_service_pb2_grpc as pbg  # noqa

app = Flask("index_http_adapter")

# ---------- gRPC 懒连接 ----------
_CHANNEL = None
_STUB = None
_GRPC_ADDR = "127.0.0.1:50051"

def get_stub():
    """懒加载并缓存 gRPC stub；若信道未就绪，尝试 3 秒，不阻塞启动。"""
    global _CHANNEL, _STUB
    if _STUB is None:
        _CHANNEL = grpc.insecure_channel(_GRPC_ADDR)
        try:
            grpc.channel_ready_future(_CHANNEL).result(timeout=3.0)
        except Exception:
            pass
        _STUB = pbg.IndexServiceStub(_CHANNEL)
    return _STUB

# ---------- 小工具 ----------
SCALE_40 = float(2 ** 40)

def _dim_dir(dim: int) -> str:
    """把维度映射到目录名：256->D0256、768->D0768 ..."""
    return f"D{dim:04d}"

def _blk_name(block_offset: int) -> str:
    """块文件名：blk-000000.dia、blk-000128.dia ..."""
    return f"blk-{block_offset:06d}.dia"

def _pick_dim(payload: dict) -> int:
    """dim 选择优先级：请求体 -> 环境变量 DIAG_DIM -> 默认 768"""
    try:
        if "dim" in payload and payload["dim"] is not None:
            return int(payload["dim"])
    except Exception:
        pass
    try:
        return int(os.environ.get("DIAG_DIM", "768"))
    except Exception:
        return 768

# ---------- 基础路由 ----------
@app.get("/health")
def health():
    return jsonify({"service": "index_http_adapter", "status": "healthy", "grpc": _GRPC_ADDR})

@app.get("/centers")
def centers():
    """调用 GetCenters，提供 centers 元信息（含 SHA 用于一致性校验）"""
    try:
        stub = get_stub()
        resp = stub.GetCenters(pb.GetCentersRequest(), timeout=3.0)
        return jsonify({
            "centers_sha":  resp.centers_sha,
            "centers_path": resp.centers_path,
            "dim":          resp.dim,
            "num_centers":  resp.num_centers,
        })
    except grpc.RpcError as e:
        return jsonify({"error": "grpc_failed", "code": e.code().name, "details": e.details()}), 502
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "server_error", "details": repr(e)}), 500

@app.get("/clusters/<cid>/candidates")
def get_candidates(cid: str):
    """
    GET /clusters/<cid>/candidates?top=R
    返回:
    {
      "padded_to": R,
      "candidates": [{"id": <uint64>}, ...]
    }
    """
    try:
        top = int(request.args.get("top", "0"))
        if top <= 0:
            return jsonify({"error": "bad_request", "details": "missing or invalid ?top="}), 400

        cluster_id = int(cid) if cid.isdigit() else 0

        stub = get_stub()
        r = pb.GetClusterCandidatesRequest(cluster_id=cluster_id, top_r=top)
        resp = stub.GetClusterCandidates(r, timeout=5.0)

        out = {"padded_to": resp.padded_to, "candidates": []}
        for c in resp.candidates:
            out["candidates"].append({"id": int(c.id)})
        return jsonify(out)
    except grpc.RpcError as e:
        return jsonify({"error": "grpc_failed", "code": e.code().name, "details": e.details()}), 502
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "server_error", "details": repr(e)}), 500

# ---------- diag-blocks（方案 B：mmap 多 block + 多维） ----------
@app.post("/diag-blocks")
def create_diag_blocks():
    """
    POST /diag-blocks
    body: { "candidates":[...], "pack_slots":4096, "dim":768 }
    返回:
    {
      "blocks":[
        {
          "block_id": "...",
          "mmap_path": "data/diag/D0768/blk-000000.dia",
          "layout": {
            "slots": 4096,
            "stride": 4096,
            "poly_modulus_degree": 8192,
            "scale": 1099511627776.0,
            "level": 0,
            "packing": "offset-major",
            "diag_offsets":[... 最多 128 个 ...]
          },
          "slot_ids":[ ...4096 ...]
        }, ...
      ]
    }
    """
    try:
        j = request.get_json(force=True, silent=False)
        if not isinstance(j, dict):
            return jsonify({"error": "bad_json"}), 400

        cand = j.get("candidates", [])
        pack_slots = int(j.get("pack_slots", 0))
        dim = _pick_dim(j)  # ★ 读取维度（请求体/环境变量/默认）

        if pack_slots <= 0:
            return jsonify({"error": "bad_json", "details": "pack_slots must be positive"}), 400
        if dim <= 0:
            return jsonify({"error": "bad_json", "details": "invalid dim"}), 400

        # 兼容对象数组 [{"id":123}, ...] 与纯 ID 列表 [123, ...]
        norm_ids = []
        if isinstance(cand, list):
            for v in cand:
                if isinstance(v, dict) and "id" in v:
                    norm_ids.append(int(v["id"]))
                else:
                    norm_ids.append(int(v))
        else:
            return jsonify({"error": "bad_json", "details": "candidates must be list"}), 400

        # 调 IndexService 生成对角计划（diag_offsets/slot_ids）
        stub = get_stub()
        req = pb.CreateDiagBlocksRequest(candidate_ids=norm_ids, pack_slots=pack_slots)
        resp = stub.CreateDiagBlocks(req, timeout=8.0)

        # 多模态对角块根路径；允许测试/部署通过环境变量覆盖。
        mode = j.get("mode", "image")

        if mode == "image":
            base_rel = Path(os.environ.get(
                "DIAG_ROOT_IMAGE", str(Path("/home/wen/Desktop/backend/ckks/image") / _dim_dir(dim))
            ))
        elif mode == "audio":
            base_rel = Path(os.environ.get(
                "DIAG_ROOT_AUDIO", str(Path("/home/wen/Desktop/backend/ckks/audio") / _dim_dir(dim))
            ))
        elif mode == "text":
            base_rel = Path(os.environ.get(
                "DIAG_ROOT_TEXT", str(Path("/home/wen/Desktop/backend/ckks/text") / _dim_dir(dim))
            ))
        else:
            return jsonify({"error": "bad_mode", "details": mode}), 400

        # The IndexService proto predates layout versioning and does not carry
        # `packing`. Recover it from the generated metadata so HECompute can
        # select the true group-level BSGS kernel while legacy blocks remain
        # compatible.
        generated_layouts = {}
        metadata_path = base_rel / "diag_blocks.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for item in metadata.get("blocks", []):
                name = Path(str(item.get("mmap_path", ""))).name
                if name:
                    generated_layouts[name] = item.get("layout", {}) or {}

        grpc_blocks = list(resp.blocks)
        blocks = []
        for i, grpc_block in enumerate(grpc_blocks):
            offs = list(grpc_block.diag_offsets)
            block_id = grpc_block.block_id or f"blk-{i:06d}"
            mmap_name = grpc_block.mmap_path or block_id
            mmap_path = str(base_rel / Path(mmap_name).name)
            slot_ids = list(grpc_block.slot_ids)
            generated_layout = generated_layouts.get(Path(mmap_name).name, {})

            layout = {
                "slots": int(grpc_block.slots or pack_slots),
                "stride": int(grpc_block.stride or pack_slots),
                "poly_modulus_degree": 8192,
                "scale": SCALE_40,
                "level": 0,
                "packing": generated_layout.get("packing", "offset-major"),
                "diag_offsets": offs
            }
            blocks.append({
                "block_id": block_id,
                "snapshot_id": "",
                "mmap_path": mmap_path,
                "layout": layout,
                "slot_ids": slot_ids
            })

        if not blocks:
            return jsonify({"error": "empty_diag_plan"}), 500

        # 日志友好：附带 echo 维度/根目录
        return jsonify({"dim": dim, "root": str(base_rel), "blocks": blocks})

    except grpc.RpcError as e:
        app.logger.error("[index_http_adapter] diag-blocks grpc error: %s", e)
        return jsonify({"error": "grpc_failed", "code": e.code().name, "details": e.details()}), 502
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "server_error", "details": repr(e)}), 500

# ---------- main ----------
def main():
    global _GRPC_ADDR
    ap = argparse.ArgumentParser()
    ap.add_argument("--grpc", default="127.0.0.1:50051", help="IndexService gRPC address")
    ap.add_argument("--host", default="127.0.0.1", help="HTTP bind address")
    ap.add_argument("--port", type=int, default=18081, help="HTTP listen port")
    args = ap.parse_args()

    _GRPC_ADDR = args.grpc
    app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
