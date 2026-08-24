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

        # ★ multimodal 根路径
        mode = j.get("mode", "image")

        if mode == "image":
            base_rel = Path("/home/wen/Desktop/backend/ckks/image") / _dim_dir(dim)
        elif mode == "audio":
            base_rel = Path("/home/wen/Desktop/backend/ckks/audio") / _dim_dir(dim)
        elif mode == "text":
            base_rel = Path("/home/wen/Desktop/backend/ckks/text") / _dim_dir(dim)
        else:
            return jsonify({"error": "bad_mode", "details": mode}), 400

        # 块数：每块 128 个 offset
        blk_cnt = ceil(dim / 128)

        grpc_blocks = list(resp.blocks)
        # slot_ids：优先使用 gRPC 返回；否则用 candidates 填满（不足补 0，超出截断）
        if grpc_blocks and len(grpc_blocks[0].slot_ids) > 0:
            slot_ids = list(grpc_blocks[0].slot_ids)
        else:
            slot_ids = (norm_ids + [0] * (pack_slots - len(norm_ids)))[:pack_slots]

        blocks = []
        for i in range(blk_cnt):
            # 该 block 覆盖的 offset 区间
            start_off = i * 128
            end_off = min(start_off + 128, dim)
            offs = list(range(start_off, end_off))

            # 若 gRPC 有对应块并给了 diag_offsets，则优先
            if i < len(grpc_blocks) and len(grpc_blocks[i].diag_offsets) > 0:
                offs = list(grpc_blocks[i].diag_offsets)

            block_offset = offs[0] if offs else start_off
            mmap_path = str(base_rel / _blk_name(block_offset))  # e.g. data/diag/D0512/blk-000128.dia

            layout = {
                "slots": 4096,
                "stride": 4096,                # ★ 一定固定 4096
                "poly_modulus_degree": 8192,
                "scale": SCALE_40,
                "level": 0,
                "packing": "offset-major",
                "diag_offsets": offs
            }
            blocks.append({
                "block_id": f"blk-{block_offset:06d}",
                "snapshot_id": "",
                "mmap_path": mmap_path,
                "layout": layout,
                "slot_ids": slot_ids
            })

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
    ap.add_argument("--port", type=int, default=18081, help="HTTP listen port")
    args = ap.parse_args()

    _GRPC_ADDR = args.grpc
    app.run(host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()

