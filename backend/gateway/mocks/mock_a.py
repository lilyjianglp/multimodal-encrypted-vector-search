#!/usr/bin/env python3
# HTTP -> gRPC adapter for IndexService
from flask import Flask, request, jsonify
import os, grpc
# 这两个文件就是你刚生成的 pb2
import index_service_pb2 as pb
import index_service_pb2_grpc as pbg

app = Flask(__name__)
# 真A地址：如果你用 socat 把 A 转到 19081，就保持默认；否则改成 127.0.0.1:50051
INDEX_ADDR = os.environ.get("INDEX_ADDR", "127.0.0.1:19081")
channel = grpc.insecure_channel(INDEX_ADDR)
stub = pbg.IndexServiceStub(channel)

@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/cluster_candidates", methods=["POST"])
def cluster_candidates():
    j = request.get_json(force=True, silent=True) or {}
    cluster_ids = j.get("cluster_ids", [])
    R = int(j.get("R", 512))
    resp = stub.GetClusterCandidates(pb.GetClusterCandidatesReq(cluster_ids=cluster_ids, R=R))
    return jsonify({"candidate_ids": list(resp.candidate_ids)}), 200

@app.route("/create_diag_blocks", methods=["POST"])
def create_diag_blocks():
    j = request.get_json(force=True, silent=True) or {}
    candidate_ids = j.get("candidate_ids", [])
    pack_slots = int(j.get("pack_slots", 4096))
    resp = stub.CreateDiagBlocks(pb.CreateDiagBlocksReq(candidate_ids=candidate_ids, pack_slots=pack_slots))
    blocks = []
    for b in resp.blocks:
        blocks.append({
            "mmap_path": b.mmap_path,
            "diag_offsets": list(b.diag_offsets),
            "slots": b.slots,
            "stride": b.stride
        })
    return jsonify({"blocks": blocks}), 200

@app.route("/get_centers", methods=["GET"])
def get_centers():
    resp = stub.GetCenters(pb.GetCentersReq())
    centers_hex = [c.hex() for c in resp.centers]  # 调试友好
    return jsonify({"centers": centers_hex}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=18081)
