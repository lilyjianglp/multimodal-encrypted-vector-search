#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
search_real_ct.py — FINAL VERSION
支持 --mode image/text/audio，自动向 Gateway 发送 x-query-mode
"""

import os
import sys
import json
import time
import grpc
import base64
import tempfile
import subprocess
import hmac
import hashlib
import argparse
from pathlib import Path

# ================== CLI 参数 ==================
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["image", "text", "audio"], default="image",
                    help="查询模态（image / text / audio）")
args_cli = parser.parse_args()
MODE = args_cli.mode
print(f"[*] Query MODE = {MODE}")

# ================== 配置 ==================
GATEWAY_ADDR = os.environ.get("GATEWAY_ADDR", "127.0.0.1:50052")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
CKKS_DIR = Path(os.environ.get("CKKS_DIR", str(SCRIPT_DIR)))

CLUSTERS_ENV = os.environ.get("CLUSTERS", "0,1")
CLUSTERS = [x.strip() for x in CLUSTERS_ENV.split(",") if x.strip()]

SCALE = int(os.environ.get("CKKS_SCALE", str(2**40)))
TOP_R = int(os.environ.get("TOP_R", "512"))

SESSION_ID = os.environ.get("SESSION_ID", "sess-001")
KEY_VER = os.environ.get("KEY_VER", "v1")
FAKE_POLICY = os.environ.get("FAKE_POLICY", "uniform")

# ================== HMAC KEY ==================
HMAC_KEY_HEX = os.environ.get("HMAC_KEY_HEX", "").strip()
if not HMAC_KEY_HEX:
    keyfile = Path.home() / ".keys" / "hmac_key.hex"
    if keyfile.exists():
        HMAC_KEY_HEX = keyfile.read_text().strip()

HMAC_KEY = None
if HMAC_KEY_HEX:
    try:
        HMAC_KEY = bytes.fromhex(HMAC_KEY_HEX)
    except:
        HMAC_KEY = None

# ================== 生成 Python stub ==================
PROTO_DIR = Path(os.environ.get(
    "GATEWAY_PROTO_DIR", str(REPO_ROOT / "backend/gateway/backend/proto")
))
PROTO_FILE = PROTO_DIR / "gateway.proto"

if not PROTO_FILE.exists():
    raise SystemExit(f"找不到 proto 文件: {PROTO_FILE}")

tmpdir = Path(tempfile.mkdtemp(prefix="gwpb_"))
subprocess.check_call([
    sys.executable, "-m", "grpc_tools.protoc",
    f"-I{PROTO_DIR}",
    f"--python_out={tmpdir}",
    f"--grpc_python_out={tmpdir}",
    str(PROTO_FILE),
])

sys.path.insert(0, str(tmpdir))
import gateway_pb2 as pb
import gateway_pb2_grpc as pbg

# ================== 读取密文 ==================
ct_path = CKKS_DIR / "ct_q.bin"
if not ct_path.exists():
    raise SystemExit(f"缺少密文文件: {ct_path}")

ct_q = ct_path.read_bytes()
print(f"[*] Loaded ct_q.bin ({len(ct_q)} bytes)")

# ================== 连接 gateway ==================
opts = [
    ("grpc.max_send_message_length", 256 * 1024 * 1024),
    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
]
chan = grpc.insecure_channel(GATEWAY_ADDR, options=opts)
grpc.channel_ready_future(chan).result(timeout=5)
stub = pbg.GatewayServiceStub(chan)
print(f"[*] Connected to Gateway @ {GATEWAY_ADDR}")

# ================== 构造 SearchRequest ==================
req = pb.SearchRequest(
    session_id=SESSION_ID,
    cluster_ids=CLUSTERS,
    ct_q=ct_q,
    key_ver=KEY_VER,
    fake_policy=FAKE_POLICY,
    scale=SCALE,
    topR=TOP_R,
    mode=MODE,      # ★★关键：写入 proto 字段
)

# ------------------- 保存 req.json -----------------------
try:
    saved = {
        "session_id": req.session_id,
        "cluster_ids": list(req.cluster_ids),
        "key_ver": req.key_ver,
        "fake_policy": req.fake_policy,
        "scale": req.scale,
        "topR": req.topR,
        "mode": req.mode,
        "ct_b64": base64.b64encode(req.ct_q).decode(),
    }
    Path("/tmp/req.json").write_text(json.dumps(saved, indent=2, ensure_ascii=False))
    Path("/tmp/req.pb").write_bytes(req.SerializeToString())
    print("[*] saved /tmp/req.json and /tmp/req.pb")
except:
    pass

# ================== HMAC / metadata ==================
metadata = []

if HMAC_KEY:
    nonce = os.urandom(16).hex()
    ts = str(int(time.time()))
    msg = b"|".join([
        ct_q,
        SESSION_ID.encode(),
        ",".join(CLUSTERS).encode(),
        KEY_VER.encode(),
        ts.encode(),
        nonce.encode()
    ])
    mac_hex = hmac.new(HMAC_KEY, msg, hashlib.sha256).hexdigest()

    metadata.extend([
        ("x-ct-mac", mac_hex),
        ("x-ct-nonce", nonce),
        ("x-ct-ts", ts),
        ("x-ct-client", os.environ.get("HE_CLIENT_ID", "gw")),
        ("x-ct-keyver", KEY_VER),
    ])
    print("[*] HMAC metadata added.")

# ================ 模态字段（最关键部分） =================
metadata.append(("x-query-mode", MODE))
print(f"[*] Metadata added: x-query-mode={MODE}")

# ================== 执行查询 ==================
t0 = time.time()
resp = stub.Search(
    req,
    timeout=float(os.environ.get("GATEWAY_TIMEOUT_SECONDS", "180")),
    metadata=metadata,
)
t1 = time.time()

# ================== 保存密态评分 ==================
print("scores_ciphertexts =", len(resp.scores_ciphertexts))
print("pack_shapes        =", [(s.batch, s.slots) for s in resp.pack_shapes])
print("latency_ms(proto)  =", resp.latency_ms)
print("latency_ms(client) =", int((t1 - t0) * 1000))

outdir = CKKS_DIR / "scores_out"
outdir.mkdir(exist_ok=True)
for i, blob in enumerate(resp.scores_ciphertexts):
    (outdir / f"scores_{i:02d}.bin").write_bytes(bytes(blob))

print(f"[OK] scores saved to {outdir}")
