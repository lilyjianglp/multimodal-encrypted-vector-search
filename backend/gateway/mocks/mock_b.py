from flask import Flask, request, jsonify
import os, base64, time, random

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "mock_b"})

@app.route("/readyz", methods=["GET"])
def readyz():
    return jsonify({"status": "ready"})

# 符合网关预期的接口：
# 入参(示例)：{ client_id, key_ver, ct_q, diag_blocks, bsgs_plan, scale, mod_chain }
# 出参：{ packed_scores: [base64...], pack_shape: [batch, slots], telemetry: {...} }
@app.route("/score/batch", methods=["POST"])
def score_batch():
    # 轻量延迟，模拟计算
    time.sleep(random.uniform(0.01, 0.03))

    body = request.get_json(force=True, silent=True) or {}

    # 从环境读取与网关一致的固定模板参数
    out_n    = int(os.getenv("OUT_CT_COUNT", "8"))     # 返回多少个密文
    ct_bytes = int(os.getenv("CT_BYTES", "16384"))     # 每个密文（解码后）的字节数
    slots    = int(os.getenv("PACK_SLOTS", "4096"))    # 仅用于 pack_shape 协议对齐

    # 构造占位密文：生成随机 bytes -> base64
    packed = []
    for _ in range(out_n):
        raw = os.urandom(ct_bytes)
        packed.append(base64.b64encode(raw).decode())

    return jsonify({
        "packed_scores": packed,
        "pack_shape": [1, slots],
        "telemetry": {"mock": "b", "ct_bytes": ct_bytes, "out_n": out_n}
    })

@app.route("/eval-keys", methods=["POST"])
def eval_keys():
    # 网关会转发这个接口；此处直接返回成功
    return jsonify({"ok": True})

@app.route("/")
def index():
    return jsonify({
        "service": "mock_b",
        "endpoints": ["/health", "/readyz", "/score/batch", "/eval-keys"]
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=18083, debug=False)

