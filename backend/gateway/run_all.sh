#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT_DIR=$(realpath "../..")
GW_ROOT="$ROOT_DIR/backend/gateway"
BUILD="$GW_ROOT/backend/build"
TESTS="$GW_ROOT/tests"
LOGDIR="$GW_ROOT/logs"
MOCK_A="$GW_ROOT/mocks/mock_a.py"
MOCK_B="$GW_ROOT/mocks/mock_b.py"
PROTO_DIR="$GW_ROOT/backend/proto"
INDEX_BUILD="$ROOT_DIR/backend/services/index/build"
COMPUTE_BUILD="$ROOT_DIR/backend/compute/build"


if [ -x "$GW_ROOT/venv/bin/python" ]; then
  PYTHON="$GW_ROOT/venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "[run_all] ERROR: Python not found. Install python3 or create venv at $GW_ROOT/venv" >&2
  exit 1
fi
echo "[run_all] Using python at $PYTHON"

OUT_CT_COUNT=${OUT_CT_COUNT:-8}
CT_BYTES=${CT_BYTES:-16384}

# ----------------- 工具函数 -----------------
ensure_free(){
  local port="$1"
  echo "[run_all] Ensuring port $port is free..."
  fuser -k "$port"/tcp >/dev/null 2>&1 || true
  while lsof -i ":$port" >/dev/null 2>&1; do sleep 0.1; done
  sleep 0.3
}

wait_port(){
  local port="$1"
  local tries=20
  for i in $(seq 1 $tries); do
    if lsof -i ":$port" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  echo "[run_all] ERROR: Port $port not available after waiting"
  exit 1
}

# ----------------- 创建目录 -----------------
mkdir -p "$LOGDIR" "$BUILD" "$TESTS"

# ----------------- 清理端口 -----------------
ensure_free 50052  # IndexService / Gateway
ensure_free 8080   # Gateway
ensure_free 50051
ensure_free 18081   # Mock A
ensure_free 18083   # Mock B

# ----------------- 启动 IndexService -----------------
echo "[run_all] Starting IndexService..."
INDEX_LOG="$LOGDIR/index_server.log"
"$INDEX_BUILD/index_server" "$INDEX_BUILD/../data" >>"$INDEX_LOG" 2>&1 &
wait_port 50051
echo "[run_all] IndexService started"

# ----------------- 启动 HECompute -----------------
echo "[run_all] Starting HECompute..."
HE_LOG="$LOGDIR/hecompute.log"

# 1) 可执行文件存在检查
if [ ! -x "$COMPUTE_BUILD/hecompute_server" ]; then
  echo "[run_all] ERROR: $COMPUTE_BUILD/hecompute_server not found or not executable"
  exit 1
fi

# 2) 后台启动 + 日志重定向 + 记录 PID
"$COMPUTE_BUILD/hecompute_server" >>"$HE_LOG" 2>&1 &
HE_PID=$!
sleep 0.3

# 3) 等端口（如果程序不是 18082，请改成实际端口）
if ! wait_port 18082; then
  echo "[run_all] ERROR: HECompute failed on 18082"
  echo "---- hecompute.log (tail) ----"
  tail -n 100 "$HE_LOG" || true
  kill "$HE_PID" >/dev/null 2>&1 || true
  exit 1
fi
echo "[run_all] HECompute started (pid=$HE_PID)"

# ----------------- 启动 Mock A/B -----------------
echo "[run_all] Starting Mock A on 18081..."
A_LOG="$LOGDIR/mock_a.log"
nohup "$PYTHON" "$MOCK_A" >>"$A_LOG" 2>&1 &
wait_port 18081

echo "[run_all] Starting Mock B on 18083..."
B_LOG="$LOGDIR/mock_b.log"
nohup "$PYTHON" "$MOCK_B" >>"$B_LOG" 2>&1 &
wait_port 18083

echo "[run_all] Mock services started successfully"

# ----------------- 启动 Gateway -----------------
echo "[run_all] Starting Gateway on 50052 (HTTP 8080)..."
G_LOG="$LOGDIR/gateway.log"
cd "$GW_ROOT/backend"
nohup "$BUILD/gateway" --config "$(realpath ./config.json)" >>"$G_LOG" 2>&1 &
cd "$GW_ROOT"
wait_port 8080
echo "[run_all] Gateway started successfully"

# ----------------- 发送测试请求 -----------------
echo "[run_all] Sending test request..."
CT=$(head -c 32768 /dev/urandom | base64 -w0)

if command -v grpcurl >/dev/null 2>&1; then
  grpcurl -plaintext \
    -import-path "$PROTO_DIR" \
    -proto gateway.proto \
    -d "{
      \"session_id\":\"sess-001\",
      \"cluster_ids\":[\"c1\",\"c2\",\"c3\",\"c4\",\"c5\",\"c6\",\"c7\",\"c8\"],
      \"ct_q\":\"$CT\",
      \"key_ver\":\"v1\",
      \"fake_policy\":\"uniform\",
      \"scale\":1099511627776,
      \"topR\":512
    }" \
    localhost:50052 gateway.v1.GatewayService/Search >"$TESTS/result.json"
else
  echo "[run_all] WARN: grpcurl not found, skip gRPC test. Install with:"
  echo "  sudo apt-get install -y golang-go && go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest"
  echo "{}" >"$TESTS/result.json"
fi

# ----------------- 验证响应 -----------------
echo "[run_all] Validating response..."
COUNT=$(jq -r '.scoresCiphertexts | length // 0' "$TESTS/result.json")
FIRST_CIPHER=$(jq -r '.scoresCiphertexts[0] // ""' "$TESTS/result.json")
if [ -n "$FIRST_CIPHER" ]; then
    FIRST_LEN=$(echo "$FIRST_CIPHER" | tr -d '\n' | base64 --decode | wc -c)
else
    FIRST_LEN=0
fi

echo "$FIRST_CIPHER" | tr -d '\n' | base64 --decode | wc -c
echo "[run_all] Response: count=$COUNT, first_cipher_len=$FIRST_LEN"

# ----------------- 状态展示 -----------------
echo
echo "[run_all] Service status:"
echo "Gateway gRPC (50052): $(lsof -i :50052 | head -1 || echo 'not running')"
echo "Gateway HTTP (8080):  $(curl -s http://127.0.0.1:8080/readyz || echo 'not running')"
echo "Mock A (18081):       $(curl -s http://127.0.0.1:18081/health || echo 'not running')"
echo "Mock B (18083):       $(curl -s http://127.0.0.1:18083/health || echo 'not running')"
echo "[run_all] Done."
