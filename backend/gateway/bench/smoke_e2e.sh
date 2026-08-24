#!/usr/bin/env bash
set -euo pipefail

# 配置
PROTO_DIR="${PROTO_DIR:-$HOME/secure-search/backend/proto}"
TESTS="${TESTS:-$HOME/secure-search/tests}"
GW="localhost:50051"
SVC="gateway.v1.GatewayService/Search"

mkdir -p "$TESTS"

echo "[smoke] proto at: $PROTO_DIR"
test -f "$PROTO_DIR/gateway.proto" || { echo "[smoke] gateway.proto not found under \$PROTO_DIR"; exit 1; }

# 5 次冒烟
echo "[smoke] running 5 requests..."
for i in {1..5}; do
  OUT="$TESTS/smoke-$i.json"
  CT=$(head -c 4096 </dev/urandom | base64 -w0)

  grpcurl -plaintext -import-path "$PROTO_DIR" -proto gateway.proto \
    -d "{\"session_id\":\"smk-$i\",\"cluster_ids\":[\"c1\",\"c2\",\"c3\"],\"ct_q\":\"$CT\",\"key_ver\":\"v1\",\"fake_policy\":\"uniform\",\"scale\":1099511627776,\"topR\":1}" \
    "$GW" "$SVC" > "$OUT"

  # 计算：密文长度、响应字节数、延迟（兼容 camel/snake）
  LEN=$(cat "$OUT" | jq -r '
    if has("scoresCiphertexts") then (.scoresCiphertexts[0] | @base64d | length)
    elif has("scores_ciphertexts") then (.scores_ciphertexts[0] | @base64d | length)
    else 0 end' 2>/dev/null || echo 0)

  # jq 无 @base64d 就用 python 兜底
  if [ -z "$LEN" ] || [ "$LEN" = "0" ]; then
    LEN=$(python3 - "$OUT" <<'PY'
import sys,json,base64
d=json.load(open(sys.argv[1]))
k='scoresCiphertexts' if 'scoresCiphertexts' in d else ('scores_ciphertexts' if 'scores_ciphertexts' in d else None)
print(len(base64.b64decode(d[k][0])) if k else 0)
PY
)
  fi

  BYTES=$(wc -c < "$OUT")
  LAT=$(cat "$OUT" | jq -r '(.latencyMs // .latency_ms // empty)')
  echo "run#$i len=$LEN bytes=$BYTES latency=$LAT"
done

echo "[smoke] recent logs (search_plan & search):"
grep -E '"event":"search_plan"|"event":"search"' "$HOME/secure-search/logs/gateway.log" | tail -10 || true

echo "[smoke] done. outputs under $TESTS"
