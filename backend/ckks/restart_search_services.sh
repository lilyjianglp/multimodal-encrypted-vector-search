#!/bin/bash
# ========================================================
# 密态向量检索系统 - 服务重启脚本
# 用途：完全重启搜索相关的所有后端服务
# 作者：自动生成
# ========================================================

SCRIPT_NAME="restart_search_services.sh"
VERSION="1.0"

echo "========================================================"
echo "🔧 $SCRIPT_NAME v$VERSION"
echo "========================================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 主函数
main() {
    log_info "开始完全重启搜索服务..."
    
    # 1. 停止所有相关服务
    log_info "停止服务..."
    pkill -f "search_service.py" && log_success "搜索服务已停止" || log_warning "搜索服务未运行"
    pkill -f "gateway" && log_success "Gateway 已停止" || log_warning "Gateway 未运行"
    pkill -f "he_http_adapter.py" && log_success "HE适配器已停止" || log_warning "HE适配器未运行"
    
    sleep 3

    # 2. 清理临时文件
    log_info "清理临时文件..."
    rm -f /tmp/q.npy /tmp/clusters.txt
    rm -f /home/wen/Desktop/backend/ckks/ct_q.bin
    rm -f /home/wen/Desktop/backend/ckks/scores_raw_raw/sess-*.json
    log_success "临时文件清理完成"

    # 3. 重新启动服务
    log_info "启动服务..."
    
    # 启动 gateway
    cd /home/wen/Desktop/backend/ckks
    ./gateway > /tmp/gateway.log 2>&1 &
    GATEWAY_PID=$!
    sleep 2
    [ -n "$GATEWAY_PID" ] && log_success "Gateway 启动完成 (PID: $GATEWAY_PID)" || log_error "Gateway 启动失败"
    
    # 启动 HE 适配器
    cd /home/wen/Desktop/adapters
    python3 he_http_adapter.py --grpc 127.0.0.1:18082 --port 18083 > /tmp/he_adapter.log 2>&1 &
    HE_PID=$!
    sleep 2
    [ -n "$HE_PID" ] && log_success "HE适配器启动完成 (PID: $HE_PID)" || log_error "HE适配器启动失败"
    
    # 启动搜索服务
    cd /home/wen/Desktop/backend
    python3 search_service.py > /tmp/search_service.log 2>&1 &
    SEARCH_PID=$!
    sleep 2
    [ -n "$SEARCH_PID" ] && log_success "搜索服务启动完成 (PID: $SEARCH_PID)" || log_error "搜索服务启动失败"

    # 4. 等待服务稳定
    log_info "等待服务稳定..."
    sleep 5

    # 5. 验证服务状态
    log_info "验证服务状态..."
    echo "----------------------------------------"
    
    if netstat -tlnp | grep 50052 > /dev/null; then
        log_success "Gateway 运行正常 (端口: 50052)"
    else
        log_error "Gateway 运行异常"
    fi
    
    if netstat -tlnp | grep 18083 > /dev/null; then
        log_success "HE适配器 运行正常 (端口: 18083)"
    else
        log_error "HE适配器 运行异常"
    fi
    
    if ps aux | grep search_service.py | grep -v grep > /dev/null; then
        log_success "搜索服务 运行正常"
    else
        log_error "搜索服务 运行异常"
    fi
    
    echo "----------------------------------------"
    log_success "服务重启完成！"
    log_info "日志文件: /tmp/gateway.log, /tmp/he_adapter.log, /tmp/search_service.log"
}

# 执行主函数
main "$@"
