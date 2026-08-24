#search_service.py
import os
import time
import json
import base64
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from flask import send_file
import sys 

# ============================================================================== 
# 0. 配置与初始化
# ==============================================================================

app = Flask(__name__)
# 允许前端跨域访问 (在开发环境中非常重要)
CORS(app) 

# --- 核心路径定义 (保持 Desktop 结构) ---
PROJECT_ROOT = Path.home()  # /home/wen
ADAPTERS_DIR = PROJECT_ROOT / "Desktop" / "adapters"  # /home/wen/Desktop/adapters
BACKEND_DIR  = PROJECT_ROOT / "Desktop" / "backend"  # /home/wen/Desktop/backend

CKKS_DIR     = BACKEND_DIR / "ckks"                   # /home/wen/Desktop/backend/ckks
SERVICES_DIR = BACKEND_DIR / "services"
INDEX_DATA_DIR = SERVICES_DIR / "index" / "data"

# --- 脚本和可执行文件（在 /home/wen/Desktop/backend/ckks 下）---
IMG_TO_VEC      = CKKS_DIR / "embed_image.py"
TEXT_TO_VEC     = CKKS_DIR / "embed_text.py"  

AUDIO_TO_VEC    = CKKS_DIR / "embed_audio.py"  
CKKS_MAKE_CTQ   = CKKS_DIR / "ckks_make_ctq_from_npy"
CLIENT_PICK_CLU = CKKS_DIR / "client_pick_clusters.py"
UPLOAD_EVALKEYS = CKKS_DIR / "upload_evalkeys_v1.py"
SEARCH_REAL_CT  = CKKS_DIR / "search_real_ct.py"
CKKS_DECRYPT    = CKKS_DIR / "ckks_decrypt_dump"
ANALYZE_TOPK    = CKKS_DIR / "analyze_topk.py"

# --- 各模态的配置文件 ---
IMAGE_CENTERS   = CKKS_DIR / "image" / "centers.npy"
AUDIO_CENTERS   = CKKS_DIR / "audio" / "centers.npy"  
TEXT_CENTERS    = CKKS_DIR / "text" / "centers.npy"

IMAGE_SLOT_IDS  = Path.home() / "Desktop" / "adapters" / "slot_ids.json"
AUDIO_SLOT_IDS  = CKKS_DIR / "audio" / "audio_id_map.json"
TEXT_SLOT_IDS   = CKKS_DIR / "text" / "slot_ids.json"

AUDIO_ID_MAP    = CKKS_DIR / "audio" / "audio_id_map.json"

# --- 密钥和数据文件 ---
CONTEXT_FILE    = CKKS_DIR / "context.seal"
PK_FILE         = CKKS_DIR / "pk.bin"
SK_FILE         = CKKS_DIR / "sk.bin"
SCORES_DIR      = CKKS_DIR / "scores_raw_raw"

AUDIO_LOOKUP = CKKS_DIR / "audio_lookup.py"
IMAGE_LOOKUP = CKKS_DIR / "image_lookup.py"
TEXT_LOOKUP = CKKS_DIR / "text_lookup.py"


# --- 临时文件路径 ---
TEMP_Q_NPY = Path("/tmp/q.npy")
TEMP_CLUSTERS_TXT = Path("/tmp/clusters.txt")
CT_Q_BIN = CKKS_DIR / "ct_q.bin"
TEMP_FILES_TO_CLEAN = [TEMP_Q_NPY, TEMP_CLUSTERS_TXT, CT_Q_BIN]

# 其他配置
GATEWAY_GRPC_ADDR = "127.0.0.1:50052"
HE_ADAPTER_HTTP_ADDR = "http://127.0.0.1:18083"

# 搜索参数
DIM = 512
TOP_R = 512
L = 32
TOP_T = 16 # 使用 TOP_T 而不是 R。R 已经是 512，可能 TOP_T 才是集群数量

# ============================================================================== 
# 权限自检函数 
# ==============================================================================

def _check_permissions():
    """
    检查关键文件和目录的读/写/执行权限。
    """
    print("--------------------------------------------------------")
    print("[INIT] 正在进行文件/目录权限自检...")

    # 定义需要检查的路径和权限要求
    checks = [
        # 1. 脚本和二进制文件的可执行性
        (IMG_TO_VEC, os.X_OK, "特征提取脚本 (图片)"),
        (TEXT_TO_VEC, os.X_OK, "特征提取脚本 (文本)"),
        (AUDIO_TO_VEC, os.X_OK, "特征提取脚本 (音频)"),
        (CKKS_MAKE_CTQ, os.X_OK, "密文生成二进制"),
        (CLIENT_PICK_CLU, os.X_OK, "簇选择脚本"),
        (SEARCH_REAL_CT, os.X_OK, "密态搜索脚本"),
        (CKKS_DECRYPT, os.X_OK, "解密二进制"),
        (ANALYZE_TOPK, os.X_OK, "Top-K 分析脚本"),

        # 2. CKKS 密钥和配置文件的可读性
        (CONTEXT_FILE, os.R_OK, "CKKS Context 文件"),
        (PK_FILE, os.R_OK, "公钥文件 (PK)"),
        (SK_FILE, os.R_OK, "私钥文件 (SK)"),
        
        # 3. 各模态的聚类中心文件
        (IMAGE_CENTERS, os.R_OK, "图像聚类中心"),
        (AUDIO_CENTERS, os.R_OK, "音频聚类中心"),
        (TEXT_CENTERS, os.R_OK, "文本聚类中心"),
        
        # 4. 各模态的slot映射文件
        (IMAGE_SLOT_IDS, os.R_OK, "图像slot映射"),
        (AUDIO_SLOT_IDS, os.R_OK, "音频slot映射"),
        (TEXT_SLOT_IDS, os.R_OK, "文本slot映射"),
        
        (AUDIO_ID_MAP, os.R_OK, "音频ID映射"),

        # 5. 临时文件和输出目录的可写性
        (TEMP_Q_NPY.parent, os.W_OK, "/tmp 目录 (用于中间文件)"), 
        (CKKS_DIR, os.W_OK, f"CKKS 目录 ({CKKS_DIR})"),
        # 检查分数目录是否存在，如果不存在则尝试创建，并检查父目录可写性
        (SCORES_DIR, os.W_OK, f"分数目录 ({SCORES_DIR})"),
    ]

    missing_paths = []
    permission_issues = []

    for path, access_mode, description in checks:
        # 如果是分数目录，尝试创建
        if path == SCORES_DIR and not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception:
                # 如果创建失败，可能是父目录权限问题
                permission_issues.append(f"{description}: 尝试创建失败，检查父目录权限。")
                continue

        if not path.exists():
            missing_paths.append(f"{description}: {path}")
            continue

        # 检查权限
        if not os.access(path, access_mode):
            mode_str = "可执行" if access_mode == os.X_OK else "可读" if access_mode == os.R_OK else "可写"
            permission_issues.append(f"{description}: {path} - 缺少 {mode_str} 权限")
        
        # 针对目录，额外的可写检查 (如果需要)
        if path.is_dir() and access_mode == os.W_OK and not os.access(path, os.W_OK):
             permission_issues.append(f"{description}: {path} - 缺少 可写 权限")

    if missing_paths or permission_issues:
        print("\n[ERROR] 权限自检失败！请修正以下问题后重新运行:")
        
        if missing_paths:
            print("\n--- 1. 路径不存在 (Path Not Found) ---")
            for issue in missing_paths:
                print(f"   ❌ {issue}")
            print("\n请检查路径配置是否正确，并确保所有文件已生成。")

        if permission_issues:
            print("\n--- 2. 权限不足 (Permission Denied) ---")
            for issue in permission_issues:
                print(f"   ❌ {issue}")
            print("\n请使用 'chmod' 命令给予程序运行用户相应的权限，例如: `chmod +x <脚本>` 或 `chmod -R u+rw <目录>`")
        
        # 权限自检失败，终止程序运行
        raise RuntimeError("系统初始化失败：关键文件/目录权限检查未通过。")
        
    print("[INIT] 权限自检通过。所有关键组件路径和权限均正常。")
    print("--------------------------------------------------------")

# ============================================================================== 
# 1. 实际外部脚本执行 (使用 subprocess)
# ==============================================================================

def _run_external_script(name, cmd_list, check=True, env=None):
    """通用函数，用于执行外部脚本或二进制程序 - 修复输出显示"""
    print(f"[PROCESS] 运行 {name}: {' '.join(map(str, cmd_list))}")
    
    cmd_list_str = [str(cmd) for cmd in cmd_list]
    process_env = env if env is not None else os.environ.copy()
    
    try:
        result = subprocess.run(
            cmd_list_str, 
            check=check, 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            env=process_env,
            cwd=CKKS_DIR
        )
        
        # 修复：显示完整输出，不再截断
        if result.stdout:
            print(f"[{name} 完整输出]:")
            print("=" * 60)
            print(result.stdout)
            print("=" * 60)
        
        if result.stderr:
            print(f"[{name} Error/Log]:")
            print("=" * 60)
            print(result.stderr)
            print("=" * 60)
        
        if name == "Analyze Top-K":
            return result.stdout.strip()
        
        return "Real execution success."
        
    except subprocess.CalledProcessError as e:
        print(f"[{name} 错误]: 命令失败，返回码 {e.returncode}")
        print(f"错误详情: {e.stderr}")
        raise RuntimeError(f"外部组件 {name} 运行失败。详细错误: {e.stderr.strip()}")
    except FileNotFoundError:
        print(f"[{name} 错误]: 找不到执行文件: {cmd_list_str[0]}")
        raise RuntimeError(f"外部组件 {name} 找不到或无执行权限。")

# ============================================================================== 
# 2. 辅助函数：临时文件清理
# ==============================================================================

def _cleanup_temp_files():
    """安全地删除在搜索流程中生成的临时文件（.npy, .txt, ct_q.bin）。"""
    print("[CLEANUP] Starting temporary file cleanup...")
    for f in TEMP_FILES_TO_CLEAN:
        if f.exists():
            try:
                f.unlink()
            except Exception as e:
                # 使用 sys.stderr 确保清理错误能被打印出来
                print(f"[CLEANUP ERROR] Failed to remove {f}: {e}", file=sys.stderr)

# ============================================================================== 
# 3. HTTP 接口
# ==============================================================================

@app.route('/search', methods=['POST'])
def search_orchestrator():
    t_start = time.time()
    data = request.json
    
    if not data or 'query_path' not in data or 'session_id' not in data:
        return jsonify({"error": "缺少 'query_path' 或 'session_id' 参数"}), 400

    query_path = data['query_path']
    session_id = data['session_id']
    key_ver = data.get('key_ver', 'v1')
    query_type = data.get('type', 'image')

    print(f"\n========================================================")
    print(f"[*] 收到查询请求: Session ID={session_id}, Path={query_path}, Type={query_type}")
    print(f"========================================================")

    # 初始化最终结果
    final_result = {}
    search_mode = query_type

    try:
        # 1. 特征提取
        if query_type == 'image':
            _run_external_script(
                "特征提取 (图像)",
                [
                    IMG_TO_VEC,
                    query_path,
                    "--pca", "/media/wen/F500/image_embeddings/image_pca_2048_to_512.pkl",
                    "--weight", "/media/wen/F500/ai_models/resnet50-0676ba61.pth",
                    "--out", TEMP_Q_NPY
                ]
            )
            centers_file = IMAGE_CENTERS
            slot_ids_file = IMAGE_SLOT_IDS
            search_mode = "image"
            
        elif query_type == 'audio':
            _run_external_script(
                "特征提取 (音频)",
                [
                    AUDIO_TO_VEC,
                    query_path,
                    TEMP_Q_NPY,
                    "--weight", "/media/wen/F500/ai_models/panns/Cnn14_16k_mAP=0.438.pth",
                    "--pca_mean", "/media/wen/F500/audio_embeddings/pca_mean.npy",
                    "--pca_comp", "/media/wen/F500/audio_embeddings/pca_components.npy"
                ]
            )
            centers_file = AUDIO_CENTERS
            slot_ids_file = AUDIO_SLOT_IDS
            search_mode = "audio"
            
        elif query_type == 'text':
            # 检查是文件路径还是直接文本内容
            if os.path.exists(query_path) and os.path.isfile(query_path):
                # 是文件路径，按原方式处理
                _run_external_script(
                    "特征提取 (文本)",
                    [
                        TEXT_TO_VEC,
                        TEMP_Q_NPY,
                        "--file", query_path,
                        "--pca", "/media/wen/F500/text_embeddings/text_pca_768_to_512.pkl"
                    ]
                )
            else:
                # 是直接文本内容，先保存到临时文件
                temp_text_file = f"/tmp/{session_id}_text_query.txt"
                with open(temp_text_file, 'w', encoding='utf-8') as f:
                    f.write(query_path)
                
                _run_external_script(
                    "特征提取 (文本)",
                    [
                        TEXT_TO_VEC,
                        TEMP_Q_NPY,
                        "--file", temp_text_file,
                        "--pca", "/media/wen/F500/text_embeddings/text_pca_768_to_512.pkl"
                    ]
                )
            centers_file = TEXT_CENTERS
            slot_ids_file = TEXT_SLOT_IDS
            search_mode = "text"
        else:
            return jsonify({
                "status": "error", 
                "message": f"不支持的查询类型: {query_type}。支持: image, audio, text"
            }), 400

        # 2. 簇选择
        _run_external_script(
            "簇选择 (client_pick_clusters)",
            [
                CLIENT_PICK_CLU,
                "--q", TEMP_Q_NPY,
                "--centers", centers_file,
                "--topT", str(TOP_T),
                "--L", str(L),
                "--session", session_id,
                "--metric", "cos",
                "--out", TEMP_CLUSTERS_TXT
            ]
        )
        
        with open(TEMP_CLUSTERS_TXT, 'r') as f:
            clusters_list = f.read().strip()
        print(f"[*] 选择的簇: {clusters_list}")

        # 3. 加密查询生成
        _run_external_script(
            "加密查询 (ckks_make_ctq)",
            [
                CKKS_MAKE_CTQ,
                "--context", CONTEXT_FILE,
                "--pk", PK_FILE,
                "--npy", TEMP_Q_NPY,
                "--dim", str(DIM),
                "--out", CT_Q_BIN
            ]
        )
        
        # 3.5 上传评估密钥
        try:
            _run_external_script("上传评估密钥", [UPLOAD_EVALKEYS])
        except RuntimeError as e:
            print(f"[WARNING] 评估密钥上传失败: {e}")

        # 4. 执行密态搜索
        env = os.environ.copy()
        env.update({
            'GATEWAY_ADDR': GATEWAY_GRPC_ADDR,
            'CKKS_DIR': str(CKKS_DIR),
            'CLUSTERS': clusters_list,
            'TOP_R': str(TOP_R),
            'SESSION_ID': session_id,
            'KEY_VER': key_ver
        })
        
        _run_external_script(
            "密态搜索 (search_real_ct)",
            [SEARCH_REAL_CT, "--mode", search_mode],
            env=env
        )
        
        # 5. 解密
        _run_external_script(
            "解密 (ckks_decrypt_dump)",
            [
                CKKS_DECRYPT,
                "--context", CONTEXT_FILE,
                "--sk", SK_FILE,
                "--scores_dir", SCORES_DIR,
                "--dim", str(DIM)
            ]
        )

        # 6. Top-K 分析 - 修复：读取带分数的文件
        temp_topk_json_file = SCORES_DIR / f"{session_id}_topk.json"
        temp_topk_with_scores_json_file = SCORES_DIR / f"{session_id}_topk_with_scores.json"

        analyze_cmd = [
            ANALYZE_TOPK,
            "--scores_dir", SCORES_DIR,
            "--slot_ids", slot_ids_file,
            "--pack_slots", "4096",
            "--topk", "20",
            "--save_json", str(temp_topk_json_file),           # 原有文件（仅ID）
            "--save_scores_json", str(temp_topk_with_scores_json_file),  # 新增文件（带分数）
            "--mode", search_mode
        ]

        if search_mode == "audio":
            analyze_cmd.extend(["--id_map_json", AUDIO_ID_MAP])
        elif search_mode == "image":
            analyze_cmd.extend(["--img_group", "12"])

        _run_external_script("Analyze Top-K", analyze_cmd)

        # 修复：读取带分数的结果文件，而不是旧文件
        if not temp_topk_with_scores_json_file.exists():
            raise RuntimeError(f"带分数的结果文件未生成: {temp_topk_with_scores_json_file}")

        with open(temp_topk_with_scores_json_file, 'r', encoding='utf-8') as f:
            final_result = json.load(f)

        # 计算总耗时
        t_end = time.time()
        final_result['e2e_latency_ms'] = int((t_end - t_start) * 1000)
        final_result['mode'] = search_mode
        
        print(f"[*] {search_mode} 流程完成，总耗时: {final_result['e2e_latency_ms']}ms")

        # 7. 自动调用 lookup 脚本（异步）
        try:
            if final_result.get('topk'):
                print(f"[*] 启动 {search_mode} lookup 脚本展示结果...")
                
                # 保存当前会话的 topk 结果供 lookup 使用
                session_topk_file = SCORES_DIR / f"{session_id}_lookup_topk.json"
                
                # 修复：lookup 脚本需要纯ID数组，从带分数的结果中提取
                id_list = [item['id'] for item in final_result['topk']]
                with open(session_topk_file, 'w') as f:
                    json.dump({"topk": id_list}, f, indent=2)
                
                # 设置 lookup 环境变量
                lookup_env = os.environ.copy()
                lookup_env['TOPK_JSON'] = str(session_topk_file)
                lookup_env['SESSION_ID'] = session_id
                
                # 选择对应的 lookup 脚本
                lookup_configs = {
                    "audio": (AUDIO_LOOKUP, "音频文件查找展示"),
                    "image": (IMAGE_LOOKUP, "图像文件查找展示"), 
                    "text": (TEXT_LOOKUP, "文本内容查找展示")
                }
                
                if search_mode in lookup_configs:
                    lookup_script, script_name = lookup_configs[search_mode]
                    if search_mode == "text":
                        import requests
                        print("\n[LOOKUP] 文本内容查找展示：使用 /api/text-content 统一逻辑...\n")

                        for item in id_list[:5]:   # 展示前5个
                            url = f"http://127.0.0.1:8081/api/text-content/{item}"
                            try:
                                r = requests.get(url)
                                if r.status_code == 200:
                                    content = r.json().get("content", "(no content)")
                                    print("=" * 80)
                                    print(f"[{item}]")
                                    print(content[:2000])  # 显示部分，防止太长
                                    print("=" * 80 + "\n")
                                else:
                                    print(f"[LOOKUP ERROR] {item}: HTTP {r.status_code}")
                            except Exception as e:
                                print(f"[LOOKUP ERROR] {item}: {e}")

                        print("[LOOKUP] 文本 lookup 展示完成（与前端完全一致）。")
                        


                    if lookup_script.exists():
                        # 使用线程异步执行 lookup（不阻塞HTTP响应）
                        import threading
                        
                        def run_lookup_background():
                            try:
                                print(f"[LOOKUP] 开始执行 {script_name}...")
                                _run_external_script(
                                    script_name,
                                    [lookup_script],
                                    check=False,  # 不检查返回值
                                    env=lookup_env
                                )
                                print(f"[LOOKUP] {script_name} 执行完成")
                            except Exception as e:
                                print(f"[LOOKUP ERROR] {e}")
                        
                        lookup_thread = threading.Thread(target=run_lookup_background)
                        lookup_thread.daemon = True  # 设置为守护线程
                        lookup_thread.start()
                        
                        print(f"[*] {script_name} 已在后台启动，请查看展示窗口...")
                    else:
                        print(f"[WARNING] {search_mode} lookup 脚本不存在: {lookup_script}")
                
                # 记录需要清理的文件
                TEMP_FILES_TO_CLEAN.append(session_topk_file)
                TEMP_FILES_TO_CLEAN.append(temp_topk_json_file)
                TEMP_FILES_TO_CLEAN.append(temp_topk_with_scores_json_file)  # 新增清理项
                
        except Exception as e:
            print(f"[WARNING] 自动调用 lookup 脚本失败: {e}")

        # 8. 返回最终结果
        return jsonify({
            "status": "success", 
            "message": f"{search_mode} 密态近邻检索完成。",
            "result": final_result
        })

    except RuntimeError as e:
        print(f"[ERROR] 流程编排失败: {e}", file=sys.stderr)
        return jsonify({
            "status": "error",
            "message": f"密态检索后端执行失败: {e}"
        }), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[CRITICAL ERROR] 发生未捕获的错误: {e}", file=sys.stderr)
        return jsonify({
            "status": "error",
            "message": f"服务器内部错误: {e}"
        }), 500
    finally:
        #_cleanup_temp_files()
        pass
@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        "status": "healthy",
        "service": "Encrypted Search HTTP Adapter",
        "timestamp": time.time()
    })
@app.route('/api/files/<path:filename>')
def serve_file(filename):
    """代理服务器文件给前端"""
    try:
        # 安全性检查：确保文件在允许的目录内
        safe_base_paths = [
            "/media/wen/F500/demo_images",
            "/media/wen/F500/audio_raw",
            "/home/wen/Desktop/backend/data/texts"
        ]
        
        full_path = None
        for base_path in safe_base_paths:
            potential_path = Path(base_path) / filename
            if potential_path.exists():
                full_path = potential_path
                break
        
        if not full_path:
            return "File not found", 404
            
        # 根据文件类型设置正确的 MIME type
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            return send_file(full_path, mimetype='image/jpeg')
        elif filename.endswith('.wav') or filename.endswith('.mp3'):
            return send_file(full_path, mimetype='audio/wav')
        elif filename.endswith('.txt'):
            return send_file(full_path, mimetype='text/plain')
        else:
            return send_file(full_path)
            
    except Exception as e:
        return str(e), 500

@app.route('/api/file-info/<path:filename>')
def get_file_info(filename):
    """获取文件信息（大小、类型等）"""
    try:
        safe_base_paths = [
            "/home/wen/pics40",
            "/media/wen/F500/audio_raw", 
            "/home/wen/Desktop/backend/data/texts"
        ]
        
        full_path = None
        for base_path in safe_base_paths:
            potential_path = Path(base_path) / filename
            if potential_path.exists():
                full_path = potential_path
                break
        
        if not full_path or not full_path.exists():
            return jsonify({"error": "File not found"}), 404
            
        return jsonify({
            "filename": filename,
            "path": str(full_path),
            "size": full_path.stat().st_size,
            "exists": True
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/text-content/<item_id>', methods=['GET'])
def get_text_by_id(item_id):
    """
    给前端返回原始文本内容（和 text_lookup.py 一样的逻辑）
    """
    try:
        import pandas as pd

        CASES_PARQUET = "/media/wen/F500/text_embeddings/cases.parquet"

        if not os.path.exists(CASES_PARQUET):
            return jsonify({"error": "文本数据库文件不存在"}), 404

        df = pd.read_parquet(CASES_PARQUET)

        # --- 工具函数 ---
        def normalize_pmc_id(c):
            if c.startswith("pmc="):
                return c.split("=", 1)[1]
            return c

        def extract_text_from_case(case):
            parts = []
            if isinstance(case.get("case_summary"), list):
                parts.append("Case Summary:\n" + "\n".join(case["case_summary"]))

            for key in ["finding", "impression", "history", "background", "text"]:
                if key in case and isinstance(case[key], str) and case[key].strip():
                    parts.append(f"{key.capitalize()}:\n{case[key]}")

            if not parts:
                parts.append(str(case))

            return "\n\n".join(parts)

        # --- 解析 ID ---
        cid_full = normalize_pmc_id(item_id)      # PMC5015624_01 
        article_id = cid_full.split("_")[0]       # PMC5015624

        # --- 查找 article ---
        rows = df[df["article_id"] == article_id]
        if len(rows) == 0:
            return jsonify({"error": f"未找到 article_id: {article_id}"}), 404

        cases = rows.iloc[0]["cases"]  # list of dict

        # --- 查找具体 case ---
        target = None
        for c in cases:
            if c.get("case_id") == cid_full:
                target = c
                break

        if target is None:
            return jsonify({"error": f"未找到 case_id: {cid_full}"}), 404

        # --- 提取内容 ---
        text_content = extract_text_from_case(target)

        return jsonify({
            "id": item_id,
            "content": text_content,
            "article_id": article_id,
            "case_id": cid_full
        })

    except Exception as e:
        return jsonify({"error": f"获取文本内容失败: {str(e)}"}), 500

if __name__ == '__main__':
    try:
        _check_permissions() 
        print(f"Search Service (HTTP Adapter) 正在 127.0.0.1:8081 上启动...")
        app.run(host='127.0.0.1', port=8081, debug=True)
    except RuntimeError as e:
        print(f"\nFATAL: {e}")
        sys.exit(1)
