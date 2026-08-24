import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="检查 WSL GPU 运行环境和关键路径")
    parser.add_argument("--csv-path", type=str, help="可选：检查 CSV 文件是否存在")
    parser.add_argument("--data-root", type=str, help="可选：检查数据根目录是否存在")
    parser.add_argument("--model-weight", type=str, help="可选：检查模型权重文件是否存在")
    parser.add_argument("--out-dir", type=str, help="可选：检查输出目录是否可创建")
    return parser.parse_args()


def check_path(path_str: str, name: str, must_exist: bool = True):
    path = Path(path_str)
    if path.exists() != must_exist:
        if must_exist:
            print(f"[ERROR] {name} 不存在: {path}")
            return False
        print(f"[WARN] {name} 不存在: {path}")
        return False
    print(f"[OK] {name}: {path}")
    return True


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    print("=== WSL GPU 环境检查 ===")
    print("Project root:", project_root)

    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch 未安装。")
        sys.exit(1)

    print("PyTorch version:", torch.__version__)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        try:
            print("GPU name:", torch.cuda.get_device_name(0))
        except Exception as exc:
            print("[WARN] 获取 GPU 名称失败：", exc)

    print("\n=== nvidia-smi 检查 ===")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("nvidia-smi 可调用，输出：")
            print(result.stdout.strip().splitlines()[0] if result.stdout else "<无输出>")
        else:
            print("[ERROR] nvidia-smi 调用失败，返回码：", result.returncode)
            print(result.stderr.strip())
    except FileNotFoundError:
        print("[ERROR] nvidia-smi 未找到，请确认 WSL 已安装 NVIDIA 驱动。")

    print("\n=== 关键项目文件检查 ===")
    checks = [
        project_root / "embeding(1)" / "audio_embed.py",
        project_root / "embeding(1)" / "image_embed_local.py",
        project_root / "requirements-wsl.txt",
    ]
    for path in checks:
        if path.exists():
            print(f"[OK] {path}")
        else:
            print(f"[ERROR] {path} 不存在")

    if args.csv_path:
        check_path(args.csv_path, "CSV 文件")
    if args.data_root:
        check_path(args.data_root, "数据根目录")
    if args.model_weight:
        check_path(args.model_weight, "模型权重文件")
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.exists():
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                print(f"[OK] 创建输出目录: {out_dir}")
            except Exception as exc:
                print(f"[ERROR] 无法创建输出目录 {out_dir}: {exc}")
        else:
            print(f"[OK] 输出目录: {out_dir}")

    print("\n检查完成。")


if __name__ == "__main__":
    main()
