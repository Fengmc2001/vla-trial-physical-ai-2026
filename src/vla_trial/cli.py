import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="尝试VLA; defaults to CPU-only preparation")
    parser.add_argument("--project-root", default=None)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("doctor", help="Read package/model metadata without loading model or querying GPU")
    ui = subs.add_parser("ui", help="Start loopback preparation UI; never starts a GPU runtime")
    ui.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    ui.add_argument("--port", type=int, default=7866)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from .config import load_config
    config = load_config(args.project_root)
    if args.command == "doctor":
        from .diagnostics import diagnose
        print(json.dumps(diagnose(config), ensure_ascii=False, indent=2))
    elif args.command == "ui":
        from .ui import build_ui
        build_ui(config).launch(server_name=args.host, server_port=args.port, share=False,
                                inbrowser=False, show_error=False, ssr_mode=False)
