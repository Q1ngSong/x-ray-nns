"""Command-line entry point: inspect a local CLIP, record one CLIP run, or serve recorded runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from xray.adapters import ClipAdapter


def main(argv: list[str] | None = None) -> int:
    """Run a supported trace command. [主线]

    Args:
        argv: list[str] | None — command arguments, or `sys.argv[1:]` when omitted.
    Returns:
        int — process exit status.

    变更: 2026-09-21 增加 `clip-run` 命令，将本地 CLIP forward 导出为离线 HTML bundle。
    变更: 2026-09-23 增加 `serve` 命令：本地服务浏览各次运行，并从页面提交新的图片和 prompts。
    变更: 2026-09-23 删除 `mock` 命令（手工构造的示例 trace）；导出格式由 pytest 覆盖。
    """
    parser = argparse.ArgumentParser(prog="xray")
    parser.add_argument("--version", action="version", version="0.1.0")
    subparsers = parser.add_subparsers(dest="command")
    clip_info = subparsers.add_parser("clip-info", help="inspect a local Hugging Face CLIP config")
    clip_info.add_argument("model_path", type=Path)
    clip_run = subparsers.add_parser("clip-run", help="run local CLIP once and export an HTML trace")
    clip_run.add_argument("--model", dest="model_path", type=Path, required=True)
    clip_run.add_argument("--image", dest="image_path", type=Path, required=True)
    clip_run.add_argument("--text", action="append", required=True, help="text prompt; repeat for a batch")
    clip_run.add_argument("--output", type=Path, default=Path("runs/clip/latest"))
    clip_run.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    clip_run.add_argument("--no-raw", action="store_true", help="keep metadata/previews but skip .pt tensors")
    clip_run.add_argument("--trace-id", default="clip-run")
    serve =subparsers.add_parser("serve", help="browse recorded runs and record new ones from the page")
    serve.add_argument("--model", dest="model_path", type=Path, required=True)
    serve.add_argument("--runs", type=Path, default=Path("runs"), help="runs root; new runs go to <runs>/clip/live/")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args(argv)
    if args.command == "clip-info":
        import json

        print(json.dumps(ClipAdapter(args.model_path).describe(), ensure_ascii=False, indent=2))
    elif args.command == "clip-run":
        from xray.recorder.clip import run_clip

        bundle = run_clip(
            args.model_path,
            args.image_path,
            args.text,
            args.output,
            device=args.device,
            save_raw=not args.no_raw,
            trace_id=args.trace_id,
        )
        print(f"wrote {bundle / 'index.html'}")
    elif args.command == "serve":
        from xray.server import serve as serve_runs

        serve_runs(args.model_path, args.runs, port=args.port, device=args.device)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
