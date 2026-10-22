import argparse
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "celloptics"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    result = subprocess.run(
        [sys.executable, *sys.argv], env={**os.environ, "PYTHONHASHSEED": "0"}
    )
    raise SystemExit(result.returncode)
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(
        description="Whole-field multimodal cell recognition"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "task",
        choices=[
            "verify-data",
            "prepare",
            "screen",
            "select",
            "fusion",
            "proxy-models",
            "evaluate",
            "proxy-features",
            "proxy-classify",
            "proxy-evaluate",
            "intervene",
            "statistics",
            "representations",
            "gradcam",
        ],
    )
    args, extra = parser.parse_known_args()
    os.environ["MMCELL_OUTPUT_ROOT"] = str(args.output.resolve())
    import workflow

    workflow.dispatch(args.task, extra)


if __name__ == "__main__":
    main()
