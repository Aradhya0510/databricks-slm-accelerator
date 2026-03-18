"""Entry point for SLM fine-tuning.

Usage (single-node, native DDP):
    python jobs/train.py --config_path configs/sft_phi3_config.yaml
    python jobs/train.py --config_path configs/sft_phi3_config.yaml --num_gpus 4

Usage (multi-node via TorchDistributor):
    python jobs/train.py --config_path configs/sft_phi3_config.yaml --distributed torchd
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    _this_file = Path(__file__).resolve()
except NameError:
    _this_file = Path(sys.argv[0]).resolve() if sys.argv else Path(os.getcwd())

_project_root = _this_file.parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))

_runtime_reqs = _project_root / "requirements_runtime.txt"
if _runtime_reqs.exists():
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(_runtime_reqs)],
        stdout=subprocess.DEVNULL,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a small language model")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs (auto-detected if omitted)")
    parser.add_argument("--distributed", type=str, default="native",
                        choices=["native", "torchd"],
                        help="Distribution strategy")
    args = parser.parse_args()

    from src.config.schema import load_config
    from src.engine import TrainingEngine

    config = load_config(args.config_path)
    engine = TrainingEngine(config)
    metrics = engine.train(num_gpus=args.num_gpus, distributed_mode=args.distributed)

    print("\nTraining complete. Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
