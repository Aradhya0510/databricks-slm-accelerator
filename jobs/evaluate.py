"""Entry point for model evaluation.

Usage:
    python jobs/evaluate.py --config_path configs/sft_phi3_config.yaml --model_path /path/to/model
    python jobs/evaluate.py --config_path configs/sft_phi3_config.yaml --run_id abc123
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
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned SLM")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--model_uri", type=str, default=None)
    parser.add_argument("--eval_type", type=str, default="all",
                        choices=["perplexity", "generation", "benchmark", "all"])
    args = parser.parse_args()

    from src.config.schema import load_config
    from src.evaluation.engine import EvaluationEngine

    config = load_config(args.config_path)
    engine = EvaluationEngine(config)

    common = dict(model_path=args.model_path, run_id=args.run_id, model_uri=args.model_uri)

    if args.eval_type in ("perplexity", "all"):
        print("\n--- Perplexity ---")
        ppl = engine.evaluate_perplexity(**common)
        print(f"  Perplexity: {ppl['perplexity']:.2f}")

    if args.eval_type in ("generation", "all"):
        print("\n--- Generation Quality ---")
        gen = engine.evaluate_generation(**common)
        for g in gen["generations"][:3]:
            print(f"  Prompt: {g['prompt'][:80]}...")
            print(f"  Response: {g['response'][:120]}...\n")

    if args.eval_type in ("benchmark", "all"):
        print("\n--- Benchmark ---")
        bench = engine.benchmark(**common)
        print(f"  Tokens/sec: {bench['tokens_per_second']:.1f}")
        print(f"  Latency p50: {bench['latency_per_generation_ms']['p50']:.1f} ms")


if __name__ == "__main__":
    main()
