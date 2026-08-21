"""Entry point for model evaluation.

Usage:
    python jobs/evaluate.py --config_path configs/sft_phi3_config.yaml --model_path /path/to/model
    python jobs/evaluate.py --config_path configs/sft_phi3_config.yaml --run_id abc123
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Support running this file straight from a repo checkout (Databricks Repos,
# ``python jobs/evaluate.py``) as well as from an installed package.  Only the
# project root goes on the path — adding ``src/`` too would make both
# ``import config`` and ``import src.config`` resolve, to two different module
# objects.  Runtime dependencies are installed from main(), not at import time.
try:
    _this_file = Path(__file__).resolve()
except NameError:  # Databricks spark_python_task exec() context
    _this_file = Path(sys.argv[0]).resolve() if sys.argv else Path(os.getcwd())

_PROJECT_ROOT = _this_file.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    from src.utils.environment import ensure_runtime_requirements

    ensure_runtime_requirements(_PROJECT_ROOT / "requirements_runtime.txt")

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
