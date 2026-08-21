"""Entry point for model registration and deployment.

Usage:
    python jobs/deploy.py --config_path configs/sft_phi3_config.yaml --run_id abc123 \
        --model_name catalog.schema.phi3_sft --endpoint_name phi3-sft-endpoint
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Support running this file straight from a repo checkout (Databricks Repos,
# ``python jobs/deploy.py``) as well as from an installed package.  Only the
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

    parser = argparse.ArgumentParser(description="Register and deploy a fine-tuned SLM")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True,
                        help="UC model name: catalog.schema.model")
    parser.add_argument("--endpoint_name", type=str, required=True)
    parser.add_argument("--model_uri", type=str, default=None)
    parser.add_argument("--skip_deploy", action="store_true")
    args = parser.parse_args()

    from src.config.schema import load_config
    from src.serving.registration import register_model
    from src.serving.deployment import deploy_endpoint, wait_for_ready

    config = load_config(args.config_path)

    print("Registering model...")
    reg_result = register_model(
        run_id=args.run_id,
        registered_model_name=args.model_name,
        task_type=config.model.task_type,
        model_uri=args.model_uri,
    )
    print(f"Model version: {reg_result['model_version']}")

    if not args.skip_deploy:
        print("Deploying endpoint...")
        deploy_endpoint(
            endpoint_name=args.endpoint_name,
            registered_model_name=args.model_name,
            model_version=str(reg_result["model_version"]),
            workload_size=config.serving.workload_size,
            workload_type=config.serving.workload_type,
            scale_to_zero=config.serving.scale_to_zero,
        )
        wait_for_ready(args.endpoint_name)


if __name__ == "__main__":
    main()
