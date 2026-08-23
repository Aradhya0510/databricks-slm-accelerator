"""Entry point for endpoint monitoring.

Usage:
    python jobs/monitor.py --endpoint_name phi3-sft-endpoint --hours 24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Support running this file straight from a repo checkout (Databricks Repos,
# ``python jobs/monitor.py``) as well as from an installed package.  Only the
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
    parser = argparse.ArgumentParser(description="Monitor a deployed SLM endpoint")
    parser.add_argument("--endpoint_name", type=str, required=True)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None,
                        help="Pipeline config supplying the monitoring thresholds")
    parser.add_argument("--fail_on_breach", action="store_true",
                        help="Exit non-zero when a threshold is breached, so a "
                             "scheduled job alerts instead of silently passing")
    args = parser.parse_args()

    from src.monitoring.endpoint_monitor import EndpointMonitor

    thresholds = None
    if args.config_path:
        from src.config.schema import load_config

        thresholds = load_config(args.config_path).monitoring

    monitor = EndpointMonitor(args.endpoint_name, thresholds=thresholds)
    report = monitor.generate_report(output_path=args.output_path)

    print(json.dumps(report, indent=2, default=str))

    breaches = report.get("threshold_breaches", [])
    if breaches and args.fail_on_breach:
        for b in breaches:
            print(f"BREACH {b['metric']}: {b['value']} exceeds {b['threshold']}")
        return 1
    return 0


if __name__ == "__main__":
    # Only a breach exits non-zero. A bare sys.exit(0) is reported as a task
    # failure when this file is exec'd as a Databricks Python task, which made
    # every healthy monitoring run look like a failed job.
    _status = main()
    if _status:
        sys.exit(_status)
