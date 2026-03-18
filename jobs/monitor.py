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

try:
    _this_file = Path(__file__).resolve()
except NameError:
    _this_file = Path(sys.argv[0]).resolve() if sys.argv else Path(os.getcwd())

_project_root = _this_file.parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))


def main():
    parser = argparse.ArgumentParser(description="Monitor a deployed SLM endpoint")
    parser.add_argument("--endpoint_name", type=str, required=True)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    from src.monitoring.endpoint_monitor import EndpointMonitor

    monitor = EndpointMonitor(args.endpoint_name)
    report = monitor.generate_report(output_path=args.output_path)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
