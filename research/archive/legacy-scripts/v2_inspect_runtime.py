#!/usr/bin/env python3
"""Print runtime capabilities as JSON."""

import json

from llm_runtime.runtime_discovery import inspect_runtime


def main():
    report = inspect_runtime()

    print(json.dumps(report.to_dict(), indent=2))

    print("\nRUNTIME INSPECTION")
    print("PASS: runtime inspection completed")

    if report.device_type != "cuda":
        print("INFO: CUDA GPU execution validation unavailable")
    else:
        print("INFO: CUDA GPU discovered")

    print("NOT YET VALIDATED: model-specific attention")
    print("NOT YET VALIDATED: actual GPU attention kernels")


if __name__ == "__main__":
    main()
