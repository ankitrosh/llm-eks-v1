#!/usr/bin/env python3
"""Inspect a local Hugging Face config.json without loading weights."""

import argparse
import json
from pathlib import Path

from llm_runtime.model_metadata import from_transformers_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    path = Path(args.config)

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    metadata = from_transformers_config(
        args.model_id, config
    )

    print(json.dumps(metadata.to_dict(), indent=2))

    print("\nVALIDATION")
    print("PASS: model metadata parsed")
    print("NOT YET VALIDATED: partitioning")
    print("NOT YET VALIDATED: optimization backends")
    print("NOT YET VALIDATED: GPU memory requirements")


if __name__ == "__main__":
    main()
