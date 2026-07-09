from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from credit_card_approval.modeling import train_best_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save the credit card approval model.")
    parser.add_argument("--data", help="Optional CSV dataset path. If omitted, sample data is used.", default=None)
    parser.add_argument("--sample-size", help="Sample rows to generate when no dataset is supplied.", type=int, default=3200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_best_model(data_path=args.data, sample_size=args.sample_size)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

