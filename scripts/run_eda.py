from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from credit_card_approval.visualization import build_eda_charts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exploratory data analysis charts.")
    parser.add_argument("--data", help="Optional CSV dataset path. If omitted, sample data is used.", default=None)
    return parser.parse_args()


def main() -> None:
    paths = build_eda_charts(data_path=parse_args().data)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

