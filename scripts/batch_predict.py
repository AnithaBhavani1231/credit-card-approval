from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from credit_card_approval.prediction import load_model, predict_application


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-screen credit card applicants from a CSV file.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path with predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_payload = load_model()
    frame = pd.read_csv(args.input)
    rows = []

    for _, row in frame.iterrows():
        result = predict_application(row.to_dict(), model_payload)
        rows.append(
            {
                **row.to_dict(),
                "decision": result["decision"],
                "approval_probability": result["approval_probability"],
                "risk_level": result["risk_level"],
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()

