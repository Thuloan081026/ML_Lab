from pathlib import Path
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "ecommerce_dataset_updated.csv"
OUTPUT = ROOT / "data" / "processed" / "data_summary.csv"

def main() -> None:
    df = pd.read_csv(SOURCE)
    summary = pd.DataFrame(
        {
            "rows": [len(df)],
            "columns": [len(df.columns)],
            "missing_values": [int(df.isna().sum().sum())],
            "duplicate_rows": [int(df.duplicated().sum())],
            "date_min": [df["Purchase_Date"].min()],
            "date_max": [df["Purchase_Date"].max()],
            "target_mean": [float(df["Final_Price(Rs.)"].mean())],
        }
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)
    print(summary.to_string(index=False))
    print(f"Đã lưu tóm tắt dữ liệu: {OUTPUT}")

if __name__ == "__main__":
    main()
