from pathlib import Path
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "ecommerce_features.csv"


def main() -> None:
    if not FEATURES.exists():
        raise SystemExit(
            "Chưa có dữ liệu đặc trưng. Hãy chạy notebook "
            "02_feature_engineering.ipynb trước."
        )
    df = pd.read_csv(FEATURES)
    print(f"Số dòng: {len(df)}")
    print(f"Số cột đặc trưng: {len(df.columns)}")
    print("Giá trị thiếu:", int(df.isna().sum().sum()))
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
