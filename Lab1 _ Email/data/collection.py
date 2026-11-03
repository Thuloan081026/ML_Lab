import sys
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "emails.csv"

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"text", "spam"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset thiếu cột bắt buộc: {sorted(missing)}")
    if frame[["text", "spam"]].isnull().any().any():
        raise ValueError("Dataset chứa giá trị rỗng trong cột text hoặc spam.")
    return frame

if __name__ == "__main__":
    data = load_raw_data()
    print(f"Đã nạp {len(data):,} email từ {RAW_DATA_PATH.name}")
    print(data["spam"].value_counts().sort_index().to_string())
