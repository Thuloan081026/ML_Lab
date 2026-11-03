from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app, load_models


def main() -> None:
    load_models()
    client = app.test_client()
    payload = {
        "Purchase_Year": 2024,
        "Purchase_Month": 12,
        "Purchase_Day": 24,
        "Purchase_DayOfWeek": 1,
        "Purchase_IsWeekend": 0,
    }
    response = client.post("/predict", json=payload)
    if response.status_code != 200:
        raise SystemExit(f"Lỗi dự đoán: {response.get_json()}")
    result = response.get_json()
    print("Dữ liệu đầu vào:")
    print(pd.Series(payload).to_string())
    print("Kết quả:")
    print(f"Số giao dịch dự báo: {result['predicted_daily_volume']:.2f}")


if __name__ == "__main__":
    main()
