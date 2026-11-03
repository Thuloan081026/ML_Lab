"""
Lab 2: Đánh giá mô hình dự báo doanh số trên tập kiểm thử độc lập (Test Set).
Nạp trực tiếp các mô hình đã lưu trong thư mục models/save_model/ để kiểm định.
"""

import sys
from pathlib import Path

# Cấu hình UTF-8 cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models.algorithms.decision_tree as _dt_module
import models.algorithms.preprocessing as _prep_module
sys.modules.setdefault("models.decision_tree", _dt_module)
sys.modules.setdefault("models.preprocessing", _prep_module)

from models.algorithms.decision_tree import DecisionTreeRegressor
from models.algorithms.linear_regression import LinearRegression
from models.algorithms.preprocessing import StandardScaler
from models.algorithms.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def main() -> None:
    test_csv = ROOT / "data" / "processed" / "test_daily.csv"
    save_dir = ROOT / "models" / "save_model"

    dt_path = save_dir / "decision_tree.pkl"
    lr_path = save_dir / "linear_regression.pkl"
    scaler_path = save_dir / "scaler.pkl"

    if not test_csv.exists() or not dt_path.exists() or not scaler_path.exists():
        print("[!] Thiếu dữ liệu test hoặc mô hình đã lưu. Đang tự động chạy pipeline huấn luyện...")
        from train import main as run_train
        run_train()

    print("=" * 75)
    print("LAB 2: ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM THỬ ĐỘC LẬP (TEST SET)")
    print("=" * 75)

    df_test = pd.read_csv(test_csv)
    print(f"[*] Đã nạp tập Test: {len(df_test)} ngày từ {test_csv.name}")

    target_col = "Daily_Volume"
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # Nạp Scaler
    scaler = StandardScaler().load(str(scaler_path))
    numeric_cols = ["Purchase_Year", "Purchase_Month", "Purchase_Day", "Purchase_DayOfWeek"]
    X_test_scaled = X_test.copy()
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Nạp các mô hình
    models = {}
    if lr_path.exists():
        models["Linear Regression (Normal Eq)"] = LinearRegression().load(str(lr_path))
    if dt_path.exists():
        models["Decision Tree Regressor"] = DecisionTreeRegressor().load(str(dt_path))

    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Mô hình": name,
            "Số mẫu Test": len(y_test),
            "MAE (Đơn hàng/ngày)": f"{mae:.3f}",
            "RMSE": f"{rmse:.3f}",
            "R2 Score": f"{r2:.4f}",
        })

    eval_table = pd.DataFrame(results)
    print("-" * 75)
    print(eval_table.to_string(index=False))
    print("=" * 75)
    print("[+] Đánh giá mô hình Lab 2 hoàn tất thành công!")

if __name__ == "__main__":
    main()
