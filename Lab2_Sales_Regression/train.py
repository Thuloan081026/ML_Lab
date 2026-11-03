"""
Lab 2: Pipeline Huấn Luyện và Đánh Giá Dự Báo Doanh Số (Daily Sales Volume)
Triển khai 100% From Scratch (Zero Scikit-Learn) bằng NumPy và Pandas.
"""

import os
import sys
from pathlib import Path

# Cấu hình UTF-8 cho stdout/stderr trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Alias để bảo đảm tính tương thích với các module pickle
import models.algorithms.decision_tree as _dt_module
import models.algorithms.preprocessing as _prep_module
sys.modules.setdefault("models.decision_tree", _dt_module)
sys.modules.setdefault("models.preprocessing", _prep_module)

from models.algorithms.linear_regression import LinearRegression
from models.algorithms.linear_regression_gd import LinearRegressionGD
from models.algorithms.decision_tree import DecisionTreeRegressor
from models.algorithms.preprocessing import time_series_train_val_test_split, StandardScaler
from models.algorithms.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from scripts.features.build_features import extract_date_features


def prepare_daily_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Gom nhóm theo ngày, đếm số giao dịch (Daily Volume) và tạo đặc trưng thời gian."""
    df_daily = df.groupby("Purchase_Date").size().reset_index(name="Daily_Volume")
    df_daily["Purchase_Date_Str"] = df_daily["Purchase_Date"].dt.strftime("%d-%m-%Y")
    df_features = extract_date_features(df_daily, date_column="Purchase_Date_Str")
    return df_features.drop(columns=["Purchase_Date", "Purchase_Date_Str"])


def main():
    print("=" * 75)
    print("LAB 2: PIPELINE DỰ BÁO DOANH SỐ SẢN PHẨM (TIME-SERIES FORECASTING)")
    print("=" * 75)

    # 1. Nạp và kiểm tra dữ liệu thô
    raw_path = PROJECT_ROOT / "data" / "raw" / "ecommerce_dataset_updated.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {raw_path}")

    df_raw = pd.read_csv(raw_path)
    print(f"[*] Đã nạp {len(df_raw):,} bản ghi giao dịch từ {raw_path.name}")
    print(f"[*] Kiểm tra giá trị khuyết: {df_raw.isna().sum().sum()} giá trị NaN")

    # Ép kiểu ngày tháng và sắp xếp theo trình tự thời gian
    df_raw["Purchase_Date"] = pd.to_datetime(df_raw["Purchase_Date"], format="%d-%m-%Y", dayfirst=True)
    df_raw = df_raw.sort_values(by="Purchase_Date").reset_index(drop=True)
    date_min = df_raw["Purchase_Date"].min().strftime("%d-%m-%Y")
    date_max = df_raw["Purchase_Date"].max().strftime("%d-%m-%Y")
    print(f"[*] Khoảng thời gian: từ {date_min} đến {date_max}")

    # 2. Phân chia chuỗi thời gian (Train 70%, Val 15%, Test 15%)
    print("\n" + "=" * 75)
    print("BƯỚC 1: PHÂN CHIA DỮ LIỆU THEO THỜI GIAN (CHRONOLOGICAL SPLIT)")
    print("=" * 75)
    df_train_raw, df_val_raw, df_test_raw = time_series_train_val_test_split(
        df_raw, val_size=0.15, test_size=0.15
    )

    # 3. Tạo Daily Volume và đặc trưng thời gian
    df_train_features = prepare_daily_volume(df_train_raw)
    df_val_features = prepare_daily_volume(df_val_raw)
    df_test_features = prepare_daily_volume(df_test_raw)

    print(f"[*] Số ngày Train: {len(df_train_features)}")
    print(f"[*] Số ngày Validation: {len(df_val_features)}")
    print(f"[*] Số ngày Test: {len(df_test_features)}")

    # Lưu dữ liệu đã xử lý
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    df_train_features.to_csv(processed_dir / "train_daily.csv", index=False)
    df_val_features.to_csv(processed_dir / "val_daily.csv", index=False)
    df_test_features.to_csv(processed_dir / "test_daily.csv", index=False)

    # 4. Tách X, y và Chuẩn hóa StandardScaler
    target_col = "Daily_Volume"
    X_train = df_train_features.drop(columns=[target_col])
    y_train = df_train_features[target_col]
    X_val = df_val_features.drop(columns=[target_col])
    y_val = df_val_features[target_col]
    X_test = df_test_features.drop(columns=[target_col])
    y_test = df_test_features[target_col]

    numeric_cols = ["Purchase_Year", "Purchase_Month", "Purchase_Day", "Purchase_DayOfWeek"]

    scaler = StandardScaler()
    scaler.fit(X_train[numeric_cols])

    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # 5. Huấn luyện các mô hình From-Scratch
    print("\n" + "=" * 75)
    print("BƯỚC 2: HUẤN LUYỆN CÁC MÔ HÌNH HỒI QUY TỰ LẬP TRÌNH (FROM SCRATCH)")
    print("=" * 75)

    models = {
        "Linear Regression (Normal Eq)": LinearRegression(),
        "Linear Regression (Gradient Descent)": LinearRegressionGD(learning_rate=0.01, epochs=1000, alpha=0.01),
        "Decision Tree Regressor (Depth=4)": DecisionTreeRegressor(max_depth=4, min_samples_split=2),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)

        # Dự đoán Val & Test
        val_pred = model.predict(X_val_scaled)
        test_pred = model.predict(X_test_scaled)

        val_mae = mean_absolute_error(y_val, val_pred)
        val_rmse = root_mean_squared_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)

        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = root_mean_squared_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)

        results.append({
            "Model": name,
            "Val_MAE": round(val_mae, 3),
            "Val_RMSE": round(val_rmse, 3),
            "Val_R2": round(val_r2, 4),
            "Test_MAE": round(test_mae, 3),
            "Test_RMSE": round(test_rmse, 3),
            "Test_R2": round(test_r2, 4),
        })

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    # Lưu bảng kết quả
    res_df.to_csv(processed_dir / "evaluation_results.csv", index=False)

    # 6. Lưu trữ mô hình triển khai
    print("\n" + "=" * 75)
    print("BƯỚC 3: ĐÓNG GÓI VÀ LƯU TRỮ MÔ HÌNH (.PKL)")
    print("=" * 75)
    save_dir = PROJECT_ROOT / "models" / "save_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    models["Linear Regression (Normal Eq)"].save(str(save_dir / "linear_regression.pkl"))
    models["Linear Regression (Gradient Descent)"].save(str(save_dir / "linear_regression_gd.pkl"))
    models["Decision Tree Regressor (Depth=4)"].save(str(save_dir / "decision_tree.pkl"))
    scaler.save(str(save_dir / "scaler.pkl"))
    print(f"[+] Đã lưu thành công các mô hình và scaler vào thư mục: {save_dir}")

    # 7. Trực quan hóa kết quả dự báo trên Test Set
    try:
        figures_dir = PROJECT_ROOT / "reports" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        days = np.arange(len(y_test))
        ax.plot(days, y_test.values, label="Thực tế (Ground Truth)", color="#111827", linewidth=2, marker="o")
        ax.plot(days, models["Linear Regression (Normal Eq)"].predict(X_test_scaled),
                label="Linear Regression", color="#2563eb", linestyle="--", linewidth=1.8)
        ax.plot(days, models["Decision Tree Regressor (Depth=4)"].predict(X_test_scaled),
                label="Decision Tree", color="#10b981", linestyle="-.", linewidth=1.8)

        ax.set_title("So sánh Dự báo Sản lượng Bán hàng trên Tập Test (Daily Volume)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Chỉ số Ngày Kiểm thử (Test Days)", fontsize=11)
        ax.set_ylabel("Số lượng đơn hàng (Daily Volume)", fontsize=11)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)

        fig_path = figures_dir / "timeseries_forecast.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"[+] Đã lưu biểu đồ dự báo: {fig_path}")
    except Exception as e:
        print(f"[!] Bỏ qua lưu biểu đồ: {e}")

    print("\n" + "=" * 75)
    print("HOÀN TẤT HUẤN LUYỆN LAB 2 THÀNH CÔNG!")
    print("=" * 75)


if __name__ == "__main__":
    main()

