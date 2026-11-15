from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app, load_models

def main() -> None:
    load_models()
    client = app.test_client()

    print("\n" + "=" * 75)
    print("DEMO DỰ BÁO DOANH SỐ BÁN HÀNG QUA FLASK API (LAB 2)")
    print("=" * 75)

    eval_csv = ROOT / "data" / "processed" / "evaluation_results.csv"
    bench_md = ""
    if eval_csv.exists():
        df_eval = pd.read_csv(eval_csv)
        print("\n--- 1. BẢNG ĐÁNH GIÁ ĐỐI CHUẨN MÔ HÌNH TRÊN TẬP TEST (CÓ MAPE) ---")
        disp_cols = ["Model", "Test_MAE", "Test_RMSE", "Test_MAPE (%)", "Test_R2"]
        cols_exist = [c for c in disp_cols if c in df_eval.columns]
        df_disp = df_eval[cols_exist].copy()
        df_disp.rename(columns={
            "Model": "Mô hình",
            "Test_MAE": "MAE (đơn)",
            "Test_RMSE": "RMSE",
            "Test_MAPE (%)": "MAPE (%)",
            "Test_R2": "R2 Score"
        }, inplace=True)
        if "MAPE (%)" in df_disp.columns:
            df_disp["MAPE (%)"] = df_disp["MAPE (%)"].apply(lambda v: f"{v:.2f}%")
        print(df_disp.to_string(index=False))

        bench_md = "## 1. Bảng Đánh Giá Đối Chuẩn Các Mô Hình Trên Tập Test (Bao gồm MAPE)\n\n"
        bench_md += "| Mô hình | MAE (đơn) | RMSE | MAPE (%) | R2 Score |\n|---|---|---|---|---|\n"
        for _, row in df_disp.iterrows():
            bench_md += f"| **{row['Mô hình']}** | {row['MAE (đơn)']:.3f} | {row['RMSE']:.3f} | {row['MAPE (%)']} | {row['R2 Score']:.4f} |\n"
        bench_md += "\n"

    scenarios = [
        {
            "name": "Kịch bản 1: Ngày thường (Thứ Ba, 24/12/2024)",
            "payload": {
                "Purchase_Year": 2024,
                "Purchase_Month": 12,
                "Purchase_Day": 24,
                "Purchase_DayOfWeek": 1,
                "Purchase_IsWeekend": 0,
            }
        },
        {
            "name": "Kịch bản 2: Cuối tuần (Chủ Nhật, 29/12/2024)",
            "payload": {
                "Purchase_Year": 2024,
                "Purchase_Month": 12,
                "Purchase_Day": 29,
                "Purchase_DayOfWeek": 6,
                "Purchase_IsWeekend": 1,
            }
        }
    ]

    print("\n--- 2. THỰC NGHIỆM DỰ BÁO CÁC KỊCH BẢN THỜI GIAN THỰC ---")
    sc_md = "## 2. Kết Quả Dự Báo Các Kịch Bản Mẫu\n\n"
    for sc in scenarios:
        p = sc["payload"]
        p_dt = {**p, "model": "decision_tree"}
        res_dt = client.post("/predict", json=p_dt).get_json()

        p_lr = {**p, "model": "linear_regression"}
        res_lr = client.post("/predict", json=p_lr).get_json()

        day_info = f"Ngày {p['Purchase_Day']:02d}/{p['Purchase_Month']:02d}/{p['Purchase_Year']} | Thứ {p['Purchase_DayOfWeek']+2} | Weekend={p['Purchase_IsWeekend']}"
        print(f"\n{sc['name']}")
        print(f"Đầu vào: {day_info}")
        print(f"  -> Linear Regression (Ols) : {res_lr['predicted_daily_volume']:.2f} đơn hàng/ngày")
        print(f"  -> Decision Tree Regressor : {res_dt['predicted_daily_volume']:.2f} đơn hàng/ngày")

        sc_md += f"### {sc['name']}\n- **Đầu vào:** {day_info}\n"
        sc_md += f"- **Linear Regression:** `{res_lr['predicted_daily_volume']:.2f} đơn/ngày`\n"
        sc_md += f"- **Decision Tree:** `{res_dt['predicted_daily_volume']:.2f} đơn/ngày`\n\n"

    rep_dir = ROOT / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    with open(rep_dir / "demo_results.md", "w", encoding="utf-8") as f:
        f.write("# KẾT QUẢ THỰC NGHIỆM DEMO — LAB 2 (SALES FORECASTING)\n\n")
        f.write(bench_md)
        f.write(sc_md)
    print(f"\n[+] Đã xuất kết quả demo có MAPE tại: {rep_dir / 'demo_results.md'}")
    print("=" * 75)
    print("[+] Hoàn tất demo thành công!")

if __name__ == "__main__":
    main()
