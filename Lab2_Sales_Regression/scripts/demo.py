from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app, load_models

def main() -> None:
    load_models()
    client = app.test_client()

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

    print("\n" + "=" * 70)
    print("DEMO DỰ BÁO DOANH SỐ BÁN HÀNG QUA FLASK API (LAB 2)")
    print("=" * 70)

    md_lines = ["# KẾT QUẢ THỰC NGHIỆM DEMO — LAB 2 (SALES FORECASTING)\n\n"]
    for sc in scenarios:
        print(f"\n--- {sc['name']} ---")
        p = sc["payload"]

        # Dự báo bằng Decision Tree
        p_dt = {**p, "model": "decision_tree"}
        res_dt = client.post("/predict", json=p_dt).get_json()

        # Dự báo bằng Linear Regression
        p_lr = {**p, "model": "linear_regression"}
        res_lr = client.post("/predict", json=p_lr).get_json()

        print(f"Đầu vào: Ngày {p['Purchase_Day']:02d}/{p['Purchase_Month']:02d}/{p['Purchase_Year']} | Thứ {p['Purchase_DayOfWeek']+2} | Weekend={p['Purchase_IsWeekend']}")
        print(f"  -> Decision Tree Regressor : {res_dt['predicted_daily_volume']:.2f} đơn hàng/ngày")
        print(f"  -> Linear Regression (Ols) : {res_lr['predicted_daily_volume']:.2f} đơn hàng/ngày")
        md_lines.append(f"### {sc['name']}\n- **Đầu vào:** Ngày {p['Purchase_Day']:02d}/{p['Purchase_Month']:02d}/{p['Purchase_Year']} (Thứ {p['Purchase_DayOfWeek']+2}, Cuối tuần: {p['Purchase_IsWeekend']})\n- **Decision Tree:** `{res_dt['predicted_daily_volume']:.2f} đơn/ngày`\n- **Linear Regression:** `{res_lr['predicted_daily_volume']:.2f} đơn/ngày`\n\n")

    rep_dir = ROOT / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    with open(rep_dir / "demo_results.md", "w", encoding="utf-8") as f:
        f.writelines(md_lines)
    print(f"[+] Đã xuất kết quả demo tại: {rep_dir / 'demo_results.md'}")

    print("\n" + "=" * 70)
    print("[+] Hoàn tất demo thành công!")

if __name__ == "__main__":
    main()
