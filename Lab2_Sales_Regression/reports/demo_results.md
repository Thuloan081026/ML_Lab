# KẾT QUẢ THỰC NGHIỆM DEMO — LAB 2 (SALES FORECASTING)

## 1. Bảng Đánh Giá Đối Chuẩn Các Mô Hình Trên Tập Test (Bao gồm MAPE)

| Mô hình | MAE (đơn) | RMSE | MAPE (%) | R2 Score |
|---|---|---|---|---|
| **Linear Regression (Normal Eq)** | 2.385 | 3.033 | 22.82% | -0.0274 |
| **Linear Regression (Gradient Descent)** | 2.552 | 3.450 | 24.21% | -0.3289 |
| **Decision Tree Regressor (Depth=4)** | 2.885 | 3.559 | 28.43% | -0.4144 |

## 2. Kết Quả Dự Báo Các Kịch Bản Mẫu

### Kịch bản 1: Ngày thường (Thứ Ba, 24/12/2024)
- **Đầu vào:** Ngày 24/12/2024 | Thứ 3 | Weekend=0
- **Linear Regression:** `12.66 đơn/ngày`
- **Decision Tree:** `12.43 đơn/ngày`

### Kịch bản 2: Cuối tuần (Chủ Nhật, 29/12/2024)
- **Đầu vào:** Ngày 29/12/2024 | Thứ 8 | Weekend=1
- **Linear Regression:** `12.78 đơn/ngày`
- **Decision Tree:** `12.75 đơn/ngày`

