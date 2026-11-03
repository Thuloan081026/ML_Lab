# Báo cáo Lab 2: Dự báo doanh số theo thời gian

## 1. Mục tiêu

Xây dựng pipeline dự báo số lượng giao dịch theo ngày từ dataset thương mại
điện tử `ecommerce_dataset_updated.csv`, sử dụng các thuật toán tự cài đặt.

## 2. Dữ liệu

Dữ liệu có 3.660 giao dịch, ngày mua trong năm 2024. Các mã định danh được
loại bỏ; ngày mua được tách thành các đặc trưng năm, tháng, ngày, thứ trong
tuần và cuối tuần.

## 3. Phương pháp

Notebook được chia thành EDA, feature engineering, modeling baseline,
time-series forecasting và online learning. Model sử dụng hồi quy tuyến tính,
cây quyết định và Gradient Descent. StandardScaler và các metric được tự cài
đặt bằng NumPy/Pandas.

## 4. Kết quả

Model triển khai là cây quyết định tại `models/decision_tree.pkl`, dùng cùng
`models/scaler.pkl`. Kết quả chi tiết được tái tạo bằng các notebook và script
trong thư mục `scripts`.

## 5. Triển khai

Chạy `python app.py` để mở API Flask. Endpoint `/predict` nhận các đặc trưng
ngày mua và trả về số giao dịch dự báo.
