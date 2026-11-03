# Mô tả dự án
## Tên dự án

Lab 2 - Dự báo doanh số sản phẩm theo chuỗi thời gian.

## Mục tiêu

Xây dựng pipeline học máy từ dữ liệu giao dịch thương mại điện tử để dự báo
số lượng đơn hàng theo ngày. Dự án minh họa đầy đủ các bước EDA, tạo đặc
trưng, chia dữ liệu theo thời gian, huấn luyện, đánh giá và triển khai API.

## Dataset

Dataset `ecommerce_dataset_updated.csv` gồm 3.660 giao dịch trong năm 2024.
Mỗi dòng tương ứng với một giao dịch mua hàng.

## Phương pháp

Các thuật toán được tự cài đặt bằng NumPy và Pandas gồm hồi quy tuyến tính,
cây quyết định hồi quy và hồi quy Gradient Descent cho Online Learning.
StandardScaler, MAE và RMSE cũng được triển khai trong mã nguồn dự án.

## Triển khai

API Flask trong `app.py` nạp cây quyết định và scaler từ
`models/save_model/`. Endpoint `/predict` nhận các đặc trưng ngày mua và trả
về số giao dịch dự báo.
