# Model Card

## Mục đích

Dự báo số lượng giao dịch theo ngày từ các đặc trưng thời gian.

## Đầu vào API

`Purchase_Year`, `Purchase_Month`, `Purchase_Day`,
`Purchase_DayOfWeek` và `Purchase_IsWeekend`.

## Model triển khai

API hiện nạp `models/decision_tree.pkl` cùng
`models/scaler.pkl`. Các model và scaler được tạo từ mã nguồn trong thư mục
`models/`.

## Giới hạn

Dữ liệu chỉ bao phủ một giai đoạn giao dịch trong năm 2024. Dự báo có thể
giảm độ chính xác khi xu hướng mua hàng thay đổi; cần huấn luyện lại khi có
dữ liệu mới.
