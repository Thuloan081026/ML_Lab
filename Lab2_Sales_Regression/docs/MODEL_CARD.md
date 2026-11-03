# Model Card
## Mục đích

Dự báo số lượng giao dịch theo ngày từ các đặc trưng thời gian (Daily Sales Volume).

## Đầu vào API

`Purchase_Year`, `Purchase_Month`, `Purchase_Day`,
`Purchase_DayOfWeek` và `Purchase_IsWeekend`.

## Model triển khai

API nạp `models/save_model/decision_tree.pkl`, `models/save_model/linear_regression.pkl` cùng
`models/save_model/scaler.pkl`. Người dùng có thể chỉ định model qua tham số `"model": "decision_tree"` hoặc `"linear_regression"`.

## Giới hạn

Dữ liệu chỉ bao phủ một giai đoạn giao dịch trong năm 2024. Dự báo có thể
giảm độ chính xác khi xu hướng mua hàng thay đổi; cần huấn luyện lại khi có
dữ liệu mới.
