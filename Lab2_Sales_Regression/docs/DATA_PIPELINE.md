# Data Pipeline

## Nguồn dữ liệu

File `data/raw/ecommerce_dataset_updated.csv` gồm 3.660 giao dịch và các cột:
`User_ID`, `Product_ID`, `Category`, `Price (Rs.)`, `Discount (%)`,
`Final_Price(Rs.)`, `Payment_Method`, `Purchase_Date`.

## Xử lý

- Parse `Purchase_Date` theo định dạng ngày-tháng-năm.
- Tạo `Purchase_Year`, `Purchase_Month`, `Purchase_Day`,
  `Purchase_DayOfWeek` và `Purchase_IsWeekend`.
- One-Hot Encoding cho `Category` và `Payment_Method`.
- Loại `User_ID` và `Product_ID` vì đây là mã định danh.
- Chuẩn hóa đặc trưng số bằng `StandardScaler` tự cài đặt.
- Chia train/validation/test theo thứ tự thời gian.
- Target của bài toán dự báo là `Daily_Volume`, tức số giao dịch theo ngày.
