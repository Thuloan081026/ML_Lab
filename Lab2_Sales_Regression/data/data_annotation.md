# Chú thích dữ liệu Lab 2

## Tập dữ liệu gốc

Tệp dữ liệu: `data/raw/ecommerce_dataset_updated.csv`

- Số dòng: 3.660 giao dịch.
- Số cột: 8.
- Khoảng thời gian: `01-01-2024` đến `31-10-2024`.
- Giá trị thiếu: 0.
- Dòng trùng lặp: 0.
- Số ngày lịch khác nhau: 326.

## Chú thích các cột

| Cột | Kiểu | Ý nghĩa | Vai trò |
|---|---|---|---|
| `User_ID` | Chuỗi | Mã duy nhất của người dùng | Định danh, không dùng trực tiếp |
| `Product_ID` | Chuỗi | Mã duy nhất của sản phẩm | Định danh, không dùng trực tiếp |
| `Category` | Chuỗi | Nhóm sản phẩm | Mã hóa One-Hot |
| `Price (Rs.)` | Số thực | Giá niêm yết bằng Rupee | Đặc trưng |
| `Discount (%)` | Số nguyên | Phần trăm giảm giá | Đặc trưng |
| `Final_Price(Rs.)` | Số thực | Giá sau khi giảm | Nhãn hồi quy cơ sở |
| `Payment_Method` | Chuỗi | Phương thức thanh toán | Mã hóa One-Hot |
| `Purchase_Date` | Chuỗi | Ngày giao dịch theo DD-MM-YYYY | Tạo đặc trưng ngày |

## Đặc trưng được tạo

Từ `Purchase_Date`, pipeline tạo:

- `Purchase_Year`
- `Purchase_Month`
- `Purchase_Day`
- `Purchase_DayOfWeek`
- `Purchase_IsWeekend`

Trong bài toán dự báo theo ngày, các giao dịch được gom theo ngày và tạo
nhãn `Daily_Volume`, là số giao dịch của ngày đó.

## Quy tắc xử lý

1. Kiểm tra giá trị thiếu và bản ghi trùng.
2. Chuyển `Purchase_Date` sang kiểu ngày với định dạng `%d-%m-%Y`.
3. Loại `User_ID` và `Product_ID` khỏi tập đặc trưng mô hình.
4. Mã hóa `Category` và `Payment_Method` bằng One-Hot Encoding.
5. Chỉ fit `StandardScaler` trên tập huấn luyện.
6. Với dự báo chuỗi thời gian, giữ nguyên thứ tự thời gian khi chia dữ liệu.
