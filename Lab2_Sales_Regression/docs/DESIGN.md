## LAB 2: DỰ BÁO DOANH SỐ BÁN SẢN PHẨM
## 1. WHAT (Dự án này là gì? Giải quyết vấn đề gì?)
- **Mục tiêu bài toán:** Xây dựng mô hình học máy hồi quy chuỗi thời gian (*Time-Series Forecasting*) để dự đoán **Tổng số lượng đơn hàng bán ra mỗi ngày (`Daily_Volume`)** dựa trên các đặc trưng thời gian (ngày, tháng, thứ trong tuần, ngày cuối tuần).
- **Yêu cầu kỹ thuật cốt lõi:**
  - Lập trình **100% From-Scratch (Zero Scikit-Learn)** bằng NumPy và Pandas: tự viết các thuật toán hồi quy tuyến tính `LinearRegression`, hạ dốc `LinearRegressionGD`, cây quyết định `DecisionTreeRegressor`, chuẩn hóa `StandardScaler` và các hàm đo sai số MAE, RMSE.
  - Tích hợp tính năng **Online Learning (Học liên tục)** bằng Gradient Descent qua hàm `partial_fit` để mô hình có thể cập nhật thêm dữ liệu mới mỗi ngày mà không cần học lại toàn bộ từ đầu.
  - Đóng gói mô hình thành một **Web API (Flask)** đơn giản với endpoint `/predict` để thử nghiệm dự đoán theo thời gian thực.
## 2. WHERE (Các thành phần nằm ở đâu trong dự án?)
Dự án được phân chia thư mục rõ ràng theo từng nhiệm vụ:

| Thành phần | File / Thư mục | Nhiệm vụ chính |
|---|---|---|
| **Dữ liệu thô** | [`data/raw/ecommerce_dataset_updated.csv`](../data/raw/ecommerce_dataset_updated.csv) | Chứa 3.660 giao dịch bán hàng gốc trong năm 2024. |
| **Đề bài môn học** | [`data/raw/ML project_ Predicting Product Sales.pdf`](../data/raw/ML%20project_%20Predicting%20Product%20Sales.pdf) | File PDF đề bài yêu cầu của bài thực hành 2. |
| **Dữ liệu theo ngày** | [`data/processed/train_daily.csv`](../data/processed/train_daily.csv), `val_daily.csv`, `test_daily.csv` | Dữ liệu sau khi gom nhóm theo ngày (`Daily_Volume`) và bóc tách ngày tháng. |
| **Thuật toán From-Scratch** | [`models/algorithms/`](../models/algorithms/) | Code tự viết các thuật toán hồi quy, cây quyết định, scaler và metrics. |
| **Mô hình đã huấn luyện** | [`models/save_model/`](../models/save_model/) | Lưu các file model đã train (`linear_regression.pkl`, `decision_tree.pkl`, `scaler.pkl`). |
| **Script chạy chính** | [`train.py`](../train.py) | File chạy tự động toàn bộ: nạp dữ liệu, gom nhóm, chia tập, train mô hình và xuất biểu đồ. |
| **Kiểm thử & Chạy thử** | [`scripts/eval.py`](../scripts/eval.py), [`scripts/demo.py`](../scripts/demo.py) | Đánh giá sai số MAE/RMSE trên 48 ngày Test và chạy thử dự báo ngày thường vs cuối tuần. |
| **Sổ tay nghiên cứu** | [`notebooks/`](../notebooks/) | 5 file Jupyter Notebook chia từng bước: EDA, trích xuất đặc trưng, thử nghiệm baseline và chuỗi thời gian. |
| **Ứng dụng API** | [`app.py`](../app.py) | Server Flask nhận dữ liệu ngày tháng qua JSON và trả về số lượng đơn hàng dự báo. |
| **Báo cáo & Biểu đồ** | [`reports/BAO_CAO_LAB2.md`](../reports/BAO_CAO_LAB2.md), [`reports/figures/`](../reports/figures/) | Báo cáo chi tiết đồ án và hình ảnh biểu đồ dự báo chuỗi thời gian (`timeseries_forecast.png`). |

## 3. WHEN (Quy trình thực hiện theo thứ tự nào?)
Quy trình thực hiện bài thực hành diễn ra tuần tự theo các bước:

```text
[Bước 1: Nạp và kiểm tra dữ liệu thô]
  └─ Đọc file CSV (3.660 giao dịch), kiểm tra giá trị thiếu (0 NaN), ép kiểu ngày tháng.
       │
[Bước 2: Gom nhóm theo ngày và Tạo đặc trưng]
  └─ Gom nhóm theo ngày giao dịch -> Đếm số đơn hàng để tạo biến mục tiêu 'Daily_Volume'.
  └─ Bóc tách ngày tháng thành: Năm, Tháng, Ngày, Thứ trong tuần, Cờ cuối tuần.
       │
[Bước 3: Chia dữ liệu theo thời gian (Chronological Split)]
  └─ Cắt theo dòng thời gian: 232 ngày đầu (Train 70%) -> 48 ngày giữa (Val 15%) -> 48 ngày cuối (Test 15%).
       │
[Bước 4: Chuẩn hóa Z-Score độc lập]
  └─ Bộ chuẩn hóa (Scaler) chỉ tính Mean và Std trên 232 ngày của tập Train, sau đó áp dụng lên Val và Test.
       │
[Bước 5: Huấn luyện các mô hình From-Scratch]
  └─ Huấn luyện Linear Regression (Normal Equation) và Decision Tree Regressor (Depth=4) trên tập Train.
       │
[Bước 6: Đánh giá trên tập Test độc lập]
  └─ Kiểm tra sai số dự đoán trên 48 ngày của tập Test (Linear Regression đạt MAE 2.38 đơn hàng/ngày).
       │
[Bước 7: Lưu mô hình và Triển khai API]
  └─ Lưu các file .pkl và khởi chạy app.py để phục vụ dự đoán.
```

---

## 4. WHY (Tại sao lại làm như vậy?)
Lý do đằng sau các quyết định kỹ thuật trong bài:

1. **Tại sao không lấy Price và Discount để đoán Final_Price?**
   - Công thức `Final_Price = Price * (1 - Discount/100)` là cố định sẵn. Nếu làm vậy, mô hình sẽ thấy trước đáp án (**Data Leakage**) và chỉ học lại công thức toán học chứ không có ý nghĩa dự báo thực tế.
2. **Tại sao chuyển sang dự đoán số lượng đơn hàng mỗi ngày (`Daily_Volume`)?**
   - Dự đoán số lượng đơn hàng hàng ngày mang lại giá trị thực tế rõ ràng: giúp doanh nghiệp dự báo trước nhu cầu để chủ động sắp xếp nhân sự đóng gói, kho bãi và xe tải vận chuyển.
3. **Tại sao phải chia dữ liệu theo thứ tự thời gian (không xáo trộn ngẫu nhiên)?**
   - Vì thời gian diễn ra một chiều từ quá khứ đến tương lai. Nếu xáo trộn ngẫu nhiên, mô hình sẽ lấy dữ liệu của tương lai để dự đoán quá khứ, làm kết quả bị sai lệch và không thực tế.
4. **Tại sao Scaler chỉ fit trên tập Train?**
   - Nếu chuẩn hóa cả bảng dữ liệu trước khi chia, các thông số trung bình của tập Test sẽ bị lộ vào tập Train. Tự viết lớp `StandardScaler` giúp đảm bảo chỉ học thông số từ tập Train.
5. **Tại sao cần tính năng Online Learning (`partial_fit`)?**
   - Khi có thêm số liệu bán hàng của ngày mới, hàm `partial_fit` giúp mô hình cập nhật nhẹ thêm trọng số mà không cần mất công huấn luyện lại toàn bộ dữ liệu lịch sử từ đầu.

---

## 5. HOW (Mô hình hoạt động và dự đoán như thế nào?)
Khi người dùng gửi thông tin một ngày cần dự đoán qua API, quá trình diễn ra qua **4 bước**:

```text
[Dữ liệu gửi lên: Ngày 24/12/2024, Thứ 3, Không phải cuối tuần]
       │
       ▼
1. Đọc dữ liệu: Nhận thông tin ngày tháng từ request JSON và đưa vào bảng dữ liệu 1 dòng.
       │
       ▼
2. Chuẩn hóa: Dùng Mean và Std đã lưu từ trước để đưa các cột số về cùng tỷ lệ (Z-Score).
       │
       ▼
3. Dự đoán:
   - Với Linear Regression: Tính tổng có trọng số y_pred = X @ W + Bias.
   - Với Decision Tree: Duyệt qua các nút điều kiện rẽ nhánh của cây để tìm giá trị dự đoán ở lá.
       │
       ▼
4. Trả kết quả: Trả về số lượng đơn hàng dự kiến bán được trong ngày đó (ví dụ: ~12.66 đơn hàng).
```
