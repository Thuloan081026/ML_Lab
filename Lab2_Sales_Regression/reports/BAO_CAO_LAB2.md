# BÁO CÁO LAB 2 - ML PROJECT: PREDICTING PRODUCT SALES

## DỰ ĐOÁN DOANH SỐ BÁN SẢN PHẨM

## 1. GIỚI THIỆU

### 1.1. Tên đề tài

**Xây dựng pipeline học máy dự đoán doanh số bán sản phẩm từ dữ liệu giao dịch,
giá bán, mức giảm giá và thời điểm mua hàng.**

### 1.2. Bối cảnh bài toán

Trong bài toán dự đoán doanh số bán sản phẩm, dữ liệu giao dịch được sử dụng
cho hai nhu cầu phân tích chính:

1. Ước lượng giá bán cuối cùng của một sản phẩm dựa trên giá niêm yết và mức
   giảm giá.
2. Dự đoán sản lượng bán theo ngày để hỗ trợ lập kế hoạch sản phẩm, kho hàng,
   nhân sự và hoạt động kinh doanh.

Lab 2 triển khai cả hai thí nghiệm hồi quy. Thí nghiệm dự đoán
`Final_Price(Rs.)` kiểm tra khả năng ước lượng giá bán sản phẩm. Thí nghiệm
dự đoán `Daily_Volume`, tức sản lượng bán theo ngày, được đóng gói thành mô
hình chính và cung cấp qua Flask API.

### 1.3. Mục tiêu của bài thực hành

1. Đọc, kiểm tra và mô tả bộ dữ liệu giao dịch bán sản phẩm.
2. Phân tích dữ liệu bằng thống kê mô tả và trực quan hóa.
3. Xử lý ngày tháng và tạo các đặc trưng thời gian.
4. Mã hóa biến phân loại bằng One-Hot Encoding.
5. Xây dựng các thuật toán hồi quy từ đầu bằng NumPy và Pandas:
   - Hồi quy tuyến tính theo phương trình Normal Equation.
   - Cây quyết định hồi quy theo tiêu chí giảm phương sai.
   - Hồi quy tuyến tính tối ưu bằng Gradient Descent.
6. Chuẩn hóa dữ liệu bằng `StandardScaler` tự cài đặt.
7. Chia dữ liệu theo đúng thứ tự thời gian để hạn chế rò rỉ dữ liệu tương lai.
8. Đánh giá mô hình bằng MAE và RMSE.
9. Mô phỏng Online Learning bằng `partial_fit`.
10. Lưu mô hình và triển khai endpoint dự đoán thời gian thực.

### 1.4. Loại bài toán học máy

Đây là bài toán **học có giám sát**, cụ thể là **hồi quy (regression)**.

- Bài toán hồi quy cơ sở:
  - Đầu vào: thông tin giá, giảm giá, ngày mua và biến phân loại.
  - Nhãn: `Final_Price(Rs.)`.
- Bài toán triển khai:
  - Đầu vào: các đặc trưng của ngày mua.
  - Nhãn: `Daily_Volume`, tức số giao dịch trong một ngày.

## 2. MÔ TẢ TẬP DỮ LIỆU

### 2.1. Nguồn dữ liệu

Dữ liệu được lưu tại
`data/raw/ecommerce_dataset_updated.csv` và được lấy từ bộ dữ liệu thương mại
điện tử dùng trong Lab 2 mẫu.

Tệp dữ liệu mô tả các giao dịch mua hàng trong khoảng thời gian từ
`01-01-2024` đến `31-10-2024`.

### 2.2. Quy mô dữ liệu

| Thuộc tính | Giá trị |
|---|---:|
| Số bản ghi | 3.660 |
| Số cột gốc | 8 |
| Số ngày khác nhau | 326 |
| Giá trị thiếu | 0 |
| Bản ghi trùng lặp | 0 |
| Giá bán trung bình | 254,80 Rs. |
| Mức giảm giá trung bình | 18,83% |
| Giá bán cuối cùng trung bình | 206,91 Rs. |

### 2.3. Ý nghĩa các thuộc tính

| Cột | Kiểu dữ liệu | Ý nghĩa | Vai trò |
|---|---|---|---|
| `User_ID` | Chuỗi | Mã người dùng | Định danh, loại khỏi đặc trưng |
| `Product_ID` | Chuỗi | Mã sản phẩm | Định danh, loại khỏi đặc trưng |
| `Category` | Chuỗi | Nhóm sản phẩm | Biến phân loại |
| `Price (Rs.)` | Số thực | Giá niêm yết | Đặc trưng số |
| `Discount (%)` | Số nguyên | Phần trăm giảm giá | Đặc trưng số |
| `Final_Price(Rs.)` | Số thực | Giá sau giảm | Nhãn của hồi quy cơ sở |
| `Payment_Method` | Chuỗi | Phương thức thanh toán | Biến phân loại |
| `Purchase_Date` | Chuỗi | Ngày giao dịch, định dạng DD-MM-YYYY | Dùng để tạo đặc trưng thời gian |

Chi tiết hơn được trình bày trong
[`data_annotation.md`](../data/data_annotation.md) và
[`data-description.html`](data-description.html).

### 2.4. Phân bố dữ liệu

Bảy nhóm sản phẩm trong dữ liệu gồm:

- Home & Kitchen: 549 giao dịch.
- Books: 534 giao dịch.
- Clothing: 531 giao dịch.
- Toys: 523 giao dịch.
- Sports: 520 giao dịch.
- Beauty: 505 giao dịch.
- Electronics: 498 giao dịch.

Năm phương thức thanh toán gồm:

- Credit Card: 760 giao dịch.
- UPI: 757 giao dịch.
- Debit Card: 731 giao dịch.
- Net Banking: 716 giao dịch.
- Cash on Delivery: 696 giao dịch.

Không có nhóm sản phẩm hoặc phương thức thanh toán nào bị thiếu dữ liệu.

## 3. QUY TRÌNH THỰC HIỆN

### 3.1. Tổng quan pipeline

```text
[Dữ liệu gốc]
      |
      v
[Kiểm tra thiếu, trùng, kiểu dữ liệu]
      |
      v
[EDA và thống kê mô tả]
      |
      v
[Tạo đặc trưng ngày + One-Hot Encoding]
      |
      +------------------------------+
      |                              |
      v                              v
[Hồi quy Final_Price]       [Gom nhóm theo ngày]
      |                              |
      v                              v
[Đánh giá MAE/RMSE]         [Chia Train/Val/Test theo thời gian]
                                     |
                                     v
                            [Dự báo Daily_Volume]
                                     |
                                     v
                             [Lưu model và triển khai API]
```

### 3.2. Các notebook thực hiện

| Notebook | Nội dung |
|---|---|
| `01_exploratory_data_analysis.ipynb` | Kiểm tra cấu trúc, thống kê, missing values, duplicate và phân phối dữ liệu |
| `02_feature_engineering.ipynb` | Tách đặc trưng ngày và mã hóa biến phân loại |
| `03_modeling_baseline.ipynb` | Huấn luyện hồi quy tuyến tính dự đoán giá sau giảm |
| `04_timeseries_forecasting.ipynb` | Gom giao dịch theo ngày, so sánh Linear Regression và Decision Tree |
| `05_online_learning.ipynb` | Mô phỏng cập nhật mô hình tuần tự bằng Gradient Descent |

## 4. KHÁM PHÁ VÀ TIỀN XỬ LÝ DỮ LIỆU

### 4.1. Kiểm tra chất lượng dữ liệu

Kết quả kiểm tra trên dữ liệu gốc:

- Tất cả 3.660 dòng đều có đủ 8 cột.
- Không có giá trị `NaN` hoặc `Null`.
- Không có dòng trùng lặp.
- `User_ID` và `Product_ID` là các mã duy nhất, không mang ý nghĩa thứ tự
  nên không đưa trực tiếp vào mô hình.
- `Purchase_Date` được đọc theo định dạng ngày-tháng-năm.

### 4.2. Xử lý ngày tháng

Từ `Purchase_Date`, pipeline tạo ra năm đặc trưng:

- `Purchase_Year`: năm giao dịch.
- `Purchase_Month`: tháng giao dịch.
- `Purchase_Day`: ngày trong tháng.
- `Purchase_DayOfWeek`: thứ trong tuần, trong đó thứ Hai bằng 0 và Chủ nhật
  bằng 6.
- `Purchase_IsWeekend`: bằng 1 nếu giao dịch vào thứ Bảy hoặc Chủ nhật,
  ngược lại bằng 0.

### 4.3. Mã hóa biến phân loại

Hai cột `Category` và `Payment_Method` được chuyển sang dạng số bằng
One-Hot Encoding. Tham số `drop_first=True` được sử dụng để loại một cột
tham chiếu và hạn chế bẫy đa cộng tuyến.

Sau xử lý, tập đặc trưng dùng trong hồi quy cơ sở có kích thước:

```text
(3660, 18)
```

### 4.4. Chuẩn hóa dữ liệu

Các đặc trưng số của ngày được chuẩn hóa bằng công thức Z-Score:

$$
x' = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}
$$

Trong đó, trung bình và độ lệch chuẩn chỉ được tính trên tập huấn luyện.
Các tập validation và test chỉ sử dụng phép `transform`, không được `fit`
lại. Cách làm này giúp tránh rò rỉ thông tin từ dữ liệu đánh giá vào quá trình
huấn luyện.

## 5. CHIA DỮ LIỆU

### 5.1. Chia dữ liệu cho hồi quy cơ sở

Trong notebook modeling baseline, dữ liệu được chia ngẫu nhiên theo tỷ lệ
80/20 với `random_state=42`:

- 80% dùng để huấn luyện.
- 20% dùng để kiểm tra.

Mục tiêu của thí nghiệm này là kiểm tra khả năng ước lượng `Final_Price(Rs.)`
từ các thông tin của giao dịch.

### 5.2. Chia dữ liệu cho chuỗi thời gian

Đối với dự báo số giao dịch, dữ liệu được sắp xếp theo `Purchase_Date` trước
khi chia. Các giao dịch được gom theo ngày và tạo nhãn `Daily_Volume`. Dữ
liệu gốc có 326 ngày lịch khác nhau. Vì notebook chia dữ liệu theo các dòng
giao dịch trước khi gom theo ngày, một ngày nằm tại ranh giới hai tập có thể
xuất hiện ở cả hai tập. Do đó, kích thước các khung đặc trưng sau khi gom
theo từng phần có tổng là 328 ngày:

| Tập | Số ngày | Tỷ lệ xấp xỉ |
|---|---:|---:|
| Train | 232 | 70% |
| Validation | 48 | 15% |
| Test | 48 | 15% |

Không xáo trộn dữ liệu thời gian là yêu cầu quan trọng vì mô hình chỉ được
phép học từ quá khứ để dự báo tương lai.

## 6. CÁC THUẬT TOÁN TỰ CÀI ĐẶT

### 6.1. Hồi quy tuyến tính

Hồi quy tuyến tính mô hình hóa quan hệ giữa đầu vào và nhãn:

$$
\hat{y} = Xw + b
$$

Trọng số được tìm bằng phương trình Normal Equation:

$$
\theta = (X^T X)^{-1}X^T y
$$

Mã nguồn nằm tại
[`linear_regression.py`](../models/algorithms/linear_regression.py).

### 6.2. Cây quyết định hồi quy

Cây quyết định chia dữ liệu thành các vùng sao cho phương sai của nhãn trong
mỗi vùng giảm nhiều nhất. Giá trị dự đoán tại nút lá là giá trị trung bình
của các mẫu thuộc nút đó.

Trong thí nghiệm chuỗi thời gian, cây được khởi tạo với `max_depth=4` để
hạn chế cây quá sâu và giảm nguy cơ overfitting.

Mã nguồn nằm tại
[`decision_tree.py`](../models/algorithms/decision_tree.py).

### 6.3. Hồi quy Gradient Descent

Gradient Descent cập nhật trọng số lặp lại theo hướng làm giảm hàm mất mát.
Phiên bản trong dự án hỗ trợ:

- L2 Regularization để hạn chế trọng số quá lớn.
- Gradient Clipping để tránh một mẫu bất thường làm thay đổi mô hình quá mạnh.
- `partial_fit` để cập nhật mô hình từng mẫu trong kịch bản dữ liệu đến liên
  tục.

Mã nguồn nằm tại
[`linear_regression_gd.py`](../models/algorithms/linear_regression_gd.py).

### 6.4. Hàm đánh giá

Sai số tuyệt đối trung bình:

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

Căn bậc hai của sai số bình phương trung bình:

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

MAE dễ diễn giải theo đơn vị của nhãn. RMSE phạt mạnh hơn các sai số lớn.
Hai hàm được cài đặt tại
[`metrics.py`](../models/algorithms/metrics.py).

## 7. KẾT QUẢ THỰC NGHIỆM

### 7.1. Hồi quy cơ sở: dự đoán giá sau giảm

Mục tiêu của thí nghiệm là `Final_Price(Rs.)`. Kết quả trên tập kiểm tra:

| Mô hình | MAE | RMSE |
|---|---:|---:|
| Linear Regression | 15,2589 Rs. | 22,6519 Rs. |

Kết quả cho thấy mô hình dự đoán giá sau giảm với sai số tuyệt đối trung
bình khoảng 15,26 Rs. Đây là mô hình baseline để kiểm tra toàn bộ quy trình
đọc dữ liệu, tạo đặc trưng, chuẩn hóa và huấn luyện.

### 7.2. Dự báo số giao dịch theo ngày

Mục tiêu của thí nghiệm là `Daily_Volume`, được tính bằng số giao dịch trong
từng ngày.

#### Kết quả trên tập validation

| Mô hình | MAE | RMSE |
|---|---:|---:|
| Linear Regression | 2,64 | 3,33 |
| Decision Tree | 3,16 | 3,93 |

#### Kết quả trên tập test

| Xếp hạng | Mô hình | MAE | RMSE |
|---:|---|---:|---:|
| 1 | Linear Regression | **2,39** | **3,03** |
| 2 | Decision Tree | 2,88 | 3,56 |

Linear Regression đạt sai số thấp hơn Decision Tree trên cả validation và
test. Tuy nhiên, Decision Tree được chọn làm model triển khai minh họa trong
API vì có khả năng mô hình hóa quan hệ phi tuyến và đã được đóng gói sẵn cùng
scaler.

### 7.3. Online Learning

Notebook Online Learning dùng `LinearRegressionGD` với:

- Learning rate: `0.01`.
- Số epoch huấn luyện ban đầu: `1000`.
- L2 regularization: `alpha=0.5`.
- Gradient clipping: `clip_value=1.0`.
- Cập nhật tuần tự bằng `partial_fit`.

Kết quả mô phỏng trên tập test:

| Mô hình | MAE | RMSE |
|---|---:|---:|
| Mô hình tĩnh | 2,46 | 3,32 |
| Mô hình cập nhật online | **2,44** | **3,22** |

Mô hình online cải thiện nhẹ so với mô hình tĩnh. Kết quả này cho thấy việc
cập nhật theo từng mẫu có thể hữu ích khi dữ liệu mới tiếp tục xuất hiện,
nhưng mức cải thiện còn hạn chế do tập dữ liệu nhỏ và các đặc trưng thời gian
đơn giản.

## 8. PHÂN TÍCH KẾT QUẢ

### 8.1. So sánh mô hình

- Linear Regression phù hợp với dữ liệu có xu hướng gần tuyến tính theo các
  đặc trưng ngày.
- Decision Tree có thể biểu diễn quan hệ phi tuyến nhưng dễ phân mảnh dữ liệu
  khi số ngày huấn luyện không lớn.
- Kết quả test cho thấy Linear Regression tốt hơn Decision Tree về cả MAE và
  RMSE.
- Online Learning giúp mô hình thích nghi dần với dữ liệu mới nhưng cần thêm
  đặc trưng như xu hướng, ngày lễ, mùa vụ và độ trễ của các ngày trước để
  cải thiện đáng kể.

### 8.2. Ý nghĩa sai số

Trong bài toán `Daily_Volume`, MAE test bằng 2,39 của Linear Regression có
nghĩa là dự báo trung bình lệch khoảng 2 đến 3 giao dịch mỗi ngày trên tập
test. RMSE bằng 3,03 cho thấy vẫn tồn tại một số ngày có sai số lớn hơn mức
trung bình.

### 8.3. Hạn chế

1. Dữ liệu chỉ kéo dài từ tháng 1 đến tháng 10 năm 2024 nên chưa đánh giá
   được đầy đủ tính mùa vụ theo năm.
2. Mô hình `Daily_Volume` mới sử dụng các đặc trưng lịch cơ bản, chưa dùng
   lag feature như số giao dịch của ngày hôm trước hoặc trung bình trượt.
3. Chưa có thông tin về ngày lễ, chiến dịch khuyến mãi hoặc tồn kho.
4. Model API hiện tại nhận các đặc trưng ngày và trả về số giao dịch dự báo;
   các biến sản phẩm và thanh toán được dùng trong thí nghiệm baseline nhưng
   không phải đầu vào của endpoint dự báo theo ngày.
5. Mô hình triển khai là Decision Tree để minh họa pipeline, dù kết quả định
   lượng tốt nhất trong thí nghiệm thuộc về Linear Regression.

## 9. LƯU TRỮ MÔ HÌNH VÀ TRIỂN KHAI

### 9.1. Cấu trúc model

```text
models/
├── algorithms/
│   ├── decision_tree.py
│   ├── linear_regression.py
│   ├── linear_regression_gd.py
│   ├── metrics.py
│   └── preprocessing.py
└── save_model/
    ├── decision_tree.pkl
    ├── linear_regression.pkl
    └── scaler.pkl
```

Mã nguồn thuật toán và file model đã huấn luyện được tách riêng để dễ đọc,
kiểm thử và thay thế.

### 9.2. Flask API

Khởi động API:

```bash
python app.py
```

Các endpoint:

| Endpoint | Phương thức | Chức năng |
|---|---|---|
| `/` | GET | Hiển thị thông tin API |
| `/health` | GET | Kiểm tra trạng thái dịch vụ |
| `/predict` | POST | Dự báo số giao dịch trong ngày |

Ví dụ request:

```json
{
  "Purchase_Year": 2024,
  "Purchase_Month": 12,
  "Purchase_Day": 24,
  "Purchase_DayOfWeek": 1,
  "Purchase_IsWeekend": 0
}
```

Kết quả kiểm thử thực tế:

```text
Số giao dịch dự báo: 12.43
```

Model và scaler được nạp từ `models/save_model/`. API đã được kiểm tra bằng
Flask test client thông qua
[`scripts/demo.py`](../scripts/demo.py).

## 10. KIỂM THỬ VÀ TÁI LẬP

Các bước chạy lại pipeline:

```bash
python scripts/data_preparation.py
python scripts/eval.py
python scripts/demo.py
```

Các notebook được chạy theo thứ tự từ 01 đến 05. Kết quả kiểm tra cuối cùng:

- Cả 5 notebook thực thi không có lỗi trong các cell.
- Dữ liệu đặc trưng được tạo thành công.
- Model và scaler được nạp thành công.
- Endpoint `/health` hoạt động.
- Endpoint `/predict` trả về kết quả dự báo hợp lệ.
- Mã nguồn Python trong project được tổ chức theo các thư mục `models` và
  `scripts`.

## 11. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
Lab2_Sales_Regression/
├── train.py
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   │   └── ecommerce_dataset_updated.csv
│   └── processed/
│       ├── ecommerce_features.csv
│       └── data_summary.csv
├── models/
│   ├── algorithms/
│   └── save_model/
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_baseline.ipynb
│   ├── 04_timeseries_forecasting.ipynb
│   └── 05_online_learning.ipynb
├── scripts/
│   ├── data/
│   ├── features/
│   ├── data_preparation.py
│   ├── eval.py
│   └── demo.py
├── docs/
└── reports/
    ├── BAO_CAO_LAB2.md
    └── data-description.html
```

## 12. KẾT LUẬN

Lab 2 đã xây dựng được một pipeline học máy hoàn chỉnh trên dữ liệu thương
mại điện tử, từ kiểm tra dữ liệu, EDA, tạo đặc trưng, huấn luyện, đánh giá,
lưu model đến triển khai API.

Các kết quả chính:

1. Dữ liệu gồm 3.660 giao dịch, không có giá trị thiếu hoặc dòng trùng.
2. Đã tạo thành công các đặc trưng ngày và biến One-Hot cho mô hình hồi quy.
3. Linear Regression dự đoán `Final_Price(Rs.)` đạt MAE 15,2589 Rs. và RMSE
   22,6519 Rs. trên tập test.
4. Trong bài toán dự báo số giao dịch theo ngày, Linear Regression đạt MAE
   2,39 và RMSE 3,03 trên tập test.
5. Online Learning cải thiện từ MAE 2,46 xuống 2,44 và từ RMSE 3,32 xuống
   3,22 trong mô phỏng.
6. Flask API nạp được model, scaler và trả về dự báo mẫu 12,43 giao dịch.

Hướng phát triển tiếp theo là bổ sung đặc trưng trễ, trung bình trượt, ngày
lễ, chương trình khuyến mãi và dữ liệu dài hơn để mô hình phản ánh tốt hơn
tính mùa vụ của doanh số.
