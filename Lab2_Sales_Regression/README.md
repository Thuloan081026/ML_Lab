<div align="center">

# Predicting Product Sales

### From-scratch Time-Series Forecasting Pipeline with Linear Regression, Decision Tree, and Online Learning (Fine-Tuning)

<p><img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" alt="Python 3.10" /> <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter" alt="Jupyter Notebook" /> <img src="https://img.shields.io/badge/Task-Time--Series%20Forecasting-brightgreen?style=for-the-badge" alt="Time-Series Forecasting" /> <img src="https://img.shields.io/badge/Core-From%20Scratch-purple?style=for-the-badge" alt="From Scratch" /> <img src="https://img.shields.io/badge/Models-Linear%20Regression%20%2B%20Decision%20Tree-red?style=for-the-badge" alt="Models" />
</p>
<table>
<tr>
<td><b>Notebook chính</b></td>
<td><code>notebooks/04_timeseries_forecasting.ipynb</code> & <code>notebooks/05_online_learning.ipynb</code></td>
</tr>
<tr>
<td><b>Dữ liệu</b></td>
<td><code>data/raw/ecommerce_dataset_updated.csv</code></td>
</tr>
<tr>
<td><b>Bài toán</b></td>
<td>Dự báo Chuỗi thời gian (Time-Series Forecasting) cho Số lượng đơn hàng (Daily Sales Volume)</td>
</tr>
<tr>
<td><b>Nguyên tắc chính</b></td>
<td>100% thuật toán tự lập trình bằng Toán học (From Scratch), không sử dụng thư viện scikit-learn.</td>
</tr>
</table>
<p>
Project này xây dựng toàn bộ quy trình dự báo Doanh số bán hàng, đi từ dữ liệu raw,
data preparing, feature engineering, model selection, training, evaluation,
kỹ thuật Online Learning đến inference trực tiếp qua Web API.
</p>
</div>

## 1. Mục tiêu dự án

Ban đầu, dữ liệu (`ecommerce_dataset_updated.csv`) cung cấp lịch sử giao dịch. Thông thường, sinh viên sẽ lấy giá trị `Price` và `Discount` để đoán `Final_Price`. Tuy nhiên, điều này gây ra hiện tượng rò rỉ dữ liệu (Data Leakage) vì công thức tính là cố định.

Để giải quyết triệt để và mang lại giá trị thực tiễn cho doanh nghiệp, bài toán đã được chuyển đổi sang: **Dự báo Chuỗi thời gian (Time-Series Forecasting) cho Tổng số lượng Đơn hàng Bán ra mỗi ngày (Daily Sales Volume).** Việc dự đoán đúng số lượng đơn hàng trong tương lai giúp công ty tối ưu hóa kho bãi, nhân sự và chiến dịch Marketing.

Mục tiêu chính:

- Xây dựng pipeline Machine Learning hoàn chỉnh cho bài toán Time-Series Forecasting.

- Trực quan hóa dữ liệu (EDA), phân tích xu hướng bán hàng theo ngày/tháng/năm.

- Triển khai các thành phần học máy cốt lõi theo hướng from scratch, không dùng model/scaler có sẵn từ scikit-learn.

- So sánh hai thuật toán: Linear Regression (Tuyến tính) và Decision Tree Regressor (Phi tuyến tính).

- Khắc phục hiện tượng Rò rỉ dữ liệu bằng Custom StandardScaler và Time-Series Split.

- Tích hợp kỹ thuật **Online Learning (Fine-Tuning)** bằng Gradient Descent để mô hình tự học thêm kiến thức mỗi ngày.

- Đóng gói hoàn chỉnh thành một ứng dụng Web API (Flask) sẵn sàng cho Production.

---

## 2. Dữ liệu sử dụng & Xử lý (Data Overview)

Nguồn dữ liệu gốc chứa lịch sử giao dịch: `data/raw/ecommerce_dataset_updated.csv`.

### 2.1. Dữ liệu gốc ban đầu

Tập dữ liệu này gồm 3660 dòng lịch sử giao dịch. Mỗi dòng là 1 đơn mua hàng.

| User_ID | Product_ID | Category | Price (Rs.) | Discount (%) | Final_Price | Payment_Method | Purchase_Date |
|---|---|---|---|---|---|---|---|
| 337c166f | f414122f-e | Sports | 36.53 | 15 | 31.05 | Net Banking | 12-11-2024 |
| d38a19bf | fde50f9c-5 | Clothing | 232.79 | 20 | 186.23 | Net Banking | 09-02-2024 |

### 2.2. Bước Kỹ nghệ đặc trưng chung (`02_feature_engineering.ipynb`)

Dữ liệu gốc sau khi nạp vào pipeline sẽ trải qua quá trình Tiền xử lý (Data Preprocessing) và Kỹ nghệ đặc trưng (Feature Engineering) nghiêm ngặt để chuyển hóa các dữ liệu thô (như văn bản, thời gian) thành các vector số học mà mô hình AI có thể hiểu được:

1. **Bóc tách thời gian (Temporal Feature Extraction):** Thay vì để nguyên một chuỗi ngày tháng vô tri giác `Purchase_Date` (ví dụ: `12-11-2024`), dự án đã thiết kế các hàm tự động bóc tách ngày này thành 5 cột giá trị:

- `Purchase_Year`, `Purchase_Month`, `Purchase_Day`: Giúp mô hình bắt được xu hướng (Trend) theo chu kỳ dài hạn.

- `Purchase_DayOfWeek`: Giúp mô hình hiểu được tính chu kỳ trong tuần (Seasonality) - ví dụ như thứ 2 thường vắng khách hơn thứ 6.

- `Purchase_IsWeekend`: Cờ (Flag) nhị phân đánh dấu các ngày thứ 7 và Chủ Nhật, giúp mô hình bắt được đột biến doanh số trong các ngày nghỉ.

2. **Mã hóa dữ liệu phân loại (One-hot Encoding):** Các biến chứa chuỗi văn bản không thể tính toán toán học. Do đó, dự án sử dụng kỹ thuật One-hot Encoding để tách cột `Category` (Sports, Clothing,...) và `Payment_Method` (Net Banking, UPI) thành ma trận thưa (Sparse Matrix) với các giá trị 0 và 1.

3. **Loại bỏ nhiễu (Noise Removal):** Các trường thông tin định danh mang tính cá nhân như `User_ID` hay `Product_ID` được loại bỏ hoàn toàn để tránh việc học máy bị "phân tâm" hoặc học thuộc lòng.

Kết quả của bước này là file trung gian `ecommerce_features.csv` với ma trận **21 chiều (21 cột)**. Ở giai đoạn này, dữ liệu vẫn giữ nguyên 3660 dòng giao dịch gốc.

**Dữ liệu đặc trưng sau khi xử lý (`ecommerce_features.csv` - Data Leakage Version):**

| Price (Rs.) | Discount (%) | Final_Price | Purchase_Year | Purchase_Month | Purchase_Day | Purchase_DayOfWeek | Purchase_IsWeekend | Category_Books | ... |
|---|---|---|---|---|---|---|---|---|---|
| 36.53 | 15 | 31.05 | 2024 | 11 | 12 | 1 | 0 | 0 | ... |
| 232.79 | 20 | 186.23 | 2024 | 2 | 9 | 4 | 0 | 0 | ... |

### 2.3. Hai luồng xử lý trên dữ liệu (Sự khác biệt giữa Lý thuyết và Thực tiễn)

Phần cốt lõi tạo nên chiều sâu của dự án này nằm ở việc xây dựng và đối chiếu 2 tư duy giải quyết vấn đề (Problem-Solving) hoàn toàn khác biệt trên cùng một tập dữ liệu:

#### Luồng 1: Tư duy "Ngây thơ" (Baseline Model - `03_modeling_baseline.ipynb`)

- **Cách tiếp cận:** Giữ nguyên 3660 dòng giao dịch. Sử dụng trực tiếp Giá gốc `Price (Rs.)` và Tỷ lệ giảm giá `Discount (%)` làm đầu vào để dự đoán Giá trị đơn hàng `Final_Price`.

- **Kết quả:** Mô hình hội tụ rất nhanh, sai số gần như bằng 0.

- **Lỗ hổng (Data Leakage):** Tuy kết quả rất đẹp, nhưng việc làm này hoàn toàn vô dụng khi áp dụng vào thực tế kinh doanh. Nguyên nhân là vì Giá cuối (`Final_Price`) thực chất được cấu thành bằng một công thức toán học bất di bất dịch: `Final_Price = Price - (Price * Discount / 100)`. Khi cung cấp Price và Discount cho mô hình, chúng ta đang bắt AI đi "giải phương trình" thay vì đi dự báo. Đây là một ví dụ kinh điển về hiện tượng **Rò rỉ dữ liệu (Data Leakage)** - khi mô hình nhìn thấy đáp án từ chính câu hỏi. Hơn nữa, việc dự đoán giá trị của một đơn hàng ngẫu nhiên không giúp ích gì cho định hướng vĩ mô của doanh nghiệp.

#### Luồng 2: Tư duy "Chuyên gia" (Time-Series Forecasting - `04_timeseries_forecasting.ipynb`)

Để khắc phục hoàn toàn lỗ hổng trên và mang lại một công cụ có giá trị thực tiễn cao cho bộ phận Vận hành (Operations) của công ty, dự án đã đập bỏ luồng 1 và thiết kế lại bài toán:

1. **Loại trừ Rò rỉ dữ liệu:** Vứt bỏ ngay lập tức toàn bộ các cột liên quan đến giá tiền (`Price`, `Discount`, `Final_Price`).

2. **Biến đổi thành Chuỗi thời gian (Time-Series Transformation):** Thay vì dự đoán cho từng đơn hàng lẻ tẻ, dự án tiến hành **Gom nhóm (Groupby)** toàn bộ dữ liệu theo từng ngày (`Purchase_Date`).

3. **Thay đổi Biến mục tiêu (Target Variable):** Trên từng ngày đã gom nhóm, dự án **Đếm tổng số lượng đơn hàng** bán được, từ đó sinh ra một Target hoàn toàn mới: `Daily_Volume`.

4. **Chia tập dữ liệu (Chronological Split):** Thay vì xáo trộn dữ liệu (Train/Test Split ngẫu nhiên) như các bài toán thông thường, dự án tự lập trình một công cụ `time_series_train_val_test_split` để chia cắt dữ liệu theo đúng **trình tự thời gian** thực tế. Điều này đảm bảo rằng Dữ liệu của Tương lai (Test Set) không bao giờ bị tuồn vào Quá khứ (Train Set) để gian lận.

**Dữ liệu Time-Series sau khi gom nhóm (Target mới: `Daily_Volume`):**

| Purchase_Year | Purchase_Month | Purchase_Day | Purchase_DayOfWeek | Purchase_IsWeekend | Target: Daily_Volume |
|---|---|---|---|---|---|
| 2024 | 1 | 1 | 0 | 0 | **12** |
| 2024 | 1 | 2 | 1 | 0 | **9** |
| 2024 | 1 | 6 | 5 | 1 | **15** |

**Bài toán mới mang tính chiến lược:** Hiện tại, mô hình sẽ sử dụng các yếu tố thời gian (Hôm nay là tháng mấy? Có phải cuối tuần không? Thứ mấy trong tuần?) để trả lời câu hỏi sống còn: **"Dự báo trong ngày mai, kho hàng của công ty sẽ xuất đi bao nhiêu đơn hàng?"**.

Kết quả thu được (MAE ~2.46 đơn/ngày) trở thành một công cụ mạnh mẽ giúp Giám đốc Vận hành dự đoán để điều tiết nhân sự đóng gói, quản lý hàng tồn kho, và lên kế hoạch vận chuyển một cách hoàn toàn tự động.

---

## 3. Cấu trúc thư mục hiện tại

```text
Lab2_Sales_Regression/
├── README.md                             # Hướng dẫn chi tiết dự án dự báo doanh số Lab 2
├── train.py                              # Pipeline huấn luyện chính toàn bộ mô hình
├── app.py                                # Web API Flask phục vụ dự đoán thời gian thực
├── requirements.txt                      # Danh sách thư viện phụ thuộc (NumPy, Pandas, Flask...)
├── data/                                 # Quản lý dữ liệu thô và dữ liệu sau xử lý
│   ├── raw/                              # Dữ liệu giao dịch gốc
│   │   ├── ecommerce_dataset_updated.csv # 3.660 giao dịch mua hàng e-commerce năm 2024
│   │   └── ML project_ Predicting Product Sales.pdf # Tài liệu đề bài yêu cầu của bài lab
│   └── processed/                        # Dữ liệu chuỗi thời gian đã làm sạch và kỹ nghệ
│       ├── ecommerce_features.csv        # Đặc trưng bóc tách thời gian (phiên bản 21 chiều)
│       ├── train_daily.csv               # Tập Train theo ngày (232 ngày đầu - 70%)
│       ├── val_daily.csv                 # Tập Validation theo ngày (48 ngày giữa - 15%)
│       ├── test_daily.csv                # Tập Test theo ngày (48 ngày cuối - 15%)
│       └── evaluation_results.csv        # Bảng tổng hợp sai số MAE, RMSE, R2 của các mô hình
├── models/                               # Thuật toán From-Scratch và mô hình lưu trữ
│   ├── algorithms/                       # Mã nguồn thuật toán tự viết 100% bằng NumPy/Pandas
│   │   ├── decision_tree.py              # Cây quyết định hồi quy (Variance Reduction)
│   │   ├── linear_regression.py          # Hồi quy tuyến tính phương trình pháp tuyến (Normal Eq)
│   │   ├── linear_regression_gd.py       # Hồi quy tuyến tính Gradient Descent (Online Learning)
│   │   ├── metrics.py                    # Các hàm đo sai số MAE, RMSE, R2
│   │   └── preprocessing.py              # StandardScaler và Time-Series Train/Val/Test Split
│   └── save_model/                       # Mô hình đã huấn luyện xuất định dạng pickle
│       ├── linear_regression.pkl         # Trọng số mô hình Linear Regression
│       ├── linear_regression_gd.pkl      # Trọng số mô hình Linear Regression GD
│       ├── decision_tree.pkl             # Cấu trúc cây quyết định Decision Tree
│       └── scaler.pkl                    # Tham số Mean/Std của bộ chuẩn hóa StandardScaler
├── notebooks/                            # Sổ tay nghiên cứu theo từng giai đoạn
│   ├── 01_exploratory_data_analysis.ipynb# Phân tích khám phá và trực quan hóa dữ liệu (EDA)
│   ├── 02_feature_engineering.ipynb      # Bóc tách đặc trưng thời gian và mã hóa One-Hot
│   ├── 03_modeling_baseline.ipynb        # Thử nghiệm Baseline và phân tích Rò rỉ dữ liệu
│   ├── 04_timeseries_forecasting.ipynb   # Huấn luyện chuỗi thời gian Daily Volume
│   └── 05_online_learning.ipynb          # Thực nghiệm Online Learning thích ứng liên tục
├── scripts/                              # Kịch bản thực thi chức năng độc lập
│   ├── data_preparation.py               # Gom nhóm theo ngày và chia tập chuỗi thời gian
│   ├── eval.py                           # Đánh giá độc lập các mô hình trên tập Test
│   ├── demo.py                           # Chạy thử nghiệm dự báo kịch bản ngày thường vs cuối tuần
│   ├── data/                             # Module hỗ trợ tiền xử lý dữ liệu
│   │   ├── missing_handler.py            # Xử lý giá trị khuyết thiếu trong dữ liệu
│   │   └── outlier_handler.py            # Phát hiện và xử lý giá trị ngoại lai (IQR)
│   └── features/                         # Module trích xuất đặc trưng
│       └── build_features.py             # Bóc tách ngày tháng thành Year, Month, Day, DayOfWeek
├── reports/                              # Báo cáo kết quả và tài liệu trực quan
│   ├── BAO_CAO_LAB2.md                   # Báo cáo chi tiết toàn bộ dự án nộp môn học
│   ├── data-description.html             # Tài liệu giao diện web mô tả tập dữ liệu
│   └── figures/                          # Biểu đồ trực quan hóa kết quả
│       └── timeseries_forecast.png       # Đồ thị so sánh dự báo và thực tế trên tập Test
└── docs/                                 # Tài liệu kiến trúc và thiết kế
├── DESIGN.md                         # Thiết kế dự án theo cấu trúc 4W1H
├── PLAN.md                           # Kế hoạch phát triển chi tiết
├── DATA_PIPELINE.md                  # Kiến trúc Pipeline chuỗi thời gian chống rò rỉ
├── ALGORITHMS.md                     # Chi tiết công thức toán học các thuật toán
└── MODEL_CARD.md                     # Thẻ thông số kỹ thuật mô hình học máy
```

| Thành phần | Vai trò |
|---|---|
| `data/raw/ecommerce_dataset_updated.csv` | File dữ liệu e-commerce gốc. |
| `notebooks/` | Chứa toàn bộ các bước nghiên cứu: EDA, Feature Engineering, Training, và Online Learning. |
| `models/algorithms/` | Mã nguồn thuật toán tự lập trình. |
| `models/save_model/` | Các model và scaler đã huấn luyện. |
| `app.py` | Web API Deployment bằng Flask để phục vụ dự đoán theo thời gian thực. |
| `scripts/` | Kiểm tra dữ liệu, đánh giá đầu ra và chạy demo. |
| `reports/` | Báo cáo, mô tả dữ liệu, kết quả đánh giá và biểu đồ. |
| `docs/` | Kế hoạch, pipeline, thuật toán và model card. |

---

## 4. From-scratch compliance (Điểm nhấn Kỹ thuật)

Các thành phần cốt lõi được tự triển khai 100% bằng NumPy và Pandas:

| Thành phần | Class / logic chính | Ghi chú |
|---|---|---|
| Chống Rò rỉ Dữ liệu | `StandardScaler` | Tự viết Class chuẩn hóa dữ liệu. Chỉ học Mean/Std trên Train set, sau đó Transform lên Test set. |
| Time-Series Split | `time_series_train_val_test_split` | Tự code hàm chia dữ liệu thành 3 tập (Train/Val/Test) theo đúng trật tự thời gian thay vì xáo trộn ngẫu nhiên. |
| Linear Regression | `LinearRegression` | Tự giải thuật toán bằng công thức Ma trận giả nghịch đảo (Normal Equation). |
| Decision Tree | `DecisionTreeRegressor` | Tự code thuật toán rẽ nhánh đệ quy dựa trên tiêu chí **Giảm phương sai (Variance Reduction)**. |
| Online Learning | `LinearRegressionGD` | Thuật toán Hạ dốc đạo hàm có tính năng `partial_fit` để học liên tục. |
| Chống Overreacting | `alpha`, `clip_value` | Áp dụng L2 Regularization và Gradient Clipping để giới hạn Trọng số, chống nhiễu dữ liệu. |
| Metrics | `mean_absolute_error`, `root_mean_squared_error` | Tự code các hàm đánh giá sai số. |

---

## 5. Hướng dẫn Chạy Dự án (How to Run)

### A. Huấn luyện và Đánh giá tự động qua CLI

1. Cài đặt các thư viện cơ bản:

```bash
pip install -r requirements.txt
```

2. Chạy toàn bộ pipeline huấn luyện (Data Prep -> Split -> Train -> Save Models):

```bash
python train.py
```

3. Đánh giá mô hình đã lưu trên tập Test độc lập:

```bash
python scripts/eval.py
```

4. Chạy kịch bản demo dự báo doanh số:

```bash
python scripts/demo.py
```

### B. Chạy Khối lượng Nghiên cứu (Notebooks)

1. Cài đặt các thư viện cơ bản:

```bash
pip install -r requirements.txt
```

2. Khởi chạy Jupyter Notebook hoặc mở trực tiếp trên VS Code.

3. Chạy lần lượt các notebook từ `01` đến `05`.

*(Lưu ý: Các thuật toán được sử dụng trực tiếp từ mã nguồn trong thư mục `models/`.)*

### C. Chạy Triển khai API (Model Deployment)

1. Tại thư mục gốc của dự án, mở Terminal và gõ:

```bash
python app.py
```

2. Máy chủ Flask sẽ khởi chạy tại `http://127.0.0.1:5000/`. Các Endpoints hỗ trợ bao gồm:

- **`GET /`**: Trang chủ, hiển thị thông tin chung.

- **`GET /health`**: Kiểm tra trạng thái server (Health Check).

- **`POST /predict`**: Nhận dữ liệu JSON và trả về dự đoán doanh số.

3. Mở một Terminal khác (hoặc dùng Postman), gửi dữ liệu ngày tháng giả định vào API để lấy kết quả dự đoán:

*(Ví dụ dùng PowerShell trên Windows)*

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body '{"Purchase_Year": 2024, "Purchase_Month": 12, "Purchase_Day": 24, "Purchase_DayOfWeek": 1, "Purchase_IsWeekend": 0}'
```

4. Ứng dụng sẽ trả về số lượng đơn hàng dự báo bán được trong ngày Noel 24/12/2024.
