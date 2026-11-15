# ML_Lab — Kho Đồ Án Thực Hành Học Máy (Machine Learning From-Scratch)

Repository tổng hợp các bài thực hành môn học **Học máy (Machine Learning)** được xây dựng và triển khai theo chuẩn kỹ thuật công nghiệp. Toàn bộ các thuật toán cốt lõi, bộ trích xuất đặc trưng và chỉ số đánh giá đều được lập trình **100% From-Scratch thuần Python, NumPy và Pandas (Không sử dụng Scikit-Learn)**.

---

## 📌 Danh Sách Các Đồ Án (Labs)

### 📧 [Lab 1: Phân Loại Email Rác (Spam / Ham Email Classification)](./Lab1%20_%20Email)
- **Mục tiêu:** Phân loại email hợp lệ (Ham) và thư rác (Spam) trên tập dữ liệu hơn 5.700 email thực tế.
- **Mô hình lập trình From-Scratch:**
  - `MultinomialNaiveBayesScratch`: Mô hình xác suất sinh mẫu với làm mịn Laplace $\alpha = 0.1$.
  - `ComplementNaiveBayesScratch`: Tối ưu hóa đặc biệt cho dữ liệu văn bản mất cân bằng lớp.
  - `LogisticRegressionScratch`: Mô hình phân biệt tuyến tính với Batch Gradient Descent và điều chuẩn L2.
  - `LinearSVMScratch`: Tối ưu hàm mất mát Soft-Margin Hinge Loss qua Subgradient Descent.
- **Kỹ thuật nổi bật:**
  - Tiền xử lý & TF-IDF Vectorizer tự lập trình (loại bỏ thẻ HTML, chuẩn hóa URL, số tiền, email).
  - Pipeline chống rò rỉ dữ liệu triệt để (*Stratified Train/Val/Test Split*).
  - Tinh chỉnh ngưỡng phân loại (*Calibrated Threshold Tuning*) tối ưu F1-Score và giảm thiểu False Positives.
- **Hiệu năng Test độc lập:** F1-Score **94.67%**, ROC-AUC **0.9892** (Mô hình Champion: Multinomial Naive Bayes).
- **Giao diện Web:** Ứng dụng Web Flask kiểm thử phân loại email theo thời gian thực và phân tích từ khóa kích hoạt (*Top Trigger Words*).

---

### 📈 [Lab 2: Dự Báo Doanh Số E-Commerce (Daily Sales Volume Forecasting)](./Lab2_Sales_Regression)
- **Mục tiêu:** Dự báo số lượng đơn hàng bán ra mỗi ngày (*Daily Sales Volume*) cho doanh nghiệp thương mại điện tử từ lịch sử giao dịch.
- **Mô hình lập trình From-Scratch:**
  - `LinearRegression`: Nghiệm giải tích chính xác qua phương trình chuẩn *Normal Equation (OLS)*.
  - `DecisionTreeRegressor`: Cây quyết định hồi quy phân tách đệ quy theo tiêu chí *Giảm phương sai (Variance Reduction)*.
  - `LinearRegressionGD`: Thuật toán hạ dốc đạo hàm tích hợp điều chuẩn L2 và Gradient Clipping.
  - `Online Learning`: Cơ chế cập nhật trọng số linh hoạt từng mẫu (*Fine-Tuning*) qua phương thức `partial_fit`.
- **Kỹ thuật nổi bật:**
  - Trích xuất đặc trưng thời gian: năm, tháng, ngày, thứ trong tuần, cờ cuối tuần (*IsWeekend*).
  - Phân chia tập dữ liệu theo chuỗi thời gian (*Temporal Train/Val/Test Split*) chống rò rỉ dữ liệu tương lai.
  - Chuẩn hóa dữ liệu `StandardScaler` tự lập trình (chỉ fit trên tập Train).
- **Hiệu năng Test độc lập:** MAE **2.385** đơn/ngày, RMSE **3.033**, MAPE **22.82%** (Mô hình Champion: Linear Regression).
- **Giao diện Web:** Ứng dụng Web Flask với bảng điều khiển chọn lịch, thử kịch bản mẫu và so sánh trực tiếp các mô hình.

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Demo

### 1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10 trở lên:
```bash
# Clone repository
git clone https://github.com/Thuloan081026/ML_Lab.git
cd ML_Lab

# Cài đặt thư viện phụ thuộc
pip install numpy pandas matplotlib flask
```

### 2. Khởi chạy Lab 1 (Phân loại Email Spam)
```bash
# Đánh giá đối chuẩn mô hình
python "Lab1 _ Email/scripts/eval.py"

# Chạy demo kiểm thử các email mẫu
python "Lab1 _ Email/scripts/demo.py"

# Khởi động ứng dụng Web Demo (Cổng 5000)
python "Lab1 _ Email/app.py"
```

### 3. Khởi chạy Lab 2 (Dự báo Doanh số Bán hàng)
```bash
# Huấn luyện toàn bộ pipeline
python "Lab2_Sales_Regression/train.py"

# Đánh giá mô hình trên tập Test (kèm MAE, RMSE, MAPE)
python "Lab2_Sales_Regression/scripts/eval.py"

# Chạy demo kịch bản dự báo
python "Lab2_Sales_Regression/scripts/demo.py"

# Khởi động ứng dụng Web Demo (Cổng 5000)
python "Lab2_Sales_Regression/app.py"
```

---

## 📊 Bảng Tổng Hợp Kết Quả Thực Nghiệm (Benchmark Summary)

### Lab 1: Phân loại Email (Test Set)
| Thuật toán | Phương pháp | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes 🏆** | Generative Probabilistic | **0.9743** | **0.9576** | **0.9360** | **0.9467** | **0.9892** |
| **Complement Naive Bayes** | Imbalance-Corrected NB | 0.9673 | 0.9320 | 0.9453 | 0.9386 | 0.9877 |
| **Logistic Regression** | Batch Gradient Descent (L2) | 0.9614 | 0.9431 | 0.8984 | 0.9202 | 0.9840 |
| **Linear SVM** | Soft-Margin Subgradient (L2) | 0.9591 | 0.9368 | 0.8984 | 0.9172 | 0.9813 |

### Lab 2: Dự báo Doanh số Hàng ngày (Test Set)
| Thuật toán | Phương pháp | MAE (đơn/ngày) | RMSE | **MAPE (%)** | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Linear Regression 🏆** | Normal Equation (OLS) | **2.385** | **3.033** | **22.82%** | **-0.0274** |
| **Linear Regression (GD)** | Gradient Descent + L2 Clip | 2.552 | 3.450 | **24.21%** | -0.3289 |
| **Decision Tree Regressor** | Variance Reduction Split (Depth=4) | 2.885 | 3.559 | **28.43%** | -0.4144 |

---

## 📜 Quy Chuẩn Kỹ Thuật
- **Zero Scikit-Learn:** Tất cả mô hình học máy, bộ tiền xử lý và chỉ số đều được viết từ đầu (From-Scratch).
- **Phòng chống Rò rỉ Dữ liệu:** Chia dữ liệu trước khi fit scaler/vectorizer; bài toán chuỗi thời gian áp dụng chia theo trật tự thời gian.
- **Tinh gọn & Tường minh:** Tuân thủ giới hạn dòng mã nguồn cho từng file thuật toán, có tài liệu giải thích chi tiết trong từng thư mục `docs/`.
