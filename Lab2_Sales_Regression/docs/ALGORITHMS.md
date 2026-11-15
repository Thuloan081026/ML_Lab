# Thuật toán

- **Hồi quy tuyến tính:** giải bằng phương trình Normal Equation.
- **Cây quyết định hồi quy:** tự cài đặt theo tiêu chí giảm phương sai.
- **Hồi quy tuyến tính Gradient Descent:** hỗ trợ `partial_fit` cho Online
  Learning.
- **StandardScaler:** chuẩn hóa theo trung bình và độ lệch chuẩn của tập
  huấn luyện.
- **MAE và RMSE:** đo sai số dự báo trên tập kiểm tra.

Các thuật toán cốt lõi được viết bằng NumPy và Pandas, không dùng
scikit-learn cho model hoặc scaler.

Mã nguồn thuật toán nằm trong `models/algorithms/`; các model đã huấn luyện
được lưu trong `models/save_model/`.

### 1. Hồi quy tuyến tính

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

### 2. Cây quyết định hồi quy

Cây quyết định chia dữ liệu thành các vùng sao cho phương sai của nhãn trong
mỗi vùng giảm nhiều nhất. Giá trị dự đoán tại nút lá là giá trị trung bình
của các mẫu thuộc nút đó.

Trong thí nghiệm chuỗi thời gian, cây được khởi tạo với `max_depth=4` để
hạn chế cây quá sâu và giảm nguy cơ overfitting.

Mã nguồn nằm tại
[`decision_tree.py`](../models/algorithms/decision_tree.py).

### 3. Hồi quy Gradient Descent

Gradient Descent cập nhật trọng số lặp lại theo hướng làm giảm hàm mất mát.
Phiên bản trong dự án hỗ trợ:

- L2 Regularization để hạn chế trọng số quá lớn.
- Gradient Clipping để tránh một mẫu bất thường làm thay đổi mô hình quá mạnh.
- `partial_fit` để cập nhật mô hình từng mẫu trong kịch bản dữ liệu đến liên
  tục.

Mã nguồn nằm tại
[`linear_regression_gd.py`](../models/algorithms/linear_regression_gd.py).

### 4. Hàm đánh giá (Evaluation Metrics)

Các hàm đánh giá đều được lập trình 100% From-Scratch thuần NumPy tại [`metrics.py`](../models/algorithms/metrics.py):

#### 4.1. Sai số tuyệt đối trung bình (MAE)
$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$
MAE phản ánh độ lệch trung bình giữa số lượng đơn thực tế và dự báo theo cùng đơn vị đo (đơn hàng/ngày).

#### 4.2. Căn bậc hai sai số bình phương trung bình (RMSE)
$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$
RMSE phạt nặng hơn đối với các sai số lớn do phép bình phương, giúp phát hiện các ngày dự báo lệch đột biến.

#### 4.3. Phần trăm sai số tuyệt đối trung bình (MAPE)
$$
MAPE = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i-\hat{y}_i}{\max(|y_i|, \epsilon)}\right|
$$
- **Ý nghĩa:** Biểu diễn mức độ sai lệch theo tỷ lệ phần trăm so với giá trị thực tế, giúp các bên kinh doanh dễ hình dung sai số tương đối bất kể quy mô doanh số lớn hay nhỏ.
- **Xử lý số học:** Áp dụng hệ số $\epsilon = 10^{-8}$ kẹp cận dưới mẫu số để triệt tiêu lỗi chia cho 0 khi ngày đó có doanh số bằng 0.

#### 4.4. Hệ số xác định ($R^2$ Score)
$$
R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}
$$
Đo lường tỷ lệ phương sai của biến mục tiêu mà các đặc trưng giải thích được so với mô hình dự đoán bằng giá trị trung bình $\bar{y}$.

