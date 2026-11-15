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

### 4. Hàm đánh giá

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
