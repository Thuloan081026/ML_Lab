# Thuật toán
## 1. TF-IDF Vectorizer

`TfidfVectorizerScratch` chuyển email đã tokenize thành vector số:

```text
TF(t, d)  = số lần t xuất hiện trong d / số token của d
IDF(t)    = log((1 + N) / (1 + DF(t))) + 1v
TF-IDF    = TF * IDF
```

Vector được chuẩn hóa L2. Vocabulary và IDF chỉ được `fit` trên tập Train;
Validation và Test chỉ được `transform`.

## 2. Multinomial Naive Bayes

`MultinomialNaiveBayesScratch` ước lượng:

```text
P(y | x) ∝ P(y) * ∏ P(x_j | y)
```

Tính toán ở miền log để tránh underflow và dùng Laplace smoothing. Đây là
model champion được đóng gói vào `models/email_classifier.json`.

## 3. Complement Naive Bayes

`ComplementNaiveBayesScratch` học xác suất từ phần dữ liệu bổ sung của từng
lớp. Cách này thường hữu ích khi dữ liệu mất cân bằng; trong dự án, nó được
dùng để so sánh với Multinomial Naive Bayes.

## 4. Logistic Regression

`LogisticRegressionScratch` dùng hàm sigmoid:

```text
σ(z) = 1 / (1 + exp(-z))
```

Trọng số được tối ưu bằng gradient descent với binary cross-entropy và
regularization L2. Xác suất Spam được dùng để hiệu chuẩn threshold.

## 5. Linear SVM

`LinearSVMScratch` tối ưu hinge loss:

```text
L = max(0, 1 - y * (wᵀx + b)) + λ/2 * ||w||²
```

Model được huấn luyện bằng stochastic gradient descent. Điểm quyết định
được chuyển đổi thành xác suất tương đối để phục vụ so sánh và đánh giá.

## 6. So sánh và chọn model

Các model được huấn luyện trên cùng một ma trận TF-IDF và đánh giá trên
Validation bằng Accuracy, Precision, Recall và F1-Score. Threshold được chọn
trên Validation, sau đó chỉ đánh giá một lần trên Test độc lập để tránh
data leakage.

## 7. BA THUẬT TOÁN HỌC MÁY TỰ LẬP TRÌNH BẰNG NUMPY
### 7.1. Multinomial Naive Bayes (`MultinomialNaiveBayesScratch`)
- Mô hình xác suất sinh mẫu.
- Xác suất tiên nghiệm: $\ln P(y = c) = \ln(N_c / N)$.
- Ma trận Log-Likelihood với hệ số làm mịn Laplace $\alpha = 0.1$:
  $$P(w_j | c) = \frac{\sum_{i: y_i=c} x_{ij} + \alpha}{\sum_{k=1}^D \sum_{i: y_i=c} x_{ik} + \alpha \cdot D}$$
- Vectorized inference: $\text{log\_probs} = X W_{\text{feat}}^T + b_{\text{prior}}$.
- Tính xác suất hậu nghiệm qua hàm **Softmax** ổn định số học.
### 7.2. Logistic Regression (`LogisticRegressionScratch`)
- Mô hình phân biệt tuyến tính, tối ưu Binary Cross-Entropy có điều chuẩn L2 Ridge ($\lambda = 0.0001$).
- Kẹp cận số học $z_{\text{clipped}} = \text{clip}(Xw + b, -20.0, 20.0)$ để triệt tiêu lỗi tràn số exp trong float32.
- Cập nhật trọng số qua **Batch Gradient Descent** ($lr = 2.0$, $n_{\text{iters}} = 400$):
  $$dw = \frac{1}{m} X^T (\hat{p} - y) + \frac{\lambda}{m} w, \quad db = \frac{1}{m} \sum (\hat{p} - y)$$
### 7.3. Linear Support Vector Machine (`LinearSVMScratch`)
- Tìm siêu phẳng tối ưu lề cực đại $w^T x + b = 0$ với nhãn $y \in \{-1, +1\}$.
- Tối ưu hàm mất mát Soft-Margin Hinge Loss có điều chuẩn L2:
  $$L(w, b) = \frac{1}{m} \sum \max(0, 1 - y_i(w^T x_i + b)) + \frac{\lambda}{2} \|w\|_2^2$$
- Cập nhật qua thuật toán **Subgradient Descent (SGD)** ($lr = 1.5$, $n_{\text{iters}} = 400$, $\lambda = 0.0001$).
- Áp dụng xấp xỉ Platt Scaling để hiệu chuẩn ngưỡng xác suất.