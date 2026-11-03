# Thuật toán
## 1. TF-IDF Vectorizer

`TfidfVectorizerScratch` chuyển email đã tokenize thành vector số:

```text
TF(t, d)  = số lần t xuất hiện trong d / số token của d
IDF(t)    = log((1 + N) / (1 + DF(t))) + 1
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
