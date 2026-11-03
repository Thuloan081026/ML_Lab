# Kế hoạch dự án: Phân loại Email Spam
## 1. Mục tiêu

Xây dựng hệ thống phân loại email nhị phân:

- `0`: Ham - email hợp lệ.
- `1`: Spam - email rác hoặc có dấu hiệu lừa đảo.

Hệ thống phải có khả năng huấn luyện, đánh giá, lưu model và dự đoán email mới.

## 2. Ràng buộc kỹ thuật

- Không dùng `scikit-learn` cho TF-IDF hoặc thuật toán học máy.
- TF-IDF, Naive Bayes, Logistic Regression và Linear SVM được cài đặt bằng NumPy.
- Dùng seed cố định để kết quả có thể tái lập.
- Chỉ fit từ điển và IDF trên tập Train để tránh data leakage.
- Model triển khai được lưu dưới dạng JSON.

## 3. Quy trình

```text
data/raw/emails.csv
        |
        v
Thu thập và kiểm tra dữ liệu
        |
        v
Deduplication + Stratified Split
        |
        v
Tiền xử lý văn bản và TF-IDF
        |
        v
Huấn luyện các model from-scratch
        |
        v
Validation + hiệu chuẩn threshold
        |
        v
Test độc lập + biểu đồ đánh giá
        |
        v
models/email_classifier.json
        |
        v
CLI/Web demo
```

## 4. Các giai đoạn triển khai

| Giai đoạn | Entry point | Đầu ra |
|---|---|---|
| Thu thập dữ liệu | `data/collection.py` | Kiểm tra raw CSV |
| EDA | `data/eda.py` | Thống kê dữ liệu |
| Chuẩn bị dữ liệu | `scripts/data_preparation.py` | Tập processed |
| Huấn luyện | `train.py` | Model JSON |
| Đánh giá | `scripts/eval.py` | Metrics và figures |
| Demo | `scripts/demo.py`, `app.py` | Dự đoán email mới |
