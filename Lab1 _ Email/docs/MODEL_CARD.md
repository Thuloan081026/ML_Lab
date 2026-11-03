# Model Card
## Mô hình

Model triển khai chính là Multinomial Naive Bayes from-scratch với đặc trưng TF-IDF.
Artifact được lưu tại [../models/email_classifier.json](../models/email_classifier.json).

Các model được so sánh:

- Multinomial Naive Bayes.
- Complement Naive Bayes.
- Logistic Regression.
- Linear SVM.

## Đánh giá

Kết quả chính thức và biểu đồ được lưu trong:

- [../reports/BAO_CAO_LAB1.md](../reports/BAO_CAO_LAB1.md)
- [../reports/figures](../reports/figures)

Threshold được hiệu chuẩn trên Validation trước khi đánh giá Test.

## Giới hạn

- Dữ liệu chủ yếu là email tiếng Anh.
- Model phụ thuộc vào vocabulary được học từ dữ liệu huấn luyện.
- Email quá ngắn hoặc không có token trong vocabulary có thể làm giảm độ tin cậy.
- Kết quả không nên được dùng như cơ chế duy nhất để quyết định an toàn email thực tế.
