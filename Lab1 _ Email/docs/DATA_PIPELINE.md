# Data Pipeline
## Dữ liệu đầu vào

File [../data/raw/emails.csv](../data/raw/emails.csv) gồm hai cột:

- `text`: nội dung email.
- `spam`: nhãn nhị phân.

## Xử lý dữ liệu

1. Kiểm tra cột bắt buộc và giá trị thiếu.
2. Loại email trùng nội dung trước khi chia tập.
3. Chia dữ liệu theo tỷ lệ 70% Train, 15% Validation và 15% Test.
4. Chuẩn hóa chữ thường.
5. Ánh xạ URL, email và số tiền thành meta-token.
6. Lọc stopword, token quá ngắn và số thuần.
7. Fit TF-IDF chỉ trên Train.
8. Transform Validation và Test bằng vocabulary của Train.

## Đầu ra

Các file sinh tại [../data/processed](../data/processed):

- `train_cleaned.csv`
- `validation.csv`
- `test.csv`
- `tfidf_splits.npz`
- `tfidf_vectorizer_params.json`
- `evaluation_results.csv`
