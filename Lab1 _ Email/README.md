# Lab 1 - Phân loại Email Spam

Dự án xây dựng hệ thống phân loại email nhị phân bằng NumPy:

- `0`: Ham - email hợp lệ.
- `1`: Spam - email rác hoặc có dấu hiệu lừa đảo.

Pipeline bao gồm đọc dữ liệu, EDA, tiền xử lý văn bản, TF-IDF, huấn luyện,
đánh giá, lưu model JSON và dự đoán email mới.

## 1. Ràng buộc kỹ thuật

- Không dùng `scikit-learn` cho TF-IDF hoặc thuật toán học máy.
- TF-IDF, Naive Bayes, Logistic Regression và Linear SVM được cài đặt bằng
  NumPy.
- Từ điển TF-IDF chỉ được `fit` trên tập Train để tránh data leakage.
- Validation dùng để chọn model và threshold; Test chỉ dùng đánh giá cuối.

## 2. Cấu trúc dự án

```text
Lab1 _ Email/
├── README.md
├── requirements.txt
├── train.py                 # Pipeline huấn luyện chính
├── predict.py               # Dự đoán bằng CLI
├── app.py                   # Giao diện web Flask
├── data/
│   ├── raw/emails.csv
│   ├── collection.py
│   ├── eda.py
│   ├── data_processing.py
│   └── processed/           # CSV, TF-IDF, metrics và tham số vectorizer
├── models/
│   ├── algorithms.py
│   ├── save_models.py
│   └── email_classifier.json
├── scripts/
│   ├── data_preparation.py
│   ├── eval.py
│   └── demo.py
├── notebooks/
│   └── Lab1_Phan_Loai_Spam_Email.ipynb
├── reports/
│   ├── BAO_CAO_LAB1.md
│   ├── data-description.html
│   ├── demo/demo_results.txt
│   └── figures/
└── docs/
    ├── PLAN.md
    ├── DATA_PIPELINE.md
    ├── ALGORITHMS.md
    └── MODEL_CARD.md
```

## 3. Dữ liệu

File [data/raw/emails.csv](data/raw/emails.csv) gồm:

- `text`: nội dung email.
- `spam`: nhãn nhị phân.
- 5.728 dòng, không có giá trị thiếu.
- 4.360 Ham và 1.368 Spam.
- 33 email trùng nội dung được loại bỏ trước khi chia tập.

Sau dedup, dữ liệu được chia stratified:

| Tập | Tổng | Ham | Spam |
|---|---:|---:|---:|
| Train | 3.985 | 3.028 | 957 |
| Validation | 854 | 649 | 205 |
| Test | 856 | 650 | 206 |

## 4. Pipeline

```text
Raw CSV
 -> kiểm tra dữ liệu và loại duplicate
 -> stratified split 70/15/15
 -> làm sạch và tokenize
 -> fit/transform TF-IDF
 -> train model
 -> chọn threshold trên Validation
 -> đánh giá Test
 -> lưu JSON và inference
```

Tiền xử lý chuyển URL, email và số tiền thành các meta-token:
`http_url`, `email_address`, `money_amount`; đồng thời loại stopword,
token quá ngắn và số thuần.

## 5. Thuật toán

- **Multinomial Naive Bayes**: model triển khai chính, dùng log-likelihood và
  Laplace smoothing.
- **Complement Naive Bayes**: model so sánh cho dữ liệu mất cân bằng.
- **Logistic Regression**: sigmoid, binary cross-entropy, gradient descent.
- **Linear SVM**: hinge loss, stochastic gradient descent.

Chi tiết công thức xem tại [docs/ALGORITHMS.md](docs/ALGORITHMS.md).

## 6. Cài đặt

Khuyến nghị Python 3.10 trở lên:

```powershell
cd "Lab1 _ Email"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 7. Chạy dự án

Huấn luyện và đánh giá đầy đủ:

```powershell
python train.py
```

Chạy các entry point theo từng giai đoạn:

```powershell
python data\collection.py
python data\eda.py
python scripts\data_preparation.py
python train.py
python scripts\eval.py
```

Dự đoán email bằng CLI:

```powershell
python predict.py --demo
python predict.py --text "Urgent: You won $1,000,000 cash!"
python scripts\demo.py --demo
```

Chạy giao diện web:

```powershell
python app.py
```

Mở `http://127.0.0.1:5000`.

## 8. Kết quả

Threshold được chọn trên Validation là `0.35`. Kết quả Test:

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 98.36% | 97.52% | 95.63% | 96.57% |
| Logistic Regression | 98.01% | 95.22% | 96.60% | 95.90% |
| Linear SVM | 95.68% | 85.36% | 99.03% | 91.69% |

Model JSON được lưu tại `models/email_classifier.json`. Kết quả demo nằm tại
[reports/demo/demo_results.txt](reports/demo/demo_results.txt).

## 9. Tài liệu

- Notebook: [notebooks/Lab1_Phan_Loai_Spam_Email.ipynb](notebooks/Lab1_Phan_Loai_Spam_Email.ipynb)
- Báo cáo: [reports/BAO_CAO_LAB1.md](reports/BAO_CAO_LAB1.md)
- Mô tả dữ liệu: [reports/data-description.html](reports/data-description.html)
- Kế hoạch: [docs/PLAN.md](docs/PLAN.md)
- Pipeline: [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)
- Thuật toán: [docs/ALGORITHMS.md](docs/ALGORITHMS.md)
- Model card: [docs/MODEL_CARD.md](docs/MODEL_CARD.md)

## 10. Giới hạn

- Dữ liệu chủ yếu là email tiếng Anh.
- Model phụ thuộc vào vocabulary học từ tập Train.
- Email quá ngắn hoặc không có token đã học có thể cho kết quả kém ổn định.
