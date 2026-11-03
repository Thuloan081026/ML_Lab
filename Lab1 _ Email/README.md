# Lab 1 - Phân loại Email Spam

Dự án bài tập lớn môn Học máy phân loại email nhị phân bằng NumPy thuần (Zero Scikit-Learn):
- `0`: Ham - email hợp lệ.
- `1`: Spam - email rác hoặc có dấu hiệu lừa đảo.

Pipeline bao gồm đọc dữ liệu, EDA, tiền xử lý văn bản, TF-IDF, huấn luyện, đánh giá, lưu model JSON và dự đoán email mới.

## 1. Ràng buộc kỹ thuật
- Không dùng `scikit-learn` cho TF-IDF hoặc thuật toán học máy.
- TF-IDF, Naive Bayes, Logistic Regression và Linear SVM được cài đặt bằng NumPy.
- Từ điển TF-IDF chỉ được `fit` trên tập Train để tránh data leakage.
- Validation dùng để chọn model và threshold; Test chỉ dùng đánh giá cuối.

## 2. Cấu trúc dự án
```text
Lab1 _ Email/
├── README.md                 # Tài liệu hướng dẫn sử dụng và báo cáo tổng quan Lab 1
├── requirements.txt          # Danh sách thư viện phụ thuộc (NumPy, Pandas, Flask...)
├── train.py                  # Pipeline huấn luyện chính (Zero Scikit-Learn)
├── predict.py                # Dự đoán bằng CLI kèm giải thích XAI Trigger Words
├── app.py                    # Giao diện web Flask tương tác phân loại email
├── data/                     # Thư mục chứa dữ liệu và mã nguồn xử lý dữ liệu
│   ├── raw/                  # Dữ liệu gốc chưa qua xử lý
│   │   └── emails.csv        # Bộ dữ liệu 5.728 email phân loại Spam/Ham
│   ├── collection.py         # Kiểm tra, thống kê và thu thập dữ liệu thô
│   ├── eda.py                # Phân tích khám phá dữ liệu (EDA) và vẽ biểu đồ
│   ├── data_processing.py    # Làm sạch text, tokenize và loại bỏ Stopwords
│   └── processed/            # Dữ liệu sau xử lý: CSV chia tập, TF-IDF và tham số
├── models/                   # Thư mục chứa thuật toán và mô hình lưu trữ
│   ├── algorithms.py         # Cài đặt From-Scratch: TF-IDF, Naive Bayes, LR, SVM
│   ├── save_models.py        # Hàm đóng gói xuất mô hình ra định dạng JSON
│   ├── metrics.py            # Hàm tính Accuracy, Precision, Recall, F1, ROC-AUC
│   └── email_classifier.json # Mô hình Champion (MNB) đã huấn luyện dạng JSON
├── scripts/                  # Kịch bản thực thi độc lập cho từng giai đoạn
│   ├── data_preparation.py   # Tiền xử lý, chia Stratified và lưu ma trận TF-IDF
│   ├── eval.py               # Đánh giá độc lập mô hình trên tập Test
│   └── demo.py               # Kịch bản chạy thử nghiệm nhanh dự đoán email mẫu
├── notebooks/                # Sổ tay nghiên cứu thực nghiệm
│   └── Lab1_Phan_Loai_Spam_Email.ipynb # Notebook thực nghiệm EDA và huấn luyện
├── templates/                # Giao diện HTML của ứng dụng web
│   └── index.html            # Giao diện người dùng Web Flask hiện đại
├── reports/                  # Báo cáo và kết quả thực nghiệm
│   ├── BAO_CAO_LAB1.md       # Báo cáo chi tiết đồ án nộp môn học
│   ├── data-description.html # Bảng mô tả chi tiết các thuộc tính dữ liệu
│   ├── demo/                 # Kết quả chạy demo thực tế
│   │   └── demo_results.txt  # Log kết quả chạy thử nghiệm các mẫu email
│   └── figures/              # Hình ảnh biểu đồ trực quan hóa dữ liệu
└── docs/                     # Tài liệu thiết kế và kiến trúc mô hình
    ├── DESIGN.md             # Thiết kế dự án theo cấu trúc 4W1H
    ├── PLAN.md               # Kế hoạch thực hiện dự án
    ├── DATA_PIPELINE.md      # Thiết kế luồng xử lý dữ liệu chống rò rỉ
    ├── ALGORITHMS.md         # Chi tiết công thức toán các thuật toán From-Scratch
    └── MODEL_CARD.md         # Thông số kỹ thuật và thẻ mô hình học máy
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
Tiền xử lý chuyển URL, email và số tiền thành các meta-token: `http_url`, `email_address`, `money_amount`; đồng thời loại stopword, token quá ngắn và số thuần.

## 5. Thuật toán
- **Multinomial Naive Bayes**: model triển khai chính, dùng log-likelihood và Laplace smoothing.
- **Complement Naive Bayes**: model so sánh cho dữ liệu mất cân bằng.
- **Logistic Regression**: sigmoid, binary cross-entropy, gradient descent.
- **Linear SVM**: hinge loss, stochastic gradient descent.

Chi tiết công thức xem tại [docs/ALGORITHMS.md](docs/ALGORITHMS.md).

## 6. Cài đặt môi trường
Khuyến nghị Python 3.10 trở lên. Bạn có thể sử dụng môi trường ảo (`venv`) để cô lập các thư viện:

### Thiết lập môi trường ảo (Khuyến nghị)
```powershell
cd "Lab1 _ Email"
# 1. Tạo môi trường ảo .venv
python -m venv .venv
# 2. Kích hoạt môi trường ảo:
# Trên Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# (Nếu PowerShell chặn script, mở quyền tạm thời: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
# Hoặc trên Windows Command Prompt (cmd.exe):
.\.venv\Scripts\activate.bat
# Hoặc trên macOS / Linux / Git Bash:
source .venv/bin/activate
# 3. Cài đặt các thư viện cần thiết:
pip install -r requirements.txt
```
> **Lưu ý:** Nếu máy tính của bạn đã cài đặt sẵn các thư viện cần thiết trong môi trường Python toàn cục (`numpy`, `pandas`, `matplotlib`, `seaborn`, `flask`), bạn có thể bỏ qua bước tạo `venv` và chạy trực tiếp các lệnh ở Mục 7.

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

Model JSON được lưu tại `models/email_classifier.json`. Kết quả demo nằm tại [reports/demo/demo_results.txt](reports/demo/demo_results.txt).

## 9. Tài liệu
- Notebook: [notebooks/Lab1_Phan_Loai_Spam_Email.ipynb](notebooks/Lab1_Phan_Loai_Spam_Email.ipynb)
- Báo cáo: [reports/BAO_CAO_LAB1.md](reports/BAO_CAO_LAB1.md)
- Mô tả dữ liệu: [reports/data-description.html](reports/data-description.html)
- Kế hoạch: [docs/PLAN.md](docs/PLAN.md)
- Pipeline: [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)
- Thuật toán: [docs/ALGORITHMS.md](docs/ALGORITHMS.md)
- Thiết kế dự án (4W1H): [docs/DESIGN.md](docs/DESIGN.md)
- Model card: [docs/MODEL_CARD.md](docs/MODEL_CARD.md)

## 10. Giới hạn
- Dữ liệu chủ yếu là email tiếng Anh.
- Model phụ thuộc vào vocabulary học từ tập Train.
- Email quá ngắn hoặc không có token đã học có thể cho kết quả kém ổn định.
