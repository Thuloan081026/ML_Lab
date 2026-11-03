## LAB 1: PHÂN LOẠI EMAIL SPAM
## 1. WHAT (Dự án này là gì? Giải quyết vấn đề gì?)
- **Mục tiêu bài toán:** Xây dựng mô hình học máy phân loại văn bản nhị phân (*Binary Text Classification*) để tự động đọc nội dung email và phân loại:
  - `0 (Ham)`: Thư hợp lệ, an toàn, trao đổi công việc/học tập thông thường.
  - `1 (Spam)`: Thư rác, quảng cáo làm phiền hoặc lừa đảo (*Phishing*).
- **Yêu cầu kỹ thuật cốt lõi:**
  - Lập trình **100% From-Scratch bằng NumPy thuần**, không dùng thư viện `scikit-learn` cho tiền xử lý, TF-IDF hay thuật toán học máy.
  - Có giải thích kết quả **Explainable AI (XAI)** qua **Top 5 Trigger Words** (5 từ khóa ảnh hưởng nhiều nhất đến quyết định).
  - Lưu trữ mô hình gọn nhẹ dưới dạng file JSON ([email_classifier.json](../models/email_classifier.json)) để có thể tái sử dụng dễ dàng.
---
## 2. WHERE (Các thành phần nằm ở đâu trong dự án?)
Dự án được phân chia thư mục rõ ràng theo từng nhiệm vụ:
| Thành phần | File / Thư mục | Nhiệm vụ chính |
|---|---|---|
| **Dữ liệu thô** | [`data/raw/emails.csv`](../data/raw/emails.csv) | Chứa 5.728 email gốc ban đầu từ Kaggle. |
| **Dữ liệu đã xử lý** | [`data/processed/`](../data/processed/) | Lưu tập Train/Val/Test và ma trận đặc trưng TF-IDF (`tfidf_splits.npz`). |
| **Tiền xử lý & EDA** | [`data/data_processing.py`](../data/data_processing.py), [`data/eda.py`](../data/eda.py) | Làm sạch văn bản, tạo meta-tokens và thống kê số lượng nhãn. |
| **Thuật toán From-Scratch** | [`models/algorithms.py`](../models/algorithms.py) | Code tự viết các thuật toán: `TfidfVectorizerScratch`, `MultinomialNaiveBayesScratch`, `ComplementNaiveBayesScratch`, `LogisticRegressionScratch`, `LinearSVMScratch`. |
| **File mô hình đã train** | [`models/email_classifier.json`](../models/email_classifier.json) | Lưu từ điển 2.500 từ, trọng số IDF và các tham số xác suất sau khi học xong. |
| **Script chạy chính** | [`train.py`](../train.py) | File chạy tự động toàn bộ bài: từ nạp dữ liệu đến train, dò threshold và xuất file model. |
| **Kiểm thử & Chạy thử** | [`scripts/eval.py`](../scripts/eval.py), [`scripts/demo.py`](../scripts/demo.py) | Đánh giá độc lập trên tập Test và chạy thử nghiệm một số email mẫu. |
| **Ứng dụng giao diện** | [`app.py`](../app.py), [`predict.py`](../predict.py) | Giao diện web Flask và công cụ dòng lệnh để người dùng nhập email vào kiểm tra. |
| **Báo cáo & Biểu đồ** | [`reports/BAO_CAO_LAB1.md`](../reports/BAO_CAO_LAB1.md), [`reports/figures/`](../reports/figures/) | Báo cáo chi tiết nộp bài và các hình ảnh biểu đồ phân tích. |
---
## 3. WHEN (Quy trình thực hiện theo thứ tự nào?)
Quy trình thực hiện bài thực hành diễn ra tuần tự từng bước để **tránh rò rỉ dữ liệu (Data Leakage)**:
```text
[Bước 1: Nạp và kiểm tra dữ liệu]
  └─ Đọc file emails.csv, kiểm tra giá trị thiếu (0 NaN), loại bỏ 33 email trùng lặp.
       │
[Bước 2: Chia dữ liệu (Stratified Split 70/15/15)]
  └─ Chia thành 3 tập Train (3.985) / Val (854) / Test (856) ngay từ đầu.
       │
[Bước 3: Học từ điển & IDF - CHỈ TRÊN TẬP TRAIN]
  └─ Làm sạch text, chọn 2.500 từ vựng phổ biến và tính trọng số IDF chỉ từ tập Train.
       │
[Bước 4: Biến đổi sang vector TF-IDF]
  └─ Dùng từ điển vừa học ở tập Train để biến đổi cả 3 tập sang ma trận số.
       │
[Bước 5: Huấn luyện và Chọn ngưỡng (Threshold Tuning)]
  └─ Huấn luyện các mô hình trên Train, sau đó thử nghiệm trên tập Val để chọn ngưỡng tối ưu (0.35).
       │
[Bước 6: Đánh giá cuối cùng trên tập Test]
  └─ Chạy kiểm thử trên tập Test độc lập để lấy điểm số thực tế (Accuracy 98.36%, F1 96.57%).
       │
[Bước 7: Lưu mô hình và Chạy thử]
  └─ Lưu file email_classifier.json và khởi chạy app.py để thử nghiệm thực tế.
```
---
## 4. WHY (Tại sao lại làm như vậy?)
Lý do đằng sau các quyết định kỹ thuật trong bài:

1. **Tại sao ưu tiên chỉ số Precision cao?**
   - Đánh dấu nhầm email quan trọng thành spam (*False Positive*) gây phiền toái hơn nhiều so với việc để lọt vài email rác (*False Negative*). Do đó mô hình cần đạt Precision > 97% để hạn chế tối đa việc bắt nhầm.
2. **Tại sao tự lập trình From-Scratch?**
   - Để hiểu sâu bản chất toán học của mô hình xác suất Naive Bayes, phương pháp hạ dốc Gradient Descent và cách biến đổi văn bản sang vector TF-IDF.
3. **Tại sao dùng Meta-Tokens (`http_url`, `email_address`, `money_amount`)?**
   - Link web lạ và số tiền là dấu hiệu rất rõ ràng của thư rác. Thay vì xóa bỏ, ta đổi chúng thành các thẻ đại diện để mô hình dễ nhận diện hơn.
4. **Tại sao chọn ngưỡng `0.35` thay vì `0.5`?**
   - Dữ liệu bị mất cân bằng (Spam chỉ chiếm ~24%). Việc hạ ngưỡng kích hoạt từ 0.5 xuống 0.35 giúp mô hình phát hiện được nhiều thư rác hơn (tăng Recall) mà vẫn giữ được độ chuẩn xác Precision rất cao.
5. **Tại sao lưu file JSON?**
   - File JSON nhẹ (~243 KB), dễ đọc, an toàn và có thể nạp vào bất kỳ ngôn ngữ nào mà không phụ thuộc vào thư viện bên ngoài.
---
## 5. HOW (Mô hình hoạt động và dự đoán như thế nào?)
Khi nhận một đoạn văn bản email mới, hàm `predict()` xử lý qua **5 bước**:
```text
[Văn bản email mới]
       │
       ▼
1. Làm sạch: Chuyển về chữ thường, thay link -> 'http_url', email -> 'email_address', tiền -> 'money_amount'.
       │
       ▼
2. Lọc từ: Loại bỏ 77 từ dừng thông dụng tiếng Anh ('the', 'is', 'at'...), giữ các từ >= 3 ký tự.
       │
       ▼
3. Đổi thành vector: Chiếu các từ vào từ điển 2.500 từ, nhân trọng số IDF và chuẩn hóa độ dài vector L2.
       │
       ▼
4. Tính xác suất: Nhân ma trận với các trọng số đã học để tính ra xác suất P(Ham) và P(Spam) qua hàm Softmax.
       │
       ▼
5. Kết luận:
   - Nếu xác suất Spam >= 0.35: Báo là SPAM (Thư rác).
   - Ngược lại: Báo là HAM (Hợp lệ).
   - Đồng thời trích xuất 5 từ khóa có trọng số chênh lệch lớn nhất để giải thích cho người dùng.
```
