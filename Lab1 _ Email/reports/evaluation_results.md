# Kết quả Lab 1: Phân loại Email Spam (100% From Scratch)

## 1. Kết quả trên tập Validation (Threshold = 0.50)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Multinomial NB | 0.9801 | 0.9796 | 0.9366 | 0.9576 |
| Complement NB | 0.9801 | 0.9796 | 0.9366 | 0.9576 |
| Logistic Regression | 0.9731 | 0.9892 | 0.8976 | 0.9412 |
| Linear SVM | 0.9895 | 0.9949 | 0.9610 | 0.9777 |

## 2. Hiệu chuẩn ngưỡng quyết định (Multinomial NB)

- Ngưỡng tối ưu (BEST_THRESHOLD): **0.24**
- Val F1-Score: **97.36%**

## 3. Kết quả chính thức trên tập Test độc lập (Threshold = 0.24)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|
| Multinomial NB | 0.9871 | 0.9621 | 0.9854 | 0.9736 | 0.9993 | 642 | 8 | 3 | 203 |
| Complement NB | 0.9871 | 0.9621 | 0.9854 | 0.9736 | 0.9993 | 642 | 8 | 3 | 203 |
| Logistic Regression | 0.9428 | 0.8078 | 1.0000 | 0.8937 | 0.9978 | 601 | 49 | 0 | 206 |
| Linear SVM | 0.8353 | 0.5937 | 1.0000 | 0.7450 | 0.9996 | 509 | 141 | 0 | 206 |
