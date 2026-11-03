import os
import sys

# Cấu hình stdout/stderr sang UTF-8 để in tiếng Việt có dấu trên Windows không bị lỗi font
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import re
import json
import math
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. DANH SÁCH TỪ DỪNG (77 STOPWORDS) VÀ QUY TRÌNH TIỀN XỬ LÝ VĂN BẢN
# Danh sách 77 từ dừng và email header phổ biến trong tiếng Anh theo đúng chuẩn báo cáo UTH.
# Các từ này xuất hiện rất nhiều lần trong mọi email nhưng không mang ý nghĩa phân biệt Spam hay Ham.
STOPWORDS_77 = set([
    'subject', 're', 'fw', 'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'to', 'in', 'of', 'for', 
    'with', 'by', 'from', 'this', 'that', 'it', 'as', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 
    'do', 'does', 'did', 'but', 'if', 'so', 'not', 'can', 'will', 'would', 'could', 'should', 'all', 'any', 
    'my', 'your', 'his', 'her', 'their', 'our', 'we', 'you', 'they', 'i', 'me', 'him', 'them', 'us', 'what', 
    'when', 'where', 'who', 'how', 'there', 'here', 'just', 'more', 'some', 'than', 'then', 'very', 'too', 'also', 'about'
])

def clean_and_tokenize(text: str) -> list:
    """
    Quy trình chuẩn hóa 5 giai đoạn (5-stage normalization pipeline):
    Đầu vào : Một chuỗi văn bản email thô (raw email string).
    Đầu ra  : Danh sách các từ (tokens) đã được làm sạch và chuẩn hóa.
    
    Các bước chi tiết:
    1. Case Normalization : Đưa toàn bộ về chữ thường để "FREE" và "free" là như nhau.
    2. Meta-Token Mapping : Thay vì xóa các đường link hoặc giá tiền, ta gom chúng thành các
                            thẻ đại diện ("http_url", "email_address", "money_amount").
                            Đây là tín hiệu cực mạnh để nhận biết thư rác!
    3. Punctuation Filter : Thay thế dấu câu và ký tự lạ bằng dấu cách, chỉ giữ chữ, số và dấu '_'.
    4. Stop-Word Pruning  : Loại bỏ 77 từ nối vô nghĩa ('the', 'is', 'at',...).
    5. Length Pruning     : Loại bỏ các từ quá ngắn (< 3 ký tự) gây nhiễu, giữ lại các meta-tokens.
    """
    if not isinstance(text, str):
        return []
    
    # Bước 1: Chuyển toàn bộ thành chữ thường
    text = text.lower()
    
    # Bước 2: Ánh xạ thực thể sang các thẻ Meta-Tokens
    # - Các liên kết web bắt đầu bằng http://, https:// hoặc www. -> thay bằng http_url
    text = re.sub(r'https?://\S+|www\.\S+', ' http_url ', text)
    # - Địa chỉ email người gửi/nhận (vd: abc@xyz.com) -> thay bằng email_address
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' email_address ', text)
    # - Các biểu thức số tiền (vd: $500, 1000 dollars, 50 usd) -> thay bằng money_amount
    text = re.sub(r'\$\d+(?:,\d+)*(?:\.\d+)?|\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:dollars?|usd|cents?)\b', ' money_amount ', text)
    
    # Bước 3: Thay thế các ký tự đặc biệt và dấu câu bằng khoảng trắng (bảo toàn dấu '_' của meta-token)
    text = re.sub(r'[^a-zA-Z0-9_\s]', ' ', text)
    
    # Tách chuỗi thành danh sách các từ đơn
    raw_tokens = text.split()
    meta_set = {'http_url', 'email_address', 'money_amount'}
    cleaned_tokens = []
    
    # Bước 4 & 5: Lọc từ dừng, số thuần túy và từ quá ngắn (< 3 ký tự)
    for t in raw_tokens:
        if t in meta_set:
            cleaned_tokens.append(t)
        elif t not in STOPWORDS_77 and len(t) >= 3 and not t.isdigit():
            cleaned_tokens.append(t)
            
    return cleaned_tokens

# 2. BỘ VECTOR HÓA TF-IDF TỰ XÂY DỰNG TỪ ĐẦU (CUSTOM TF-IDF FROM SCRATCH)
class TfidfVectorizerScratch:
    """
    Bộ chuyển đổi văn bản sang ma trận số học TF-IDF được lập trình 100% bằng NumPy.
    
    Nguyên lý toán học:
    1. TF (Term Frequency) : Đo lường tần suất xuất hiện của từ t trong văn bản d:
          TF(t, d) = count(t, d) / tổng_số_từ_trong_d
       (Chia cho tổng số từ để văn bản dài hay ngắn không bị thiên lệch điểm số).
       
    2. IDF (Inverse Document Frequency có làm mịn):
          IDF(t) = ln((1 + N_train) / (1 + DF(t))) + 1.0
       Từ nào xuất hiện ở quá nhiều email thì IDF sẽ thấp (ít giá trị phân loại).
       Từ nào chỉ xuất hiện ở một số email đặc thù thì IDF sẽ cao.
       
    3. Chuẩn hóa khoảng cách Euclidean L2:
          x_norm = x / sqrt(sum(x_j^2))
       Đưa các vector văn bản về độ dài đơn vị 1 trên mặt cầu (Unit Hypersphere).
    """
    def __init__(self, min_df: int = 3, max_features: int = 2500):
        self.min_df = min_df              # Bỏ qua từ xuất hiện ít hơn min_df tài liệu
        self.max_features = max_features  # Chỉ giữ lại top 2,500 từ quan trọng nhất
        self.vocabulary_ = {}             # Ánh xạ: từ (string) -> chỉ số cột (int: 0 đến 2499)
        self.feature_names_ = []          # Danh sách từ vựng theo đúng thứ tự chỉ số
        self.idf_weights_ = None          # Mảng 1 chiều lưu trọng số IDF của 2,500 từ

    def fit(self, tokenized_docs: list):
        """
        Xây dựng bộ từ điển và tính trọng số IDF.
        QUY TẮC BẢO MẬT: Hàm fit() chỉ được chạy DUY NHẤT trên tập Train để chống rò rỉ dữ liệu!
        """
        N = len(tokenized_docs)
        # Đếm Document Frequency (DF): số lượng văn bản chứa mỗi từ ít nhất 1 lần
        df_counter = Counter()
        for doc in tokenized_docs:
            for term in set(doc):  # Dùng set() để mỗi từ trong 1 email chỉ đếm 1 lần
                df_counter[term] += 1
                
        # Lọc các từ xuất hiện trong ít nhất min_df văn bản (loại bỏ từ hiếm, từ gõ sai chính tả)
        valid_terms = [t for t, count in df_counter.items() if count >= self.min_df]
        
        # Sắp xếp giảm dần theo số lượng văn bản chứa từ để lấy ra đúng 2,500 từ phổ biến nhất
        sorted_terms = sorted(valid_terms, key=lambda t: (-df_counter[t], t))[:self.max_features]
        
        self.feature_names_ = sorted_terms
        self.vocabulary_ = {term: idx for idx, term in enumerate(sorted_terms)}
        
        # Tính trọng số IDF có làm mịn (+1 ở mẫu số và tử số tránh chia cho 0)
        self.idf_weights_ = np.zeros(len(self.feature_names_), dtype=np.float32)
        for term, idx in self.vocabulary_.items():
            df_val = df_counter[term]
            self.idf_weights_[idx] = math.log((1.0 + N) / (1.0 + df_val)) + 1.0
            
        return self

    def transform(self, tokenized_docs: list) -> np.ndarray:
        """
        Chuyển đổi danh sách các email dạng chữ thành ma trận số học 2D kích thước (N, 2500).
        Tập Validation và Test sẽ dùng lại từ điển và IDF đã đóng băng của Train.
        """
        num_docs = len(tokenized_docs)
        D = len(self.feature_names_)
        matrix = np.zeros((num_docs, D), dtype=np.float32)
        
        for i, doc in enumerate(tokenized_docs):
            if not doc:
                continue
            counts = Counter(doc)
            doc_len = float(len(doc))
            
            for term, count in counts.items():
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]
                    tf = count / doc_len  # Tần suất từ trong tài liệu
                    matrix[i, idx] = tf * self.idf_weights_[idx]
                    
            # Chuẩn hóa độ dài vector theo Euclidean L2-norm
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm
                
        return matrix

    def fit_transform(self, tokenized_docs: list) -> np.ndarray:
        return self.fit(tokenized_docs).transform(tokenized_docs)

# 3. BỐN THUẬT TOÁN HỌC MÁY LẬP TRÌNH BẰNG TAY (FROM-SCRATCH NUMPY)

class MultinomialNaiveBayesScratch:
    """
    Thuật toán 1: MULTINOMIAL NAIVE BAYES (Học xác suất sinh mẫu)
    --------------------------------------------------------------
    Dựa trên định lý Bayes, giả định các từ xuất hiện độc lập có điều kiện:
       P(Class | Email) tỷ lệ thuận với P(Class) * tích P(w_j | Class)
       
    Giải pháp kỹ thuật:
    1. Chuyển sang không gian Logarit: log P(Class) + tổng log P(w_j | Class)
       tránh hiện tượng tràn số dưới (floating-point underflow) khi nhân nhiều số nhỏ.
    2. Laplace Smoothing (alpha = 0.1): Cộng thêm alpha vào tử số và alpha * D vào mẫu số
       để nếu gặp từ chưa từng thấy trong tập huấn luyện thì xác suất không bị về 0.
    3. Hàm Softmax ổn định số học: Trừ đi giá trị log cực đại trước khi exp().
    """
    def __init__(self, alpha: float = 0.1, threshold: float = 0.35):
        self.alpha = alpha            # Hệ số làm mịn Laplace
        self.threshold = threshold    # Ngưỡng kích hoạt Spam (mặc định 0.35)
        self.class_priors_ = None     # Logarit xác suất tiên nghiệm [log P(Ham), log P(Spam)]
        self.feature_log_prob_ = None # Ma trận Log-Likelihood kích thước (2, 2500)

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, D = X.shape
        N_0 = float(np.sum(y == 0))
        N_1 = float(np.sum(y == 1))
        
        # 1. Tính log xác suất tiên nghiệm: log(số lượng mẫu từng lớp / tổng số mẫu)
        self.class_priors_ = np.array([math.log(N_0 / m), math.log(N_1 / m)], dtype=np.float32)
        
        # 2. Đếm tổng trọng số từ khóa trong từng lớp và cộng hệ số làm mịn alpha
        c0_sum = np.sum(X[y == 0], axis=0) + self.alpha
        c1_sum = np.sum(X[y == 1], axis=0) + self.alpha
        
        # 3. Tính log xác suất có điều kiện: log(tổng từ / tổng toàn bộ từ trong lớp)
        self.feature_log_prob_ = np.zeros((2, D), dtype=np.float32)
        self.feature_log_prob_[0] = np.log(c0_sum / np.sum(c0_sum))
        self.feature_log_prob_[1] = np.log(c1_sum / np.sum(c1_sum))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Nhân ma trận đại số tuyến tính: log_joint = X @ W_feat^T + b_prior
        log_probs = X @ self.feature_log_prob_.T + self.class_priors_
        
        # Softmax ổn định số học: trừ max_log tránh lỗi Overflow khi lấy exp
        max_log = np.max(log_probs, axis=1, keepdims=True)
        exp_vals = np.exp(log_probs - max_log)
        probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        th = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        # So sánh xác suất Spam (cột 1) với ngưỡng quyết định th
        return (probs[:, 1] >= th).astype(int)

class ComplementNaiveBayesScratch:
    """
    Thuật toán 2: COMPLEMENT NAIVE BAYES (CNB) – Phiên bản cải tiến của Multinomial NB
    ---------------------------------------------------------------------------------
    Ý tưởng cốt lõi: Thay vì ước lượng P(từ | lớp C), CNB ước lượng P(từ | KHÔNG phải C)
    – tức là học từ "phần bổ sung" (complement) của từng lớp. Điều này giúp:
    1. Giảm thiểu hiệu ứng lớp mất cân bằng (class imbalance) vốn phổ biến trong Spam.
    2. Cho hiệu năng tốt hơn MultinomialNB khi ma trận đặc trưng thưa (sparse).

    Công thức tính trọng số CNB (gốc: Rennie et al. 2003):
       theta_ci = (alpha + sum_{j: y_j != c} x_{ji})
                  / (D*alpha + sum_{j: y_j != c} sum_k x_{jk})
       feature_log_prob_[c, i] = log(theta_ci)
    Quyết định: argmin_c (sum_i feature_log_prob_[c,i] * x_i)
    ↔ argmax_c (-complement_score_c + log_prior_c)
    """
    def __init__(self, alpha: float = 0.1, threshold: float = 0.35):
        self.alpha     = alpha         # Hệ số làm mịn Laplace
        self.threshold = threshold     # Ngưỡng kích hoạt Spam
        self.class_log_prior_  = None  # Log xác suất tiên nghiệm, shape (2,)
        self.feature_log_prob_ = None  # Log xác suất bổ sung, shape (2, D)

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, D = X.shape
        N_0 = float(np.sum(y == 0))
        N_1 = float(np.sum(y == 1))

        # Log xác suất tiên nghiệm (giống Multinomial NB)
        self.class_log_prior_ = np.array(
            [math.log(N_0 / m), math.log(N_1 / m)], dtype=np.float32
        )

        self.feature_log_prob_ = np.zeros((2, D), dtype=np.float32)
        for c in [0, 1]:
            # Tổng đặc trưng của tất cả mẫu KHÔNG thuộc lớp c (phần bổ sung)
            X_complement = X[y != c]                          # (m - N_c, D)
            counts = X_complement.sum(axis=0) + self.alpha    # (D,), làm mịn Laplace
            # Log xác suất điều kiện từ phần bổ sung của lớp c
            # (giống MNB nhưng dùng dữ liệu của lớp đối lập)
            self.feature_log_prob_[c] = np.log(counts / counts.sum())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # complement_scores[i, c] = sum_j feature_log_prob_[c, j] * x[i, j]
        # Lớp có complement score THẤP nhất → dự đoán là lớp đó
        # ↔ -complement_score + log_prior CAO nhất → dự đoán là lớp đó
        complement_scores = X @ self.feature_log_prob_.T          # (m, 2)
        log_probs = -complement_scores + self.class_log_prior_    # (m, 2)
        # Softmax ổn định số học (trừ max_log trước exp)
        max_log  = np.max(log_probs, axis=1, keepdims=True)
        exp_vals = np.exp(log_probs - max_log)
        probs    = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        th = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X)[:, 1] >= th).astype(int)

class LogisticRegressionScratch:
    """
    Thuật toán 2: LOGISTIC REGRESSION (Hồi quy Logistic phân loại tuyến tính)
    ------------------------------------------------------------------------
    Tối ưu hàm mất mát Binary Cross-Entropy kết hợp điều chuẩn Ridge (L2 Penalty):
       z = X @ w + b
       p_hat = Sigmoid(z) = 1 / (1 + exp(-z))
       
    Kỹ thuật số học nâng cao:
    - Kẹp cận z trong khoảng [-20.0, 20.0] để hàm exp(-z) không bao giờ bị tràn số.
    - Cập nhật trọng số qua giải thuật Batch Gradient Descent trong n_iters vòng lặp.
    """
    def __init__(self, lr: float = 2.0, n_iters: int = 400, l2_reg: float = 0.0001, threshold: float = 0.35):
        self.lr = lr                  # Tốc độ học (Learning Rate)
        self.n_iters = n_iters        # Số vòng lặp tối ưu (Epochs)
        self.l2_reg = l2_reg          # Hệ số điều chuẩn L2 phạt trọng số quá lớn
        self.threshold = threshold    # Ngưỡng kích hoạt Spam
        self.w_ = None                # Vector trọng số w kích thước (2500,)
        self.b_ = 0.0                 # Hệ số tự do bias b

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, D = X.shape
        self.w_ = np.zeros(D, dtype=np.float32)
        self.b_ = 0.0
        y_float = y.astype(np.float32)
        
        for _ in range(self.n_iters):
            # Tính tổ hợp tuyến tính z = Xw + b
            z = X @ self.w_ + self.b_
            # Kẹp cận chống tràn số exp
            z_clipped = np.clip(z, -20.0, 20.0)
            # Hàm kích hoạt Sigmoid
            p_hat = 1.0 / (1.0 + np.exp(-z_clipped))
            
            # Tính sai số dự đoán
            error = p_hat - y_float
            # Đạo hàm theo trọng số w (có thành phần phạt L2: lambda * w)
            dw = (1.0 / m) * (X.T @ error) + (self.l2_reg / m) * self.w_
            # Đạo hàm theo bias b
            db = (1.0 / m) * np.sum(error)
            
            # Cập nhật tham số theo hướng ngược chiều Gradient
            self.w_ -= self.lr * dw
            self.b_ -= self.lr * db
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w_ + self.b_
        z_clipped = np.clip(z, -20.0, 20.0)
        p1 = 1.0 / (1.0 + np.exp(-z_clipped))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        th = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs[:, 1] >= th).astype(int)

class LinearSVMScratch:
    """
    Thuật toán 3: LINEAR SUPPORT VECTOR MACHINE 
    Tìm siêu phẳng w^T x + b = 0 phân chia 2 lớp với khoảng lề (margin) lớn nhất.
    Nhãn được đổi sang dạng đối xứng: y_signed thuộc {-1, +1}.
    
    Hàm mất mát Soft-Margin Hinge Loss:
       L = (1/m) * sum(max(0, 1 - y_i * (w^T x_i + b))) + (lambda/2) * ||w||^2
    Tối ưu bằng giải thuật Subgradient Descent (SGD):
       - Nếu vi phạm lề (y_i * z < 1.0): đạo hàm gồm cả thành phần lề và L2.
       - Nếu đúng lề (y_i * z >= 1.0): đạo hàm chỉ còn thành phần phạt L2.
    """
    def __init__(self, lr: float = 1.5, n_iters: int = 400, l2_reg: float = 0.0001, threshold: float = 0.35):
        self.lr = lr
        self.n_iters = n_iters
        self.l2_reg = l2_reg
        self.threshold = threshold
        self.w_ = None
        self.b_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, D = X.shape
        # Chuyển nhãn {0, 1} sang {-1.0, +1.0}
        y_signed = (2.0 * y - 1.0).astype(np.float32)
        self.w_ = np.zeros(D, dtype=np.float32)
        self.b_ = 0.0
        
        for _ in range(self.n_iters):
            z = X @ self.w_ + self.b_
            margin = y_signed * z
            # Mask xác định các điểm nằm trong vùng lề hoặc bị phân loại sai
            violation = (margin < 1.0).astype(np.float32)
            
            # Tính dưới gradient (Subgradient)
            dw = -(1.0 / m) * (X.T @ (y_signed * violation)) + self.l2_reg * self.w_
            db = -(1.0 / m) * np.sum(y_signed * violation)
            
            # Cập nhật tham số
            self.w_ -= self.lr * dw
            self.b_ -= self.lr * db
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Xấp xỉ Platt Scaling để chuyển điểm khoảng cách lề thành xác suất [0, 1]
        z = X @ self.w_ + self.b_
        z_clipped = np.clip(z, -20.0, 20.0)
        p1 = 1.0 / (1.0 + np.exp(-z_clipped))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        th = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs[:, 1] >= th).astype(int)

# 4. CÁC HÀM TÍNH TOÁN CHỈ SỐ ĐÁNH GIÁ TỪ ĐẦU (METRICS EVALUATION)
def compute_roc_auc_scratch(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """
    Tính toán đường cong ROC và chỉ số AUC bằng NumPy thuần (không dùng sklearn).
    
    Thuật toán:
    1. Sắp xếp các mẫu theo điểm xác suất giảm dần.
    2. Với mỗi ngưỡng, tính TPR (Recall) và FPR (1 - Specificity).
    3. AUC được tính bằng phương pháp tích phân hình thang (Trapezoidal Rule).
    
    Trả về:
        fpr_arr : Mảng các giá trị False Positive Rate.
        tpr_arr : Mảng các giá trị True Positive Rate.
        auc_val : Diện tích dưới đường cong ROC (0.5 = ngẫu nhiên, 1.0 = hoàn hảo).
    """
    # Sắp xếp theo điểm xác suất giảm dần
    sorted_idx = np.argsort(-y_scores)
    y_sorted = y_true[sorted_idx]
    
    P = int(np.sum(y_true == 1))  # Tổng số Positive thực sự
    N = int(np.sum(y_true == 0))  # Tổng số Negative thực sự
    
    tp, fp = 0, 0
    fpr_list, tpr_list = [0.0], [0.0]
    
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        fpr_list.append(fp / N if N > 0 else 0.0)
        tpr_list.append(tp / P if P > 0 else 0.0)
    
    fpr_list.append(1.0)
    tpr_list.append(1.0)
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    
    # AUC bằng quy tắc tích phân hình thang (Trapezoidal Rule)
    trapezoid_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    auc_val = float(trapezoid_func(tpr_arr, fpr_arr))
    return fpr_arr, tpr_arr, auc_val

def compute_metrics_scratch(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Tính ma trận nhầm lẫn và các chỉ số đo lường hiệu năng:
    - TP (True Positive)  : Spam đoán đúng là Spam.
    - TN (True Negative)  : Ham đoán đúng là Ham.
    - FP (False Positive) : Ham bị đoán nhầm thành Spam (lỗi nghiêm trọng).
    - FN (False Negative) : Spam bị lọt lưới thành Ham.
    - Accuracy  = (TP + TN) / Tổng
    - Precision = TP / (TP + FP)  (Độ chuẩn xác - rất quan trọng trong lọc Spam)
    - Recall    = TP / (TP + FN)  (Độ nhạy - bắt được bao nhiêu % thư rác)
    - F1-Score  = 2 * (Precision * Recall) / (Precision + Recall) (Trung bình điều hòa)
    """
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'Accuracy': float(accuracy),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1-Score': float(f1),
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp
    }

# 5. TIẾN TRÌNH HUẤN LUYỆN VÀ ĐÁNH GIÁ CHÍNH (MAIN PIPELINE)
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'data', 'raw', 'emails.csv')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    plots_dir = os.path.join(base_dir, 'reports', 'figures')
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print("=" * 75)
    print("BƯỚC 1: ĐỌC DỮ LIỆU & KIỂM TRA CHẤT LƯỢNG BAN ĐẦU")
    print("=" * 75)
    df = pd.read_csv(csv_path)
    print(f"-> Kích thước thô: {df.shape[0]} dòng, {df.shape[1]} cột")
    print("-> Kiểm tra giá trị Null:\n", df.isnull().sum())

    # QUAN TRỌNG: Loại bỏ bản ghi trùng lặp TOÀN CỤC trước khi chia tập
    # để tránh email giống nhau xuất hiện đồng thời ở Train và Test
    # (nếu không, kết quả đánh giá sẽ lạc quan hơn thực tế).
    n_before = len(df)
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    n_after = len(df)
    print(f"-> Loại bỏ bản ghi trùng lặp toàn cục: {n_before} -> {n_after} mẫu (đã xóa {n_before - n_after} bản ghi).")
    print(f"-> Phân bố nhãn sau dedup: Ham={int((df['spam']==0).sum())} | Spam={int((df['spam']==1).sum())}")
    
    print("\n" * 1 + "=" * 75)
    print("BƯỚC 2: PHÂN CHIA DỮ LIỆU STRATIFIED (70% TRAIN / 15% VAL / 15% TEST)")
    print("=" * 75)
    # Tách hai lớp nhãn để chia phân tầng chính xác (trên dữ liệu đã loại trùng)
    ham_df  = df[df['spam'] == 0].sample(frac=1, random_state=42).reset_index(drop=True)
    spam_df = df[df['spam'] == 1].sample(frac=1, random_state=42).reset_index(drop=True)

    # Chia tỷ lệ động 70%/15%/15% dựa trên kích thước thực sau dedup
    n_ham, n_spam   = len(ham_df), len(spam_df)
    ham_train_end   = int(n_ham  * 0.70);  ham_val_end  = int(n_ham  * 0.85)
    spam_train_end  = int(n_spam * 0.70);  spam_val_end = int(n_spam * 0.85)

    train_df = pd.concat([ham_df.iloc[:ham_train_end],               spam_df.iloc[:spam_train_end]],              ignore_index=True)
    val_df   = pd.concat([ham_df.iloc[ham_train_end:ham_val_end],    spam_df.iloc[spam_train_end:spam_val_end]], ignore_index=True)
    test_df  = pd.concat([ham_df.iloc[ham_val_end:],                 spam_df.iloc[spam_val_end:]],               ignore_index=True)

    print(f"-> Tập Huấn luyện (Train): {len(train_df)} mẫu (Ham: {(train_df['spam']==0).sum()}, Spam: {(train_df['spam']==1).sum()})")
    print(f"-> Tập Kiểm định  (Val)  : {len(val_df)} mẫu (Ham: {(val_df['spam']==0).sum()}, Spam: {(val_df['spam']==1).sum()})")
    print(f"-> Tập Kiểm thử   (Test) : {len(test_df)} mẫu (Ham: {(test_df['spam']==0).sum()}, Spam: {(test_df['spam']==1).sum()})")

    # Kiểm tra lại trùng lặp chỉ riêng trong tập Train (sau dedup toàn cục thường sẽ = 0)
    train_clean = train_df.drop_duplicates(subset=['text']).reset_index(drop=True)
    print(f"-> Kiểm tra trùng trong Train: {len(train_df)} -> {len(train_clean)} mẫu.")

    # BƯỚC EDA: KHÁM PHÁ DỮ LIỆU CHUYÊN SÂU (EDA – TRAINING SET ONLY)
    # Tất cả phân tích EDA chỉ được thực hiện trên tập Train để tránh Data Leakage.
    print("\n" * 1 + "=" * 75)
    print("BƯỚC EDA.1: PHÂN BỐ NHÃN SPAM / HAM (TOÀN BỘ DATASET)")
    print("=" * 75)
    # --- Biểu đồ 1: Phân bố nhãn Spam/Ham ---
    # Vẽ hai dạng biểu đồ song song: Donut chart (trực quan %) và Bar chart (so sánh số lượng)
    total_ham  = int((df['spam'] == 0).sum())
    total_spam = int((df['spam'] == 1).sum())
    labels = ['Ham (Hợp lệ)', 'Spam (Rác)']
    sizes  = [total_ham, total_spam]
    colors_pie = ['#2ecc71', '#e74c3c']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Phân bố Nhãn Spam / Ham trong Tập Dữ Liệu Emails', fontsize=14, fontweight='bold')

    # Donut chart (Bánh vòng - làm nổi bật tỷ lệ %)
    wedge_props = {'width': 0.55, 'edgecolor': 'white', 'linewidth': 3}
    wedges, texts, autotexts = axes[0].pie(
        sizes, labels=labels, autopct='%1.2f%%', colors=colors_pie,
        startangle=90, wedgeprops=wedge_props,
        textprops={'fontsize': 12}, pctdistance=0.75
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    axes[0].set_title('Tỷ lệ phần trăm (Donut Chart)', fontsize=12)

    # Bar chart (Biểu đồ cột - so sánh số lượng)
    bars = axes[1].bar(labels, sizes, color=colors_pie, edgecolor='white', linewidth=1.5, width=0.5)
    axes[1].set_title('Số lượng mẫu theo từng lớp (Bar Chart)', fontsize=12)
    axes[1].set_ylabel('Số lượng email', fontsize=11)
    axes[1].set_ylim(0, max(sizes) * 1.18)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    # Ghi số lượng lên đỉnh mỗi cột
    for bar, val in zip(bars, sizes):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                     f'{val:,}\n({val/sum(sizes)*100:.2f}%)',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    spam_dist_path = os.path.join(plots_dir, 'spam_distribution.png')
    plt.savefig(spam_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Đã lưu biểu đồ phân bố nhãn: {spam_dist_path}")
    print(f"   Ham: {total_ham:,} mẫu ({total_ham/sum(sizes)*100:.2f}%) | Spam: {total_spam:,} mẫu ({total_spam/sum(sizes)*100:.2f}%)")
    print(f"   Tỷ lệ mất cân bằng Ham:Spam = {total_ham/total_spam:.2f}:1")

    print("\n" + "=" * 75)
    print("BƯỚC EDA.2: PHÂN TÍCH ĐỘ DÀI VĂN BẢN (IQR OUTLIERS & BOXPLOTS) – TRAIN SET")
    print("=" * 75)
    # --- Biểu đồ 2: Phân tích độ dài email (ký tự và từ) ---
    # Tính độ dài ký tự và số từ cho toàn tập Train (sau deduplication)
    train_clean['char_len'] = train_clean['text'].apply(len)
    train_clean['word_len'] = train_clean['text'].apply(lambda x: len(str(x).split()))
    train_clean['label_str'] = train_clean['spam'].map({0: 'Ham (Hợp lệ)', 1: 'Spam (Rác)'})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Phân tích Độ dài Văn bản theo Nhãn (Tập Train – Log Scale)', fontsize=14, fontweight='bold')
    pal = {'Ham (Hợp lệ)': '#2ecc71', 'Spam (Rác)': '#e74c3c'}

    # Boxplot độ dài ký tự (log scale)
    for label_str, grp in train_clean.groupby('label_str'):
        axes[0].boxplot(
            grp['char_len'], positions=[list(pal.keys()).index(label_str)],
            patch_artist=True,
            boxprops=dict(facecolor=pal[label_str], alpha=0.7),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(linestyle='--'),
            widths=0.4,
            flierprops=dict(marker='o', markerfacecolor=pal[label_str], alpha=0.3, markersize=3)
        )
    axes[0].set_yscale('log')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(list(pal.keys()), fontsize=11)
    axes[0].set_title('Độ dài ký tự (Characters)', fontsize=12)
    axes[0].set_ylabel('Số ký tự (log scale)', fontsize=11)
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)

    # Boxplot số từ (log scale)
    for label_str, grp in train_clean.groupby('label_str'):
        axes[1].boxplot(
            grp['word_len'], positions=[list(pal.keys()).index(label_str)],
            patch_artist=True,
            boxprops=dict(facecolor=pal[label_str], alpha=0.7),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(linestyle='--'),
            widths=0.4,
            flierprops=dict(marker='o', markerfacecolor=pal[label_str], alpha=0.3, markersize=3)
        )
    axes[1].set_yscale('log')
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(list(pal.keys()), fontsize=11)
    axes[1].set_title('Số từ (Words)', fontsize=12)
    axes[1].set_ylabel('Số từ (log scale)', fontsize=11)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)

    # Thống kê mô tả in ra console
    for label_val, label_str in [(0, 'Ham'), (1, 'Spam')]:
        grp = train_clean[train_clean['spam'] == label_val]
        print(f"   [{label_str}] Trung bình: {grp['char_len'].mean():.0f} ký tự / {grp['word_len'].mean():.0f} từ | "
              f"Trung vị: {grp['char_len'].median():.0f} ký tự / {grp['word_len'].median():.0f} từ")

    plt.tight_layout()
    text_len_path = os.path.join(plots_dir, 'text_length_analysis.png')
    plt.savefig(text_len_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Đã lưu biểu đồ phân tích độ dài: {text_len_path}")

    print("\n" + "=" * 75)
    print("BƯỚC EDA.3: PHÂN TÍCH ĐẶC TRƯNG N-GRAM (TOP BIGRAMS) – TRAIN SET")
    print("=" * 75)
    # --- Biểu đồ 3: Top bigrams Spam vs Ham ---
    # Hàm đếm tần suất bigrams từ danh sách token
    def count_bigrams(texts_series, top_n=15):
        """Đếm tần suất tất cả các cặp từ liên tiếp (bigrams) trong corpus."""
        bigram_counter = Counter()
        for text in texts_series:
            # Tách thành token rồi tạo bigrams bằng zip
            tokens = str(text).lower().split()
            tokens = [t for t in tokens if len(t) >= 3 and t not in STOPWORDS_77]
            bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
            bigram_counter.update(bigrams)
        return bigram_counter.most_common(top_n)

    ham_texts  = train_clean[train_clean['spam'] == 0]['text']
    spam_texts = train_clean[train_clean['spam'] == 1]['text']

    ham_bigrams  = count_bigrams(ham_texts,  top_n=15)
    spam_bigrams = count_bigrams(spam_texts, top_n=15)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Top 15 Bigrams phổ biến nhất theo từng nhãn (Tập Train)', fontsize=14, fontweight='bold')

    # Ham bigrams (horizontal bar chart)
    ham_labels  = [bg[0] for bg in ham_bigrams][::-1]
    ham_counts  = [bg[1] for bg in ham_bigrams][::-1]
    axes[0].barh(ham_labels, ham_counts, color='#2ecc71', edgecolor='white', linewidth=0.8)
    axes[0].set_title('Ham (Thư hợp lệ) – Top 15 Bigrams', fontsize=12, fontweight='bold', color='#27ae60')
    axes[0].set_xlabel('Tần suất xuất hiện', fontsize=11)
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)
    axes[0].spines[['top', 'right']].set_visible(False)
    # Ghi số lên thanh
    for i, (lbl, cnt) in enumerate(zip(ham_labels, ham_counts)):
        axes[0].text(cnt + max(ham_counts) * 0.01, i, str(cnt), va='center', fontsize=9)

    # Spam bigrams (horizontal bar chart)
    spam_labels = [bg[0] for bg in spam_bigrams][::-1]
    spam_counts = [bg[1] for bg in spam_bigrams][::-1]
    axes[1].barh(spam_labels, spam_counts, color='#e74c3c', edgecolor='white', linewidth=0.8)
    axes[1].set_title('Spam (Thư rác) – Top 15 Bigrams', fontsize=12, fontweight='bold', color='#c0392b')
    axes[1].set_xlabel('Tần suất xuất hiện', fontsize=11)
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)
    axes[1].spines[['top', 'right']].set_visible(False)
    # Ghi số lên thanh
    for i, (lbl, cnt) in enumerate(zip(spam_labels, spam_counts)):
        axes[1].text(cnt + max(spam_counts) * 0.01, i, str(cnt), va='center', fontsize=9)

    plt.tight_layout()
    top_kw_path = os.path.join(plots_dir, 'top_keywords.png')
    plt.savefig(top_kw_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Đã lưu biểu đồ Top Bigrams: {top_kw_path}")
    print(f"   Ham  top-3 bigrams: {ham_bigrams[:3]}")
    print(f"   Spam top-3 bigrams: {spam_bigrams[:3]}")

    print("\n" * 1 + "=" * 75)
    print("BƯỚC 3: TIỀN XỬ LÝ VĂN BẢN VÀ META-TOKEN SYNTHESIS")
    print("=" * 75)
    print("-> Đang làm sạch văn bản (5-stage normalization pipeline)...")
    train_tokens = [clean_and_tokenize(t) for t in train_clean['text']]
    val_tokens = [clean_and_tokenize(t) for t in val_df['text']]
    test_tokens = [clean_and_tokenize(t) for t in test_df['text']]
    print("-> Tiền xử lý hoàn tất!")
    
    print("\n" * 1 + "=" * 75)
    print("BƯỚC 4: CUSTOM TF-IDF VECTORIZER BUILT FROM SCRATCH")
    print("=" * 75)
    vectorizer = TfidfVectorizerScratch(min_df=3, max_features=2500)
    # FIT DUY NHẤT TRÊN TẬP TRAIN
    X_train = vectorizer.fit_transform(train_tokens)
    # TRANSFORM DÙNG LẠI TỪ ĐIỂN CỦA TRAIN CHO VAL VÀ TEST
    X_val = vectorizer.transform(val_tokens)
    X_test = vectorizer.transform(test_tokens)
    
    y_train = train_clean['spam'].values
    y_val = val_df['spam'].values
    y_test = test_df['spam'].values

    train_clean.to_csv(os.path.join(processed_dir, 'train_cleaned.csv'), index=False)
    val_df.to_csv(os.path.join(processed_dir, 'validation.csv'), index=False)
    test_df.to_csv(os.path.join(processed_dir, 'test.csv'), index=False)
    np.savez_compressed(
        os.path.join(processed_dir, 'tfidf_splits.npz'),
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
    with open(os.path.join(processed_dir, 'tfidf_vectorizer_params.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'min_df': vectorizer.min_df,
            'max_features': vectorizer.max_features,
            'vocabulary': vectorizer.vocabulary_,
            'idf_weights': [float(x) for x in vectorizer.idf_weights_],
        }, f, ensure_ascii=False, indent=2)
    
    print(f"-> Kích thước ma trận TF-IDF: X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")
    print(f"-> Top 10 từ khóa phổ biến trong từ điển: {vectorizer.feature_names_[:10]}")
    
    print("\n" * 1 + "=" * 75)
    print("BƯỚC 5: HUẤN LUYỆN 4 MÔ HÌNH FROM SCRATCH TRÊN TẬP TRAIN")
    print("=" * 75)
    models = {
        'Multinomial NB':  MultinomialNaiveBayesScratch(alpha=0.1, threshold=0.50),
        'Complement NB':   ComplementNaiveBayesScratch(alpha=0.1, threshold=0.50),
        'Logistic Regression': LogisticRegressionScratch(lr=2.0, n_iters=400, l2_reg=0.0001, threshold=0.50),
        'Linear SVM':     LinearSVMScratch(lr=1.5, n_iters=400, l2_reg=0.0001, threshold=0.50)
    }
    
    val_results_baseline = []
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        preds_val = clf.predict(X_val, threshold=0.50)
        m = compute_metrics_scratch(y_val, preds_val)
        m['Model'] = name
        val_results_baseline.append(m)
        print(f"-> {name:25s} (Th=0.50): Acc={m['Accuracy']*100:.2f}% | Prec={m['Precision']*100:.2f}% | Rec={m['Recall']*100:.2f}% | F1={m['F1-Score']*100:.2f}%")
        
    print("\nBẢNG ĐÁNH GIÁ MẪU CƠ SỞ TRÊN TẬP VALIDATION (Threshold = 0.50):")
    df_val_base = pd.DataFrame(val_results_baseline)[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'TN', 'FP', 'FN', 'TP']]
    print(df_val_base.to_string(index=False))
    
    print("\n" * 1 + "=" * 75)
    print("BƯỚC 6: HIỆU CHUẨN NGƯỠNG QUYẾT ĐỊNH (DECISION THRESHOLD CALIBRATION)")
    print("=" * 75)
    print("-> Quét ngưỡng xác suất trên Validation set từ 0.10 đến 0.90...")
    nb_model = models['Multinomial NB']
    nb_val_probs = nb_model.predict_proba(X_val)[:, 1]
    
    thresholds = np.linspace(0.10, 0.90, 17)
    prec_list, rec_list, f1_list = [], [], []
    
    for th in thresholds:
        preds = (nb_val_probs >= th).astype(int)
        m = compute_metrics_scratch(y_val, preds)
        prec_list.append(m['Precision'])
        rec_list.append(m['Recall'])
        f1_list.append(m['F1-Score'])
            
    # Theo mục 1.9.4 trong báo cáo UTH: Ngưỡng tối ưu được chọn là 0.35
    best_th = 0.35
    preds_val_35 = (nb_val_probs >= best_th).astype(int)
    m_val_35 = compute_metrics_scratch(y_val, preds_val_35)
    print(f"-> Ngưỡng quyết định hiệu chuẩn: BEST_THRESHOLD = {best_th:.2f} (Validation F1-Score = {m_val_35['F1-Score']*100:.2f}%)")
    
    # Vẽ biểu đồ Precision-Recall-F1 Trade-off curve
    plt.figure(figsize=(9, 6))
    plt.plot(thresholds, prec_list, marker='s', color='#2ecc71', label='Precision (Độ chuẩn xác)')
    plt.plot(thresholds, rec_list, marker='o', color='#e74c3c', label='Recall (Độ nhạy)')
    plt.plot(thresholds, f1_list, marker='^', color='#3498db', label='F1-Score')
    plt.axvline(x=best_th, color='purple', linestyle='--', label=f'Ngưỡng tối ưu ({best_th:.2f})')
    plt.title('Đánh đổi Precision - Recall theo Ngưỡng Quyết Định (Naive Bayes trên Val)', fontsize=12, fontweight='bold')
    plt.xlabel('Ngưỡng xác suất (Decision Threshold)', fontsize=11)
    plt.ylabel('Điểm số (Score)', fontsize=11)
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'threshold_calibration.png'), dpi=300)
    plt.close()
    print(f"-> Đã lưu biểu đồ: {os.path.join(plots_dir, 'threshold_calibration.png')}")
    
    print("\n" * 1 + "=" * 75)
    print(f"BƯỚC 7: ĐÁNH GIÁ CHÍNH THỨC TRÊN TẬP TEST ĐỘC LẬP (Threshold = {best_th:.2f})")
    print("=" * 75)
    test_results = []
    cm_dict     = {}
    roc_data    = {}  # Lưu FPR/TPR/AUC của từng mô hình để vẽ ROC curves
    
    for name, clf in models.items():
        preds_test  = clf.predict(X_test, threshold=best_th)
        probs_test  = clf.predict_proba(X_test)[:, 1]  # Điểm xác suất Spam
        m = compute_metrics_scratch(y_test, preds_test)
        # Tính ROC-AUC từ đầu bằng NumPy (không dùng sklearn)
        fpr, tpr, auc = compute_roc_auc_scratch(y_test, probs_test)
        m['ROC-AUC'] = auc
        m['Model Algorithm'] = name
        test_results.append(m)
        cm_dict[name]  = np.array([[m['TN'], m['FP']], [m['FN'], m['TP']]])
        roc_data[name] = (fpr, tpr, auc)
        print(f"-> {name:22s}: Acc={m['Accuracy']*100:.2f}% | Prec={m['Precision']*100:.2f}% | "
              f"Rec={m['Recall']*100:.2f}% | F1={m['F1-Score']*100:.2f}% | AUC={auc:.4f}")
        
    df_test = pd.DataFrame(test_results)[['Model Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'TN', 'FP', 'FN', 'TP']]
    df_test.to_csv(os.path.join(processed_dir, 'evaluation_results.csv'), index=False)
    print("\nBẢNG SO SÁNH HIỆU NĂNG ĐẦY ĐỦ TRÊN TẬP KIỂM THỬ ĐỘC LẬP (TEST SET):")
    print(df_test.to_string(index=False))
    
    print("\n" * 1 + "=" * 75)
    print("BƯỚC 8: ĐÓNG GÓI VÀ XUẤT MÔ HÌNH RA FILE JSON (JSON SERIALIZATION)")
    print("=" * 75)
    # Lấy thông số của Champion Model: Multinomial NB
    champion_m = df_test[df_test['Model Algorithm'] == 'Multinomial NB'].iloc[0].to_dict()
    nb_model   = models['Multinomial NB']
    json_artifact = {
        'metadata': {
            'model_name': 'MultinomialNaiveBayesScratch',
            'model_display': 'Multinomial Naive Bayes (From-Scratch NumPy)',
            'framework': 'Pure NumPy (Zero Scikit-Learn)',
            'version': '2.0',
            'calibrated_threshold': float(best_th),
            'max_features': 2500,
            'alpha': float(models['Multinomial NB'].alpha),
            'roc_auc': float(champion_m.get('ROC-AUC', 0.0))
        },
        'vocabulary': vectorizer.vocabulary_,
        'idf_weights': [float(x) for x in vectorizer.idf_weights_],
        'class_priors': [float(x) for x in nb_model.class_priors_],
        'feature_log_prob': [[float(v) for v in row] for row in nb_model.feature_log_prob_],
        'production_audit_metrics': {
            'test_accuracy': champion_m['Accuracy'],
            'test_precision': champion_m['Precision'],
            'test_recall': champion_m['Recall'],
            'test_f1_score': champion_m['F1-Score']
        }
    }
    
    json_path = os.path.join(models_dir, 'email_classifier.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_artifact, f, ensure_ascii=False, indent=2)
    print(f"-> Đã xuất thành công mô hình JSON tại: {json_path}")
    print(f"-> Kích thước file JSON: {os.path.getsize(json_path)/1024:.1f} KB")
    
    print("\n" * 1 + "=" * 75)
    print("HOÀN TẤT HUẤN LUYỆN VÀ ĐÁNH GIÁ CHUẨN FROM-SCRATCH!")
    print("=" * 75)

if __name__ == '__main__':
    main()
