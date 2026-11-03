"""Dự đoán email và giải thích bằng Top 5 Trigger Words."""

import os
import sys

# Cấu hình stdout/stderr sang UTF-8 để in tiếng Việt chuẩn trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import re
import json
import math
import argparse
from collections import Counter
import numpy as np

# 1. BỘ TỪ DỪNG (77 STOPWORDS) VÀ HÀM TIỀN XỬ LÝ ĐỒNG BỘ VỚI TẬP HUẤN LUYỆN
STOPWORDS_77 = set([
    'subject', 're', 'fw', 'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'to', 'in', 'of', 'for', 
    'with', 'by', 'from', 'this', 'that', 'it', 'as', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 
    'do', 'does', 'did', 'but', 'if', 'so', 'not', 'can', 'will', 'would', 'could', 'should', 'all', 'any', 
    'my', 'your', 'his', 'her', 'their', 'our', 'we', 'you', 'they', 'i', 'me', 'him', 'them', 'us', 'what', 
    'when', 'where', 'who', 'how', 'there', 'here', 'just', 'more', 'some', 'than', 'then', 'very', 'too', 'also', 'about'
])

def clean_and_tokenize(text: str) -> list:
    """
    Hàm làm sạch và chuẩn hóa email đầu vào giống hệt lúc huấn luyện:
    1. Chuyển thành chữ thường.
    2. Quy đổi URL -> 'http_url', Email -> 'email_address', Số tiền -> 'money_amount'.
    3. Xóa dấu câu, chỉ giữ chữ cái, số và dấu '_'.
    4. Lọc bỏ 77 từ dừng và các từ ngắn hơn 3 ký tự (trừ meta-tokens).
    """
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' http_url ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' email_address ', text)
    text = re.sub(r'\$\d+(?:,\d+)*(?:\.\d+)?|\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:dollars?|usd|cents?)\b', ' money_amount ', text)
    text = re.sub(r'[^a-zA-Z0-9_\s]', ' ', text)
    raw_tokens = text.split()
    meta_set = {'http_url', 'email_address', 'money_amount'}
    cleaned = []
    for t in raw_tokens:
        if t in meta_set:
            cleaned.append(t)
        elif t not in STOPWORDS_77 and len(t) >= 3 and not t.isdigit():
            cleaned.append(t)
    return cleaned

# 2. CƠ CHẾ SUY DIỄN PHI TRẠNG THÁI (STATELESS INFERENCE ENGINE)
class StatelessSpamClassifier:
    """
    Lớp thực thi phân loại nạp trực tiếp từ file JSON 'models/email_classifier.json':
    - Không cần cài đặt scikit-learn, joblib hay pickle.
    - Chạy độc lập, an toàn và có thể tái lập trên mọi hệ điều hành hoặc ngôn ngữ khác.
    """
    def __init__(self, json_path: str = None):
        if json_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(base_dir, 'models', 'email_classifier.json')
            
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại '{json_path}'. Hãy chạy 'python train.py' trước!")
            
        print(f"[*] Đang nạp mô hình JSON từ: {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            artifact = json.load(f)
            
        # BƯỚC 1: Khôi phục tham số mô hình đã đóng băng (Parameter Restitution)
        self.metadata = artifact['metadata']
        self.vocabulary = artifact['vocabulary']  # Ánh xạ từ vựng (từ -> chỉ số cột 0..2499)
        self.idf_weights = np.array(artifact['idf_weights'], dtype=np.float32)  # Mảng trọng số IDF
        self.class_priors = np.array(artifact['class_priors'], dtype=np.float32) # Log xác suất tiên nghiệm
        self.feature_log_prob = np.array(artifact['feature_log_prob'], dtype=np.float32) # Ma trận log-likelihood (2, 2500)
        self.threshold = float(self.metadata.get('calibrated_threshold', 0.35))   # Ngưỡng kích hoạt Spam
        self.idx_to_word = {i: w for w, i in self.vocabulary.items()}             # Ánh xạ ngược chỉ số -> từ vựng
        print(f"[+] Đã khôi phục thành công mô hình: {self.metadata['model_name']} (Ngưỡng kích hoạt: {self.threshold:.2f})")

    def predict(self, raw_text: str) -> dict:
        """
        Quy trình suy diễn thời gian thực gồm 5 bước nguyên tử (Atomic Steps):
        -------------------------------------------------------------------
        Bước 1: Làm sạch và tách từ (Text Sanitization & Tokenization).
        Bước 2: Vector hóa TF-IDF và chuẩn hóa Euclidean L2 (Vectorized TF-IDF Projection).
        Bước 3: Tính toán điểm Log Joint Likelihood và hàm Softmax để ra xác suất hậu nghiệm.
        Bước 4: So sánh xác suất Spam với ngưỡng hiệu chuẩn 0.35 (Decision Classification).
        Bước 5: Trích xuất khả năng giải thích XAI (Top 5 Trigger Words).
        """
        # Bước 1: Làm sạch văn bản thô
        tokens = clean_and_tokenize(raw_text)
        
        # Bước 2: Chiếu vào không gian vector 2,500 chiều
        D = len(self.vocabulary)
        vec = np.zeros(D, dtype=np.float32)
        counts = Counter(tokens)
        total_tokens = float(len(tokens)) if tokens else 1.0
        
        present_indices = []
        for term, count in counts.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                present_indices.append(idx)
                tf = count / total_tokens
                vec[idx] = tf * self.idf_weights[idx]
                
        # Chuẩn hóa L2-norm
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
            
        # Bước 3: Tính xác suất hậu nghiệm qua Softmax
        # Công thức: log_probs = vec @ W_feat^T + b_prior
        log_probs = vec @ self.feature_log_prob.T + self.class_priors
        max_log = np.max(log_probs)
        exp_vals = np.exp(log_probs - max_log)  # Trừ max_log để chống tràn số float32
        probs = exp_vals / np.sum(exp_vals)
        
        ham_prob = float(probs[0])
        spam_prob = float(probs[1])
        
        # Bước 4: Ra quyết định phân loại dựa trên ngưỡng 0.35
        is_spam = 1 if spam_prob >= self.threshold else 0
        label_text = "🚨 SPAM (THƯ RÁC)" if is_spam == 1 else "✅ HAM (THƯ HỢP LỆ)"
        
        # Bước 5: Explainable AI (XAI) - Trích xuất Top 5 Trigger Words
        # Tính sai phân log-likelihood: diff_j = log P(w_j | Spam) - log P(w_j | Ham)
        # - diff_j > 0 càng lớn: từ này là bằng chứng mạnh mẽ chỉ ra Spam (vd: claim, click, cash)
        # - diff_j < 0 càng nhỏ: từ này là bằng chứng mạnh mẽ chỉ ra Ham (vd: meeting, report, team)
        trigger_words = []
        if present_indices:
            diffs = []
            for idx in set(present_indices):
                diff_j = float(self.feature_log_prob[1, idx] - self.feature_log_prob[0, idx])
                term_word = self.idx_to_word[idx]
                diffs.append((term_word, diff_j))
                
            # Nếu dự đoán là Spam -> lấy các từ có diff_j cao nhất
            # Nếu dự đoán là Ham  -> lấy các từ có diff_j thấp nhất (thiên về Ham nhất)
            if is_spam == 1:
                diffs.sort(key=lambda x: x[1], reverse=True)
            else:
                diffs.sort(key=lambda x: x[1])
                
            trigger_words = [w for w, score in diffs[:5]]
            
        return {
            'label': label_text,
            'is_spam': is_spam,
            'spam_probability': spam_prob,
            'ham_probability': ham_prob,
            'calibrated_threshold': self.threshold,
            'tokens_count': len(tokens),
            'top_trigger_words': trigger_words,
            'cleaned_tokens_preview': tokens[:10]
        }

def interactive_mode(classifier: StatelessSpamClassifier):
    """Giao diện dòng lệnh tương tác trực tiếp người dùng"""
    print("\n" + "=" * 70)
    print("   HỆ THỐNG PHÂN LOẠI EMAIL RÁC (FROM-SCRATCH NUMPY) - XAI INTERFACE")
    print("=" * 70)
    print("Gõ hoặc dán nội dung email vào đây (nhập 'exit' hoặc 'quit' để thoát):")
    
    while True:
        try:
            print("-" * 70)
            text = input("Email: ").strip()
            if not text:
                continue
            if text.lower() in ['exit', 'quit']:
                print("[*] Kết thúc phiên làm việc!")
                break
                
            res = classifier.predict(text)
            print("\n--- KẾT QUẢ PHÂN LOẠI TỨC THÌ ---")
            print(f"-> Phán đoán     : {res['label']}")
            print(f"-> Xác suất Spam : {res['spam_probability']*100:.2f}% (Ngưỡng kích hoạt: {res['calibrated_threshold']*100:.0f}%)")
            print(f"-> Xác suất Ham  : {res['ham_probability']*100:.2f}%")
            if res['top_trigger_words']:
                print(f"-> Top Trigger Words (Từ khóa kích hoạt quyết định): {', '.join(res['top_trigger_words'])}")
            print(f"-> Tokens hợp lệ : {res['cleaned_tokens_preview']}")
            
        except (KeyboardInterrupt, EOFError):
            print("\n[*] Tạm biệt!")
            break

def main():
    parser = argparse.ArgumentParser(description="Dự đoán Email Spam hoặc Ham bằng mô hình From-Scratch thuần NumPy")
    parser.add_argument("--text", type=str, help="Chuỗi nội dung email cần kiểm tra")
    parser.add_argument("--file", type=str, help="Đường dẫn đến file .txt chứa email")
    parser.add_argument("--demo", action="store_true", help="Chạy kiểm tra trên 4 kịch bản thực tế của UTH Benchmark")
    
    args = parser.parse_args()
    classifier = StatelessSpamClassifier()
    
    if args.demo:
        demo_cases = [
            (
                "Kịch bản 1: Lottery Phishing (Lừa đảo trúng thưởng)",
                "Subject: URGENT! You have won a $1,000,000 lottery prize! Click here immediately to claim your free cash prize before it expires. No credit check needed!"
            ),
            (
                "Kịch bản 2: Commercial Spam (Quảng cáo thuốc & giảm giá)",
                "Subject: Exclusive special discount on luxury watches and cheap meds without prescription online. Order now for 80% discount and free worldwide shipping."
            ),
            (
                "Kịch bản 3: Internal Meeting (Họp nội bộ doanh nghiệp)",
                "Subject: Team meeting tomorrow at 10 AM. Please review the attached quarterly revenue report and project documentation before attending. Best regards."
            ),
            (
                "Kịch bản 4: Work Collaboration (Trao đổi chuyên môn / Học tập)",
                "Subject: Hi John, please send updated python code for machine learning lab assignment. Let me know if you need any assistance with the algorithms."
            )
        ]
        
        print("\n" + "=" * 75)
        print("BẢNG KIỂM CHỨNG THỰC NGHIỆM TRÊN CÁC KỊCH BẢN THỬ NGHIỆM (UTH BENCHMARK)")
        print("=" * 75)
        for name, snippet in demo_cases:
            res = classifier.predict(snippet)
            print(f"\n[Test Scenario]  : {name}")
            print(f"Trích đoạn email : {snippet[:80]}...")
            print(f"-> Phán đoán     : {res['label']}")
            print(f"-> Xác suất Spam : {res['spam_probability']*100:.2f}% (Hợp lệ: {res['ham_probability']*100:.2f}%)")
            print(f"-> Trigger Words : {', '.join(res['top_trigger_words'])}")
        print("\n" + "=" * 75)
        return
        
    if args.text:
        res = classifier.predict(args.text)
        print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
        print(f"Email            : {args.text}")
        print(f"Phán đoán        : {res['label']}")
        print(f"Xác suất Spam    : {res['spam_probability']*100:.2f}%")
        print(f"Xác suất Hợp lệ  : {res['ham_probability']*100:.2f}%")
        print(f"Trigger Words    : {', '.join(res['top_trigger_words'])}")
        return
        
    if args.file:
        if not os.path.exists(args.file):
            print(f"[!] Không tìm thấy file: {args.file}")
            return
        with open(args.file, 'r', encoding='utf-8') as f:
            content = f.read()
        res = classifier.predict(content)
        print(f"\n--- KẾT QUẢ DỰ ĐOÁN TỪ FILE ({args.file}) ---")
        print(f"Phán đoán        : {res['label']}")
        print(f"Xác suất Spam    : {res['spam_probability']*100:.2f}%")
        print(f"Xác suất Hợp lệ  : {res['ham_probability']*100:.2f}%")
        print(f"Trigger Words    : {', '.join(res['top_trigger_words'])}")
        return

    # Chế độ tương tác mặc định
    interactive_mode(classifier)

if __name__ == '__main__':
    main()
