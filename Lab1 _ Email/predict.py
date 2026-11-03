"""
Dự đoán email và giải thích bằng Top 5 Trigger Words (From Scratch, <200 dòng).
"""

import os
import sys
import json
import argparse
from collections import Counter
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.data_processing import clean_and_tokenize

class StatelessSpamClassifier:
    """Bộ suy diễn phân loại email nạp từ file JSON Artifact độc lập."""
    def __init__(self, json_path: str = None):
        if json_path is None:
            json_path = os.path.join(PROJECT_ROOT, 'models', 'email_classifier.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại '{json_path}'. Hãy chạy 'python train.py' trước!")

        with open(json_path, 'r', encoding='utf-8') as f:
            artifact = json.load(f)

        self.metadata = artifact['metadata']
        self.vocabulary = artifact['vocabulary']
        self.idf_weights = np.array(artifact['idf_weights'], dtype=np.float32)
        self.class_priors = np.array(artifact['class_priors'], dtype=np.float32)
        self.feature_log_prob = np.array(artifact['feature_log_prob'], dtype=np.float32)
        self.threshold = float(self.metadata.get('calibrated_threshold', 0.35))
        self.idx_to_word = {i: w for w, i in self.vocabulary.items()}

    def predict(self, raw_text: str) -> dict:
        tokens = clean_and_tokenize(raw_text)
        D = len(self.vocabulary)
        vec = np.zeros(D, dtype=np.float32)
        counts = Counter(tokens)
        total_tokens = float(len(tokens)) if tokens else 1.0

        present_indices = []
        for term, count in counts.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                present_indices.append(idx)
                vec[idx] = (count / total_tokens) * self.idf_weights[idx]

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        log_probs = vec @ self.feature_log_prob.T + self.class_priors
        exp_vals = np.exp(log_probs - np.max(log_probs))
        probs = exp_vals / np.sum(exp_vals)
        spam_prob = float(probs[1])
        is_spam = 1 if spam_prob >= self.threshold else 0

        trigger_words = []
        if present_indices:
            diffs = [(self.idx_to_word[i], float(self.feature_log_prob[1, i] - self.feature_log_prob[0, i]))
                     for i in set(present_indices)]
            diffs.sort(key=lambda x: x[1], reverse=(is_spam == 1))
            trigger_words = [w for w, _ in diffs[:5]]

        return {
            'label': "🚨 SPAM (THƯ RÁC)" if is_spam == 1 else "✅ HAM (THƯ HỢP LỆ)",
            'is_spam': is_spam,
            'spam_probability': spam_prob,
            'ham_probability': float(probs[0]),
            'top_trigger_words': trigger_words
        }

def run_benchmark_demo(classifier: StatelessSpamClassifier):
    scenarios = [
        ("Lottery Phishing", "Subject: URGENT! You have won a $1,000,000 lottery prize! Click here to claim your cash reward."),
        ("Commercial Promo", "Subject: Exclusive special discount on luxury watches and cheap meds with free shipping worldwide."),
        ("Internal Meeting", "Subject: Team meeting tomorrow at 10 AM. Please review the attached quarterly revenue report."),
        ("Work Task", "Subject: Hi John, please send updated python code for machine learning lab assignment before friday.")
    ]
    print("=" * 70)
    print("KIỂM CHỨNG THỰC NGHIỆM TRÊN CÁC KỊCH BẢN THỬ NGHIỆM")
    print("=" * 70)
    for name, text in scenarios:
        res = classifier.predict(text)
        print(f"\n[{name}]")
        print(f"Trích đoạn : {text[:65]}...")
        print(f"Phán đoán  : {res['label']} (Spam: {res['spam_probability']*100:.2f}%)")
        print(f"Triggers   : {', '.join(res['top_trigger_words'])}")
    print("\n" + "=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Dự đoán Email Spam/Ham")
    parser.add_argument("--text", type=str, help="Nội dung email cần kiểm tra")
    parser.add_argument("--demo", action="store_true", help="Chạy 4 kịch bản thử nghiệm mẫu")
    args = parser.parse_args()

    clf = StatelessSpamClassifier()
    if args.demo or not args.text:
        run_benchmark_demo(clf)
    else:
        res = clf.predict(args.text)
        print(f"\nKết quả: {res['label']}")
        print(f"Xác suất Spam: {res['spam_probability']*100:.2f}% | Hợp lệ: {res['ham_probability']*100:.2f}%")
        print(f"Từ kích hoạt: {', '.join(res['top_trigger_words'])}\n")

if __name__ == "__main__":
    main()
