"""
Các thuật toán học máy và Vector hóa văn bản tự lập trình (100% From Scratch bằng NumPy).
"""

import math
from collections import Counter
import numpy as np

class TfidfVectorizerScratch:
    """Bộ chuyển đổi văn bản sang ma trận số học TF-IDF chuẩn hóa L2."""
    def __init__(self, min_df: int = 3, max_features: int = 2500):
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary_ = {}
        self.feature_names_ = []
        self.idf_weights_ = None

    def fit(self, tokenized_docs: list):
        doc_count = len(tokenized_docs)
        df_counter = Counter()
        for doc in tokenized_docs:
            df_counter.update(set(doc))

        valid_words = [w for w, cnt in df_counter.items() if cnt >= self.min_df]
        valid_words.sort(key=lambda w: (-df_counter[w], w))
        self.feature_names_ = valid_words[:self.max_features]
        self.vocabulary_ = {w: i for i, w in enumerate(self.feature_names_)}

        num_feats = len(self.feature_names_)
        self.idf_weights_ = np.zeros(num_feats, dtype=np.float32)
        for i, word in enumerate(self.feature_names_):
            df_val = df_counter[word]
            self.idf_weights_[i] = math.log((1.0 + doc_count) / (1.0 + df_val)) + 1.0
        return self

    def transform(self, tokenized_docs: list) -> np.ndarray:
        m = len(tokenized_docs)
        D = len(self.feature_names_)
        X = np.zeros((m, D), dtype=np.float32)

        for i, tokens in enumerate(tokenized_docs):
            if not tokens:
                continue
            tf_counter = Counter(tokens)
            doc_len = float(len(tokens))
            for word, count in tf_counter.items():
                if word in self.vocabulary_:
                    j = self.vocabulary_[word]
                    X[i, j] = (count / doc_len) * self.idf_weights_[j]

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return X / norms

    def fit_transform(self, tokenized_docs: list) -> np.ndarray:
        return self.fit(tokenized_docs).transform(tokenized_docs)

class MultinomialNaiveBayesScratch:
    """Mô hình xác suất sinh mẫu Naive Bayes đa thức với làm mịn Laplace."""
    def __init__(self, alpha: float = 0.1, threshold: float = 0.35):
        self.alpha = alpha
        self.threshold = threshold
        self.class_priors_ = None
        self.feature_log_prob_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, D = X.shape
        n_0 = float(np.sum(y == 0))
        n_1 = float(np.sum(y == 1))
        self.class_priors_ = np.array([math.log(n_0 / m), math.log(n_1 / m)], dtype=np.float32)

        self.feature_log_prob_ = np.zeros((2, D), dtype=np.float32)
        for c in [0, 1]:
            counts = X[y == c].sum(axis=0) + self.alpha
            self.feature_log_prob_[c] = np.log(counts / counts.sum())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_probs = X @ self.feature_log_prob_.T + self.class_priors_
        max_log = np.max(log_probs, axis=1, keepdims=True)
        exp_vals = np.exp(log_probs - max_log)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        th = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X)[:, 1] >= th).astype(int)

class ComplementNaiveBayesScratch:
    """Complement Naive Bayes tối ưu cho dữ liệu nhãn mất cân bằng."""
    def __init__(self, alpha: float = 0.1, threshold: float = 0.35):
        self.alpha = alpha
        self.threshold = threshold
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, D = X.shape
        self.class_log_prior_ = np.array([math.log(np.sum(y == 0) / m), math.log(np.sum(y == 1) / m)], dtype=np.float32)
        self.feature_log_prob_ = np.zeros((2, D), dtype=np.float32)
        for c in [0, 1]:
            counts = X[y != c].sum(axis=0) + self.alpha
            self.feature_log_prob_[c] = np.log(counts / counts.sum())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = - (X @ self.feature_log_prob_.T) + self.class_log_prior_
        max_log = np.max(scores, axis=1, keepdims=True)
        exp_vals = np.exp(scores - max_log)
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        th = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X)[:, 1] >= th).astype(int)

class LogisticRegressionScratch:
    """Hồi quy Logistic phân loại nhị phân với L2 Regularization."""
    def __init__(self, lr: float = 2.0, n_iters: int = 400, l2_reg: float = 0.0001, threshold: float = 0.35):
        self.lr = lr
        self.n_iters = n_iters
        self.l2_reg = l2_reg
        self.threshold = threshold
        self.w_ = None
        self.b_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, D = X.shape
        self.w_ = np.zeros(D, dtype=np.float32)
        self.b_ = 0.0
        y_flt = y.astype(np.float32)

        for _ in range(self.n_iters):
            z = np.clip(X @ self.w_ + self.b_, -20.0, 20.0)
            p_hat = 1.0 / (1.0 + np.exp(-z))
            err = p_hat - y_flt
            grad_w = (X.T @ err) / m + self.l2_reg * self.w_
            grad_b = np.sum(err) / m
            self.w_ -= self.lr * grad_w
            self.b_ -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.clip(X @ self.w_ + self.b_, -20.0, 20.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        th = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X)[:, 1] >= th).astype(int)

class LinearSVMScratch:
    """Linear SVM tối ưu hàm mất mát Hinge Loss qua Stochastic Gradient Descent."""
    def __init__(self, lr: float = 0.05, n_epochs: int = 40, l2_reg: float = 0.0005, threshold: float = 0.35):
        self.lr = lr
        self.n_epochs = n_epochs
        self.l2_reg = l2_reg
        self.threshold = threshold
        self.w_ = None
        self.b_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, D = X.shape
        self.w_ = np.zeros(D, dtype=np.float32)
        self.b_ = 0.0
        y_sgn = np.where(y == 1, 1.0, -1.0).astype(np.float32)

        for _ in range(self.n_epochs):
            for i in range(m):
                margin = y_sgn[i] * (np.dot(X[i], self.w_) + self.b_)
                if margin < 1.0:
                    self.w_ -= self.lr * (self.l2_reg * self.w_ - y_sgn[i] * X[i])
                    self.b_ += self.lr * y_sgn[i]
                else:
                    self.w_ -= self.lr * (self.l2_reg * self.w_)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w_ + self.b_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.clip(self.decision_function(X), -20.0, 20.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        th = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X)[:, 1] >= th).astype(int)
