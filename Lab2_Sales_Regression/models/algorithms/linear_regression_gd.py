import numpy as np
import pandas as pd

class LinearRegressionGD:
    """
    Hồi quy Tuyến tính giải bằng Gradient Descent (Hạ dốc Đạo hàm).
    Hỗ trợ Online Learning thông qua hàm partial_fit().
    """
    def __init__(self, learning_rate=0.01, epochs=1000, alpha=0.0, clip_value=1.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.clip_value = clip_value
        self.weights = None
        self.bias = None

    def _initialize_parameters(self, num_features):
        """Khởi tạo trọng số ngẫu nhiên nếu chưa có (lần đầu học)."""
        if self.weights is None:
            self.weights = np.zeros(num_features)
            self.bias = 0.0

    def fit(self, X, y):
        """
        Huấn luyện mô hình từ đầu (Offline Learning).
        """
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_vec = y.values if isinstance(y, pd.Series) else np.array(y)
        
        num_samples, num_features = X_mat.shape
        self._initialize_parameters(num_features)
        
        for epoch in range(self.epochs):
            # Tính đạo hàm và cập nhật
            self._update_weights(X_mat, y_vec, num_samples)
            
        return self

    def partial_fit(self, X, y, epochs=1):
        """
        Tính năng Fine-tuning (Online Learning).
        Chỉ lặp một số ít vòng (epochs=1) để cập nhật nhẹ trọng số W dựa trên dữ liệu mới,
        không làm quên mất kiến thức cũ.
        """
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_vec = y.values if isinstance(y, pd.Series) else np.array(y)
        
        num_samples, num_features = X_mat.shape
        self._initialize_parameters(num_features)
        
        for epoch in range(epochs):
            self._update_weights(X_mat, y_vec, num_samples)
            
        return self

    def _update_weights(self, X, y, num_samples):
        """Tính toán đạo hàm hàm mất mát (MSE) và cập nhật Trọng số."""
        y_pred = X.dot(self.weights) + self.bias
        error = y_pred - y
        
        # Đạo hàm của MSE theo Weights + L2 Regularization Penalty
        dw = (2 / num_samples) * X.T.dot(error) + 2 * self.alpha * self.weights
        
        # Đạo hàm của MSE theo Bias
        db = (2 / num_samples) * np.sum(error)
        
        # Gradient Clipping: Giới hạn độ giật của đạo hàm
        if self.clip_value is not None:
            dw = np.clip(dw, -self.clip_value, self.clip_value)
            db = np.clip(db, -self.clip_value, self.clip_value)
        
        # Cập nhật W và B (hạ dốc)
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("Mô hình chưa được huấn luyện.")
            
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        return X_mat.dot(self.weights) + self.bias

    def save(self, filepath: str):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'weights': self.weights, 'bias': self.bias}, f)

    def load(self, filepath: str):
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self.bias = data['bias']
        return self
