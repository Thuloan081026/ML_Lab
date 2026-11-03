import numpy as np
import pandas as pd

class LinearRegression:
    """
    Mô hình Hồi quy tuyến tính đa biến (Multiple Linear Regression).
    Giải bằng phương pháp Ma trận Normal Equation (Ma trận giả nghịch đảo).
    """
    def __init__(self):
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """
        Huấn luyện mô hình tìm trọng số W.
        Công thức: W = (X^T * X)^(-1) * X^T * Y
        """
        # Chuyển đổi sang numpy array nếu đầu vào là DataFrame/Series
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_vec = y.values if isinstance(y, pd.Series) else np.array(y)
        
        # Thêm cột toàn số 1 vào X_mat để biểu diễn cho Bias (Intercept)
        X_b = np.c_[np.ones((len(X_mat), 1)), X_mat]
        
        # Tính toán ma trận giả nghịch đảo (pseudo-inverse) để tránh lỗi ma trận suy biến
        # theta = (X_b.T.dot(X_b))^-1 . X_b.T . y
        theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_vec)
        
        # Bóc tách Bias (phần tử đầu tiên) và Weights (các phần tử còn lại)
        self.bias = theta[0]
        self.weights = theta[1:]
        
        return self
        
    def predict(self, X):
        """
        Dự đoán giá trị mục tiêu.
        Công thức: y_pred = X * W + Bias
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Mô hình chưa được huấn luyện. Gọi fit() trước.")
            
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        
        y_pred = X_mat.dot(self.weights) + self.bias
        return y_pred

    def save(self, filepath: str):
        """Lưu trữ mô hình ra file (sử dụng pickle)."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'weights': self.weights, 'bias': self.bias}, f)

    def load(self, filepath: str):
        """Nạp lại mô hình từ file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self.bias = data['bias']
        return self
