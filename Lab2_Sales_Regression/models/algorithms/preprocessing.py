import numpy as np
import pandas as pd

def train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = None):
    """
    Chia tập dữ liệu thành tập Huấn luyện (Train) và Kiểm thử (Test) bằng cách xáo trộn ngẫu nhiên.
    Sử dụng cho các bài toán phân loại hoặc hồi quy cơ bản không phụ thuộc thời gian.
    """
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.random.permutation(len(X))
    test_samples = int(len(X) * test_size)

    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]

    X_train = X.iloc[train_indices].copy()
    X_test = X.iloc[test_indices].copy()
    y_train = y.iloc[train_indices].copy()
    y_test = y.iloc[test_indices].copy()

    return X_train, X_test, y_train, y_test

def time_series_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """
    Chia tập dữ liệu theo trình tự thời gian (không xáo trộn).
    Sử dụng cho chuỗi thời gian (Time-Series Forecasting) để tránh rò rỉ dữ liệu tương lai vào quá khứ.

    Lưu ý: Yêu cầu X và y phải được sắp xếp theo thời gian từ trước.
    """
    test_samples = int(len(X) * test_size)
    train_samples = len(X) - test_samples

    X_train = X.iloc[:train_samples].copy()
    X_test = X.iloc[train_samples:].copy()
    y_train = y.iloc[:train_samples].copy()
    y_test = y.iloc[train_samples:].copy()

    return X_train, X_test, y_train, y_test

def time_series_train_val_test_split(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15):
    """
    Chia tập dữ liệu DataFrame theo trình tự thời gian thành 3 phần: Train, Validation và Test.
    """
    test_samples = int(len(df) * test_size)
    val_samples = int(len(df) * val_size)
    train_samples = len(df) - test_samples - val_samples

    df_train = df.iloc[:train_samples].copy()
    df_val = df.iloc[train_samples:train_samples + val_samples].copy()
    df_test = df.iloc[train_samples + val_samples:].copy()

    return df_train, df_val, df_test

class StandardScaler:
    """
    Chuẩn hóa Z-Score: X_new = (X - Mean) / Std
    """
    def __init__(self):
        self.means = None
        self.stds = None
        self.columns = None

    def fit(self, X: pd.DataFrame):
        """Tính toán Mean và Std trên tập Train."""
        self.columns = X.columns
        self.means = X.mean(axis=0)
        self.stds = X.std(axis=0)
        self.stds = self.stds.replace(0, 1e-8)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Áp dụng chuẩn hóa lên dữ liệu."""
        if self.means is None or self.stds is None:
            raise ValueError("Cần gọi hàm fit() trước khi transform().")

        X_scaled = X.copy()
        for col in self.columns:
            if col in X_scaled.columns:
                X_scaled[col] = (X_scaled[col] - self.means[col]) / self.stds[col]

        return X_scaled

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

    def save(self, filepath: str):
        """Lưu trữ thông số scaler ra file (sử dụng pickle)."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'means': self.means, 'stds': self.stds, 'columns': self.columns}, f)

    def load(self, filepath: str):
        """Nạp lại thông số scaler từ file."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.means = data['means']
            self.stds = data['stds']
            self.columns = data['columns']
        return self
