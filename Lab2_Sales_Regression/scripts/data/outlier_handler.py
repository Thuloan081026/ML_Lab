import pandas as pd
import numpy as np

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> tuple:
    """
    Phát hiện số lượng và tỷ lệ Outliers trong một cột dữ liệu bằng phương pháp IQR.
    
    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu.
        column (str): Tên cột cần kiểm tra.
        
    Returns:
        tuple: (số lượng outlier, tỷ lệ phần trăm outlier)
    """
    if column not in df.columns:
        raise ValueError(f"Cột '{column}' không tồn tại trong DataFrame.")
        
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / len(df)) * 100
    
    return outlier_count, outlier_percentage

def handle_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', cap_percentile: float = 0.01) -> pd.DataFrame:
    """
    Xử lý Outliers trong một cột dữ liệu bằng các phương pháp không dùng scikit-learn.
    
    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        column (str): Tên cột cần xử lý.
        method (str): Phương pháp xử lý. Chọn một trong các giá trị:
                      - 'iqr': Xóa các dòng chứa outlier (Trimming).
                      - 'cap': Chặn trần/sàn (Winsorization) dựa trên cap_percentile.
                      - 'log': Biến đổi logarit tự nhiên log(1+x).
        cap_percentile (float): Mức phân vị để chặn trần/sàn (vd: 0.01 cho 1% và 99%).
        
    Returns:
        pd.DataFrame: DataFrame mới đã qua xử lý.
    """
    if column not in df.columns:
        raise ValueError(f"Cột '{column}' không tồn tại trong DataFrame.")
        
    df_out = df.copy()
    
    if method == 'iqr':
        Q1 = df_out[column].quantile(0.25)
        Q3 = df_out[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Lọc bỏ outlier
        df_out = df_out[(df_out[column] >= lower_bound) & (df_out[column] <= upper_bound)]
        
    elif method == 'cap':
        lower_limit = df_out[column].quantile(cap_percentile)
        upper_limit = df_out[column].quantile(1 - cap_percentile)
        
        # Capping dữ liệu
        df_out[column] = np.clip(df_out[column], lower_limit, upper_limit)
        
    elif method == 'log':
        # Dùng log1p để tránh lỗi log(0)
        df_out[f"{column}_log"] = np.log1p(df_out[column])
        
    else:
        raise ValueError("Phương pháp (method) không hợp lệ. Vui lòng chọn 'iqr', 'cap', hoặc 'log'.")
        
    return df_out
