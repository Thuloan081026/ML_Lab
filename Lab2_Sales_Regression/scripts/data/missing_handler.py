import pandas as pd
import numpy as np

def drop_missing(df: pd.DataFrame, threshold_percent: float = 5.0) -> pd.DataFrame:
    """
    Xóa các dòng chứa giá trị khuyết nếu tỷ lệ khuyết của cột đó nhỏ hơn ngưỡng cho phép.
    
    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        threshold_percent (float): Ngưỡng tỷ lệ phần trăm tối đa cho phép xóa (mặc định 5%).
        
    Returns:
        pd.DataFrame: DataFrame sau khi đã xử lý (hoặc giữ nguyên nếu vượt ngưỡng).
    """
    df_out = df.copy()
    
    # Tính tỷ lệ phần trăm missing ở mỗi dòng
    missing_ratio = (df_out.isnull().sum() / len(df_out)) * 100
    
    # Lấy các cột có missing value và tỷ lệ nhỏ hơn threshold
    cols_to_drop_na = missing_ratio[(missing_ratio > 0) & (missing_ratio <= threshold_percent)].index.tolist()
    
    if cols_to_drop_na:
        df_out = df_out.dropna(subset=cols_to_drop_na)
        
    return df_out

def impute_numerical(df: pd.DataFrame, columns: list, strategy: str = 'median') -> pd.DataFrame:
    """
    Điền các giá trị khuyết (Missing Values) cho các cột dạng Số.
    
    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        columns (list): Danh sách các cột số cần điền khuyết.
        strategy (str): Chiến lược điền ('mean' hoặc 'median').
        
    Returns:
        pd.DataFrame: DataFrame đã được điền khuyết.
    """
    df_out = df.copy()
    
    for col in columns:
        if col not in df_out.columns:
            continue
            
        # Kiểm tra kiểu dữ liệu có phải là số không
        if not pd.api.types.is_numeric_dtype(df_out[col]):
            raise TypeError(f"Cột '{col}' không phải là dữ liệu dạng số.")
            
        if strategy == 'median':
            fill_val = df_out[col].median()
        elif strategy == 'mean':
            fill_val = df_out[col].mean()
        else:
            raise ValueError("Chiến lược (strategy) phải là 'mean' hoặc 'median'.")
            
        df_out[col] = df_out[col].fillna(fill_val)
        
    return df_out

def impute_categorical(df: pd.DataFrame, columns: list, fill_value: str = 'mode') -> pd.DataFrame:
    """
    Điền các giá trị khuyết (Missing Values) cho các cột Phân loại (Categorical).
    
    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        columns (list): Danh sách các cột phân loại cần điền khuyết.
        fill_value (str): 'mode' để điền giá trị phổ biến nhất, hoặc truyền vào chuỗi (vd: 'Unknown').
        
    Returns:
        pd.DataFrame: DataFrame đã được điền khuyết.
    """
    df_out = df.copy()
    
    for col in columns:
        if col not in df_out.columns:
            continue
            
        if fill_value == 'mode':
            # Lấy giá trị xuất hiện nhiều nhất (bỏ qua NaN)
            if not df_out[col].mode().empty:
                val = df_out[col].mode()[0]
                df_out[col] = df_out[col].fillna(val)
        else:
            # Điền bằng chuỗi cứng (ví dụ: 'Unknown')
            df_out[col] = df_out[col].fillna(fill_value)
            
    return df_out
