import pandas as pd

def extract_date_features(df: pd.DataFrame, date_column: str, date_format: str = "%d-%m-%Y") -> pd.DataFrame:
    """
    Chuyển đổi cột chuỗi ngày tháng sang Datetime và trích xuất các đặc trưng (features) mới.
    
    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        date_column (str): Tên cột chứa ngày tháng cần xử lý.
        date_format (str): Định dạng ngày tháng gốc trong dữ liệu.
        
    Returns:
        pd.DataFrame: DataFrame đã được bổ sung các cột thời gian mới.
    """
    if date_column not in df.columns:
        raise ValueError(f"Cột '{date_column}' không tồn tại trong DataFrame.")
        
    df_out = df.copy()
    
    # Ép kiểu dữ liệu sang dạng Datetime chuẩn của pandas
    # dayfirst=True giúp pandas hiểu định dạng DD-MM-YYYY an toàn hơn
    df_out[date_column] = pd.to_datetime(df_out[date_column], format=date_format, dayfirst=True)
    
    # Trích xuất thông tin
    df_out['Purchase_Year'] = df_out[date_column].dt.year
    df_out['Purchase_Month'] = df_out[date_column].dt.month
    df_out['Purchase_Day'] = df_out[date_column].dt.day
    df_out['Purchase_DayOfWeek'] = df_out[date_column].dt.dayofweek # 0 = Monday, 6 = Sunday
    df_out['Purchase_IsWeekend'] = df_out['Purchase_DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    return df_out

def encode_categorical(df: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    """
    Mã hóa One-Hot Encoding cho các cột biến phân loại mà không dùng scikit-learn.
    
    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        categorical_columns (list): Danh sách tên các cột phân loại cần mã hóa.
        
    Returns:
        pd.DataFrame: DataFrame với các cột mới đã mã hóa (cột cũ sẽ bị bỏ đi tự động).
    """
    df_out = df.copy()
    
    # Dùng hàm get_dummies có sẵn của Pandas để biến đổi One-Hot
    # drop_first=True giúp tránh bẫy đa cộng tuyến (Dummy Variable Trap)
    df_out = pd.get_dummies(df_out, columns=categorical_columns, drop_first=True, dtype=int)
    
    return df_out
