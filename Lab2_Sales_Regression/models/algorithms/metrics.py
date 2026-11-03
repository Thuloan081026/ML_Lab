import numpy as np

def mean_absolute_error(y_true, y_pred) -> float:
    """
    Tính Mean Absolute Error (MAE).
    """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    return np.mean(np.abs(y_true_np - y_pred_np))

def root_mean_squared_error(y_true, y_pred) -> float:
    """
    Tính Root Mean Squared Error (RMSE).
    """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    mse = np.mean(np.square(y_true_np - y_pred_np))
    return np.sqrt(mse)
