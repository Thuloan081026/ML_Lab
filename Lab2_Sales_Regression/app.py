from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Các model pickle được nạp từ thư mục models.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.algorithms.decision_tree import DecisionTreeRegressor, Node
from models.algorithms.linear_regression import LinearRegression
from models.algorithms.preprocessing import StandardScaler

# Tương thích với các file pickle được tạo trước khi mã nguồn được tách thư mục.
import models.algorithms.decision_tree as _decision_tree_module
import models.algorithms.preprocessing as _preprocessing_module
sys.modules.setdefault("models.decision_tree", _decision_tree_module)
sys.modules.setdefault("models.preprocessing", _preprocessing_module)

app = Flask(__name__)

# Khởi tạo biến toàn cục cho các model và scaler
dt_model = None
lr_model = None
scaler = None

def load_models():
    """Hàm nạp mô hình và scaler khi server khởi động"""
    global dt_model, lr_model, scaler
    try:
        dt_path = os.path.join(PROJECT_ROOT, "models", "save_model", "decision_tree.pkl")
        lr_path = os.path.join(PROJECT_ROOT, "models", "save_model", "linear_regression.pkl")
        scaler_path = os.path.join(PROJECT_ROOT, "models", "save_model", "scaler.pkl")

        if os.path.exists(dt_path):
            dt_model = DecisionTreeRegressor().load(dt_path)
        if os.path.exists(lr_path):
            lr_model = LinearRegression().load(lr_path)
        if os.path.exists(scaler_path):
            scaler = StandardScaler().load(scaler_path)

        print("Đã nạp mô hình Decision Tree, Linear Regression và Scaler thành công!")
    except Exception as e:
        print(f"Lỗi nạp mô hình: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    API Endpoint dự đoán doanh số.
    Nhận dữ liệu dạng JSON:
    {
        "Purchase_Year": 2024,
        "Purchase_Month": 1,
        "Purchase_Day": 15,
        "Purchase_DayOfWeek": 1,
        "Purchase_IsWeekend": 0,
        "model": "decision_tree" (tuỳ chọn: "decision_tree" hoặc "linear_regression")
    }
    """
    if scaler is None or (dt_model is None and lr_model is None):
        return jsonify({"error": "Mô hình chưa được nạp (Model not loaded)"}), 500

    try:
        # Lấy dữ liệu từ Request
        data = request.get_json()
        selected_model_type = data.pop("model", "decision_tree").lower()

        active_model = dt_model if selected_model_type == "decision_tree" else lr_model
        if active_model is None:
            active_model = dt_model if dt_model is not None else lr_model

        # Đưa vào DataFrame (1 dòng)
        df = pd.DataFrame([data])

        # Các cột dạng số cần chuẩn hóa
        numeric_cols = ['Purchase_Year', 'Purchase_Month', 'Purchase_Day', 'Purchase_DayOfWeek']

        # Transform dữ liệu (Không fit lại)
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])

        # Dự đoán
        prediction = active_model.predict(df_scaled)

        # Trả về kết quả
        return jsonify({
            "status": "success",
            "model_used": "DecisionTreeRegressor" if active_model == dt_model else "LinearRegression",
            "input_data": data,
            "predicted_daily_volume": float(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "API đang hoạt động bình thường!",
        "models_loaded": {
            "decision_tree": dt_model is not None,
            "linear_regression": lr_model is not None,
            "scaler": scaler is not None
        }
    })

@app.route('/', methods=['GET'])
def index():
    return """
    <h1>API dự báo doanh số sản phẩm</h1>
    <p>Hệ thống dự báo số lượng đơn hàng (Daily Volume) đang hoạt động.</p>
    <ul>
        <li>Trạng thái: <a href="/health">/health</a> (GET)</li>
        <li>Dự đoán: <code>/predict</code> (POST)</li>
    </ul>
    """

if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=True)
