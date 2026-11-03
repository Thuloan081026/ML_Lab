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
from models.algorithms.preprocessing import StandardScaler

# Tương thích với các file pickle được tạo trước khi mã nguồn được tách thư mục.
import models.algorithms.decision_tree as _decision_tree_module
import models.algorithms.preprocessing as _preprocessing_module
sys.modules.setdefault("models.decision_tree", _decision_tree_module)
sys.modules.setdefault("models.preprocessing", _preprocessing_module)

app = Flask(__name__)

# Khởi tạo biến toàn cục cho model và scaler
model = None
scaler = None

def load_models():
    """Hàm nạp mô hình và scaler khi server khởi động"""
    global model, scaler
    try:
        model = DecisionTreeRegressor().load(
            os.path.join(PROJECT_ROOT, "models", "save_model", "decision_tree.pkl")
        )
        scaler = StandardScaler().load(
            os.path.join(PROJECT_ROOT, "models", "save_model", "scaler.pkl")
        )
        print("Đã nạp mô hình Decision Tree và Scaler thành công!")
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
        "Purchase_IsWeekend": 0
    }
    """
    if model is None or scaler is None:
        return jsonify({"error": "Mô hình chưa được nạp (Model not loaded)"}), 500

    try:
        # Lấy dữ liệu từ Request
        data = request.get_json()
        
        # Đưa vào DataFrame (1 dòng)
        df = pd.DataFrame([data])
        
        # Các cột dạng số cần chuẩn hóa
        numeric_cols = ['Purchase_Year', 'Purchase_Month', 'Purchase_Day', 'Purchase_DayOfWeek']
        
        # Transform dữ liệu (Không fit lại)
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])
        
        # Dự đoán
        prediction = model.predict(df_scaled)
        
        # Trả về kết quả
        return jsonify({
            "status": "success",
            "input_data": data,
            "predicted_daily_volume": float(prediction[0])
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "API đang hoạt động bình thường!"})

@app.route('/', methods=['GET'])
def index():
    return """
    <h1>API dự báo doanh số sản phẩm</h1>
    <p>Hệ thống dự báo số lượng đơn hàng đang hoạt động.</p>
    <ul>
        <li>Trạng thái: <a href="/health">/health</a> (GET)</li>
        <li>Dự đoán: <code>/predict</code> (POST)</li>
    </ul>
    """

if __name__ == '__main__':
    load_models()
    # Chạy server ở port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
