# Thuật toán

- **Hồi quy tuyến tính:** giải bằng phương trình Normal Equation.
- **Cây quyết định hồi quy:** tự cài đặt theo tiêu chí giảm phương sai.
- **Hồi quy tuyến tính Gradient Descent:** hỗ trợ `partial_fit` cho Online
  Learning.
- **StandardScaler:** chuẩn hóa theo trung bình và độ lệch chuẩn của tập
  huấn luyện.
- **MAE và RMSE:** đo sai số dự báo trên tập kiểm tra.

Các thuật toán cốt lõi được viết bằng NumPy và Pandas, không dùng
scikit-learn cho model hoặc scaler.

Mã nguồn thuật toán nằm trong `models/algorithms/`; các model đã huấn luyện
được lưu trong `models/save_model/`.
