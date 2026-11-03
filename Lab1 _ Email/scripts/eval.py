"""
Đánh giá mô hình đã lưu trên tập kiểm thử độc lập (Test Set).
Chạy độc lập để kiểm định hiệu năng mô hình email_classifier.json.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from predict import StatelessSpamClassifier


def main():
    test_csv = PROJECT_ROOT / "data" / "processed" / "test.csv"
    model_json = PROJECT_ROOT / "models" / "email_classifier.json"

    if not test_csv.exists() or not model_json.exists():
        print("[!] Chưa tìm thấy test.csv hoặc model JSON. Đang chạy pipeline huấn luyện trước...")
        from train import main as run_train
        run_train()

    print("=" * 75)
    print("ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM THỬ ĐỘC LẬP (TEST SET)")
    print("=" * 75)

    df_test = pd.read_csv(test_csv)
    print(f"[*] Đã nạp tập Test: {len(df_test):,} mẫu từ {test_csv.name}")

    classifier = StatelessSpamClassifier(str(model_json))
    threshold = classifier.threshold

    y_true = df_test["spam"].values
    y_pred = []
    y_scores = []

    for text in df_test["text"]:
        res = classifier.predict(str(text))
        y_pred.append(res["is_spam"])
        y_scores.append(res["spam_probability"])

    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Tính Confusion Matrix
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Tính ROC-AUC
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    fps, tps = 0, 0
    fpr_list, tpr_list = [0.0], [0.0]
    for label in y_true_sorted:
        if label == 1:
            tps += 1
        else:
            fps += 1
        fpr_list.append(fps / N if N > 0 else 0.0)
        tpr_list.append(tps / P if P > 0 else 0.0)
    fpr_list.append(1.0)
    tpr_list.append(1.0)

    trapezoid_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc_val = float(trapezoid_func(tpr_list, fpr_list))

    print("-" * 75)
    print(f"Ngưỡng kích hoạt (Threshold) : {threshold:.2f}")
    print(f"Confusion Matrix (TN/FP/FN/TP): TN={tn} | FP={fp} | FN={fn} | TP={tp}")
    print(f"Độ chính xác (Accuracy)       : {accuracy * 100:.2f}%")
    print(f"Độ chuẩn xác (Precision)      : {precision * 100:.2f}%")
    print(f"Độ bao phủ (Recall)           : {recall * 100:.2f}%")
    print(f"Điểm F1-Score                 : {f1 * 100:.2f}%")
    print(f"Chỉ số ROC-AUC                : {auc_val:.4f}")
    print("=" * 75)
    print("[+] Hoàn tất đánh giá thành công!")


if __name__ == "__main__":
    main()
