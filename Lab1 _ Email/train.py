"""
Pipeline huấn luyện và đánh giá hệ thống phân loại Email Spam (100% From Scratch).
Tuân thủ giới hạn < 200 dòng theo quy chuẩn môn học.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.collection import load_raw_data
from data.eda import generate_eda_plots
from data.data_processing import clean_and_tokenize
from models.algorithms import (
    TfidfVectorizerScratch, MultinomialNaiveBayesScratch,
    ComplementNaiveBayesScratch, LogisticRegressionScratch, LinearSVMScratch
)
from models.metrics import compute_metrics, compute_roc_auc
from models.save_models import save_json_artifact
from scripts.data_preparation import prepare_and_save_data

# [THÊM MỚI] Hàm chuyển DataFrame thành bảng Markdown (không cần cài thêm thư viện tabulate)
def df_to_markdown(df, digits=4):
    header = '| ' + ' | '.join(str(c) for c in df.columns) + ' |'
    divider = '|' + '|'.join(['---'] * len(df.columns)) + '|'
    rows = ['| ' + ' | '.join(f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
                              for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join([header, divider] + rows)

def main():
    print("=" * 75)
    print("LAB 1: PIPELINE PHÂN LOẠI EMAIL SPAM (100% FROM SCRATCH BẰNG NUMPY)")
    print("=" * 75)

    raw_df = load_raw_data()
    train_df, val_df, test_df, vec, X_tr, X_val, X_te = prepare_and_save_data()
    y_tr, y_val, y_te = train_df['spam'].values, val_df['spam'].values, test_df['spam'].values

    # 1. Sinh biểu đồ EDA
    plots_dir = os.path.join(PROJECT_ROOT, 'reports', 'figures')
    generate_eda_plots(raw_df, train_df, plots_dir)
    print(f"[+] Đã xuất các biểu đồ EDA vào: {plots_dir}")

    # 2. Huấn luyện 4 mô hình From Scratch
    print("\n" + "=" * 75)
    print("HUẤN LUYỆN VÀ ĐÁNH GIÁ TRÊN TẬP VALIDATION (Threshold = 0.50)")
    print("=" * 75)
    models = {
        'Multinomial NB': MultinomialNaiveBayesScratch(alpha=0.1),
        'Complement NB': ComplementNaiveBayesScratch(alpha=0.1),
        'Logistic Regression': LogisticRegressionScratch(lr=2.0, n_iters=400, l2_reg=0.0001),
        'Linear SVM': LinearSVMScratch(lr=0.05, n_epochs=40, l2_reg=0.0005)
    }

    val_res = []
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        m_eval = compute_metrics(y_val, m.predict(X_val, threshold=0.5))
        val_res.append({'Model': name, 'Accuracy': m_eval['accuracy'], 'Precision': m_eval['precision'],
                        'Recall': m_eval['recall'], 'F1-Score': m_eval['f1_score']})
    print(pd.DataFrame(val_res).to_string(index=False))

    # 3. Hiệu chuẩn ngưỡng (Threshold Calibration) trên Validation với Multinomial NB
    print("\n" + "=" * 75)
    print("HIỆU CHUẨN NGƯỠNG QUYẾT ĐỊNH (DECISION THRESHOLD CALIBRATION)")
    print("=" * 75)
    best_th, best_f1 = 0.50, 0.0
    mnb = models['Multinomial NB']
    val_probs = mnb.predict_proba(X_val)[:, 1]

    for th in np.arange(0.10, 0.91, 0.01):
        f1 = compute_metrics(y_val, (val_probs >= th).astype(int))['f1_score']
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
    print(f"[+] Ngưỡng tối ưu tìm được: BEST_THRESHOLD = {best_th:.2f} (Val F1: {best_f1*100:.2f}%)")

    # 4. Đánh giá chính thức trên tập Test độc lập (Threshold tối ưu)
    print("\n" + "=" * 75)
    print(f"ĐÁNH GIÁ CHÍNH THỨC TRÊN TẬP TEST ĐỘC LẬP (Threshold = {best_th:.2f})")
    print("=" * 75)
    test_res = []
    for name, m in models.items():
        probs = m.predict_proba(X_te)
        preds = (probs[:, 1] >= best_th).astype(int)
        m_eval = compute_metrics(y_te, preds)
        _, _, auc_val = compute_roc_auc(y_te, probs[:, 1])
        test_res.append({
            'Model': name, 'Accuracy': m_eval['accuracy'], 'Precision': m_eval['precision'],
            'Recall': m_eval['recall'], 'F1-Score': m_eval['f1_score'], 'ROC-AUC': auc_val,
            'TN': m_eval['tn'], 'FP': m_eval['fp'], 'FN': m_eval['fn'], 'TP': m_eval['tp']
        })

    df_test_res = pd.DataFrame(test_res)
    print(df_test_res.to_string(index=False))
    df_test_res.to_csv(PROJECT_ROOT / 'data' / 'processed' / 'evaluation_results.csv', index=False)

    # 5. Xuất mô hình JSON Artifact
    json_artifact = {
        'metadata': {'model_name': 'MultinomialNaiveBayesScratch', 'calibrated_threshold': best_th, 'alpha': 0.1},
        'vocabulary': vec.vocabulary_,
        'idf_weights': vec.idf_weights_.tolist(),
        'class_priors': mnb.class_priors_.tolist(),
        'feature_log_prob': mnb.feature_log_prob_.tolist(),
        'test_metrics': test_res[0]
    }
    save_json_artifact(json_artifact)
    print(f"[+] Đã xuất mô hình JSON thành công vào: models/email_classifier.json")

    # 6.Xuất kết quả đánh giá ra file Markdown 
    report_path = PROJECT_ROOT / 'reports' / 'evaluation_results.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# Kết quả Lab 1: Phân loại Email Spam (100% From Scratch)", "",
        "## 1. Kết quả trên tập Validation (Threshold = 0.50)", "",
        df_to_markdown(pd.DataFrame(val_res)), "",
        "## 2. Hiệu chuẩn ngưỡng quyết định (Multinomial NB)", "",
        f"- Ngưỡng tối ưu (BEST_THRESHOLD): **{best_th:.2f}**",
        f"- Val F1-Score: **{best_f1 * 100:.2f}%**", "",
        f"## 3. Kết quả chính thức trên tập Test độc lập (Threshold = {best_th:.2f})", "",
        df_to_markdown(df_test_res), "",
    ]
    report_path.write_text("\n".join(md_lines), encoding='utf-8')
    print(f"[+] Đã xuất báo cáo kết quả Markdown vào: {report_path}")

    print("=" * 75)
    print("HOÀN TẤT HUẤN LUYỆN LAB 1 THÀNH CÔNG (<200 DÒNG)!")
    print("=" * 75)

if __name__ == '__main__':
    main()