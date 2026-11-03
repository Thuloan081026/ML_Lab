"""
Khám phá và trực quan hóa dữ liệu (EDA - Exploratory Data Analysis).
"""

import os
import sys
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .collection import load_raw_data
except ImportError:
    from data.collection import load_raw_data

def summarize_data() -> dict:
    frame = load_raw_data()
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_values": {k: int(v) for k, v in frame.isnull().sum().items()},
        "label_counts": {str(k): int(v) for k, v in frame["spam"].value_counts().sort_index().items()},
        "duplicate_texts": int(frame.duplicated(subset=["text"]).sum()),
    }

def generate_eda_plots(df_raw: pd.DataFrame, train_df: pd.DataFrame, output_dir: str):
    """Vẽ và lưu 3 biểu đồ EDA: Phân bố nhãn, Phân tích độ dài và Top Bigrams."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Biểu đồ Donut & Bar phân bố nhãn
    total_ham = int((df_raw['spam'] == 0).sum())
    total_spam = int((df_raw['spam'] == 1).sum())
    sizes = [total_ham, total_spam]
    colors = ['#2ecc71', '#e74c3c']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Phân bố Nhãn Spam / Ham trong Emails Dataset', fontsize=13, fontweight='bold')
    axes[0].pie(sizes, labels=['Ham', 'Spam'], autopct='%1.2f%%', colors=colors, startangle=90,
                wedgeprops={'width': 0.55, 'edgecolor': 'white', 'linewidth': 2})
    axes[0].set_title('Tỷ lệ phần trăm', fontsize=11)

    axes[1].bar(['Ham', 'Spam'], sizes, color=colors, width=0.45)
    axes[1].set_title('Số lượng mẫu', fontsize=11)
    for i, v in enumerate(sizes):
        axes[1].text(i, v + 40, f"{v:,}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spam_distribution.png'), dpi=200)
    plt.close()

    # 2. Boxplot độ dài văn bản (ký tự & từ) trên tập Train
    df_train_copy = train_df.copy()
    df_train_copy['char_len'] = df_train_copy['text'].astype(str).apply(len)
    df_train_copy['word_len'] = df_train_copy['text'].astype(str).apply(lambda x: len(x.split()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Phân tích Độ dài Văn bản theo Nhãn (Train Set - Log Scale)', fontsize=13, fontweight='bold')
    for ax, col, title in [(axes[0], 'char_len', 'Ký tự (Characters)'), (axes[1], 'word_len', 'Số từ (Words)')]:
        data_ham = df_train_copy[df_train_copy['spam'] == 0][col]
        data_spam = df_train_copy[df_train_copy['spam'] == 1][col]
        ax.boxplot([data_ham, data_spam], tick_labels=['Ham', 'Spam'], patch_artist=True)
        ax.set_yscale('log')
        ax.set_title(title, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'text_length_analysis.png'), dpi=200)
    plt.close()

    # 3. Top Bigrams
    def extract_bigrams(texts):
        cnt = Counter()
        for t in texts:
            words = str(t).lower().split()
            cnt.update(zip(words[:-1], words[1:]))
        return cnt

    ham_bg = extract_bigrams(train_df[train_df['spam'] == 0]['text']).most_common(10)
    spam_bg = extract_bigrams(train_df[train_df['spam'] == 1]['text']).most_common(10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Top 10 Bigrams Xuất Hiện Nhiều Nhất (Train Set)', fontsize=13, fontweight='bold')
    if ham_bg:
        axes[0].barh([f"{w1} {w2}" for (w1, w2), _ in reversed(ham_bg)], [c for _, c in reversed(ham_bg)], color='#2ecc71')
        axes[0].set_title('Top Bigrams (Ham)', fontsize=11)
    if spam_bg:
        axes[1].barh([f"{w1} {w2}" for (w1, w2), _ in reversed(spam_bg)], [c for _, c in reversed(spam_bg)], color='#e74c3c')
        axes[1].set_title('Top Bigrams (Spam)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_keywords.png'), dpi=200)
    plt.close()

if __name__ == "__main__":
    print("=" * 60)
    print("THỐNG KÊ DỮ LIỆU BAN ĐẦU (EDA - LAB 1)")
    print("=" * 60)
    for k, v in summarize_data().items():
        print(f"{k}: {v}")
