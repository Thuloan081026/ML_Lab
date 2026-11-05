import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.collection import load_raw_data
from data.data_processing import clean_and_tokenize
from models.algorithms import TfidfVectorizerScratch

def prepare_and_save_data():
    raw_df = load_raw_data()
    clean_df = raw_df.drop_duplicates(subset=['text']).reset_index(drop=True)

    np.random.seed(42)
    ham_indices = clean_df[clean_df['spam'] == 0].index.to_numpy()
    spam_indices = clean_df[clean_df['spam'] == 1].index.to_numpy()
    np.random.shuffle(ham_indices)
    np.random.shuffle(spam_indices)

    def split_idx(arr):
        n = len(arr)
        t_end = int(0.70 * n)
        v_end = int(0.85 * n)
        return arr[:t_end], arr[t_end:v_end], arr[v_end:]

    tr_ham, val_ham, te_ham = split_idx(ham_indices)
    tr_spm, val_spm, te_spm = split_idx(spam_indices)

    train_idx = np.concatenate([tr_ham, tr_spm])
    val_idx = np.concatenate([val_ham, val_spm])
    test_idx = np.concatenate([te_ham, te_spm])
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    train_df = clean_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = clean_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = clean_df.iloc[test_idx].copy().reset_index(drop=True)

    proc_dir = PROJECT_ROOT / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(proc_dir / "train_cleaned.csv", index=False)
    val_df.to_csv(proc_dir / "validation.csv", index=False)
    test_df.to_csv(proc_dir / "test.csv", index=False)

    train_tokens = [clean_and_tokenize(t) for t in train_df['text']]
    val_tokens = [clean_and_tokenize(t) for t in val_df['text']]
    test_tokens = [clean_and_tokenize(t) for t in test_df['text']]

    vec = TfidfVectorizerScratch(min_df=3, max_features=2500)
    X_train = vec.fit_transform(train_tokens)
    X_val = vec.transform(val_tokens)
    X_test = vec.transform(test_tokens)

    np.savez_compressed(proc_dir / "tfidf_splits.npz",
                        X_train=X_train, y_train=train_df['spam'].values,
                        X_val=X_val, y_val=val_df['spam'].values,
                        X_test=X_test, y_test=test_df['spam'].values)

    with open(proc_dir / "tfidf_vectorizer_params.json", 'w', encoding='utf-8') as f:
        json.dump({'vocabulary': vec.vocabulary_, 'idf_weights': vec.idf_weights_.tolist()}, f)

    print(f"[+] Đã hoàn thành chuẩn bị dữ liệu: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df, vec, X_train, X_val, X_test

if __name__ == "__main__":
    prepare_and_save_data()
