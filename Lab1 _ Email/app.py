import os
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from collections import Counter
import numpy as np
from flask import Flask, jsonify, render_template, request

from data.data_processing import clean_and_tokenize

BASE_DIR = Path(__file__).resolve().parent
JSON_PATH = BASE_DIR / 'models' / 'email_classifier.json'

if not JSON_PATH.exists():
    raise FileNotFoundError(f"Chưa tìm thấy mô hình tại {JSON_PATH}. Chạy python train.py trước!")

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    artifact = json.load(f)

metadata = artifact['metadata']
vocabulary = artifact['vocabulary']
idf_weights = np.array(artifact['idf_weights'], dtype=np.float32)
class_priors = np.array(artifact['class_priors'], dtype=np.float32)
feature_log_prob = np.array(artifact['feature_log_prob'], dtype=np.float32)
calibrated_threshold = float(metadata.get('calibrated_threshold', 0.35))
idx_to_word = {i: w for w, i in vocabulary.items()}

app = Flask(__name__, template_folder=str(BASE_DIR / 'templates'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'Nội dung email trống'}), 400

    tokens = clean_and_tokenize(text)
    D = len(vocabulary)
    vec = np.zeros(D, dtype=np.float32)
    counts = Counter(tokens)
    total_tokens = float(len(tokens)) if tokens else 1.0

    present_indices = []
    for term, count in counts.items():
        if term in vocabulary:
            idx = vocabulary[term]
            present_indices.append(idx)
            vec[idx] = (count / total_tokens) * idf_weights[idx]

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    log_probs = vec @ feature_log_prob.T + class_priors
    max_log = np.max(log_probs)
    exp_vals = np.exp(log_probs - max_log)
    probs = exp_vals / np.sum(exp_vals)

    ham_prob = float(probs[0])
    spam_prob = float(probs[1])
    is_spam = 1 if spam_prob >= calibrated_threshold else 0

    trigger_words = []
    if present_indices:
        diffs = [(idx_to_word[idx], float(feature_log_prob[1, idx] - feature_log_prob[0, idx]))
                 for idx in set(present_indices)]
        diffs.sort(key=lambda x: x[1], reverse=(is_spam == 1))
        trigger_words = [w for w, _ in diffs[:5]]

    return jsonify({
        'is_spam': is_spam,
        'label': 'SPAM' if is_spam == 1 else 'HAM',
        'spam_probability': spam_prob,
        'ham_probability': ham_prob,
        'calibrated_threshold': calibrated_threshold,
        'model_name': metadata.get('model_display', metadata.get('model_name', 'Multinomial Naive Bayes')),
        'top_trigger_words': trigger_words,
        'tokens_count': len(tokens)
    })

if __name__ == '__main__':
    port = 5000
    print(f"[+] Server khởi chạy tại http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False)
