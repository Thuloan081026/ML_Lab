
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import re
import json
import numpy as np
from collections import Counter
from flask import Flask, request, jsonify, render_template_string

STOPWORDS_77 = set([
    'subject', 're', 'fw', 'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'to', 'in', 'of', 'for', 
    'with', 'by', 'from', 'this', 'that', 'it', 'as', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 
    'do', 'does', 'did', 'but', 'if', 'so', 'not', 'can', 'will', 'would', 'could', 'should', 'all', 'any', 
    'my', 'your', 'his', 'her', 'their', 'our', 'we', 'you', 'they', 'i', 'me', 'him', 'them', 'us', 'what', 
    'when', 'where', 'who', 'how', 'there', 'here', 'just', 'more', 'some', 'than', 'then', 'very', 'too', 'also', 'about'
])

def clean_and_tokenize(text: str) -> list:
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' http_url ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' email_address ', text)
    text = re.sub(r'\$\d+(?:,\d+)*(?:\.\d+)?|\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:dollars?|usd|cents?)\b', ' money_amount ', text)
    text = re.sub(r'[^a-zA-Z0-9_\s]', ' ', text)
    raw_tokens = text.split()
    meta_set = {'http_url', 'email_address', 'money_amount'}
    cleaned = []
    for t in raw_tokens:
        if t in meta_set:
            cleaned.append(t)
        elif t not in STOPWORDS_77 and len(t) >= 3 and not t.isdigit():
            cleaned.append(t)
    return cleaned

base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, 'models', 'email_classifier.json')

if not os.path.exists(json_path):
    raise FileNotFoundError(f"Chưa tìm thấy mô hình JSON tại {json_path}. Hãy chạy python train.py trước!")

print(f"[*] Đang tải mô hình JSON từ: {json_path}...")
with open(json_path, 'r', encoding='utf-8') as f:
    artifact = json.load(f)

metadata = artifact['metadata']
vocabulary = artifact['vocabulary']
idf_weights = np.array(artifact['idf_weights'], dtype=np.float32)
class_priors = np.array(artifact['class_priors'], dtype=np.float32)
feature_log_prob = np.array(artifact['feature_log_prob'], dtype=np.float32)
calibrated_threshold = float(metadata.get('calibrated_threshold', 0.35))
idx_to_word = {i: w for w, i in vocabulary.items()}
print(f"[+] Mô hình sẵn sàng: {metadata['model_name']} (Ngưỡng hiệu chuẩn: {calibrated_threshold})")

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hệ Thống Phân Loại Email Spam - UTH Machine Learning Lab 1</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --card-bg: rgba(22, 30, 49, 0.78);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-primary: #3b82f6;
            --spam-color: #ef4444;
            --spam-bg: rgba(239, 68, 68, 0.15);
            --ham-color: #10b981;
            --ham-bg: rgba(16, 185, 129, 0.15);
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        }

        body {
            background-color: var(--bg);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(circle at 10% 10%, rgba(59, 130, 246, 0.15), transparent 45%),
                radial-gradient(circle at 90% 90%, rgba(139, 92, 246, 0.12), transparent 45%);
            padding: 30px 20px;
        }

        .container {
            max-width: 920px;
            margin: 0 auto;
            width: 100%;
        }

        header {
            text-align: center;
            margin-bottom: 30px;
        }

        .badge-header {
            display: inline-block;
            background: rgba(59, 130, 246, 0.12);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
            padding: 6px 16px;
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 12px;
            letter-spacing: 0.8px;
        }

        h1 {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 30%, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 0.98rem;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            backdrop-filter: blur(16px);
            border-radius: 20px;
            padding: 26px;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
            margin-bottom: 22px;
        }

        .samples-label {
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 12px;
            display: block;
        }

        .samples-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }

        .sample-btn {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--card-border);
            color: var(--text-main);
            padding: 10px 14px;
            border-radius: 12px;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 500;
            text-align: left;
            transition: all 0.2s ease;
        }

        .sample-btn:hover {
            background: rgba(59, 130, 246, 0.15);
            border-color: rgba(59, 130, 246, 0.4);
            transform: translateY(-2px);
        }

        label.input-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 600;
            margin-bottom: 10px;
            font-size: 0.95rem;
        }

        textarea {
            width: 100%;
            height: 150px;
            background: rgba(11, 15, 25, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 14px;
            color: #fff;
            padding: 16px;
            font-size: 0.95rem;
            resize: vertical;
            outline: none;
            transition: border-color 0.2s;
            line-height: 1.5;
        }

        textarea:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        }

        .actions {
            display: flex;
            gap: 12px;
            margin-top: 16px;
        }

        .btn-predict {
            flex: 2;
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: #fff;
            border: none;
            padding: 14px 20px;
            border-radius: 14px;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
        }

        .btn-predict:hover {
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            transform: translateY(-2px);
        }

        .btn-clear {
            flex: 1;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            color: var(--text-muted);
            padding: 14px 16px;
            border-radius: 14px;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
        }

        #result-box {
            display: none;
            animation: fadeIn 0.35s ease forwards;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 22px;
            border-radius: 9999px;
            font-size: 1.15rem;
            font-weight: 800;
        }

        .status-spam {
            background: var(--spam-bg);
            color: var(--spam-color);
            border: 1px solid rgba(239, 68, 68, 0.4);
        }

        .status-ham {
            background: var(--ham-bg);
            color: var(--ham-color);
            border: 1px solid rgba(16, 185, 129, 0.4);
        }

        .prob-section {
            margin-bottom: 20px;
        }

        .prob-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .progress-bar-bg {
            height: 14px;
            background: rgba(16, 185, 129, 0.25);
            border-radius: 999px;
            overflow: hidden;
            position: relative;
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #ef4444, #f97316);
            width: 0%;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            border-radius: 999px;
        }

        .trigger-section {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 16px;
            border-radius: 14px;
            margin-top: 18px;
        }

        .trigger-title {
            font-size: 0.85rem;
            font-weight: 700;
            color: #94a3b8;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .trigger-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .trigger-tag {
            background: rgba(59, 130, 246, 0.18);
            color: #93c5fd;
            border: 1px solid rgba(59, 130, 246, 0.35);
            padding: 5px 12px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .details-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-top: 16px;
        }

        .detail-item {
            background: rgba(0, 0, 0, 0.2);
            padding: 12px 14px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.04);
        }

        .detail-title {
            font-size: 0.78rem;
            color: var(--text-muted);
            margin-bottom: 4px;
            text-transform: uppercase;
        }

        .detail-val {
            font-size: 1.05rem;
            font-weight: 700;
            color: #fff;
        }

        footer {
            text-align: center;
            margin-top: auto;
            color: #64748b;
            font-size: 0.82rem;
            padding-top: 25px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge-header">UTH • MACHINE LEARNING PROJECT • LAB 1</div>
            <h1>Email Spam Classifier (From-Scratch)</h1>
            <p class="subtitle">Triển khai 100% bằng NumPy thuần (Zero Scikit-Learn) • Ngưỡng hiệu chuẩn 35%</p>
        </header>

        <div class="card">
            <span class="samples-label">⚡ Thử nghiệm nhanh các kịch bản thực tế (UTH Benchmark):</span>
            <div class="samples-grid">
                <button class="sample-btn" onclick="loadSample(1)">🎣 Lottery Phishing (Lừa đảo)</button>
                <button class="sample-btn" onclick="loadSample(2)">🛍️ Commercial Spam (Quảng cáo)</button>
                <button class="sample-btn" onclick="loadSample(3)">👥 Internal Meeting (Họp)</button>
                <button class="sample-btn" onclick="loadSample(4)">📝 Work Collaboration (Công việc)</button>
            </div>

            <label class="input-label" for="email-input">
                <span>Nội dung thư điện tử (Email Content)</span>
                <span id="char-counter" style="color: var(--text-muted); font-size: 0.85rem;">0 ký tự</span>
            </label>
            <textarea id="email-input" placeholder="Dán hoặc gõ nội dung email cần kiểm tra vào đây..."></textarea>

            <div class="actions">
                <button class="btn-predict" id="predict-btn" onclick="classifyEmail()">
                    🔍 Phân Loại Email & Trích Xuất XAI
                </button>
                <button class="btn-clear" onclick="clearText()">Xóa nội dung</button>
            </div>
        </div>

        <div class="card" id="result-box">
            <div class="result-header">
                <div>
                    <div style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 4px;">KẾT QUẢ PHÂN LOẠI:</div>
                    <div id="status-badge" class="status-badge">--</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 4px;">MÔ HÌNH CHAMPION:</div>
                    <div style="font-weight: 700; color: #60a5fa;" id="model-name">Multinomial Naive Bayes</div>
                    <div style="font-size: 0.78rem; color: #a1a1aa;">Ngưỡng quyết định: 35.0%</div>
                </div>
            </div>

            <div class="prob-section">
                <div class="prob-labels">
                    <span style="color: #ef4444;" id="spam-prob-text">Xác suất Spam: 0%</span>
                    <span style="color: #10b981;" id="ham-prob-text">Xác suất Hợp lệ: 100%</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" id="prob-bar"></div>
                </div>
            </div>

            <div class="trigger-section" id="trigger-box">
                <div class="trigger-title">
                    💡 EXPLAINABLE AI (XAI) - TOP TỪ KHÓA KÍCH HOẠT (TRIGGER WORDS):
                </div>
                <div class="trigger-badges" id="trigger-badges"></div>
            </div>

            <div class="details-grid">
                <div class="detail-item">
                    <div class="detail-title">Số lượng từ hợp lệ</div>
                    <div class="detail-val" id="stat-words">0 từ</div>
                </div>
                <div class="detail-item">
                    <div class="detail-title">Độ dài văn bản</div>
                    <div class="detail-val" id="stat-chars">0 ký tự</div>
                </div>
                <div class="detail-item">
                    <div class="detail-title">Ký hiệu đặc biệt</div>
                    <div class="detail-val" id="stat-special">0 (!) | 0 ($)</div>
                </div>
            </div>
        </div>

        <footer>
            Báo cáo Thực hành Máy học: Email Spam Classification • Trường ĐH Giao thông vận tải TP.HCM (UTH)
        </footer>
    </div>

    <script>
        const samples = {
            1: "Subject: URGENT! You have won a $1,000,000 lottery prize! Click here immediately to claim your free cash prize before it expires. No credit check needed!",
            2: "Subject: Exclusive special discount on luxury watches and cheap meds without prescription online. Order now for 80% discount and free worldwide shipping.",
            3: "Subject: Team meeting tomorrow at 10 AM. Please review the attached quarterly revenue report and project documentation before attending. Best regards.",
            4: "Subject: Hi John, please send updated python code for machine learning lab assignment. Let me know if you need any assistance with the algorithms."
        };

        const emailInput = document.getElementById('email-input');
        const charCounter = document.getElementById('char-counter');

        emailInput.addEventListener('input', () => {
            charCounter.textContent = emailInput.value.length + " ký tự";
        });

        function loadSample(id) {
            emailInput.value = samples[id];
            charCounter.textContent = emailInput.value.length + " ký tự";
            classifyEmail();
        }

        function clearText() {
            emailInput.value = "";
            charCounter.textContent = "0 ký tự";
            document.getElementById('result-box').style.display = 'none';
        }

        async function classifyEmail() {
            const text = emailInput.value.trim();
            if (!text) {
                alert("Vui lòng nhập nội dung email cần phân loại!");
                return;
            }

            const btn = document.getElementById('predict-btn');
            btn.textContent = "⏳ Đang phân tích XAI...";
            btn.disabled = true;

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();
                renderResult(data, text);
            } catch (err) {
                alert("Lỗi khi kết nối với máy chủ phân loại: " + err.message);
            } finally {
                btn.textContent = "🔍 Phân Loại Email & Trích Xuất XAI";
                btn.disabled = false;
            }
        }

        function renderResult(data, rawText) {
            const resultBox = document.getElementById('result-box');
            const badge = document.getElementById('status-badge');
            const probBar = document.getElementById('prob-bar');
            const spamText = document.getElementById('spam-prob-text');
            const hamText = document.getElementById('ham-prob-text');
            const modelName = document.getElementById('model-name');
            const triggerBadges = document.getElementById('trigger-badges');

            resultBox.style.display = 'block';
            modelName.textContent = data.model_name || "Multinomial Naive Bayes (Scratch)";

            const spamPct = (data.spam_probability * 100).toFixed(1);
            const hamPct = (data.ham_probability * 100).toFixed(1);

            probBar.style.width = spamPct + "%";
            spamText.textContent = `Xác suất Spam: ${spamPct}%`;
            hamText.textContent = `Xác suất Hợp lệ: ${hamPct}%`;

            if (data.is_spam === 1) {
                badge.className = "status-badge status-spam";
                badge.innerHTML = "🚨 PHÁT HIỆN THƯ RÁC (SPAM)";
            } else {
                badge.className = "status-badge status-ham";
                badge.innerHTML = "✅ THƯ HỢP LỆ (HAM / NOT SPAM)";
            }

            // Hiển thị Trigger Words XAI
            triggerBadges.innerHTML = "";
            if (data.top_trigger_words && data.top_trigger_words.length > 0) {
                data.top_trigger_words.forEach(word => {
                    const tag = document.createElement('span');
                    tag.className = 'trigger-tag';
                    tag.textContent = word;
                    triggerBadges.appendChild(tag);
                });
            } else {
                triggerBadges.innerHTML = "<span style='color: #71717a; font-size: 0.85rem;'>Không phát hiện từ khóa thiên lệch mạnh.</span>";
            }

            // Thống kê nhanh
            const words = rawText.trim().split(/\\s+/).filter(Boolean).length;
            const excl = (rawText.match(/!/g) || []).length;
            const dollar = (rawText.match(/\\$/g) || []).length;

            document.getElementById('stat-words').textContent = words + " từ";
            document.getElementById('stat-chars').textContent = rawText.length + " ký tự";
            document.getElementById('stat-special').textContent = `${excl} (!) | ${dollar} ($)`;

            resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

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
            tf = count / total_tokens
            vec[idx] = tf * idf_weights[idx]

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    # Softmax
    log_probs = vec @ feature_log_prob.T + class_priors
    max_log = np.max(log_probs)
    exp_vals = np.exp(log_probs - max_log)
    probs = exp_vals / np.sum(exp_vals)

    ham_prob = float(probs[0])
    spam_prob = float(probs[1])

    is_spam = 1 if spam_prob >= calibrated_threshold else 0

    # Top Trigger Words
    trigger_words = []
    if present_indices:
        diffs = []
        for idx in set(present_indices):
            diff_j = float(feature_log_prob[1, idx] - feature_log_prob[0, idx])
            diffs.append((idx_to_word[idx], diff_j))
        if is_spam == 1:
            diffs.sort(key=lambda x: x[1], reverse=True)
        else:
            diffs.sort(key=lambda x: x[1])
        trigger_words = [w for w, _ in diffs[:5]]

    return jsonify({
        'is_spam': is_spam,
        'label': 'SPAM' if is_spam == 1 else 'HAM',
        'spam_probability': spam_prob,
        'ham_probability': ham_prob,
        'calibrated_threshold': calibrated_threshold,
        'model_name': metadata.get('model_display', metadata.get('model_name', 'Multinomial Naive Bayes (From-Scratch NumPy)')),
        'top_trigger_words': trigger_words,
        'tokens_count': len(tokens)
    })

if __name__ == '__main__':
    port = 5000
    print(f"\n[+] Khởi chạy UTH From-Scratch Web Server tại: http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False)
