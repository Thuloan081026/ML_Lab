"""
Tiền xử lý văn bản: Làm sạch chuỗi, chuẩn hóa meta-tokens và lọc từ dừng.
"""

import re

# 77 từ dừng và email header phổ biến tiếng Anh
STOPWORDS_77 = set([
    'subject', 're', 'fw', 'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'to', 'in', 'of', 'for',
    'with', 'by', 'from', 'this', 'that', 'it', 'as', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
    'do', 'does', 'did', 'but', 'if', 'so', 'not', 'can', 'will', 'would', 'could', 'should', 'all', 'any',
    'my', 'your', 'his', 'her', 'their', 'our', 'we', 'you', 'they', 'i', 'me', 'him', 'them', 'us', 'what',
    'when', 'where', 'who', 'how', 'there', 'here', 'just', 'more', 'some', 'than', 'then', 'very', 'too', 'also', 'about'
])

def clean_and_tokenize(text: str) -> list:
    """
    Quy trình chuẩn hóa 5 giai đoạn:
    1. Chuyển chữ thường
    2. Ánh xạ meta-tokens: URL -> http_url, Email -> email_address, Tiền -> money_amount
    3. Xóa dấu câu và ký tự đặc biệt (giữ dấu _)
    4. Tách từ
    5. Lọc từ dừng, số thuần túy và từ < 3 ký tự (bảo toàn meta-tokens)
    """
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' http_url ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' email_address ', text)
    text = re.sub(r'\$\d+(?:,\d+)*(?:\.\d+)?|\b\d+(?:,\d+)*(?:\.\d+)?\s*(?:dollars?|usd|cents?)\b', ' money_amount ', text)
    text = re.sub(r'[^a-zA-Z0-9_\s]', ' ', text)

    raw_tokens = text.split()
    meta_set = {'http_url', 'email_address', 'money_amount'}
    cleaned_tokens = []

    for t in raw_tokens:
        if t in meta_set:
            cleaned_tokens.append(t)
        elif t not in STOPWORDS_77 and len(t) >= 3 and not t.isdigit():
            cleaned_tokens.append(t)

    return cleaned_tokens

if __name__ == '__main__':
    sample = "URGENT: Claim $1,000 lottery cash at http://example.com"
    print("Mẫu thử:", sample)
    print("Tokens :", clean_and_tokenize(sample))
