import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "email_classifier.json"

def save_json_artifact(artifact: dict, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
    return path

def load_json_artifact(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy mô hình: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

if __name__ == "__main__":
    if MODEL_PATH.exists():
        art = load_json_artifact()
        print(f"Mô hình JSON: {art['metadata']['model_name']} | Kích thước từ vựng: {len(art['vocabulary'])}")
    else:
        print(f"Chưa có file mô hình tại {MODEL_PATH}")
