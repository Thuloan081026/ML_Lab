import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def clean_and_tokenize(text: str) -> list:
    spec = importlib.util.spec_from_file_location("lab1_train", PROJECT_ROOT / "train.py")
    if spec is None or spec.loader is None:
        raise ImportError("Không thể nạp bộ tiền xử lý của train.py.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.clean_and_tokenize(text)


if __name__ == "__main__":
    print(clean_and_tokenize("Urgent: claim your $1,000 prize at https://example.com"))
