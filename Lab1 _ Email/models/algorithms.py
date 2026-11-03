import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("lab1_train", PROJECT_ROOT / "train.py")
if _spec is None or _spec.loader is None:
    raise ImportError("Không thể nạp các thuật toán từ train.py.")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

TfidfVectorizerScratch = _module.TfidfVectorizerScratch
MultinomialNaiveBayesScratch = _module.MultinomialNaiveBayesScratch
ComplementNaiveBayesScratch = _module.ComplementNaiveBayesScratch
LogisticRegressionScratch = _module.LogisticRegressionScratch
LinearSVMScratch = _module.LinearSVMScratch
