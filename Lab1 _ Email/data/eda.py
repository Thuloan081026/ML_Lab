from .collection import load_raw_data


def summarize_data() -> dict:
    frame = load_raw_data()
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_values": {key: int(value) for key, value in frame.isnull().sum().items()},
        "label_counts": {str(key): int(value) for key, value in frame["spam"].value_counts().sort_index().items()},
        "duplicate_texts": int(frame.duplicated(subset=["text"]).sum()),
    }


if __name__ == "__main__":
    for key, value in summarize_data().items():
        print(f"{key}: {value}")
