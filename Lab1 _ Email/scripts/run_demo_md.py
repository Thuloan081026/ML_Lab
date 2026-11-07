"""
Chạy nguyên bản scripts/demo.py (KHÔNG chỉnh sửa demo.py) và xuất kết quả ra file Markdown.

Cách dùng (từ thư mục gốc project):
    python scripts\\run_demo_md.py
    python scripts\\run_demo_md.py --demo
    python scripts\\run_demo_md.py --text "Urgent: You won $1,000,000 cash!"

Mọi tham số phía sau được chuyển nguyên cho demo.py.
File kết quả: reports/demo/demo_results.md
"""

import re
import runpy
import sys
import traceback
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = PROJECT_ROOT / "scripts" / "demo.py"
OUTPUT_MD = PROJECT_ROOT / "reports" / "demo" / "demo_results.md"

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class _Tee:
    """Vẫn in ra màn hình, đồng thời ghi lại nội dung."""

    def __init__(self, original, log):
        self._original = original
        self._log = log

    def write(self, text):
        self._log.append(text)
        return self._original.write(text)

    def flush(self):
        self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


def main() -> int:
    if Path(__file__).resolve() == DEMO_PATH.resolve():
        print(
            "[!] File này đang được đặt tên là scripts/demo.py nên sẽ tự gọi lại chính nó.\n"
            "    Hãy khôi phục demo.py về bản gốc của nhóm (8 dòng, gọi predict.main),\n"
            "    và lưu file này với tên khác: scripts/run_demo_md.py."
        )
        return 2
    if not DEMO_PATH.exists():
        print(f"[!] Không tìm thấy {DEMO_PATH}")
        return 2

    captured = []
    started = datetime.now()
    real_stdout, real_stderr, real_argv = sys.stdout, sys.stderr, sys.argv
    sys.stdout = _Tee(real_stdout, captured)
    sys.stderr = _Tee(real_stderr, captured)
    sys.argv = [str(DEMO_PATH)] + real_argv[1:]

    exit_code = 0
    try:
        runpy.run_path(str(DEMO_PATH), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            exit_code = 1
    except Exception:
        exit_code = 1
        captured.append(traceback.format_exc())
    finally:
        sys.stdout, sys.stderr, sys.argv = real_stdout, real_stderr, real_argv

    text = _ANSI.sub("", "".join(captured)).strip("\n")
    status = "Chạy thành công" if exit_code == 0 else "Có lỗi khi chạy"
    command = "python scripts/demo.py" + "".join(f" {a}" for a in real_argv[1:])

    md = (
        "# Kết quả chạy demo\n\n"
        f"- **Thời gian chạy:** {started:%d/%m/%Y %H:%M:%S}\n"
        f"- **Lệnh:** `{command}`\n"
        f"- **Trạng thái:** {status}\n\n"
        "## Kết quả\n\n"
        "~~~text\n"
        f"{text}\n"
        "~~~\n"
    )
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text(md, encoding="utf-8")
    print(f"\n[+] Đã xuất kết quả ra file: {OUTPUT_MD}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())