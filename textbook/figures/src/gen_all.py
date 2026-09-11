# -*- coding: utf-8 -*-
"""一键重建全部章节插图：依次运行 gen_ch*.py。"""

import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    scripts = sorted(SRC_DIR.glob("gen_ch*.py"))
    failed = []
    for script in scripts:
        print(f"=== {script.name} ===")
        result = subprocess.run([sys.executable, script.name], cwd=SRC_DIR)
        if result.returncode != 0:
            failed.append(script.name)
    if failed:
        print(f"失败 {len(failed)} 个: {failed}")
        sys.exit(1)
    print(f"全部完成，共 {len(scripts)} 个章节脚本。")
