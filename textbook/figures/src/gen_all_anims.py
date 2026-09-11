# -*- coding: utf-8 -*-
"""一键重建全部 GIF 动画：在进程内依次运行 gen_anim_ch*.py。

GIF 生成比静态 SVG 慢（几秒到几十秒一个），因此不并入 gen_all.py，
需要更新动画时单独运行本脚本。
"""

import runpy
import traceback
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    scripts = sorted(SRC_DIR.glob("gen_anim_ch*.py"))
    failed = []
    for script in scripts:
        print(f"=== {script.name} ===")
        try:
            runpy.run_path(str(script), run_name="__main__")
        except Exception:
            failed.append(script.name)
            traceback.print_exc()
    if failed:
        print(f"失败 {len(failed)} 个: {failed}")
        raise SystemExit(1)
    print(f"全部完成，共 {len(scripts)} 个动画脚本。")
