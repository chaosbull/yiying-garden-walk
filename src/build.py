# Author: ZengWenquan
# https://github.com/chaosbull
# License: Apache-2.0
# -*- coding: utf-8 -*-
"""Build the five figure sets and write them under output/."""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import viz_common  # noqa: E402
from viz_common import configure_park, ensure_dirs, setup_font  # noqa: E402

MOD_NAMES = (
    "01_network_map",
    "02_recommend_paths",
    "03_walk_animation",
    "04_reroute_animation",
    "05_aimless",
)


def _sync_park_bindings():
    for name in MOD_NAMES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        for attr in ("OUT", "DATA", "RESULTS", "PARK_NAME", "PARK_ID", "LANG", "STR"):
            if hasattr(viz_common, attr):
                setattr(mod, attr, getattr(viz_common, attr))


def build_one(lang: str = "en", n_frames: int = 400):
    cfg = configure_park(1, lang=lang)
    setup_font()
    ensure_dirs()
    t0 = time.time()
    print(f"\n--- {viz_common.PARK_NAME} ({viz_common.LANG}) -> {viz_common.OUT} ---")

    for name in MOD_NAMES:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
        else:
            importlib.import_module(name)
    configure_park(1, lang=lang)
    _sync_park_bindings()

    print("=== 01 network map ===")
    importlib.import_module("01_network_map").build_network_map()

    print("=== 02 recommend paths ===")
    packs = importlib.import_module("02_recommend_paths").build_recommend_paths()
    best_nodes = packs[0]["nodes"]

    print("=== 05 aimless ===")
    mid = best_nodes[len(best_nodes) // 2]
    importlib.import_module("05_aimless").build_aimless_viz(current=mid)

    print("=== 03 walk animation ===")
    importlib.import_module("03_walk_animation").build_walk_animation(
        path_nodes=best_nodes, n_frames=n_frames
    )

    print("=== 04 reroute ===")
    importlib.import_module("04_reroute_animation").build_reroute_animation(n_frames=n_frames)

    readme = viz_common.OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# {viz_common.PARK_NAME} Walk-Along Visualization\n",
                f"- Park: {viz_common.PARK_NAME}",
                f"- Language: {viz_common.LANG}",
                "- Data: `data/`",
                "- Model results: `results/`\n",
                "## Folders\n",
                "1. `01_network_scores/` — network + mid-segment Q scores",
                "2. `02_recommended_paths/` — three gate→exit recommendations",
                "3. `03_frame_walk/` — ~400 walk frames; recommendations only on arrival",
                "4. `04_midtrip_reroute/` — once / twice destination changes",
                "5. `05_aimless_twostep/` — aimless two-step local options\n",
                f"Elapsed ~ {time.time() - t0:.1f} s.\n",
            ]
        ),
        encoding="utf-8",
    )
    print("DONE ->", viz_common.OUT, "elapsed", round(time.time() - t0, 1), "s")
    return viz_common.OUT


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build Yiying Garden visualizations into output/")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--lang", type=str, default="en", choices=["en", "zh"])
    args = ap.parse_args()
    build_one(lang=args.lang, n_frames=args.frames)
