# Author: ZengWenquan
# https://github.com/chaosbull
# License: Apache-2.0
# -*- coding: utf-8 -*-
"""Full garden network map with Q scores on path midpoints."""
from __future__ import annotations

import matplotlib.pyplot as plt

from viz_common import (
    BODY_FONTSIZE,
    COLORS,
    DEFAULT_HALF_WIDTH,
    LANG,
    PARK_NAME,
    STR,
    TITLE_FONTSIZE,
    GardenMap,
    add_legend,
    ensure_dirs,
    load_tables,
    out_subdir,
    setup_font,
)


def build_network_map():
    setup_font()
    ensure_dirs()
    spots, roads, _ = load_tables()
    gmap = GardenMap(spots, roads)

    fig, ax = plt.subplots(figsize=(15, 12), dpi=160)
    gmap.draw_atmosphere(ax)
    gmap.draw_roads(ax, show_scores=True, score_fontsize=8.0, half_width=DEFAULT_HALF_WIDTH)
    gmap.draw_spots(ax, show_names=True, radius=68)
    add_legend(ax, "network")
    ax.set_title(
        STR["network_title"].format(park=PARK_NAME),
        fontsize=TITLE_FONTSIZE + 1,
        fontweight="bold",
        color=COLORS["ink"],
        pad=14,
    )
    ax.text(
        0.5,
        -0.02,
        STR["network_footer"],
        transform=ax.transAxes,
        ha="center",
        fontsize=BODY_FONTSIZE + 0.5,
        color=COLORS["muted"],
    )
    out = out_subdir("01") / ("network_scores_map.png" if LANG == "en" else "全园路径评分总图.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    roads[["road_id", "from_id", "to_id", "road_index_r", "length_L", "Q"]].to_csv(
        out_subdir("01") / ("road_scores.csv" if LANG == "en" else "道路评分明细.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print("bend_stats", gmap._bend_stats)
    print("saved", out)
    return out


if __name__ == "__main__":
    build_network_map()
