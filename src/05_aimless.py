# Author: ZengWenquan
# https://github.com/chaosbull
# License: Apache-2.0
# -*- coding: utf-8 -*-
"""Aimless two-step neighborhood options from the current spot."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from importlib import import_module
from matplotlib.gridspec import GridSpec

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
    bottom_text_panel,
    ensure_dirs,
    load_model,
    load_tables,
    out_subdir,
    road_key,
    setup_font,
)

walk = import_module("03_walk_animation")


def build_aimless_viz(current: int = 12, visited=None, t0: int = 55):
    setup_font()
    ensure_dirs()
    spots, roads_df, _ = load_tables()
    model = load_model()
    gmap = GardenMap(spots, roads_df)

    if visited is None:
        rec = import_module("02_recommend_paths")
        packs = rec.pick_three_paths(model, s=1, g=22, t0=40, main_route_id=1)
        nodes = [int(x) for x in str(packs[0]["path_nodes"]).split("->")]
        if current in nodes:
            visited = set(nodes[: nodes.index(current) + 1])
        else:
            visited = {1, 2, 7, 11, current}
    visited = set(visited)

    rec_df, s1, s2, edge_hl = walk.two_step_rec(model, current, visited - {current}, t0=t0, topn=8)
    visited_edges = set()
    for a in visited:
        for b in visited:
            if a >= b:
                continue
            for r in (1, 2):
                if road_key(a, b, r) in gmap.centerlines:
                    visited_edges.add(road_key(a, b, r))

    fig = plt.figure(figsize=(13, 12), dpi=150)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3.2, 1.55], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1])

    gmap.draw_atmosphere(ax)
    gmap.draw_roads(
        ax,
        visited_edges=visited_edges,
        highlight_edges=edge_hl,
        show_scores=True,
        score_fontsize=7.5,
        half_width=DEFAULT_HALF_WIDTH,
        dim_others=True,
        focus_edges=visited_edges | set(edge_hl.keys()),
    )
    gmap.draw_spots(
        ax,
        visited=visited - {current},
        current=current,
        rec_step1=s1,
        rec_step2=s2,
        show_names=True,
        radius=78,
    )
    gmap.draw_tourist(ax, gmap.xy[current])
    add_legend(ax, "walk")
    ax.set_title(
        STR["aimless_title"].format(park=PARK_NAME),
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        color=COLORS["ink"],
    )

    sep = ", " if LANG == "en" else "、"
    lines = [
        STR["aimless_here"].format(name=gmap.names[current]),
        STR["aimless_visited"].format(names=sep.join(gmap.names[s] for s in sorted(visited))),
        STR["aimless_header"],
    ]
    rows = []
    for _, r in rec_df.iterrows():
        tau = int(r["step1_arrive_min"])
        _, lo, hi = walk.crowd_interval(model, int(r["step1_spot"]), t0, tau)
        a_name = gmap.names[int(r["step1_spot"])]
        b_name = gmap.names[int(r["step2_spot"])]
        lines.append(
            STR["aimless_line"].format(
                rank=int(r["rank"]), a=a_name, b=b_name, tau=tau, lo=lo, hi=hi, q=float(r["path_Q"]), jr=float(r["JR"])
            )
        )
        rows.append(
            {
                "current": gmap.names[current],
                "rank": int(r["rank"]),
                "step1": a_name,
                "step2": b_name,
                "arrive_min": tau,
                "crowd_lo": round(lo, 2),
                "crowd_hi": round(hi, 2),
                "path_Q": r["path_Q"],
                "JR": r["JR"],
                "path_L": r["path_L"],
            }
        )

    bottom_text_panel(
        ax_t,
        lines[:11],
        title=STR["walk_panel_title"],
        fontsize=BODY_FONTSIZE,
        title_fontsize=TITLE_FONTSIZE - 3,
        wrap_width=88 if LANG == "en" else 44,
    )

    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in gmap.names[current])
    out = out_subdir("05") / (f"aimless_from_{safe}.png" if LANG == "en" else f"漫游推荐_自{gmap.names[current]}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(
        out_subdir("05") / ("twostep_data.csv" if LANG == "en" else "两步推荐数据.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    (out_subdir("05") / ("readme.md" if LANG == "en" else "说明.md")).write_text(
        "\n".join(
            [
                f"# {STR['aimless_title'].format(park=PARK_NAME)}\n",
                f"Visitor at **{gmap.names[current]}** with no final destination.",
                "Shows two-step reachables excluding visited spots, with travel time and crowd interval.",
                f"Visited: {sep.join(gmap.names[s] for s in sorted(visited))}",
            ]
        ),
        encoding="utf-8",
    )
    print("aimless saved", out)
    return out


if __name__ == "__main__":
    build_aimless_viz()
