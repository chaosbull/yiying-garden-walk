# Author: ZengWenquan
# https://github.com/chaosbull
# License: Apache-2.0
# -*- coding: utf-8 -*-
"""Three recommended gate-to-exit paths with maps and captions."""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import pandas as pd
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
    bottom_text_panel,
    ensure_dirs,
    load_model,
    load_tables,
    out_subdir,
    road_key,
    setup_font,
)


def _path_edges(nodes, roads_choice):
    return {road_key(nodes[i], nodes[i + 1], r) for i, r in enumerate(roads_choice)}


def pick_three_paths(model, s=1, g=22, t0=40, main_route_id=1):
    df = model.shortest_paths_to_goal(s, g, t0=t0, main_route_id=main_route_id, max_depth=8)
    if df.empty:
        raise RuntimeError("no paths found")
    chosen, used = [], []
    for _, row in df.iterrows():
        nodes = [int(x) for x in str(row["path_nodes"]).split("->")]
        sset = set(nodes)
        if any(len(sset & prev) / max(len(sset | prev), 1) > 0.72 for prev in used):
            continue
        chosen.append(row)
        used.append(sset)
        if len(chosen) >= 3:
            break
    if len(chosen) < 3:
        for _, row in df.iterrows():
            if any(str(row["path_nodes"]) == str(c["path_nodes"]) for c in chosen):
                continue
            chosen.append(row)
            if len(chosen) >= 3:
                break
    return chosen


def explain_lines(row, rank: int, names: dict):
    nodes = [int(x) for x in str(row["path_nodes"]).split("->")]
    route = " → ".join(names[n] for n in nodes)
    bullets = []
    if rank == 1:
        bullets.append(STR["why_jr"])
    if float(row["Q"]) >= 1.2:
        bullets.append(STR["why_q_hi"].format(q=float(row["Q"])))
    else:
        bullets.append(STR["why_q_mid"].format(q=float(row["Q"])))
    if float(row["P"]) < 50:
        bullets.append(STR["why_p_lo"])
    else:
        bullets.append(STR["why_p_hi"])
    if int(row["H"]) >= 3:
        bullets.append(STR["why_h"].format(h=int(row["H"])))
    if float(row["R_main"]) >= 0.35:
        bullets.append(STR["why_main"])
    bullets.append(STR["why_stats"].format(L=float(row["L"]), C=float(row["C"]), T=float(row["T"])))
    width = 56 if LANG == "en" else 42
    route_wrapped = "\n".join(textwrap.wrap(route, width=width))
    body = route_wrapped + "\n" + "\n".join(f"· {b}" for b in bullets)
    return route, body


def build_recommend_paths():
    setup_font()
    ensure_dirs()
    spots, roads, _ = load_tables()
    model = load_model()
    gmap = GardenMap(spots, roads)
    names = gmap.names
    chosen = pick_three_paths(model, s=1, g=22, t0=40, main_route_id=1)

    path_pack = []
    for rank, row in enumerate(chosen, start=1):
        nodes = [int(x) for x in str(row["path_nodes"]).split("->")]
        road_choices = [model._best_road(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
        route, body = explain_lines(row, rank, names)
        path_pack.append(
            {
                "rank": rank,
                "nodes": nodes,
                "roads": road_choices,
                "row": row,
                "route": route,
                "body": body,
                "explain": body,
            }
        )

    fig = plt.figure(figsize=(18, 10.2), dpi=150)
    outer = GridSpec(1, 3, figure=fig, wspace=0.12)
    for i, pack in enumerate(path_pack):
        inner = outer[i].subgridspec(2, 1, height_ratios=[3.2, 1.55], hspace=0.10)
        ax = fig.add_subplot(inner[0])
        ax_t = fig.add_subplot(inner[1])
        gmap.draw_atmosphere(ax)
        edges = _path_edges(pack["nodes"], pack["roads"])
        gmap.draw_roads(
            ax,
            show_scores=False,
            half_width=DEFAULT_HALF_WIDTH,
            dim_others=True,
            focus_edges=edges,
            highlight_edges={e: "active" for e in edges},
        )
        gmap.draw_spots(
            ax,
            visited=set(pack["nodes"][1:-1]),
            current=pack["nodes"][0],
            goal=pack["nodes"][-1],
            show_names=True,
            radius=58,
        )
        ax.set_title(
            STR["rec_path_title"].format(rank=pack["rank"]),
            fontsize=TITLE_FONTSIZE - 2,
            fontweight="bold",
            color=COLORS["ink"],
        )
        metrics = (
            f"JR={pack['row']['JR']:.3f}   Q={pack['row']['Q']:.2f}   "
            f"L={pack['row']['L']:.0f}   T≈{pack['row']['T']:.0f} min\n"
            f"N≈{pack['row']['N']:.1f}   P={pack['row']['P']:.1f}"
        )
        bottom_text_panel(
            ax_t,
            [metrics],
            title=STR["rec_metrics_title"].format(rank=pack["rank"]),
            fontsize=BODY_FONTSIZE,
            title_fontsize=TITLE_FONTSIZE - 4,
            wrap_width=40,
        )

    fig.suptitle(
        STR["rec_overview_title"].format(park=PARK_NAME),
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        color=COLORS["ink"],
        y=0.98,
    )
    out_overview = out_subdir("02") / ("three_paths_overview.png" if LANG == "en" else "三条推荐路径总览.png")
    fig.savefig(out_overview, bbox_inches="tight")
    plt.close(fig)

    lines = [f"# {STR['rec_overview_title'].format(park=PARK_NAME)}\n"]
    rows = []
    for pack in path_pack:
        lines.append(f"## Path {pack['rank']}\n")
        lines.append(pack["body"] + "\n")
        rows.append(
            {
                "rank": pack["rank"],
                "nodes": "->".join(map(str, pack["nodes"])),
                "names": pack["route"],
                "JR": pack["row"]["JR"],
                "Q": pack["row"]["Q"],
                "L": pack["row"]["L"],
                "C": pack["row"]["C"],
                "T": pack["row"]["T"],
                "N": pack["row"]["N"],
                "P": pack["row"]["P"],
                "H": pack["row"]["H"],
                "explain": pack["body"].replace("\n", " | "),
            }
        )
        fig = plt.figure(figsize=(12, 13.5), dpi=150)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[3.3, 1.7], hspace=0.08)
        ax = fig.add_subplot(gs[0])
        ax_t = fig.add_subplot(gs[1])
        gmap.draw_atmosphere(ax)
        edges = _path_edges(pack["nodes"], pack["roads"])
        gmap.draw_roads(
            ax,
            show_scores=True,
            score_fontsize=7.5,
            half_width=DEFAULT_HALF_WIDTH,
            dim_others=True,
            focus_edges=edges,
            highlight_edges={e: "active" for e in edges},
        )
        gmap.draw_spots(
            ax,
            current=1,
            goal=22,
            visited=set(pack["nodes"]) - {1, 22},
            show_names=True,
            radius=62,
        )
        ax.set_title(
            STR["rec_path_title"].format(rank=pack["rank"]),
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
            color=COLORS["ink"],
            pad=10,
        )
        bottom_text_panel(
            ax_t,
            pack["body"].split("\n"),
            title=STR["rec_explain_title"].format(rank=pack["rank"]),
            fontsize=BODY_FONTSIZE,
            title_fontsize=TITLE_FONTSIZE - 3,
            wrap_width=78 if LANG == "en" else 42,
        )
        p = out_subdir("02") / (f"recommended_path_{pack['rank']}.png" if LANG == "en" else f"推荐路径{pack['rank']}.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(rows).to_csv(
        out_subdir("02") / ("three_paths.csv" if LANG == "en" else "三条推荐路径说明.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    (out_subdir("02") / ("three_paths.md" if LANG == "en" else "三条推荐路径说明.md")).write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("saved recommend paths ->", out_overview)
    return path_pack


if __name__ == "__main__":
    build_recommend_paths()
