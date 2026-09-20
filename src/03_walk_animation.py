# Author: ZengWenquan
# https://github.com/chaosbull
# License: Apache-2.0
# -*- coding: utf-8 -*-
"""Walk-along animation; two-step recommendations only on arrival."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz_common import (
    BODY_FONTSIZE,
    COLORS,
    LANG,
    OUT,
    STR,
    TITLE_FONTSIZE,
    GardenMap,
    add_legend,
    ensure_dirs,
    load_model,
    load_tables,
    out_subdir,
    road_key,
    setup_font,
)


def build_itinerary(model, nodes: List[int], t0: int = 35):
    roads = [model._best_road(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    # segment travel minutes
    taus = [model.edge_lookup[(nodes[i], nodes[i + 1], roads[i])]["tau"] for i in range(len(nodes) - 1)]
    stays = []
    for i, sid in enumerate(nodes):
        if i == 0 or i == len(nodes) - 1:
            stays.append(0)
        else:
            stays.append(int(round(float(model.spot_meta[sid]["mu_stay"]))))
    return roads, taus, stays


def timeline_to_frames(
    nodes: List[int],
    roads: List[int],
    taus: List[int],
    stays: List[int],
    n_frames: int = 400,
):
    """
    Build a continuous timeline:
      at node0 (brief) -> move edge0 -> arrive node1 (dwell for recommendation) -> ...
    Return list of frame descriptors.
    """
    # relative weights: walking gets most frames; arrivals get dedicated still frames
    move_w = list(taus)
    # arrival dwell weight proportional to stay but capped; start also has short dwell
    arrive_w = [max(2, int(s)) for s in stays]
    arrive_w[0] = 3
    arrive_w[-1] = 5

    units = []
    # start dwell at node 0
    units.append(("arrive", 0, None, arrive_w[0]))
    for i in range(len(roads)):
        units.append(("move", i, None, move_w[i] * 4))  # walking emphasized
        units.append(("arrive", i + 1, None, arrive_w[i + 1] * 3))

    total_w = sum(u[3] for u in units)
    frames = []
    # allocate integer frames
    alloc = [max(1, int(round(n_frames * u[3] / total_w))) for u in units]
    # fix sum
    while sum(alloc) > n_frames:
        j = int(np.argmax(alloc))
        alloc[j] -= 1
    while sum(alloc) < n_frames:
        j = int(np.argmax([u[3] for u in units]))
        alloc[j] += 1

    fid = 0
    for (kind, idx, _, _), n in zip(units, alloc):
        for k in range(n):
            frac = (k + 0.5) / n
            if kind == "arrive":
                frames.append(
                    {
                        "frame_id": fid,
                        "kind": "arrive",
                        "node_index": idx,
                        "spot": nodes[idx],
                        "frac": 1.0,
                        "edge_index": None,
                    }
                )
            else:
                frames.append(
                    {
                        "frame_id": fid,
                        "kind": "move",
                        "node_index": idx,  # edge from nodes[idx] -> nodes[idx+1]
                        "spot": None,
                        "frac": frac,
                        "edge_index": idx,
                    }
                )
            fid += 1
    return frames[:n_frames]


def visited_state(nodes, roads, frame) -> Tuple[Set[int], Set[Tuple[int, int, int]], Optional[Tuple[int, int, int]], int]:
    """Return visited spots, visited edges, active edge, current spot (last arrived)."""
    visited_spots: Set[int] = set()
    visited_edges: Set[Tuple[int, int, int]] = set()
    active = None
    current_spot = nodes[0]

    if frame["kind"] == "arrive":
        ni = frame["node_index"]
        visited_spots = set(nodes[: ni + 1])
        for i in range(ni):
            visited_edges.add(road_key(nodes[i], nodes[i + 1], roads[i]))
        current_spot = nodes[ni]
    else:
        ei = frame["edge_index"]
        visited_spots = set(nodes[: ei + 1])
        for i in range(ei):
            visited_edges.add(road_key(nodes[i], nodes[i + 1], roads[i]))
        active = road_key(nodes[ei], nodes[ei + 1], roads[ei])
        current_spot = nodes[ei]
    return visited_spots, visited_edges, active, current_spot


def two_step_rec(model, spot: int, visited: Set[int], t0: int, topn: int = 4):
    df = model.two_step_recommendations(spot, t0=t0, main_route_id=1, visit_middle=True)
    if df.empty:
        return df, set(), set(), {}
    # filter visited
    m = (~df["step1_spot"].isin(visited)) & (~df["step2_spot"].isin(visited))
    df = df[m].copy()
    df = df.head(topn)
    s1 = set(df["step1_spot"].astype(int))
    s2 = set(df["step2_spot"].astype(int)) - s1
    edge_hl = {}
    for _, r in df.iterrows():
        e1 = road_key(spot, int(r["step1_spot"]), int(r["road1_r"]))
        e2 = road_key(int(r["step1_spot"]), int(r["step2_spot"]), int(r["road2_r"]))
        edge_hl[e1] = "rec"
        edge_hl[e2] = "rec"
    return df, s1, s2, edge_hl


def crowd_interval(model, spot: int, t0: int, tau: int) -> Tuple[float, float, float]:
    """Crowd display in 0–50 with per-spot variation from estimated N series."""
    t0 = int(np.clip(t0, 1, max(1, model.T - 1)))
    tau = int(np.clip(tau, 0, max(0, model.T - t0)))
    t_arr = int(np.clip(t0 + tau, 1, model.T))
    i = model.id2idx[int(spot)]
    # Prefer estimated occupancy; fall back to predictive Nbar
    n_est = float(model.N[i, t_arr]) if model.N is not None else None
    Nhat, Nbar = model.predict_N_at(spot, t0, tau)
    if n_est is None or not np.isfinite(n_est):
        center = float(Nbar)
    else:
        # Blend so display tracks spot/time series but stays in 0–50
        center = 0.65 * n_est + 0.35 * float(Nbar)
    center = float(np.clip(center, 0.0, 50.0))
    spread = max(2.0, 0.12 * center + 1.5)
    lo = float(np.clip(center - spread, 0.0, 50.0))
    hi = float(np.clip(center + spread * 0.85, 0.0, 50.0))
    if hi - lo < 2.0:
        hi = min(50.0, lo + 2.0)
    return center, lo, hi


def render_frame(
    gmap: GardenMap,
    model,
    nodes,
    roads,
    frame,
    visited_spots,
    visited_edges,
    active,
    current_spot,
    tourist_xy,
    show_rec: bool,
    rec_df,
    rec1,
    rec2,
    rec_edges,
    clock_min: float,
    goal: int,
    title: str,
    out_path: Path,
    extra_lines=None,
    half_width: float = 20.0,
    map_only: bool = False,
):
    """Render one frame. If map_only=True, save only the map (no title / bottom panel) for GIFs."""
    from matplotlib.gridspec import GridSpec
    from viz_common import DEFAULT_HALF_WIDTH, bottom_text_panel

    hw = half_width if half_width is not None else DEFAULT_HALF_WIDTH

    if map_only:
        fig, ax = plt.subplots(figsize=(11.5, 9.2), dpi=100)
    else:
        fig = plt.figure(figsize=(12, 12.2), dpi=100)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[3.15, 1.65], hspace=0.08)
        ax = fig.add_subplot(gs[0])
        ax_t = fig.add_subplot(gs[1])

    gmap.draw_atmosphere(ax)
    hl = dict(rec_edges) if show_rec else {}
    if active is not None:
        hl[active] = "active"
    gmap.draw_roads(
        ax,
        visited_edges=visited_edges,
        highlight_edges=hl,
        show_scores=False,
        half_width=hw,
        dim_others=True,
        focus_edges=visited_edges | set(hl.keys()) | ({active} if active else set()),
    )
    gmap.draw_spots(
        ax,
        visited=visited_spots - ({current_spot} if show_rec else set()),
        current=current_spot if frame["kind"] == "arrive" else None,
        rec_step1=rec1 if show_rec else set(),
        rec_step2=rec2 if show_rec else set(),
        goal=goal,
        show_names=True,
        radius=62,
    )
    gmap.draw_tourist(ax, (float(tourist_xy[0]), float(tourist_xy[1])))
    add_legend(ax, "walk")

    if map_only:
        fig.tight_layout(pad=0.15)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        return

    ax.set_title(STR["walk_title"], fontsize=TITLE_FONTSIZE - 1, fontweight="bold", color=COLORS["ink"], pad=8)

    mode = STR["walk_arrive"] if show_rec else STR["walk_move"]
    lines = [
        title,
        STR["walk_frame"].format(fid=frame["frame_id"] + 1, mode=mode, mins=clock_min),
        STR["walk_current"].format(name=gmap.names[current_spot]),
    ]
    if show_rec and rec_df is not None and len(rec_df):
        lines.append(STR["walk_rec_header"])
        for _, r in rec_df.head(3).iterrows():
            tau = int(r["step1_arrive_min"])
            _, lo, hi = crowd_interval(model, int(r["step1_spot"]), 35 + int(min(clock_min, model.T - 2)), tau)
            a_name = gmap.names[int(r["step1_spot"])]
            b_name = gmap.names[int(r["step2_spot"])]
            lines.append(
                STR["walk_rec_line"].format(a=a_name, b=b_name, tau=tau, lo=lo, hi=hi, q=float(r["path_Q"]))
            )
        _, lo0, hi0 = crowd_interval(model, current_spot, 35, int(round(min(clock_min, model.T - 2))))
        lines.append(STR["walk_crowd_here"].format(lo=lo0, hi=hi0))
    if extra_lines:
        lines.extend(extra_lines)

    bottom_text_panel(
        ax_t,
        lines,
        title=STR["walk_panel_title"],
        fontsize=BODY_FONTSIZE,
        title_fontsize=TITLE_FONTSIZE - 3,
        wrap_width=86 if LANG == "en" else 44,
    )
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


def _assemble_gif(png_paths, gif_path: Path):
    raw = [imageio.imread(p) for p in png_paths]
    if not raw:
        return
    th = max(im.shape[0] for im in raw)
    tw = max(im.shape[1] for im in raw)
    imgs = []
    for im in raw:
        canvas = np.full((th, tw, im.shape[2] if im.ndim == 3 else 3), 238, dtype=im.dtype)
        ih, iw = im.shape[:2]
        if im.ndim == 2:
            canvas[:ih, :iw, 0] = im
            canvas[:ih, :iw, 1] = im
            canvas[:ih, :iw, 2] = im
        else:
            canvas[:ih, :iw, : im.shape[2]] = im
        imgs.append(canvas)
    imageio.mimsave(gif_path, imgs, duration=0.08)
    print("gif", gif_path)


def build_walk_animation(
    path_nodes: Optional[List[int]] = None,
    n_frames: int = 400,
    make_gif: bool = True,
    gif_only: bool = False,
):
    """If gif_only=True, skip full info-panel frames and only rebuild the map-only GIF."""
    setup_font()
    ensure_dirs()
    spots, roads_df, _ = load_tables()
    model = load_model()
    gmap = GardenMap(spots, roads_df)

    if path_nodes is None:
        from importlib import import_module

        rec = import_module("02_recommend_paths")
        packs = rec.build_recommend_paths()
        path_nodes = packs[0]["nodes"]

    goal = path_nodes[-1]
    road_ids, taus, stays = build_itinerary(model, path_nodes)
    frames = timeline_to_frames(path_nodes, road_ids, taus, stays, n_frames=n_frames)

    frame_dir = out_subdir("03_frames")
    if not gif_only:
        for p in frame_dir.glob("*.png"):
            try:
                p.unlink()
            except OSError:
                pass

    rec_log = []
    paths_png = []
    gif_pngs = []
    gif_dir = out_subdir("03") / "_gif_frames"
    gif_dir.mkdir(parents=True, exist_ok=True)
    for p in gif_dir.glob("*.png"):
        try:
            p.unlink()
        except OSError:
            pass

    clock = 0.0
    for i, fr in enumerate(frames):
        visited_spots, visited_edges, active, current_spot = visited_state(path_nodes, road_ids, fr)
        show_rec = fr["kind"] == "arrive"
        rec_df = None
        rec1 = set()
        rec2 = set()
        rec_edges = {}
        if show_rec and not gif_only:
            rec_df, rec1, rec2, rec_edges = two_step_rec(
                model, fr["spot"], visited_spots - {fr["spot"]}, t0=35 + int(clock), topn=4
            )
            for _, r in rec_df.iterrows():
                tau = int(r["step1_arrive_min"])
                _, lo, hi = crowd_interval(model, int(r["step1_spot"]), 35 + int(clock), tau)
                rec_log.append(
                    {
                        "frame": fr["frame_id"] + 1,
                        "spot_id": fr["spot"],
                        "spot": gmap.names[fr["spot"]],
                        "elapsed_min": round(clock, 2),
                        "step1": gmap.names[int(r["step1_spot"])],
                        "step2": gmap.names[int(r["step2_spot"])],
                        "arrive_min": tau,
                        "crowd_lo": round(lo, 2),
                        "crowd_hi": round(hi, 2),
                        "path_Q": r["path_Q"],
                        "JR": r["JR"],
                    }
                )
        elif show_rec and gif_only:
            # Still need rec highlights on arrive frames for the map GIF
            rec_df, rec1, rec2, rec_edges = two_step_rec(
                model, fr["spot"], visited_spots - {fr["spot"]}, t0=35 + int(clock), topn=4
            )

        if fr["kind"] == "move":
            ei = fr["edge_index"]
            tourist_xy = gmap.point_on_road(path_nodes[ei], path_nodes[ei + 1], road_ids[ei], fr["frac"])
            clock += taus[ei] / max(1, sum(1 for f in frames if f["kind"] == "move" and f["edge_index"] == ei))
        else:
            tourist_xy = np.array(gmap.xy[fr["spot"]])
            ni = fr["node_index"]
            n_arrive = sum(1 for f in frames if f["kind"] == "arrive" and f["node_index"] == ni)
            clock += stays[ni] / max(1, n_arrive)

        title = STR["walk_main_path"].format(route=" → ".join(gmap.names[n] for n in path_nodes))
        common = dict(
            gmap=gmap,
            model=model,
            nodes=path_nodes,
            roads=road_ids,
            frame=fr,
            visited_spots=visited_spots,
            visited_edges=visited_edges,
            active=active,
            current_spot=current_spot,
            tourist_xy=tourist_xy,
            show_rec=show_rec,
            rec_df=rec_df,
            rec1=rec1,
            rec2=rec2,
            rec_edges=rec_edges,
            clock_min=clock,
            goal=goal,
            title=title,
        )
        if not gif_only:
            out_path = frame_dir / f"frame_{fr['frame_id']+1:04d}.png"
            render_frame(**common, out_path=out_path, map_only=False)
            paths_png.append(out_path)
        if make_gif and i % 2 == 0:
            gpath = gif_dir / f"gif_{fr['frame_id']+1:04d}.png"
            render_frame(**common, out_path=gpath, map_only=True)
            gif_pngs.append(gpath)
        if (i + 1) % 40 == 0:
            print(f"walk frames {i+1}/{n_frames}")

    if not gif_only:
        csv_name = "arrival_recommendations.csv" if LANG == "en" else "抵达时两步推荐数据.csv"
        pd.DataFrame(rec_log).to_csv(out_subdir("03") / csv_name, index=False, encoding="utf-8-sig")
        meta = {
            "path_nodes": path_nodes,
            "path_names": [gmap.names[n] for n in path_nodes],
            "n_frames": n_frames,
            "note": "Recommendations only on arrive frames. GIF is map-only.",
            "lang": LANG,
        }
        (out_subdir("03") / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if make_gif:
        gif_name = "walk_animation.gif" if LANG == "en" else "循路动画.gif"
        gif_path = out_subdir("03") / gif_name
        _assemble_gif(gif_pngs, gif_path)
        for p in gif_pngs:
            try:
                p.unlink()
            except OSError:
                pass
        try:
            gif_dir.rmdir()
        except OSError:
            pass

    if not gif_only:
        seen = set()
        key_frames = []
        for f in frames:
            if f["kind"] != "arrive":
                continue
            if f["spot"] in seen:
                continue
            seen.add(f["spot"])
            key_frames.append(f["frame_id"])
        list_name = "key_arrival_frames.txt" if LANG == "en" else "关键抵达帧列表.txt"
        (out_subdir("03") / list_name).write_text(
            "\n".join(f"frame_{i+1:04d}.png  spot {frames[i]['spot']}" for i in key_frames),
            encoding="utf-8",
        )
    print("walk animation done", n_frames, "gif_only" if gif_only else "")
    return path_nodes, frames, road_ids


if __name__ == "__main__":
    build_walk_animation(n_frames=400)
