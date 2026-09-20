# Author: ZengWenquan
# https://github.com/chaosbull
# License: Apache-2.0
# -*- coding: utf-8 -*-
"""Mid-trip destination changes (once or twice)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from importlib import import_module

from viz_common import (
    DEFAULT_HALF_WIDTH,
    LANG,
    OUT,
    STR,
    GardenMap,
    ensure_dirs,
    load_model,
    load_tables,
    out_subdir,
    road_key,
    setup_font,
)

walk = import_module("03_walk_animation")


def _shortest_nodes(model, s: int, g: int, t0: int = 50) -> List[int]:
    df = model.shortest_paths_to_goal(s, g, t0=t0, main_route_id=1, max_depth=8)
    if df.empty:
        raise RuntimeError(f"no path {s}->{g}")
    return [int(x) for x in str(df.iloc[0]["path_nodes"]).split("->")]


def _build_phases(model, start: int, changes: Sequence[Tuple[int, int]], final_goal: int) -> List[List[int]]:
    """
    changes: list of (change_spot, new_goal_after_change) in chronological order.
    After last change, continue to that goal (final_goal should match last new_goal).
    """
    rec = import_module("02_recommend_paths")
    packs = rec.pick_three_paths(model, s=start, g=22, t0=40, main_route_id=1)
    orig = [int(x) for x in str(packs[0]["path_nodes"]).split("->")]

    phases: List[List[int]] = []
    cursor_path = orig
    cursor_start = start
    for change_spot, new_goal in changes:
        if change_spot not in cursor_path:
            # fall back to mid of current planned path
            change_spot = cursor_path[max(1, len(cursor_path) // 2)]
        idx = cursor_path.index(change_spot)
        phase = cursor_path[: idx + 1]
        if phases:
            # drop duplicated start node already counted
            phase = phase  # includes change_spot as end
        phases.append(phase)
        # replan from change_spot toward new_goal
        cursor_path = _shortest_nodes(model, change_spot, new_goal, t0=55)
        cursor_start = change_spot
    # final phase: remaining path to last goal
    phases.append(cursor_path)
    return phases


def _stitch_full(phases: List[List[int]]) -> List[int]:
    full = list(phases[0])
    for ph in phases[1:]:
        full.extend(ph[1:])
    return full


def build_reroute_scenario(
    out_dir: Path,
    changes: Sequence[Tuple[int, int]],
    n_frames: int = 320,
    scenario_title: str = "",
    gif_only: bool = False,
):
    """
    changes: [(spot_at_change, new_goal), ...]
    Produces frames + gif + csv under out_dir.
    If gif_only=True, only rebuild map-only GIF (skip full info-panel frames).
    """
    setup_font()
    ensure_dirs()
    spots, roads_df, _ = load_tables()
    model = load_model()
    gmap = GardenMap(spots, roads_df)

    phases = _build_phases(model, start=1, changes=changes, final_goal=changes[-1][1])
    full_nodes = _stitch_full(phases)
    n_phases = len(phases)

    itineraries = []
    for ph in phases:
        roads, taus, stays = walk.build_itinerary(model, ph)
        stays[-1] = max(stays[-1], 4)
        itineraries.append((roads, taus, stays))
    itineraries[-1] = (
        itineraries[-1][0],
        itineraries[-1][1],
        [0] + [int(round(float(model.spot_meta[s]["mu_stay"]))) for s in phases[-1][1:-1]] + [3],
    )

    weights = [1.15] * (n_phases - 1) + [1.0]
    wsum = sum(weights)
    alloc = [max(40, int(round(n_frames * w / wsum))) for w in weights]
    while sum(alloc) > n_frames:
        alloc[int(np.argmax(alloc))] -= 1
    while sum(alloc) < n_frames:
        alloc[int(np.argmin(alloc))] += 1

    frame_lists = []
    for ph, (roads, taus, stays), nf in zip(phases, itineraries, alloc):
        frame_lists.append(walk.timeline_to_frames(ph, roads, taus, stays, n_frames=nf))

    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    if not gif_only:
        for p in frame_dir.glob("*.png"):
            try:
                p.unlink()
            except OSError:
                pass

    rec_log = []
    gif_pngs = []
    clock = 0.0
    global_id = 0
    carry_visited: set = set()
    carry_edges: set = set()
    gif_dir = out_dir / "_gif_frames"
    gif_dir.mkdir(parents=True, exist_ok=True)
    for p in gif_dir.glob("*.png"):
        try:
            p.unlink()
        except OSError:
            pass

    change_spots = {c[0] for c in changes}
    change_goal_at = {c[0]: c[1] for c in changes}
    current_goal = 22

    def advance_clock(fr, taus, stays, frames_local):
        nonlocal clock
        if fr["kind"] == "move":
            ei = fr["edge_index"]
            n_move = sum(1 for f in frames_local if f["kind"] == "move" and f["edge_index"] == ei)
            clock += taus[ei] / max(1, n_move)
        else:
            ni = fr["node_index"]
            n_arr = sum(1 for f in frames_local if f["kind"] == "arrive" and f["node_index"] == ni)
            clock += stays[ni] / max(1, n_arr)

    for pi, (ph, frames_local, (roads, taus, stays)) in enumerate(zip(phases, frame_lists, itineraries)):
        end_spot = ph[-1]
        is_change_phase = pi < n_phases - 1 and end_spot in change_spots
        n_arrive_end = sum(1 for f in frames_local if f["kind"] == "arrive" and f["spot"] == end_spot)
        arrive_end_seen = 0
        next_goal = change_goal_at.get(end_spot, current_goal)

        for fr0 in frames_local:
            fr = dict(fr0)
            fr["frame_id"] = global_id
            visited_spots, visited_edges, active, current_spot = walk.visited_state(ph, roads, fr)
            visited_spots = visited_spots | carry_visited
            visited_edges = visited_edges | carry_edges

            show_rec = fr["kind"] == "arrive"
            is_decision = False
            if is_change_phase and show_rec and fr["spot"] == end_spot:
                arrive_end_seen += 1
                is_decision = arrive_end_seen > max(1, n_arrive_end // 2)

            if pi == 0 and not is_decision:
                goal_now = 22
            elif is_decision:
                goal_now = next_goal
            else:
                completed = min(pi, len(changes))
                goal_now = changes[completed - 1][1] if completed >= 1 else 22

            rec_df = None
            rec1 = set()
            rec2 = set()
            rec_edges: Dict = {}
            extra = None
            if show_rec:
                excl = visited_spots - {fr["spot"]}
                rec_df, rec1, rec2, rec_edges = walk.two_step_rec(
                    model, fr["spot"], excl, t0=35 + int(min(clock, model.T - 2))
                )
                if not gif_only:
                    for _, r in rec_df.iterrows():
                        tau = int(r["step1_arrive_min"])
                        _, lo, hi = walk.crowd_interval(
                            model, int(r["step1_spot"]), 35 + int(min(clock, model.T - 2)), tau
                        )
                        rec_log.append(
                            {
                                "scenario": scenario_title,
                                "phase": pi + 1,
                                "frame": global_id + 1,
                                "spot": gmap.names[fr["spot"]],
                                "goal": gmap.names.get(goal_now, str(goal_now)),
                                "elapsed_min": round(clock, 2),
                                "step1": gmap.names[int(r["step1_spot"])],
                                "step2": gmap.names[int(r["step2_spot"])],
                                "arrive_min": tau,
                                "crowd_lo": round(lo, 2),
                                "crowd_hi": round(hi, 2),
                                "changing_goal": int(is_decision),
                            }
                        )
            if is_decision and not gif_only:
                extra = [
                    STR["reroute_star"].format(goal=gmap.names[next_goal]),
                    STR["reroute_kept"].format(
                        names=(", " if LANG == "en" else "、").join(gmap.names[s] for s in sorted(visited_spots))
                    ),
                ]

            if fr["kind"] == "move":
                ei = fr["edge_index"]
                tourist_xy = gmap.point_on_road(ph[ei], ph[ei + 1], roads[ei], fr["frac"])
            else:
                tourist_xy = np.array(gmap.xy[fr["spot"]])
            advance_clock(fr, taus, stays, frames_local)

            title = STR["reroute_phase"].format(
                scenario=scenario_title, pi=pi + 1, n=n_phases, goal=gmap.names[goal_now]
            )
            if is_decision:
                title += STR["reroute_changing"]

            common = dict(
                gmap=gmap,
                model=model,
                nodes=ph,
                roads=roads,
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
                goal=goal_now,
                title=title,
                extra_lines=extra,
                half_width=DEFAULT_HALF_WIDTH,
            )
            if not gif_only:
                out_path = frame_dir / f"frame_{global_id+1:04d}.png"
                walk.render_frame(**common, out_path=out_path, map_only=False)
            if global_id % 2 == 0:
                gpath = gif_dir / f"gif_{global_id+1:04d}.png"
                walk.render_frame(**common, out_path=gpath, map_only=True)
                gif_pngs.append(gpath)
            global_id += 1
            if global_id % 40 == 0:
                print(f"[{scenario_title}] frames {global_id}/{n_frames}")

        carry_visited |= set(ph)
        carry_edges |= {road_key(ph[i], ph[i + 1], roads[i]) for i in range(len(roads))}
        if is_change_phase:
            current_goal = next_goal

    if not gif_only:
        pd.DataFrame(rec_log).to_csv(
            out_dir / ("reroute_recommendations.csv" if LANG == "en" else "改目标过程推荐数据.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        meta = {
            "scenario": scenario_title,
            "changes": [{"spot": s, "new_goal": g} for s, g in changes],
            "phases": phases,
            "full_narrative": full_nodes,
            "n_frames": n_frames,
            "lang": LANG,
            "note": "GIF is map-only (no title / bottom panel).",
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        change_desc = []
        for i, (s, g) in enumerate(changes, 1):
            if LANG == "en":
                change_desc.append(f"{i}. At “{gmap.names[s]}” change goal to “{gmap.names[g]}”")
            else:
                change_desc.append(f"{i}. 在「{gmap.names[s]}」改为目标「{gmap.names[g]}」")
        readme = out_dir / ("readme.md" if LANG == "en" else "改目标说明.md")
        readme.write_text(
            "\n".join(
                [
                    f"# {scenario_title}\n",
                    f"- Route: {' → '.join(gmap.names[n] for n in full_nodes)}",
                    *change_desc,
                    f"- Frames ≈ {n_frames}; recommendations only on arrival frames",
                ]
            ),
            encoding="utf-8",
        )

    gif_path = out_dir / ("reroute_animation.gif" if LANG == "en" else "改目标循路动画.gif")
    walk._assemble_gif(gif_pngs, gif_path)
    for p in gif_pngs:
        try:
            p.unlink()
        except OSError:
            pass
    try:
        gif_dir.rmdir()
    except OSError:
        pass
    print("saved", gif_path)
    return {"scenario": scenario_title, "n_frames": n_frames}


def build_reroute_animation(n_frames: int = 320, gif_only: bool = False):
    """Build both once / twice destination-change scenarios."""
    ensure_dirs()
    spots, _, _ = load_tables()
    names = {int(r.spot_id): str(r.name) for r in spots.itertuples()}
    meta1 = build_reroute_scenario(
        out_subdir("04_once"),
        changes=[(9, 18)],
        n_frames=n_frames,
        scenario_title=STR["reroute_once"],
        gif_only=gif_only,
    )
    meta2 = build_reroute_scenario(
        out_subdir("04_twice"),
        changes=[(4, 14), (14, 19)],
        n_frames=n_frames,
        scenario_title=STR["reroute_twice"],
        gif_only=gif_only,
    )
    if not gif_only:
        (out_subdir("04") / "README.md").write_text(
            "\n".join(
                [
                    f"# Mid-trip destination change\n" if LANG == "en" else "# 中途改目标可视化\n",
                    "## once/" if LANG == "en" else "## 单次修改/",
                    f"- Original goal: {names[22]}; at {names[9]} change to {names[18]}",
                    "## twice/" if LANG == "en" else "## 两次修改/",
                    f"- First: at {names[4]} → {names[14]}",
                    f"- Second: at {names[14]} → {names[19]}",
                ]
            ),
            encoding="utf-8",
        )
    return meta1, meta2


# backward-compatible alias
def build_reroute_animation_legacy(*args, **kwargs):
    return build_reroute_animation(*args, **kwargs)


if __name__ == "__main__":
    build_reroute_animation(n_frames=320)
