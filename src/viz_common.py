# Author: ZengWenquan
# https://github.com/chaosbull
# License: Apache-2.0
# -*- coding: utf-8 -*-
"""Shared map drawing helpers for the garden walk visualizations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent

# Mutable park context (set via configure_park)
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUT = ROOT / "output"
PARK_ID = 1
PARK_NAME = "Yiying Garden"
LANG = "en"
STR: dict = {}

COLORS = {
    "bg": "#EEF2EA",
    "panel": "#F7FAF5",
    "ink": "#243028",
    "muted": "#6B756E",
    "path_fill": "#C8D0BC",
    "path_edge": "#2F3D34",
    "path_visited_fill": "#8FB58A",
    "path_visited_edge": "#2F6B3A",
    "path_rec_fill": "#E8C56A",
    "path_rec_edge": "#A06A10",
    "spot": "#3A5F7A",
    "spot_edge": "#1E3344",
    "spot_visited": "#3F8F5A",
    "spot_current": "#C45C26",
    "spot_rec1": "#D4A017",
    "spot_rec2": "#E07A3A",
    "spot_goal": "#7A3E9D",
    "tourist": "#C0392B",
    "water": "#A9C9D9",
    "forest": "#B7C9A8",
    "score_hi": "#1F6B3A",
    "score_lo": "#7A5A20",
    "text": "#1A2420",
}

# Spot name fontsize
SPOT_NAME_FONTSIZE = 10.0
TITLE_FONTSIZE = 18.0
BODY_FONTSIZE = 8.0
DEFAULT_HALF_WIDTH = 15.0
FRAME_PAD = 320.0
PATH_INSET = 40.0


def configure_park(park_id: int = 1, lang: str = "en"):
    """Point DATA / RESULTS / OUT at this repo's single park folders."""
    global DATA, RESULTS, OUT, PARK_ID, PARK_NAME, LANG, STR
    import sys

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from generate_scenic_data import PARK
    from viz_i18n import PARK_NAME_EN, strings

    PARK_ID = 1
    LANG = "en" if str(lang).lower().startswith("en") else "zh"
    STR = strings(LANG)
    if LANG == "en":
        PARK_NAME = PARK_NAME_EN.get(1, PARK.get("name_en", "Yiying Garden"))
    else:
        PARK_NAME = str(PARK["name"])
    DATA = ROOT / "data"
    RESULTS = ROOT / "results"
    OUT = ROOT / "output"
    return PARK


def out_subdir(key: str) -> Path:
    """Resolve a logical folder key (01, 02, 03_frames, 04_once, ...) under OUT."""
    from viz_i18n import DIRS_EN, DIRS_ZH

    mapping = DIRS_EN if LANG == "en" else DIRS_ZH
    return OUT / mapping[key]


def setup_font():
    # Prefer fonts that cover both English and CJK when needed
    if LANG == "en":
        plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei"]
    else:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Segoe UI"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = COLORS["bg"]
    plt.rcParams["savefig.facecolor"] = COLORS["bg"]
    plt.rcParams["axes.facecolor"] = COLORS["bg"]


def load_tables():
    spots = pd.read_csv(DATA / "spots.csv")
    roads = pd.read_csv(DATA / "roads.csv")
    q = pd.read_csv(RESULTS / "road_yibu_yijing_Q.csv")
    qcol = "移步异景综合评分Q" if "移步异景综合评分Q" in q.columns else "Q_yibu_yijing"
    rid = "道路编号" if "道路编号" in q.columns else "road_id"
    qmap = {int(r[rid]): float(r[qcol]) for _, r in q.iterrows()}
    roads = roads.copy()
    roads["Q"] = roads["road_id"].map(qmap)
    if LANG == "en":
        from viz_i18n import SPOT_NAMES_EN

        en = SPOT_NAMES_EN.get(PARK_ID, {})
        spots = spots.copy()
        spots["name"] = spots["spot_id"].map(lambda s: en.get(int(s), str(s)))
    return spots, roads, q


def load_model():
    import sys

    sys.path.insert(0, str(SRC))
    from model_core import ModelParams, ScenicModel

    xlsx = DATA / "scenic_generated.xlsx"
    spots = pd.read_excel(xlsx, sheet_name="景点信息")
    roads = pd.read_excel(xlsx, sheet_name="道路信息")
    landscape = pd.read_excel(xlsx, sheet_name="景观元素")
    aij = pd.read_excel(xlsx, sheet_name="方向到达系数")
    br = pd.read_excel(xlsx, sheet_name="道路选择比例")
    mains = pd.read_excel(xlsx, sheet_name="主线路")
    monitor = pd.read_excel(xlsx, sheet_name="监测人数")
    params = ModelParams()
    pjson = RESULTS / "optimal_parameters.json"
    if pjson.exists():
        d = json.loads(pjson.read_text(encoding="utf-8"))["params"]
        lw = {k[3:]: v for k, v in d.items() if k.startswith("wq_")}
        kwargs = {k: v for k, v in d.items() if not k.startswith("wq_")}
        params = ModelParams(**kwargs, landscape_weights=lw)
    model = ScenicModel(spots, roads, landscape, aij, br, mains, monitor, params)
    model.estimate_N()
    model.invert_IO()
    model.compute_road_Q()
    return model


# ---------- geometry: unique winding garden paths (曲径通幽) ---------- #

def _hash_unit(*vals: int) -> float:
    h = 2166136261
    for v in vals:
        h ^= int(v) + 0x9E3779B9 + (h << 6) + (h >> 2)
        h &= 0xFFFFFFFF
    return (h % 10007) / 10007.0


def _soft_polyline(ctrl: np.ndarray, n: int, rounds: int = 1) -> np.ndarray:
    pts = ctrl.astype(float).copy()
    for _ in range(max(0, rounds)):
        if len(pts) < 3:
            break
        new = [pts[0]]
        for i in range(len(pts) - 1):
            p, q = pts[i], pts[i + 1]
            new.append(0.75 * p + 0.25 * q)
            new.append(0.25 * p + 0.75 * q)
        new.append(pts[-1])
        pts = np.asarray(new, dtype=float)
    dense = [pts[0]]
    for i in range(len(pts) - 1):
        steps = 12
        for k in range(1, steps + 1):
            t = k / steps
            dense.append((1 - t) * pts[i] + t * pts[i + 1])
    arr = np.asarray(dense, dtype=float)
    seg = np.sqrt(((np.diff(arr, axis=0)) ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1] + 1e-9
    targets = np.linspace(0, total, n)
    res = np.column_stack(
        [np.interp(targets, cum, arr[:, 0]), np.interp(targets, cum, arr[:, 1])]
    )
    res[0] = ctrl[0]
    res[-1] = ctrl[-1]
    return res


def bent_centerline(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    road_r: int = 1,
    n: int = 80,
    lane_sign: float = 1.0,
    edge_a: int = 0,
    edge_b: int = 0,
    amp_scale: float = 0.72,
) -> np.ndarray:
    """Unique 曲径通幽 centerline. amp_scale < 1 keeps bends inside the guide frame."""
    dx, dy = x1 - x0, y1 - y0
    length = float(np.hypot(dx, dy)) + 1e-9
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    nvec = np.array([nx, ny], dtype=float)
    uvec = np.array([ux, uy], dtype=float)
    p0 = np.array([x0, y0], dtype=float)
    p1 = np.array([x1, y1], dtype=float)

    u0 = _hash_unit(edge_a, edge_b, road_r, 11)
    u1 = _hash_unit(edge_a, edge_b, road_r, 29)
    u2 = _hash_unit(edge_a, edge_b, road_r, 47)
    u3 = _hash_unit(edge_a, edge_b, road_r, 67)
    u4 = _hash_unit(edge_a, edge_b, road_r, 89)
    u5 = _hash_unit(edge_a, edge_b, road_r, 109)

    if u0 < 0.82:
        n_bends = 2
    elif u0 < 0.92:
        n_bends = 3
    else:
        n_bends = 1

    style = int(u1 * 8) % 8
    base_side = float(lane_sign) * (1.0 if road_r == 1 else -1.15)
    lane0 = length * (0.04 + 0.05 * (road_r - 1) + 0.02 * u2) * base_side * amp_scale

    def at(frac: float, lat: float, along: float = 0.0) -> np.ndarray:
        return p0 + uvec * (length * frac + along) + nvec * (lat + lane0)

    ctrl: list = [p0]

    if style in (0, 1) and n_bends >= 2:
        mx = x0 + (x1 - x0) * (0.30 + 0.35 * u2)
        my = y0 + (y1 - y0) * (0.30 + 0.35 * u3)
        jog = length * (0.10 + 0.10 * u4) * base_side * amp_scale
        if style == 0:
            ctrl += [
                np.array([mx, y0]) + nvec * (lane0 * 0.4),
                np.array([mx, my]) + nvec * jog,
                np.array([x1, my]) + nvec * (lane0 * 0.35 + jog * 0.15),
            ]
        else:
            ctrl += [
                np.array([x0, my]) + nvec * (lane0 * 0.35),
                np.array([mx, my]) + nvec * (-jog),
                np.array([mx, y1]) + nvec * (lane0 * 0.3 - jog * 0.1),
            ]
    elif n_bends == 1:
        amp = length * (0.14 + 0.10 * u2) * base_side * (1.0 if style % 2 == 0 else -1.0) * amp_scale
        ctrl.append(at(0.25 + 0.45 * u3, amp))
    elif n_bends >= 3 or style == 7:
        fracs = [0.17 + 0.05 * u1, 0.46 + 0.06 * u2, 0.75 + 0.05 * u3]
        amps = [
            length * (0.12 + 0.08 * u2) * amp_scale,
            length * (0.16 + 0.09 * u3) * amp_scale,
            length * (0.12 + 0.08 * u4) * amp_scale,
        ]
        signs = [base_side, -base_side, base_side if style % 2 == 0 else -base_side]
        if style == 7:
            signs = [-base_side, base_side, -base_side]
        for f, a, s in zip(fracs, amps, signs):
            ctrl.append(at(f, a * s))
    else:
        a1 = length * (0.12 + 0.10 * u2) * amp_scale
        a2 = length * (0.11 + 0.11 * u3) * amp_scale
        t1 = 0.22 + 0.14 * u3
        t2 = 0.60 + 0.18 * u4
        if t2 - t1 < 0.26:
            t2 = min(0.84, t1 + 0.28)
        if style == 2:
            s1, s2 = base_side, -base_side
            a1 *= 1.2
            a2 *= 1.15
        elif style == 3:
            s1, s2 = -base_side, base_side
            t1, t2 = 0.24 + 0.1 * u2, 0.70 + 0.12 * u4
        elif style == 4:
            s1, s2 = base_side, base_side * 0.65
            a1 *= 1.25
        elif style == 5:
            s1, s2 = base_side * 1.15, -base_side
            t1 = 0.16 + 0.08 * u2
        else:
            s1, s2 = -base_side * 0.9, base_side * 1.15
            t2 = 0.72 + 0.12 * u3
        ctrl.append(at(t1, a1 * s1))
        ctrl.append(at((t1 + t2) * 0.5, 0.03 * length * (u5 - 0.5) * base_side * amp_scale))
        ctrl.append(at(t2, a2 * s2))

    ctrl.append(p1)
    cleaned = [ctrl[0]]
    for q in ctrl[1:]:
        if np.hypot(*(np.asarray(q) - cleaned[-1])) > 10.0:
            cleaned.append(np.asarray(q, dtype=float))
    cleaned[-1] = p1
    return _soft_polyline(np.asarray(cleaned, dtype=float), n=n, rounds=2)


def _clamp_polyline(cl: np.ndarray, bounds: Tuple[float, float, float, float], keep_ends: bool = True) -> np.ndarray:
    xmin, ymin, xmax, ymax = bounds
    out = cl.copy()
    out[:, 0] = np.clip(out[:, 0], xmin, xmax)
    out[:, 1] = np.clip(out[:, 1], ymin, ymax)
    if keep_ends and len(out) >= 2:
        out[0] = cl[0]
        out[-1] = cl[-1]
        # If endpoints themselves are outside (shouldn't), still clip them
        out[0, 0] = np.clip(out[0, 0], xmin, xmax)
        out[0, 1] = np.clip(out[0, 1], ymin, ymax)
        out[-1, 0] = np.clip(out[-1, 0], xmin, xmax)
        out[-1, 1] = np.clip(out[-1, 1], ymin, ymax)
    return out


def parallel_band(center: np.ndarray, half_width: float) -> Tuple[np.ndarray, np.ndarray]:
    d = np.diff(center, axis=0)
    d = np.vstack([d[:1], d, d[-1:]])
    tang = d[:-1] + d[1:]
    norms = np.hypot(tang[:, 0], tang[:, 1]) + 1e-9
    tang = tang / norms[:, None]
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])
    left = center + nrm * half_width
    right = center - nrm * half_width
    return left, right


def road_key(a: int, b: int, r: int) -> Tuple[int, int, int]:
    u, v = (a, b) if a < b else (b, a)
    return u, v, r


class GardenMap:
    """Reusable canvas for a synthetic classical garden."""

    def __init__(self, spots: pd.DataFrame, roads: pd.DataFrame):
        self.spots = spots
        self.roads = roads
        self.xy = {int(r.spot_id): (float(r.x), float(r.y)) for r in spots.itertuples()}
        self.names = {int(r.spot_id): str(r.name) for r in spots.itertuples()}
        self.env = {int(r.spot_id): str(r.env_type) for r in spots.itertuples()}
        self.centerlines: Dict[Tuple[int, int, int], np.ndarray] = {}
        self.centerline_from: Dict[Tuple[int, int, int], Tuple[int, int]] = {}
        self.q_of: Dict[Tuple[int, int, int], float] = {}
        self._label_xy: Dict[Tuple[int, int, int], np.ndarray] = {}

        xs = [p[0] for p in self.xy.values()]
        ys = [p[1] for p in self.xy.values()]
        self.frame_bounds = (
            min(xs) - FRAME_PAD,
            min(ys) - FRAME_PAD,
            max(xs) + FRAME_PAD,
            max(ys) + FRAME_PAD,
        )
        # Path samples stay inside frame with inset so double-line width also fits
        inset = PATH_INSET + DEFAULT_HALF_WIDTH + 8
        path_bounds = (
            self.frame_bounds[0] + inset,
            self.frame_bounds[1] + inset,
            self.frame_bounds[2] - inset,
            self.frame_bounds[3] - inset,
        )

        two_bend = 0
        total = 0
        for r in roads.itertuples():
            a, b, rr = int(r.from_id), int(r.to_id), int(r.road_index_r)
            x0, y0 = self.xy[a]
            x1, y1 = self.xy[b]
            u, v = (a, b) if a < b else (b, a)
            lane_sign = 1.0 if _hash_unit(u, v, 0) >= 0.5 else -1.0
            key = road_key(a, b, rr)
            cl = bent_centerline(
                x0, y0, x1, y1, road_r=rr, n=72, lane_sign=lane_sign, edge_a=a, edge_b=b, amp_scale=0.68
            )
            cl = _clamp_polyline(cl, path_bounds, keep_ends=True)
            self.centerlines[key] = cl
            # Remember build direction (from_id → to_id) for correct walk interpolation
            self.centerline_from[key] = (a, b)
            self.q_of[key] = float(getattr(r, "Q", 1.0))
            total += 1
            if _hash_unit(a, b, rr, 11) < 0.92:
                two_bend += 1

        self._place_score_labels()
        self._bend_stats = {
            "total": total,
            "two_or_more_bend_edges": two_bend,
            "ratio": two_bend / max(total, 1),
        }

    def _sample_on_path(self, cl: np.ndarray, frac: float) -> Tuple[np.ndarray, np.ndarray]:
        """Point and unit normal at arc-length fraction frac ∈ (0,1)."""
        seg = np.sqrt(((np.diff(cl, axis=0)) ** 2).sum(axis=1))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1] + 1e-9
        target = float(np.clip(frac, 0.08, 0.92)) * total
        i = int(np.searchsorted(cum, target) - 1)
        i = int(np.clip(i, 0, len(cl) - 2))
        local = (target - cum[i]) / (seg[i] + 1e-9)
        pt = (1 - local) * cl[i] + local * cl[i + 1]
        d = cl[min(len(cl) - 1, i + 1)] - cl[max(0, i - 1)]
        nrm = np.array([-d[1], d[0]], dtype=float)
        nrm = nrm / (np.hypot(*nrm) + 1e-9)
        return pt, nrm

    def _place_score_labels(self, min_sep: float = 280.0):
        """Place Q labels on the road midline; resolve overlaps by sliding along the path."""
        keys = list(self.centerlines.keys())
        fracs = {}
        for key in keys:
            # Stagger multi-road pairs and hash-offset others to reduce initial collisions
            base = 0.42 + 0.20 * _hash_unit(*key, 5)
            if key[2] == 2:
                base = 0.28 + 0.18 * _hash_unit(*key, 7)
            fracs[key] = float(np.clip(base, 0.18, 0.82))

        positions: Dict[Tuple[int, int, int], np.ndarray] = {}

        def _pos_for(key, frac):
            pt, nrm = self._sample_on_path(self.centerlines[key], frac)
            nudge = (0.0 if key[2] == 1 else 12.0) * (1.0 if _hash_unit(*key, 3) > 0.5 else -1.0)
            return pt + nrm * nudge

        for key in keys:
            positions[key] = _pos_for(key, fracs[key])

        for round_i in range(40):
            moved = False
            ordered = sorted(keys, key=lambda k: (float(positions[k][0]), float(positions[k][1])))
            for i, ka in enumerate(ordered):
                for kb in ordered[i + 1 :]:
                    delta = positions[ka] - positions[kb]
                    d = float(np.hypot(*delta))
                    if d >= min_sep:
                        continue
                    # Prefer moving the longer / secondary road
                    la = float(np.sum(np.sqrt(((np.diff(self.centerlines[ka], axis=0)) ** 2).sum(axis=1))))
                    lb = float(np.sum(np.sqrt(((np.diff(self.centerlines[kb], axis=0)) ** 2).sum(axis=1))))
                    victim = kb if (kb[2] > ka[2] or (kb[2] == ka[2] and lb >= la)) else ka
                    other = ka if victim is kb else kb
                    # Slide along path away from the other label's projection
                    step = 0.055 if d < min_sep * 0.5 else 0.035
                    # Choose direction that increases separation in xy
                    cand = []
                    for sgn in (-1.0, 1.0):
                        f2 = float(np.clip(fracs[victim] + sgn * step, 0.12, 0.88))
                        p2 = _pos_for(victim, f2)
                        cand.append((float(np.hypot(*(p2 - positions[other]))), f2, p2))
                    cand.sort(key=lambda x: -x[0])
                    best_d, best_f, best_p = cand[0]
                    if best_d > d + 1.0 or abs(best_f - fracs[victim]) > 1e-6:
                        fracs[victim] = best_f
                        positions[victim] = best_p
                        moved = True
            if not moved:
                break

        # Lateral micro-nudge + extreme frac as last resort for stubborn pairs
        for _ in range(20):
            moved = False
            ordered = sorted(keys, key=lambda k: (float(positions[k][0]), float(positions[k][1])))
            for i, ka in enumerate(ordered):
                for kb in ordered[i + 1 :]:
                    d = float(np.hypot(*(positions[ka] - positions[kb])))
                    if d >= min_sep * 0.95:
                        continue
                    victim = kb if kb[2] >= ka[2] else ka
                    other = ka if victim is kb else kb
                    best = (d, fracs[victim], positions[victim])
                    for ftry in (0.14, 0.22, 0.78, 0.86, fracs[victim]):
                        for lat in (0.0, 22.0, -22.0, 40.0, -40.0, 60.0, -60.0):
                            pt, nrm = self._sample_on_path(self.centerlines[victim], ftry)
                            p2 = pt + nrm * lat
                            dd = float(np.hypot(*(p2 - positions[other])))
                            if dd > best[0]:
                                best = (dd, ftry, p2)
                    if best[0] > d + 0.5:
                        fracs[victim] = best[1]
                        positions[victim] = best[2]
                        moved = True
            if not moved:
                break

        xmin, ymin, xmax, ymax = self.frame_bounds
        pad = 55.0
        for key, pos in positions.items():
            positions[key] = np.array(
                [
                    float(np.clip(pos[0], xmin + pad, xmax - pad)),
                    float(np.clip(pos[1], ymin + pad, ymax - pad)),
                ]
            )
        self._label_xy = positions
        self._label_frac = fracs

    def draw_atmosphere(self, ax):
        xmin, ymin, xmax, ymax = self.frame_bounds
        ax.set_xlim(xmin - 40, xmax + 40)
        ax.set_ylim(ymin - 40, ymax + 40)
        ax.set_aspect("equal")
        ax.axis("off")

        rng = np.random.default_rng(3 + PARK_ID)
        for sid, (x, y) in self.xy.items():
            if self.env[sid] == "W":
                circ = Circle((x + 40, y - 30), 220, facecolor=COLORS["water"], edgecolor="none", alpha=0.35, zorder=0)
                ax.add_patch(circ)
            elif self.env[sid] == "F":
                for _ in range(3):
                    jx, jy = rng.normal(0, 90, 2)
                    circ = Circle(
                        (x + jx, y + jy),
                        rng.uniform(60, 140),
                        facecolor=COLORS["forest"],
                        edgecolor="none",
                        alpha=0.28,
                        zorder=0,
                    )
                    ax.add_patch(circ)

        ax.add_patch(
            FancyBboxPatch(
                (xmin, ymin),
                (xmax - xmin),
                (ymax - ymin),
                boxstyle="round,pad=20,rounding_size=40",
                facecolor="none",
                edgecolor=COLORS["ink"],
                linewidth=1.6,
                alpha=0.25,
                zorder=0,
            )
        )

    def draw_roads(
        self,
        ax,
        visited_edges: Optional[Set[Tuple[int, int, int]]] = None,
        highlight_edges: Optional[Dict[Tuple[int, int, int], str]] = None,
        show_scores: bool = True,
        score_fontsize: float = 8.0,
        half_width: float = DEFAULT_HALF_WIDTH,
        dim_others: bool = False,
        focus_edges: Optional[Set[Tuple[int, int, int]]] = None,
    ):
        visited_edges = visited_edges or set()
        highlight_edges = highlight_edges or {}
        focus_edges = focus_edges

        for key, cl in self.centerlines.items():
            if focus_edges is not None and key not in focus_edges and key not in visited_edges and key not in highlight_edges:
                if dim_others:
                    fill, edge, lw, alpha = "#D5DACF", "#9AA396", 0.9, 0.30
                else:
                    fill, edge, lw, alpha = COLORS["path_fill"], COLORS["path_edge"], 1.2, 0.95
            elif key in highlight_edges:
                mode = highlight_edges[key]
                if mode == "rec":
                    fill, edge, lw, alpha = COLORS["path_rec_fill"], COLORS["path_rec_edge"], 1.6, 1.0
                elif mode == "active":
                    fill, edge, lw, alpha = "#F0D48A", "#8A5A12", 1.8, 1.0
                else:
                    fill, edge, lw, alpha = COLORS["path_visited_fill"], COLORS["path_visited_edge"], 1.5, 1.0
            elif key in visited_edges:
                fill, edge, lw, alpha = COLORS["path_visited_fill"], COLORS["path_visited_edge"], 1.5, 1.0
            else:
                fill, edge, lw, alpha = COLORS["path_fill"], COLORS["path_edge"], 1.2, 0.95

            left, right = parallel_band(cl, half_width)
            poly = np.vstack([left, right[::-1]])
            ax.fill(poly[:, 0], poly[:, 1], facecolor=fill, edgecolor="none", alpha=alpha, zorder=2)
            ax.plot(left[:, 0], left[:, 1], color=edge, lw=lw, solid_capstyle="round", alpha=alpha, zorder=3)
            ax.plot(right[:, 0], right[:, 1], color=edge, lw=lw, solid_capstyle="round", alpha=alpha, zorder=3)

            if show_scores and not (
                dim_others
                and focus_edges is not None
                and key not in focus_edges
                and key not in visited_edges
                and key not in highlight_edges
            ):
                q = self.q_of.get(key, 1.0)
                pos = self._label_xy.get(key)
                if pos is None:
                    pos, _ = self._sample_on_path(cl, 0.5)
                t = np.clip((q - 0.3) / 2.0, 0, 1)
                tc = (1 - t) * np.array([0.48, 0.35, 0.12]) + t * np.array([0.12, 0.42, 0.23])
                ax.text(
                    float(pos[0]),
                    float(pos[1]),
                    f"Q {q:.2f}",
                    fontsize=score_fontsize,
                    fontweight="bold",
                    color=tc,
                    ha="center",
                    va="center",
                    zorder=6,
                    bbox=dict(
                        boxstyle="round,pad=0.16",
                        facecolor="#F4F7F1",
                        edgecolor=edge,
                        linewidth=0.75,
                        alpha=0.94,
                    ),
                )

    def draw_spots(
        self,
        ax,
        visited: Optional[Set[int]] = None,
        current: Optional[int] = None,
        rec_step1: Optional[Set[int]] = None,
        rec_step2: Optional[Set[int]] = None,
        goal: Optional[int] = None,
        show_names: bool = True,
        radius: float = 78.0,
        name_fontsize: float = SPOT_NAME_FONTSIZE,
    ):
        visited = visited or set()
        rec_step1 = rec_step1 or set()
        rec_step2 = rec_step2 or set()
        for sid, (x, y) in self.xy.items():
            if sid == current:
                fc, ec, rw = COLORS["spot_current"], "#5A2208", 2.4
            elif goal is not None and sid == goal:
                fc, ec, rw = COLORS["spot_goal"], "#3A1A50", 2.2
            elif sid in visited:
                fc, ec, rw = COLORS["spot_visited"], "#1E4A2A", 2.0
            elif sid in rec_step1:
                fc, ec, rw = COLORS["spot_rec1"], "#6A4A08", 2.2
            elif sid in rec_step2:
                fc, ec, rw = COLORS["spot_rec2"], "#6A3A10", 2.0
            else:
                fc, ec, rw = COLORS["spot"], COLORS["spot_edge"], 1.6
            ax.add_patch(Circle((x, y), radius, facecolor=fc, edgecolor=ec, lw=rw, zorder=8))
            ax.text(
                x,
                y,
                str(sid),
                color="white",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=9,
            )
            if show_names:
                ax.text(
                    x,
                    y - radius - 55,
                    self.names[sid],
                    color=COLORS["text"],
                    fontsize=name_fontsize,
                    ha="center",
                    va="top",
                    zorder=9,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#F4F7F1", edgecolor="none", alpha=0.85),
                )

    def draw_tourist(self, ax, xy: Tuple[float, float], size: float = 55):
        ax.scatter(
            [xy[0]],
            [xy[1]],
            s=size * 4,
            c=COLORS["tourist"],
            edgecolors="white",
            linewidths=1.8,
            zorder=20,
            marker="o",
        )
        ax.scatter([xy[0]], [xy[1]], s=size, c="white", zorder=21, marker="o")

    def point_on_road(self, a: int, b: int, r: int, frac: float) -> np.ndarray:
        """Interpolate along the road from spot a toward spot b (fixes reverse/teleport)."""
        key = road_key(a, b, r)
        cl = self.centerlines[key]
        built = self.centerline_from.get(key)
        f = float(np.clip(frac, 0.0, 1.0))
        if built is not None:
            fa, tb = built
            if (a, b) == (tb, fa):
                f = 1.0 - f
            elif (a, b) != (fa, tb):
                # Fallback: orient by endpoint proximity
                if np.hypot(*(cl[0] - np.array(self.xy[a]))) > np.hypot(*(cl[-1] - np.array(self.xy[a]))):
                    f = 1.0 - f
        else:
            if np.hypot(*(cl[0] - np.array(self.xy[a]))) > np.hypot(*(cl[-1] - np.array(self.xy[a]))):
                f = 1.0 - f
        seg = np.sqrt(((np.diff(cl, axis=0)) ** 2).sum(axis=1))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1] + 1e-9
        target = f * total
        i = int(np.searchsorted(cum, target) - 1)
        i = int(np.clip(i, 0, len(cl) - 2))
        local = (target - cum[i]) / (seg[i] + 1e-9)
        return (1 - local) * cl[i] + local * cl[i + 1]


def add_legend(ax, kind: str = "network"):
    s = STR or {}
    handles = []
    if kind == "network":
        handles = [
            mpatches.Patch(
                facecolor=COLORS["path_fill"],
                edgecolor=COLORS["path_edge"],
                label=s.get("legend_road", "Garden path"),
            ),
            mpatches.Patch(
                facecolor=COLORS["spot"],
                edgecolor=COLORS["spot_edge"],
                label=s.get("legend_spot", "Scenic spot"),
            ),
            mpatches.Patch(
                facecolor="#F4F7F1",
                edgecolor=COLORS["path_edge"],
                label=s.get("legend_q", "Path score Q"),
            ),
        ]
    elif kind == "walk":
        handles = [
            mpatches.Patch(facecolor=COLORS["spot_visited"], label=s.get("legend_visited", "Visited")),
            mpatches.Patch(facecolor=COLORS["spot_current"], label=s.get("legend_current", "Current")),
            mpatches.Patch(facecolor=COLORS["spot_rec1"], label=s.get("legend_rec1", "1-step")),
            mpatches.Patch(facecolor=COLORS["spot_rec2"], label=s.get("legend_rec2", "2-step")),
            mpatches.Patch(
                facecolor=COLORS["path_visited_fill"],
                edgecolor=COLORS["path_visited_edge"],
                label=s.get("legend_path_visited", "Walked"),
            ),
            mpatches.Patch(facecolor=COLORS["tourist"], label=s.get("legend_tourist", "Visitor")),
        ]
    ax.legend(handles=handles, loc="upper right", frameon=True, fancybox=True, fontsize=8, framealpha=0.92)


def info_panel(ax, lines: Sequence[str], xy_axes=(0.02, 0.02), fontsize: float = 9.0):
    import textwrap

    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(str(line), width=52) or [""])
    wrapped = wrapped[:14]
    text = "\n".join(wrapped)
    ax.text(
        xy_axes[0],
        xy_axes[1],
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        va="bottom",
        ha="left",
        color=COLORS["text"],
        linespacing=1.35,
        clip_on=True,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor=COLORS["panel"],
            edgecolor=COLORS["ink"],
            linewidth=1.0,
            alpha=0.94,
        ),
        zorder=30,
    )


def bottom_text_panel(
    ax_text,
    lines: Sequence[str],
    title: Optional[str] = None,
    fontsize: float = BODY_FONTSIZE,
    title_fontsize: Optional[float] = None,
    wrap_width: int = 88,
):
    """Caption panel: text is wrapped and clipped strictly inside the white box."""
    import textwrap

    ax_text.clear()
    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(0, 1)
    ax_text.axis("off")
    ax_text.set_facecolor(COLORS["bg"])

    box_x, box_y, box_w, box_h = 0.02, 0.05, 0.96, 0.90
    ax_text.add_patch(
        FancyBboxPatch(
            (box_x, box_y),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=COLORS["panel"],
            edgecolor=COLORS["ink"],
            linewidth=1.1,
            transform=ax_text.transAxes,
            clip_on=False,
            zorder=0,
        )
    )

    t_fs = title_fontsize if title_fontsize is not None else max(fontsize + 3.5, 12.0)
    pad_x = box_x + 0.03
    y = box_y + box_h - 0.06
    y_min = box_y + 0.05
    line_step = max(0.055, fontsize * 0.0072)
    title_step = max(0.09, t_fs * 0.0075)

    if title:
        ax_text.text(
            pad_x,
            y,
            title,
            fontsize=t_fs,
            fontweight="bold",
            color=COLORS["ink"],
            va="top",
            ha="left",
            transform=ax_text.transAxes,
            clip_on=True,
            zorder=2,
        )
        y -= title_step

    wrapped: List[str] = []
    for line in lines:
        parts = textwrap.wrap(str(line), width=wrap_width, replace_whitespace=False, drop_whitespace=False)
        wrapped.extend(parts if parts else [""])

    for wline in wrapped:
        if y < y_min + line_step * 0.5:
            ax_text.text(
                pad_x,
                y_min,
                "…",
                fontsize=fontsize,
                color=COLORS["muted"],
                va="bottom",
                ha="left",
                transform=ax_text.transAxes,
                clip_on=True,
                zorder=2,
            )
            break
        ax_text.text(
            pad_x,
            y,
            wline,
            fontsize=fontsize,
            color=COLORS["text"],
            va="top",
            ha="left",
            transform=ax_text.transAxes,
            clip_on=True,
            zorder=2,
            linespacing=1.25,
        )
        y -= line_step


def ensure_dirs(out: Optional[Path] = None):
    root = Path(out) if out is not None else OUT
    from viz_i18n import DIRS_EN, DIRS_ZH

    mapping = DIRS_EN if LANG == "en" else DIRS_ZH
    dirs = []
    for key in (
        "01",
        "02",
        "03",
        "03_frames",
        "04",
        "04_once",
        "04_once_frames",
        "04_twice",
        "04_twice_frames",
        "05",
    ):
        d = root / mapping[key]
        d.mkdir(parents=True, exist_ok=True)
        dirs.append(d)
    return dirs


# Initialize English strings by default
try:
    from viz_i18n import strings as _strings_init

    STR = _strings_init("en")
except Exception:
    STR = {}
