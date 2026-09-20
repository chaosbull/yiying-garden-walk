# Author: ZengWenquan
# https://github.com/chaosbull
# License: Apache-2.0
# -*- coding: utf-8 -*-
"""Generate Yiying Garden network tables into data/."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Yiying Garden (怡影园)
# --------------------------------------------------------------------------- #
SPOTS = [
    (1, "正门石坊", 1000, 4600, "B", 160, 55, 48, 3.5, 1.0, 1),
    (2, "玉兰堂", 1700, 4400, "B", 210, 70, 50, 6.0, 1.6, 0),
    (3, "藕花水榭", 2500, 4550, "W", 180, 60, 45, 7.5, 2.0, 0),
    (4, "月到风来亭", 3300, 4300, "W", 130, 45, 38, 5.5, 1.4, 0),
    (5, "金粟亭", 4100, 4100, "B", 150, 50, 42, 6.5, 1.7, 0),
    (6, "桐荫斋", 1400, 3700, "F", 260, 80, 50, 5.0, 1.3, 0),
    (7, "面壁亭", 2200, 3800, "B", 170, 55, 45, 7.0, 1.9, 0),
    (8, "锁绿轩", 3000, 3600, "B", 200, 65, 48, 8.0, 2.1, 0),
    (9, "湛露堂", 3800, 3400, "B", 240, 75, 50, 9.0, 2.4, 0),
    (10, "南雪亭", 1200, 3000, "F", 220, 70, 48, 5.5, 1.5, 0),
    (11, "碧梧栖凤", 2000, 2900, "F", 280, 85, 50, 6.0, 1.6, 0),
    (12, "石听琴室", 2800, 2800, "B", 190, 60, 46, 7.5, 2.0, 0),
    (13, "坡仙琴馆", 3600, 2700, "B", 210, 70, 50, 8.5, 2.2, 0),
    (14, "拜石轩", 4400, 2900, "B", 230, 75, 50, 9.5, 2.5, 0),
    (15, "锄月轩", 1600, 2200, "B", 180, 60, 45, 6.0, 1.5, 0),
    (16, "金栗斋", 2400, 2100, "B", 200, 65, 48, 7.0, 1.8, 0),
    (17, "小沧浪馆", 3200, 2000, "W", 170, 55, 42, 8.0, 2.0, 0),
    (18, "螺髻亭", 4000, 1800, "F", 150, 50, 40, 6.5, 1.7, 0),
    (19, "岁寒草庐", 2600, 1400, "B", 220, 70, 50, 8.0, 2.1, 0),
    (20, "藕香深处", 3400, 1200, "W", 190, 60, 46, 9.0, 2.3, 0),
    (21, "碧梧台", 4200, 1300, "F", 210, 70, 48, 7.0, 1.8, 0),
    (22, "园门茶寮", 4800, 1000, "B", 140, 50, 40, 3.5, 0.9, 0),
]

BASE_EDGES = [
    (1, 2), (1, 6),
    (2, 3), (2, 6), (2, 7),
    (3, 4), (3, 7), (3, 8),
    (4, 5), (4, 8), (4, 9),
    (5, 9), (5, 14),
    (6, 7), (6, 10), (6, 11),
    (7, 8), (7, 11), (7, 12),
    (8, 9), (8, 12), (8, 13),
    (9, 13), (9, 14),
    (10, 11), (10, 15),
    (11, 12), (11, 15), (11, 16),
    (12, 13), (12, 16), (12, 17),
    (13, 14), (13, 17), (13, 18),
    (14, 18), (14, 21),
    (15, 16), (15, 19),
    (16, 17), (16, 19),
    (17, 18), (17, 20),
    (18, 20), (18, 21),
    (19, 20), (19, 22),
    (20, 21), (20, 22),
    (21, 22),
]
MULTI_ROAD = {(2, 7), (7, 12), (12, 17), (16, 19), (8, 13)}
MAIN_ROUTES = {
    1: [1, 2, 3, 4, 5, 14, 18, 21, 22],
    2: [1, 6, 10, 15, 19, 20, 22],
    3: [1, 2, 7, 12, 17, 20, 22],
}

PARK = {
    "key": "yiying",
    "name": "怡影园",
    "name_en": "Yiying Garden",
    "data_dir": ROOT / "data",
    "results_dir": ROOT / "results",
    "viz_out": ROOT / "output",
    "spots": SPOTS,
    "edges": BASE_EDGES,
    "multi": MULTI_ROAD,
    "mains": MAIN_ROUTES,
    "seed": 137,
}
# keep PARKS = {1: PARK} for minimal breakage if anything still indexes park_id 1
PARKS = {1: PARK}


def _euclid(a, b, coords):
    xa, ya = coords[a]
    xb, yb = coords[b]
    return float(np.hypot(xa - xb, ya - yb))


def build_spots_df(spots: Sequence[tuple]) -> pd.DataFrame:
    rows = []
    for sid, name, x, y, env, area, ac, cap, mu, sigma, ent in spots:
        rows.append(
            {
                "spot_id": sid,
                "name": name,
                "x": x,
                "y": y,
                "env_type": env,
                "area_Ai": area,
                "monitor_area_Ac": ac,
                "capacity_Si": cap,
                "mu_stay": mu,
                "sigma_stay": sigma,
                "is_entrance": ent,
                "theta1": 0.55,
                "theta2": 0.90,
            }
        )
    return pd.DataFrame(rows)


def build_roads_df(
    spots: pd.DataFrame,
    edges: Sequence[Tuple[int, int]],
    multi: set,
    rng: np.random.Generator,
) -> pd.DataFrame:
    coords = {int(r.spot_id): (float(r.x), float(r.y)) for r in spots.itertuples()}
    rows = []
    road_id = 1
    for i, j in edges:
        pairs = [(i, j, 1)]
        if (i, j) in multi or (j, i) in multi:
            pairs.append((i, j, 2))
        for a, b, r in pairs:
            d = _euclid(a, b, coords)
            scale = 0.52 if r == 1 else 0.78
            noise = float(rng.uniform(0.85, 1.25))
            length = max(90.0, d * scale * noise)
            # Varied walking speeds → spread travel minutes (≈3–18)
            speed = float(rng.uniform(48.0, 140.0))
            if r == 2:
                speed *= 0.82  # alternate path slower / longer feel
            tau = int(np.clip(round(length / speed), 3, 18))
            energy = round(length / 105.0 + float(rng.uniform(0.3, 2.2)), 2)
            B = round(length * float(rng.uniform(0.07, 0.28)), 2)
            rows.append(
                {
                    "road_id": road_id,
                    "from_id": a,
                    "to_id": b,
                    "road_index_r": r,
                    "length_L": round(length, 2),
                    "travel_time_tau": tau,
                    "energy_C": energy,
                    "boundary_B": B,
                    "bidirectional": 1,
                }
            )
            road_id += 1
    return pd.DataFrame(rows)


def build_landscape_df(spots: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    types = [
        ("半开放建筑", 1.0),
        ("实体建筑", 1.2),
        ("道路", 0.4),
        ("假山", 1.1),
        ("水体", 1.3),
        ("植物", 0.9),
    ]
    xs = spots["x"].to_numpy()
    ys = spots["y"].to_numpy()
    xmin, xmax = xs.min() - 200, xs.max() + 200
    ymin, ymax = ys.min() - 200, ys.max() + 200
    rows = []
    qid = 1
    for tname, base_w in types:
        n = int(rng.integers(8, 14))
        for _ in range(n):
            x = float(rng.uniform(xmin, xmax))
            y = float(rng.uniform(ymin, ymax))
            area = float(rng.uniform(80, 450))
            sigma = float(rng.uniform(180, 520))
            rows.append(
                {
                    "element_id": qid,
                    "type_name": tname,
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "area_Aq": round(area, 2),
                    "sigma_q": round(sigma, 2),
                    "weight_wq": base_w,
                }
            )
            qid += 1
    return pd.DataFrame(rows)


def build_direction_coef(spots: pd.DataFrame, roads: pd.DataFrame) -> pd.DataFrame:
    undirected: Dict[int, set] = {}
    for r in roads.itertuples():
        a, b = int(r.from_id), int(r.to_id)
        undirected.setdefault(a, set()).add(b)
        undirected.setdefault(b, set()).add(a)

    rows = []
    for i, neigh in undirected.items():
        neigh = sorted(neigh)
        raw = np.array([1.2 if j > i else 0.8 for j in neigh], dtype=float)
        if i == 1:
            raw = np.array([1.5 if j in neigh[:2] else 1.0 for j in neigh], dtype=float)
        raw = raw / raw.sum()
        for j, aij in zip(neigh, raw):
            rows.append({"from_id": i, "to_id": j, "a_ij": round(float(aij), 4)})
    return pd.DataFrame(rows)


def build_road_choice(roads: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = roads.groupby(["from_id", "to_id"])
    for (a, b), g in grouped:
        rs = g["road_index_r"].tolist()
        if len(rs) == 1:
            rows.append({"from_id": a, "to_id": b, "road_index_r": rs[0], "b_r": 1.0})
            rows.append({"from_id": b, "to_id": a, "road_index_r": rs[0], "b_r": 1.0})
        else:
            w = np.array([0.62, 0.38][: len(rs)], dtype=float)
            w = w / w.sum()
            for r, br in zip(rs, w):
                rows.append({"from_id": a, "to_id": b, "road_index_r": int(r), "b_r": round(float(br), 4)})
                rows.append({"from_id": b, "to_id": a, "road_index_r": int(r), "b_r": round(float(br), 4)})
    return pd.DataFrame(rows)


def build_main_routes(mains: Dict[int, List[int]]) -> pd.DataFrame:
    rows = []
    for mid, seq in mains.items():
        for order, sid in enumerate(seq, start=1):
            rows.append({"main_route_id": mid, "order": order, "spot_id": sid})
    return pd.DataFrame(rows)


def build_monitor_series(spots: pd.DataFrame, rng: np.random.Generator, T: int = 90) -> pd.DataFrame:
    """Synthetic monitor counts; after model expansion N stays roughly in 0–50."""
    rows = []
    for r in spots.itertuples():
        # Smaller base → estimated N ≈ 0–50 after area expansion
        base = 2.5 + 0.018 * r.area_Ai + float(rng.uniform(0, 4))
        if r.is_entrance:
            base += 6
        # Per-spot personality so neighboring spots differ clearly
        personality = float(rng.uniform(0.55, 1.45))
        for t in range(1, T + 1):
            diurnal = 0.45 + 0.55 * np.sin(np.pi * (t - 8) / max(T - 8, 1))
            diurnal = max(0.2, diurnal)
            # Second soft peak mid-afternoon
            peak2 = 0.15 * np.exp(-((t - 55) / 18.0) ** 2)
            noise = float(rng.normal(0, 1.4))
            nc = max(0, int(round(base * personality * (diurnal + peak2) + noise)))
            rows.append({"spot_id": int(r.spot_id), "minute_t": t, "monitor_count_nc": nc})
    return pd.DataFrame(rows)


def generate_park(park_id: int = 1) -> Path:
    """Write Yiying Garden CSV/XLSX tables under data/.

    park_id other than 1 is ignored (this repo only ships Yiying Garden).
    """
    if int(park_id) != 1:
        print(f"[warn] park_id={park_id} ignored; using Yiying Garden (park_id=1)")
    cfg = PARK
    park_id = 1
    data_dir: Path = cfg["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(cfg["seed"]))

    spots = build_spots_df(cfg["spots"])
    roads = build_roads_df(spots, cfg["edges"], cfg["multi"], rng)
    landscape = build_landscape_df(spots, rng)
    aij = build_direction_coef(spots, roads)
    br = build_road_choice(roads)
    mains = build_main_routes(cfg["mains"])
    monitor = build_monitor_series(spots, rng, T=90)

    out = data_dir / "scenic_generated.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        spots.to_excel(writer, sheet_name="景点信息", index=False)
        roads.to_excel(writer, sheet_name="道路信息", index=False)
        landscape.to_excel(writer, sheet_name="景观元素", index=False)
        aij.to_excel(writer, sheet_name="方向到达系数", index=False)
        br.to_excel(writer, sheet_name="道路选择比例", index=False)
        mains.to_excel(writer, sheet_name="主线路", index=False)
        monitor.to_excel(writer, sheet_name="监测人数", index=False)

    spots.to_csv(data_dir / "spots.csv", index=False, encoding="utf-8-sig")
    roads.to_csv(data_dir / "roads.csv", index=False, encoding="utf-8-sig")
    landscape.to_csv(data_dir / "landscape.csv", index=False, encoding="utf-8-sig")
    aij.to_csv(data_dir / "aij.csv", index=False, encoding="utf-8-sig")
    br.to_csv(data_dir / "br.csv", index=False, encoding="utf-8-sig")
    mains.to_csv(data_dir / "main_routes.csv", index=False, encoding="utf-8-sig")
    monitor.to_csv(data_dir / "monitor.csv", index=False, encoding="utf-8-sig")

    meta = {
        "park_id": park_id,
        "park_key": cfg["key"],
        "park_name": cfg["name"],
        "n_spots": len(spots),
        "n_roads": len(roads),
    }
    (data_dir / "park_meta.json").write_text(
        __import__("json").dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[park {park_id}] {cfg['name']} -> {out}  spots={len(spots)} roads={len(roads)}")
    return out


def main(park_id: int | None = None):
    generate_park(1 if park_id is None else park_id)


if __name__ == "__main__":
    generate_park(1)
