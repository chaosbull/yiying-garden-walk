# Author: ZengWenquan
# https://github.com/chaosbull
# License: Apache-2.0
# -*- coding: utf-8 -*-
"""Crowd-aware path scoring for the garden network."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls


@dataclass
class ModelParams:
    # Crowd non-uniform estimation (eq 3.2-3.4)
    alpha_B: float = 1.05
    beta_B: float = 0.92
    alpha_F: float = 1.10
    beta_F: float = 0.88
    alpha_W: float = 1.00
    beta_W: float = 0.95

    # Inflow inversion smooth (eq 4.11)
    lambda_I: float = 0.35

    # Prediction risk (eq 5.14)
    lambda_eps: float = 0.8

    # Congestion defaults (eq 6.3) — can be overridden per spot via spots table
    lambda_cong: float = 2.5
    kappa_cong: float = 3.0

    # 移步异景 Q weights (eq 8.5), sum to 1
    alpha_Q: float = 0.35
    beta_Q: float = 0.40
    gamma_Q: float = 0.25

    # Path objective weights (eq 11.7 / 12.4)
    omega_L: float = 0.18
    omega_C: float = 0.12
    omega_N: float = 0.18
    omega_P: float = 0.22
    omega_H: float = 0.12
    omega_Q: float = 0.18
    omega_R: float = 0.15

    # Landscape type weight overrides (optional)
    landscape_weights: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "alpha_B": self.alpha_B,
            "beta_B": self.beta_B,
            "alpha_F": self.alpha_F,
            "beta_F": self.beta_F,
            "alpha_W": self.alpha_W,
            "beta_W": self.beta_W,
            "lambda_I": self.lambda_I,
            "lambda_eps": self.lambda_eps,
            "lambda_cong": self.lambda_cong,
            "kappa_cong": self.kappa_cong,
            "alpha_Q": self.alpha_Q,
            "beta_Q": self.beta_Q,
            "gamma_Q": self.gamma_Q,
            "omega_L": self.omega_L,
            "omega_C": self.omega_C,
            "omega_N": self.omega_N,
            "omega_P": self.omega_P,
            "omega_H": self.omega_H,
            "omega_Q": self.omega_Q,
            "omega_R": self.omega_R,
        }
        d.update({f"wq_{k}": v for k, v in self.landscape_weights.items()})
        return d


class ScenicModel:
    def __init__(
        self,
        spots: pd.DataFrame,
        roads: pd.DataFrame,
        landscape: pd.DataFrame,
        aij: pd.DataFrame,
        br: pd.DataFrame,
        main_routes: pd.DataFrame,
        monitor: pd.DataFrame,
        params: Optional[ModelParams] = None,
        grid_mn: Tuple[int, int] = (24, 24),
    ):
        self.spots = spots.copy()
        self.roads = roads.copy()
        self.landscape = landscape.copy()
        self.aij = aij.copy()
        self.br = br.copy()
        self.main_routes = main_routes.copy()
        self.monitor = monitor.copy()
        self.params = params or ModelParams()
        self.grid_mn = grid_mn

        self.spot_ids = sorted(self.spots["spot_id"].astype(int).tolist())
        self.id2idx = {sid: i for i, sid in enumerate(self.spot_ids)}
        self.n = len(self.spot_ids)
        self.T = int(self.monitor["minute_t"].max())

        self._prepare_graph()
        self._prepare_stay()
        self._prepare_monitor_matrix()

        self.N = None  # actual counts over time
        self.I_hat = None
        self.O_hat = None
        self.Q_roads = None  # road_id -> Q
        self.road_metrics = None

    # ------------------------------------------------------------------ #
    # Graph / stay helpers
    # ------------------------------------------------------------------ #
    def _prepare_graph(self):
        self.neighbors: Dict[int, List[int]] = {sid: [] for sid in self.spot_ids}
        self.edge_lookup: Dict[Tuple[int, int, int], dict] = {}
        self.edges_between: Dict[Tuple[int, int], List[int]] = {}

        for r in self.roads.itertuples():
            a, b, rr = int(r.from_id), int(r.to_id), int(r.road_index_r)
            info = {
                "road_id": int(r.road_id),
                "L": float(r.length_L),
                "tau": int(r.travel_time_tau),
                "C": float(r.energy_C),
                "B": float(r.boundary_B),
                "r": rr,
            }
            for u, v in ((a, b), (b, a)):
                self.edge_lookup[(u, v, rr)] = info
                self.edges_between.setdefault((u, v), [])
                if rr not in self.edges_between[(u, v)]:
                    self.edges_between[(u, v)].append(rr)
            if b not in self.neighbors[a]:
                self.neighbors[a].append(b)
            if a not in self.neighbors[b]:
                self.neighbors[b].append(a)

        self.a_mat = {(int(r.from_id), int(r.to_id)): float(r.a_ij) for r in self.aij.itertuples()}
        self.b_mat = {
            (int(r.from_id), int(r.to_id), int(r.road_index_r)): float(r.b_r)
            for r in self.br.itertuples()
        }

        self.main_sets = {}
        for mid, g in self.main_routes.groupby("main_route_id"):
            self.main_sets[int(mid)] = set(g["spot_id"].astype(int).tolist())

        self.spot_meta = self.spots.set_index("spot_id").to_dict("index")

    def _prepare_stay(self):
        """Discrete stay pmf f_i[k] from (mu, sigma), truncated at K_i."""
        self.f: Dict[int, np.ndarray] = {}
        self.S: Dict[int, np.ndarray] = {}
        self.K: Dict[int, int] = {}
        for sid in self.spot_ids:
            mu = float(self.spot_meta[sid]["mu_stay"])
            sigma = float(self.spot_meta[sid]["sigma_stay"])
            K = max(8, int(np.ceil(mu + 3 * sigma)))
            ks = np.arange(1, K + 1)
            # Discretized Gaussian support on 1..K, renormalized.
            pdf = np.exp(-0.5 * ((ks - mu) / max(sigma, 1e-6)) ** 2)
            pdf = pdf / pdf.sum()
            # S[0]=1; S[k]=P(T>k)=1-sum_{h=1..k} f[h]
            S = np.zeros(K + 1)
            S[0] = 1.0
            cdf = 0.0
            for k in range(1, K + 1):
                cdf += pdf[k - 1]
                S[k] = max(0.0, 1.0 - cdf)
            self.f[sid] = pdf
            self.S[sid] = S
            self.K[sid] = K

    def _prepare_monitor_matrix(self):
        self.nc = np.zeros((self.n, self.T + 1))
        for r in self.monitor.itertuples():
            i = self.id2idx[int(r.spot_id)]
            t = int(r.minute_t)
            self.nc[i, t] = float(r.monitor_count_nc)

    # ------------------------------------------------------------------ #
    # Section 3: actual counts
    # ------------------------------------------------------------------ #
    def estimate_N(self) -> np.ndarray:
        p = self.params
        N = np.zeros_like(self.nc)
        for sid in self.spot_ids:
            i = self.id2idx[sid]
            meta = self.spot_meta[sid]
            Ai = float(meta["area_Ai"])
            Ac = float(meta["monitor_area_Ac"])
            z = str(meta["env_type"])
            if z == "B":
                alpha, beta = p.alpha_B, p.beta_B
            elif z == "F":
                alpha, beta = p.alpha_F, p.beta_F
            else:
                alpha, beta = p.alpha_W, p.beta_W
            ratio = (Ai / max(Ac, 1e-6)) ** beta
            N[i, :] = np.clip(alpha * self.nc[i, :] * ratio, 0.0, 50.0)
        self.N = N
        return N

    # ------------------------------------------------------------------ #
    # Section 4: inflow / outflow inversion
    # ------------------------------------------------------------------ #
    def invert_IO(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.N is None:
            self.estimate_N()
        p = self.params
        I = np.zeros((self.n, self.T + 1))
        O = np.zeros((self.n, self.T + 1))

        for sid in self.spot_ids:
            i = self.id2idx[sid]
            Svec = self.S[sid]
            K = self.K[sid]
            Ni = self.N[i, 1:]  # length T
            T = self.T
            # Build H (T x T): H[t,s] = S[t-s] for s<=t
            H = np.zeros((T, T))
            for t in range(T):
                for s in range(t + 1):
                    lag = t - s
                    H[t, s] = Svec[lag] if lag <= K else 0.0
            # Smoothness: ||D I|| with first difference
            D = np.zeros((T - 1, T))
            for t in range(T - 1):
                D[t, t] = -1.0
                D[t, t + 1] = 1.0
            A = np.vstack([H, np.sqrt(p.lambda_I) * D])
            b = np.concatenate([Ni, np.zeros(T - 1)])
            Ii, _ = nnls(A, b)
            I[i, 1:] = Ii
            # Outflow eq 4.12
            f = self.f[sid]
            for t in range(1, T + 1):
                val = 0.0
                for k in range(1, min(K, t) + 1):
                    val += I[i, t - k] * f[k - 1]
                O[i, t] = val

        self.I_hat = I
        self.O_hat = O
        return I, O

    # ------------------------------------------------------------------ #
    # Sections 7-8: 移步异景
    # ------------------------------------------------------------------ #
    def _enclosure_and_grad(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """E[p] and G[p]=||grad E|| at points (P,2)."""
        p = self.params
        land = self.landscape.copy()
        if p.landscape_weights:
            for tname, w in p.landscape_weights.items():
                land.loc[land["type_name"] == tname, "weight_wq"] = w

        E = np.zeros(len(points))
        dEx = np.zeros(len(points))
        dEy = np.zeros(len(points))
        for r in land.itertuples():
            wq = float(r.weight_wq)
            Aq = float(r.area_Aq)
            sig2 = float(r.sigma_q) ** 2
            dx = points[:, 0] - float(r.x)
            dy = points[:, 1] - float(r.y)
            hq = (1.0 / (2 * np.pi * sig2)) * np.exp(-(dx * dx + dy * dy) / (2 * sig2))
            contrib = wq * Aq * hq
            E += contrib
            dEx += -contrib * dx / sig2
            dEy += -contrib * dy / sig2
        G = np.sqrt(dEx * dEx + dEy * dEy)
        return E, G

    def compute_road_Q(self, samples_per_100: int = 4) -> pd.DataFrame:
        rows = []
        spot_xy = {int(r.spot_id): (float(r.x), float(r.y)) for r in self.spots.itertuples()}
        alpha, beta, gamma = self.params.alpha_Q, self.params.beta_Q, self.params.gamma_Q
        s = alpha + beta + gamma
        alpha, beta, gamma = alpha / s, beta / s, gamma / s

        for r in self.roads.itertuples():
            a, b = int(r.from_id), int(r.to_id)
            xa, ya = spot_xy[a]
            xb, yb = spot_xy[b]
            L = float(r.length_L)
            K = max(5, int(round(L / 100.0 * samples_per_100)))
            ts = np.linspace(0, 1, K)
            pts = np.column_stack([xa + ts * (xb - xa), ya + ts * (yb - ya)])
            E, G = self._enclosure_and_grad(pts)
            Ebar = float(E.mean())
            V = float(((E - Ebar) ** 2).mean())
            Gbar = float(G.mean())
            B = float(r.boundary_B)
            # Normalize components to comparable scale within this computation batch later
            rows.append(
                {
                    "road_id": int(r.road_id),
                    "from_id": a,
                    "to_id": b,
                    "road_index_r": int(r.road_index_r),
                    "length_L": L,
                    "V_enclosure_var": V,
                    "G_mean_grad": Gbar,
                    "B_boundary": B,
                    "E_mean": Ebar,
                }
            )

        df = pd.DataFrame(rows)
        # Scale each component to [0,1] across roads then combine (stable Q).
        def _norm(col):
            v = df[col].to_numpy(dtype=float)
            lo, hi = v.min(), v.max()
            if hi - lo < 1e-12:
                return np.ones_like(v) * 0.5
            return (v - lo) / (hi - lo)

        Vn, Gn, Bn = _norm("V_enclosure_var"), _norm("G_mean_grad"), _norm("B_boundary")
        Q = alpha * Vn + beta * Gn + gamma * Bn
        # Map to a score range similar to reference view scores (~0.3–3).
        df["Q_yibu_yijing"] = 0.3 + 2.7 * Q
        self.Q_roads = {int(r.road_id): float(r.Q_yibu_yijing) for r in df.itertuples()}
        self.road_metrics = df
        return df

    def get_Q(self, u: int, v: int, r: int) -> float:
        info = self.edge_lookup[(u, v, r)]
        return float(self.Q_roads[info["road_id"]])

    # ------------------------------------------------------------------ #
    # Sections 5-6: future crowd & congestion
    # ------------------------------------------------------------------ #
    def predict_N_at(self, g: int, t: int, tau: int) -> Tuple[float, float]:
        """Return (Nhat, Nbar) at g for arrival time t+tau (eq 5.11, 5.14)."""
        if self.I_hat is None:
            self.invert_IO()
        t = int(np.clip(t, 1, self.T))
        tau = int(max(0, tau))
        gi = self.id2idx[g]
        Kg = self.K[g]
        Sg = self.S[g]
        # Surviving historical inflow
        Ns = 0.0
        for k in range(0, Kg + 1):
            tk = t - k
            if tk < 1 or tk > self.T:
                continue
            lag = k + tau
            s_val = Sg[lag] if lag <= Kg else 0.0
            Ns += self.I_hat[gi, tk] * s_val

        # New arrivals during prediction window (simplified using current outflows)
        Na = 0.0
        for i in self.neighbors[g]:
            for r in self.edges_between.get((i, g), [1]):
                aig = self.a_mat.get((i, g), 0.0)
                brig = self.b_mat.get((i, g, r), 1.0)
                tau_r = self.edge_lookup[(i, g, r)]["tau"]
                t_dep = t + tau - tau_r
                if t_dep < 1 or t_dep > self.T:
                    continue
                ii = self.id2idx[i]
                Na += aig * brig * self.O_hat[ii, t_dep]

        Nhat = Ns + Na
        # Empirical uncertainty proxy
        hist = self.N[gi, max(1, t - 10) : t + 1]
        sigma = float(np.std(hist)) if len(hist) > 1 else 1.0
        Nbar = Nhat + self.params.lambda_eps * sigma
        return Nhat, Nbar

    def congestion_penalty(self, spot_id: int, Nbar: float) -> float:
        meta = self.spot_meta[spot_id]
        Si = float(meta["capacity_Si"])
        theta1 = float(meta.get("theta1", 0.55))
        theta2 = float(meta.get("theta2", 0.90))
        rho = Nbar / max(Si, 1e-6)
        if rho <= theta1:
            return 0.0
        if rho >= theta2:
            return 9999.0
        lam = self.params.lambda_cong
        kap = self.params.kappa_cong
        return float(lam * (np.exp(kap * (rho - theta1)) - 1.0))

    # ------------------------------------------------------------------ #
    # Path enumeration (two-step + short paths to goal)
    # ------------------------------------------------------------------ #
    def enumerate_two_step(self, s: int) -> List[dict]:
        """All (j,k) with j neighbor of s, k neighbor of j, k!=s."""
        cands = []
        for j in self.neighbors[s]:
            for k in self.neighbors[j]:
                if k == s:
                    continue
                cands.append({"s": s, "j": j, "k": k})
        return cands

    def _best_road(self, u: int, v: int) -> int:
        rs = self.edges_between[(u, v)]
        # Prefer higher Q / lower length combo
        best, best_score = rs[0], -1e18
        for r in rs:
            info = self.edge_lookup[(u, v, r)]
            q = self.get_Q(u, v, r)
            score = q - 0.001 * info["L"]
            if score > best_score:
                best_score, best = score, r
        return best

    def path_attributes(
        self,
        nodes: Sequence[int],
        road_choices: Sequence[int],
        visit_flags: Sequence[int],
        t0: int,
    ) -> dict:
        """Compute L,C,T,H,Q,N,P for a path (eq 9-10)."""
        assert len(nodes) == len(road_choices) + 1
        assert len(visit_flags) == len(nodes)
        L = C = T_travel = 0.0
        Q_num = 0.0
        for k, r in enumerate(road_choices):
            u, v = nodes[k], nodes[k + 1]
            info = self.edge_lookup[(u, v, r)]
            L += info["L"]
            C += info["C"]
            T_travel += info["tau"]
            Q_num += info["L"] * self.get_Q(u, v, r)
        Qpi = Q_num / max(L, 1e-6)
        # Stay times for intermediate visited nodes
        T_stay = 0.0
        H = 0
        for k in range(1, len(nodes) - 1):
            if visit_flags[k]:
                T_stay += float(self.spot_meta[nodes[k]]["mu_stay"])
                H += 1
        Tpi = T_travel + T_stay

        # Arrival times & crowd / congestion on visited intermediate + goal
        Npi = 0.0
        Ppi = 0.0
        arrive = t0
        detail_N = []
        for k in range(1, len(nodes)):
            r = road_choices[k - 1]
            arrive += self.edge_lookup[(nodes[k - 1], nodes[k], r)]["tau"]
            # Evaluate crowd at arrival (before stay); stay only delays onward travel
            tau_from_now = arrive - t0
            Nhat, Nbar = self.predict_N_at(nodes[k], t0, max(0, tau_from_now))
            P = self.congestion_penalty(nodes[k], Nbar)
            detail_N.append(
                {
                    "spot_id": nodes[k],
                    "arrive_offset_min": tau_from_now,
                    "Nhat": Nhat,
                    "Nbar": Nbar,
                    "P": P,
                }
            )
            if visit_flags[k] or k == len(nodes) - 1:
                Npi += Nbar
                Ppi += P
            if k < len(nodes) - 1 and visit_flags[k]:
                arrive += int(round(float(self.spot_meta[nodes[k]]["mu_stay"])))

        return {
            "L": L,
            "C": C,
            "T": Tpi,
            "H": H,
            "Q": Qpi,
            "N": Npi,
            "P": Ppi,
            "nodes": list(nodes),
            "roads": list(road_choices),
            "visit": list(visit_flags),
            "detail_N": detail_N,
        }

    def score_paths(self, attrs_list: List[dict], main_route_id: int = 1) -> List[dict]:
        """Normalize and compute J and J^R (eq 11-12)."""
        if not attrs_list:
            return []
        keys_cost = ["L", "C", "N", "P"]
        keys_ben = ["H", "Q"]
        mins, maxs = {}, {}
        for k in keys_cost + keys_ben:
            vals = np.array([a[k] for a in attrs_list], dtype=float)
            mins[k], maxs[k] = float(vals.min()), float(vals.max())

        def norm(k, v, benefit=False):
            lo, hi = mins[k], maxs[k]
            if hi - lo < 1e-12:
                return 0.0
            x = (v - lo) / (hi - lo)
            return x

        p = self.params
        wsum = p.omega_L + p.omega_C + p.omega_N + p.omega_P + p.omega_H + p.omega_Q
        wL, wC, wN, wP, wH, wQ = (
            p.omega_L / wsum,
            p.omega_C / wsum,
            p.omega_N / wsum,
            p.omega_P / wsum,
            p.omega_H / wsum,
            p.omega_Q / wsum,
        )
        V0 = self.main_sets.get(main_route_id, set())
        out = []
        for a in attrs_list:
            J = (
                wL * norm("L", a["L"])
                + wC * norm("C", a["C"])
                + wN * norm("N", a["N"])
                + wP * norm("P", a["P"])
                - wH * norm("H", a["H"], True)
                - wQ * norm("Q", a["Q"], True)
            )
            Vpi = set(a["nodes"])
            if not Vpi and not V0:
                R = 0.0
            else:
                R = len(Vpi & V0) / max(len(Vpi | V0), 1)
            JR = J + p.omega_R * (1.0 - R)
            b = dict(a)
            b["J"] = J
            b["JR"] = JR
            b["R_main"] = R
            out.append(b)
        out.sort(key=lambda x: x["JR"])
        return out

    def two_step_recommendations(
        self, s: int, t0: int, main_route_id: int = 1, visit_middle: bool = True
    ) -> pd.DataFrame:
        if self.Q_roads is None:
            self.compute_road_Q()
        if self.I_hat is None:
            self.invert_IO()

        attrs = []
        for cand in self.enumerate_two_step(s):
            j, k = cand["j"], cand["k"]
            r1 = self._best_road(s, j)
            r2 = self._best_road(j, k)
            y = [0, 1 if visit_middle else 0, 1]  # visit j optional, always "target" k
            a = self.path_attributes([s, j, k], [r1, r2], y, t0)
            a["from_spot"] = s
            a["step1_spot"] = j
            a["step2_spot"] = k
            a["road1_r"] = r1
            a["road2_r"] = r2
            # one-step path score component
            a1 = self.path_attributes([s, j], [r1], [0, 1], t0)
            a["J_step1_raw"] = a1  # filled after batch? store attrs first
            attrs.append(a)

        scored = self.score_paths(attrs, main_route_id=main_route_id)
        rows = []
        name = {int(r.spot_id): r.name for r in self.spots.itertuples()}
        for rank, a in enumerate(scored, start=1):
            d1 = a["detail_N"][0]
            d2 = a["detail_N"][1]
            rows.append(
                {
                    "rank": rank,
                    "current_spot": a["from_spot"],
                    "current_name": name[a["from_spot"]],
                    "step1_spot": a["step1_spot"],
                    "step1_name": name[a["step1_spot"]],
                    "step2_spot": a["step2_spot"],
                    "step2_name": name[a["step2_spot"]],
                    "path_L": round(a["L"], 2),
                    "path_C": round(a["C"], 2),
                    "path_T": round(a["T"], 2),
                    "path_Q": round(a["Q"], 4),
                    "path_H": a["H"],
                    "path_N": round(a["N"], 3),
                    "path_P": round(a["P"], 3),
                    "J": round(a["J"], 6),
                    "JR": round(a["JR"], 6),
                    "R_main": round(a["R_main"], 4),
                    "step1_arrive_min": d1["arrive_offset_min"],
                    "step1_Nhat": round(d1["Nhat"], 3),
                    "step1_Nbar": round(d1["Nbar"], 3),
                    "step1_P": round(d1["P"], 3),
                    "step2_arrive_min": d2["arrive_offset_min"],
                    "step2_Nhat": round(d2["Nhat"], 3),
                    "step2_Nbar": round(d2["Nbar"], 3),
                    "step2_P": round(d2["P"], 3),
                    "road1_r": a["road1_r"],
                    "road2_r": a["road2_r"],
                    "Q_road1": round(self.get_Q(a["from_spot"], a["step1_spot"], a["road1_r"]), 4),
                    "Q_road2": round(self.get_Q(a["step1_spot"], a["step2_spot"], a["road2_r"]), 4),
                }
            )
        return pd.DataFrame(rows)

    def adjacent_summary(self, s: int, t0: int) -> pd.DataFrame:
        """One-hop neighbor scores and arrival crowd."""
        if self.Q_roads is None:
            self.compute_road_Q()
        if self.I_hat is None:
            self.invert_IO()
        name = {int(r.spot_id): r.name for r in self.spots.itertuples()}
        attrs = []
        meta = []
        for j in self.neighbors[s]:
            r = self._best_road(s, j)
            a = self.path_attributes([s, j], [r], [0, 1], t0)
            a["from_spot"] = s
            a["to_spot"] = j
            a["road_r"] = r
            attrs.append(a)
            meta.append((j, r))
        scored = self.score_paths(attrs, main_route_id=1)
        rows = []
        for rank, a in enumerate(scored, start=1):
            d = a["detail_N"][0]
            rows.append(
                {
                    "rank": rank,
                    "current_spot": a["from_spot"],
                    "current_name": name[a["from_spot"]],
                    "adj_spot": a["to_spot"],
                    "adj_name": name[a["to_spot"]],
                    "road_r": a["road_r"],
                    "L": round(a["L"], 2),
                    "C": round(a["C"], 2),
                    "tau": int(self.edge_lookup[(a["from_spot"], a["to_spot"], a["road_r"])]["tau"]),
                    "Q_road": round(self.get_Q(a["from_spot"], a["to_spot"], a["road_r"]), 4),
                    "path_Q": round(a["Q"], 4),
                    "J": round(a["J"], 6),
                    "JR": round(a["JR"], 6),
                    "arrive_min": d["arrive_offset_min"],
                    "Nhat_at_arrive": round(d["Nhat"], 3),
                    "Nbar_at_arrive": round(d["Nbar"], 3),
                    "P_at_arrive": round(d["P"], 3),
                }
            )
        return pd.DataFrame(rows)

    def shortest_paths_to_goal(
        self, s: int, g: int, t0: int, main_route_id: int = 1, max_depth: int = 6
    ) -> pd.DataFrame:
        """Enumerate simple paths s->g up to max_depth edges, score them."""
        if self.Q_roads is None:
            self.compute_road_Q()
        if self.I_hat is None:
            self.invert_IO()

        paths = []

        def dfs(node, path, depth):
            if depth > max_depth:
                return
            if node == g and len(path) > 1:
                paths.append(list(path))
                return
            for nxt in self.neighbors[node]:
                if nxt in path:
                    continue
                path.append(nxt)
                dfs(nxt, path, depth + 1)
                path.pop()

        dfs(s, [s], 0)
        attrs = []
        for nodes in paths:
            roads = [self._best_road(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
            visit = [0] + [1] * (len(nodes) - 2) + [1]
            a = self.path_attributes(nodes, roads, visit, t0)
            attrs.append(a)
        scored = self.score_paths(attrs, main_route_id=main_route_id)
        name = {int(r.spot_id): r.name for r in self.spots.itertuples()}
        rows = []
        for rank, a in enumerate(scored[:30], start=1):
            rows.append(
                {
                    "rank": rank,
                    "path_nodes": "->".join(str(x) for x in a["nodes"]),
                    "path_names": "->".join(name[x] for x in a["nodes"]),
                    "L": round(a["L"], 2),
                    "C": round(a["C"], 2),
                    "T": round(a["T"], 2),
                    "H": a["H"],
                    "Q": round(a["Q"], 4),
                    "N": round(a["N"], 3),
                    "P": round(a["P"], 3),
                    "J": round(a["J"], 6),
                    "JR": round(a["JR"], 6),
                    "R_main": round(a["R_main"], 4),
                }
            )
        return pd.DataFrame(rows)
