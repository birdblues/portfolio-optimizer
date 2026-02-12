#!/usr/bin/env python3
"""
Black-Litterman + Minsky Overlay Portfolio Optimizer (v5.0-MH)
Agent Team 진단 확장: 상관행렬, π-μ_BL 괴리, 위험기여도, VaR/CVaR, 제약근접도, JSON 출력
멀티 지평(1m/3m/6m) 지원, EWMA 공분산 기반
- views.yaml에서 뷰 로드 (절대/상대/바스켓 뷰 지원)
- 뷰 변경 이력 자동 저장
- primary_horizon(기본 3m)으로 최종 비중 결정, 나머지 지평은 진단/참조용
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import cvxpy as cp
import yfinance as yf
import yaml
from scipy.stats import norm


# ============================================================
# 1. 설정 및 뷰 로드
# ============================================================
def load_config(config_path: str = "config.yaml") -> dict:
    """config.yaml 로드"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_views(views_path: str = "views.yaml") -> dict:
    """views.yaml 로드"""
    with open(views_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_view_matrices(views_data: dict, Sigma: np.ndarray, horizon: str = "3m") -> tuple:
    """
    views.yaml에서 P, Q, Ω 행렬 생성 (멀티 지평 지원)
    - 절대뷰: P = [1,0,0,0]
    - 상대뷰: P = [1,0,0,-1] (SPY - DXY)
    - 바스켓뷰: P = [0.5,0.5,-1,0]
    - horizon: "1m", "3m", "6m" — Q_{horizon}, conf_{horizon} 키 사용
    """
    views = views_data["views"]
    n_views = len(views)
    n_assets = len(views_data["assets"])

    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)
    conf = np.zeros(n_views)

    q_key = f"Q_{horizon}"
    conf_key = f"conf_{horizon}"

    for i, v in enumerate(views):
        P[i, :] = v["P"]
        # 지평별 Q: Q_{horizon} 우선, 없으면 Q_3m 폴백, 그것도 없으면 0
        Q[i] = v.get(q_key, v.get("Q_3m", 0.0))
        # 지평별 conf: conf_{horizon} 우선, 없으면 기본 conf 폴백
        conf[i] = v.get(conf_key, v.get("conf", 0.5))

    # Omega 계산: Ω_ii = k × (1-conf_i)/conf_i
    k = np.median(np.diag(Sigma))
    ratio = (1 - conf) / conf
    Omega = np.diag(k * ratio)

    return P, Q, conf, Omega


def save_views_snapshot(views_path: str = "views.yaml", history_dir: str = "views_history"):
    """뷰 변경 이력 저장"""
    Path(history_dir).mkdir(exist_ok=True)

    kst = ZoneInfo("Asia/Seoul")
    now = datetime.now(kst)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    snapshot_path = Path(history_dir) / f"{timestamp}_views.yaml"

    shutil.copy(views_path, snapshot_path)
    return snapshot_path


def load_previous_views(history_dir: str = "views_history") -> dict | None:
    """가장 최근 뷰 스냅샷 로드"""
    history_path = Path(history_dir)
    if not history_path.exists():
        return None

    snapshots = sorted(history_path.glob("*_views.yaml"))
    if len(snapshots) < 2:
        return None

    # 현재 저장 전 가장 최근 스냅샷
    with open(snapshots[-2], "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def diff_views(prev: dict | None, curr: dict, horizons: list[str] = None) -> list[str]:
    """뷰 변경사항 비교 (멀티 지평 지원)"""
    if horizons is None:
        horizons = ["1m", "3m", "6m"]

    changes = []

    if prev is None:
        changes.append("(이전 뷰 기록 없음 - 첫 실행)")
        return changes

    prev_views = {v["id"]: v for v in prev.get("views", [])}
    curr_views = {v["id"]: v for v in curr.get("views", [])}

    # 추가된 뷰
    for vid in curr_views:
        if vid not in prev_views:
            changes.append(f"[추가] {vid}: {curr_views[vid]['name']}")

    # 삭제된 뷰
    for vid in prev_views:
        if vid not in curr_views:
            changes.append(f"[삭제] {vid}: {prev_views[vid]['name']}")

    # 변경된 뷰
    for vid in curr_views:
        if vid in prev_views:
            pv = prev_views[vid]
            cv = curr_views[vid]

            # 지평별 Q 비교
            for hz in horizons:
                q_key = f"Q_{hz}"
                pq = pv.get(q_key, pv.get("Q_3m"))
                cq = cv.get(q_key, cv.get("Q_3m"))
                if pq is not None and cq is not None and pq != cq:
                    changes.append(f"[Q_{hz} 변경] {vid}: {pq*100:.4f}% → {cq*100:.4f}%")

            # 지평별 conf 비교
            for hz in horizons:
                conf_key = f"conf_{hz}"
                pc = pv.get(conf_key, pv.get("conf"))
                cc = cv.get(conf_key, cv.get("conf"))
                if pc is not None and cc is not None and pc != cc:
                    changes.append(f"[conf_{hz} 변경] {vid}: {pc*100:.0f}% → {cc*100:.0f}%")

            # 기본 conf 비교 (하위호환)
            if pv.get("conf") != cv.get("conf"):
                changes.append(f"[conf 변경] {vid}: {pv['conf']*100:.0f}% → {cv['conf']*100:.0f}%")

            if pv["P"] != cv["P"]:
                changes.append(f"[P 변경] {vid}: {pv['P']} → {cv['P']}")

    if not changes:
        changes.append("(변경사항 없음)")

    return changes


# ============================================================
# 2. 데이터 수집
# ============================================================
def fetch_prices(config: dict) -> pd.DataFrame:
    """yfinance에서 가격 데이터 수집"""
    symbols = config["symbols"]
    assets = config["assets"]
    lookback = config["data"]["lookback_days"]

    end = datetime.now()
    start = end - pd.Timedelta(days=int(lookback * 1.5))

    tickers = [symbols[a] for a in assets]
    print(f"  티커 다운로드 중: {tickers}")

    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError("데이터를 가져올 수 없습니다.")

    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"].copy()
    else:
        prices = df[["Close"]].copy()

    prices.columns = [assets[tickers.index(col)] if col in tickers else col for col in prices.columns]
    prices = prices[assets]
    prices = prices.dropna()

    print(f"  총 {len(prices)}일 데이터 수집 완료")
    return prices


# ============================================================
# 3. 로그수익률 계산
# ============================================================
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """로그수익률 계산: r_t = log(P_t / P_{t-1})"""
    return np.log(prices / prices.shift(1)).dropna()


# ============================================================
# 4. EWMA 공분산
# ============================================================
def compute_ewma_cov(returns: pd.DataFrame, lam: float, init_window: int) -> np.ndarray:
    """EWMA 공분산 계산"""
    n = len(returns)
    if n < init_window:
        raise ValueError(f"데이터 부족: {n}일 < 필요 {init_window}일")

    S = returns.iloc[:init_window].cov().values

    for t in range(init_window, n):
        r_t = returns.iloc[t].values.reshape(-1, 1)
        S = lam * S + (1 - lam) * (r_t @ r_t.T)

    return S


# ============================================================
# 5. Black-Litterman (일반화된 P 행렬 지원)
# ============================================================
def black_litterman(
    Sigma: np.ndarray,
    w_mkt: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray,
    tau: float,
    delta: float
) -> tuple:
    """
    Black-Litterman posterior 기대수익률 계산
    - π = δ Σ w_mkt (시장균형)
    - μ_BL = [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1} × [(τΣ)^{-1}π + P^T Ω^{-1}Q]
    """
    # 시장균형 기대수익
    pi = delta * Sigma @ w_mkt

    # BL posterior
    tau_Sigma = tau * Sigma
    inv_tau_Sigma = np.linalg.inv(tau_Sigma)
    inv_Omega = np.linalg.inv(Omega)

    A = inv_tau_Sigma + P.T @ inv_Omega @ P
    b = inv_tau_Sigma @ pi + P.T @ inv_Omega @ Q

    mu_bl = np.linalg.solve(A, b)

    return mu_bl.flatten(), pi.flatten()


# ============================================================
# 6. QP 최적화
# ============================================================
def optimize_qp(
    mu_bl: np.ndarray,
    Sigma: np.ndarray,
    delta: float,
    lb: np.ndarray,
    ub: np.ndarray
) -> np.ndarray:
    """QP 최적화: max μ_BL^T w - (δ/2) w^T Σ w"""
    n = len(mu_bl)
    w = cp.Variable(n)

    objective = cp.Maximize(mu_bl @ w - 0.5 * delta * cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1, w >= lb, w <= ub]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"QP 최적화 실패: {prob.status}")

    return np.array(w.value).flatten()


# ============================================================
# 7. 민스키 오버레이
# ============================================================
def apply_minsky_overlay(
    w_star: np.ndarray,
    gld_target: float,
    lb: np.ndarray,
    ub: np.ndarray
) -> np.ndarray:
    """민스키 GLD 오버레이 적용"""
    w = w_star.copy()
    w[2] = gld_target

    rest_idx = [0, 1, 3]
    rest_sum = w_star[rest_idx].sum()

    if rest_sum > 0:
        scale = (1 - gld_target) / rest_sum
        w[rest_idx] = w_star[rest_idx] * scale
    else:
        w[rest_idx] = (1 - gld_target) / 3

    for _ in range(10):
        w_clipped = np.clip(w, lb, ub)
        if np.abs(w_clipped.sum() - 1.0) < 1e-6:
            break
        w = w_clipped / w_clipped.sum()
        w = np.clip(w, lb, ub)

    w = w / w.sum()

    return w


# ============================================================
# 8. 진단 함수 (Agent Team 지원)
# ============================================================
def compute_correlation_matrix(Sigma: np.ndarray) -> np.ndarray:
    """공분산 → 상관관계 행렬"""
    vol = np.sqrt(np.diag(Sigma))
    outer = np.outer(vol, vol)
    outer = np.where(outer == 0, 1e-10, outer)
    return Sigma / outer


def compute_pi_bl_divergence(
    pi: np.ndarray, mu_bl: np.ndarray, Sigma: np.ndarray
) -> list[dict]:
    """π(균형) vs μ_BL(사후) 괴리 분석"""
    vol = np.sqrt(np.diag(Sigma))
    diff = mu_bl - pi
    z_scores = np.where(vol > 0, diff / vol, 0.0)

    results = []
    for i in range(len(pi)):
        direction = "상방" if diff[i] > 0 else "하방" if diff[i] < 0 else "중립"
        strength = "약" if abs(z_scores[i]) < 0.3 else "중" if abs(z_scores[i]) < 0.7 else "강"
        results.append({
            "pi": float(pi[i]),
            "mu_bl": float(mu_bl[i]),
            "diff": float(diff[i]),
            "diff_pct": float(diff[i] * 100),
            "z_score": float(z_scores[i]),
            "direction": direction,
            "strength": strength,
        })
    return results


def compute_risk_contribution(w: np.ndarray, Sigma: np.ndarray) -> dict:
    """
    위험기여도(Risk Contribution)
    RC_i = w_i × (Σw)_i / σ_p
    """
    port_var = w @ Sigma @ w
    port_vol = np.sqrt(port_var)
    n = len(w)

    if port_vol < 1e-10:
        return {
            "marginal_risk": np.zeros(n).tolist(),
            "risk_contribution": np.zeros(n).tolist(),
            "risk_pct": np.zeros(n).tolist(),
            "port_vol": 0.0,
            "hhi": 0.0
        }

    marginal = (Sigma @ w) / port_vol
    rc = w * marginal
    rc_pct = rc / port_vol
    hhi = float(np.sum(rc_pct ** 2))

    return {
        "marginal_risk": marginal.tolist(),
        "risk_contribution": rc.tolist(),
        "risk_pct": rc_pct.tolist(),
        "port_vol": float(port_vol),
        "hhi": hhi
    }


def compute_constraint_proximity(
    w: np.ndarray, lb: np.ndarray, ub: np.ndarray, threshold: float = 0.02
) -> list[dict]:
    """제약조건 근접도 분석"""
    results = []
    for i in range(len(w)):
        dist_lb = w[i] - lb[i]
        dist_ub = ub[i] - w[i]
        range_total = ub[i] - lb[i]
        utilization = (w[i] - lb[i]) / range_total if range_total > 0 else 0.5

        status = "정상"
        if dist_lb <= threshold:
            status = "⚠ 하한 바인딩"
        elif dist_ub <= threshold:
            status = "⚠ 상한 바인딩"
        elif utilization > 0.85:
            status = "상한 근접"
        elif utilization < 0.15:
            status = "하한 근접"

        results.append({
            "weight": float(w[i]),
            "lb": float(lb[i]),
            "ub": float(ub[i]),
            "dist_lb": float(dist_lb),
            "dist_ub": float(dist_ub),
            "utilization": float(utilization),
            "status": status
        })
    return results


def compute_var_cvar(
    w: np.ndarray, mu_bl: np.ndarray, Sigma: np.ndarray,
    confidence_levels: list[float] = [0.95, 0.99]
) -> dict:
    """
    파라메트릭 VaR/CVaR (정규분포 가정)
    VaR_α = -(μ_p - z_α × σ_p)
    CVaR_α = -(μ_p - σ_p × φ(z_α)/(1-α))
    """
    mu_p = float(w @ mu_bl)
    port_vol = float(np.sqrt(w @ Sigma @ w))

    results = {}
    for cl in confidence_levels:
        z = norm.ppf(cl)
        var_val = -(mu_p - z * port_vol)
        cvar_val = -(mu_p - port_vol * norm.pdf(z) / (1 - cl))
        key = f"{int(cl*100)}%"
        results[key] = {
            "VaR": float(var_val),
            "CVaR": float(cvar_val),
            "VaR_pct": float(var_val * 100),
            "CVaR_pct": float(cvar_val * 100)
        }

    results["mu_p"] = mu_p
    results["sigma_p"] = port_vol
    return results


def load_previous_weights(report_dir: str = "reports") -> dict | None:
    """가장 최근 JSON 리포트에서 전회 비중 로드"""
    report_path = Path(report_dir)
    if not report_path.exists():
        return None

    json_files = sorted(report_path.glob("*_bl_report.json"))
    if not json_files:
        return None

    try:
        with open(json_files[-1], "r", encoding="utf-8") as f:
            prev = json.load(f)
        return prev
    except (json.JSONDecodeError, KeyError):
        return None


def build_json_output(
    config, views_data, prices, Sigma_daily, horizons_results,
    primary_horizon, w_final, minsky_state, view_changes,
    divergence, risk_contrib, constraint_prox, var_cvar, prev_weights
) -> dict:
    """에이전트 파싱용 JSON 구조화 출력 (멀티 지평)"""
    assets = config["assets"]
    primary = primary_horizon
    pr = horizons_results[primary]
    Sigma_primary = pr["Sigma"]
    corr = compute_correlation_matrix(Sigma_primary)
    kst = ZoneInfo("Asia/Seoul")
    now = datetime.now(kst)

    prev_w = None
    if prev_weights and "weights" in prev_weights:
        prev_w = prev_weights["weights"].get("w_final")

    # 지평별 BL 결과
    bl_by_horizon = {}
    for hz_name, hr in horizons_results.items():
        bl_by_horizon[hz_name] = {
            "T": hr["T"],
            "pi": {a: float(hr["pi"][i]) for i, a in enumerate(assets)},
            "mu_bl": {a: float(hr["mu_bl"][i]) for i, a in enumerate(assets)},
            "w_star": {a: float(hr["w_star"][i]) for i, a in enumerate(assets)},
        }

    return {
        "meta": {
            "version": "v5.0-MH",
            "timestamp": now.isoformat(),
            "minsky_state": minsky_state,
            "assets": assets,
            "primary_horizon": primary,
            "horizons": list(horizons_results.keys())
        },
        "prices": {
            "date": prices.index[-1].strftime("%Y-%m-%d"),
            "values": {a: float(prices[a].iloc[-1]) for a in assets}
        },
        "covariance": {
            "correlation_primary": corr.tolist(),
            "vol_daily": {a: float(np.sqrt(Sigma_daily[i, i])) for i, a in enumerate(assets)},
            "vol_by_horizon": {
                hz: {a: float(np.sqrt(hr["Sigma"][i, i])) for i, a in enumerate(assets)}
                for hz, hr in horizons_results.items()
            }
        },
        "bl_results": bl_by_horizon,
        "primary_horizon": primary,
        "weights": {
            "w_mkt": {a: float(config["w_mkt"][i]) for i, a in enumerate(assets)},
            "w_star": {a: float(pr["w_star"][i]) for i, a in enumerate(assets)},
            "w_final": {a: float(w_final[i]) for i, a in enumerate(assets)},
            "w_prev": {a: float(prev_w[a]) for a in assets} if prev_w else None,
            "w_change": {a: float(w_final[i] - prev_w[a]) for i, a in enumerate(assets)} if prev_w else None
        },
        "risk": {
            "risk_contribution": {a: risk_contrib["risk_contribution"][i] for i, a in enumerate(assets)},
            "risk_pct": {a: risk_contrib["risk_pct"][i] for i, a in enumerate(assets)},
            "marginal_risk": {a: risk_contrib["marginal_risk"][i] for i, a in enumerate(assets)},
            "port_vol_primary": risk_contrib["port_vol"],
            "port_vol_annual": risk_contrib["port_vol"] * np.sqrt(252 / pr["T"]),
            "hhi": risk_contrib["hhi"],
            "var_cvar": var_cvar
        },
        "constraints": {a: constraint_prox[i] for i, a in enumerate(assets)},
        "views": {
            v["id"]: {
                "name": v["name"],
                "P": v["P"],
                **{f"Q_{hz}": v.get(f"Q_{hz}", v.get("Q_3m")) for hz in horizons_results.keys()},
                **{f"conf_{hz}": v.get(f"conf_{hz}", v.get("conf")) for hz in horizons_results.keys()},
            }
            for v in views_data["views"]
        },
        "view_changes": view_changes
    }


def save_json_output(data: dict, output_dir: str = "reports") -> Path:
    """JSON 리포트 저장"""
    Path(output_dir).mkdir(exist_ok=True)
    kst = ZoneInfo("Asia/Seoul")
    now = datetime.now(kst)
    filename = now.strftime("%Y%m%d_%H%M%S_bl_report.json")
    filepath = Path(output_dir) / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    return filepath


# ============================================================
# 9. 리포트 생성 (멀티 지평 확장판)
# ============================================================
def generate_report(
    config, views_data, prices, Sigma_daily, horizons_results,
    primary_horizon, w_final, minsky_state, view_changes,
    divergence=None, risk_contrib=None, constraint_prox=None,
    var_cvar=None, prev_weights=None
) -> str:
    """한국어 리포트 생성 (v5.0-MH 멀티 지평 확장판)"""
    kst = ZoneInfo("Asia/Seoul")
    now = datetime.now(kst)
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S KST")

    assets = config["assets"]
    primary = primary_horizon
    pr = horizons_results[primary]
    Sigma_primary = pr["Sigma"]
    T_primary = pr["T"]
    w_star = pr["w_star"]
    pi = pr["pi"]
    mu_bl = pr["mu_bl"]
    lb = np.array([config["constraints"][a][0] for a in assets])
    ub = np.array([config["constraints"][a][1] for a in assets])

    port_vol_primary = np.sqrt(w_final @ Sigma_primary @ w_final)
    port_vol_annual = port_vol_primary * np.sqrt(252 / T_primary)

    hz_names = list(horizons_results.keys())

    report = f"""
================================================================================
        BL + 민스키 오버레이 일간 리포트 (v5.0-MH)
        멀티 지평: {', '.join(hz_names)} | Primary: {primary}
================================================================================
생성 시각: {timestamp}

--------------------------------------------------------------------------------
1. 시장 데이터
--------------------------------------------------------------------------------
최근 종가 ({prices.index[-1].strftime('%Y-%m-%d')}):
"""
    for i, asset in enumerate(assets):
        report += f"  {asset}: {prices[asset].iloc[-1]:,.2f}\n"

    report += f"""
--------------------------------------------------------------------------------
2. 공분산 (멀티 지평)
--------------------------------------------------------------------------------
일간 변동성 (σ_daily):
"""
    for i, asset in enumerate(assets):
        report += f"  {asset}: {np.sqrt(Sigma_daily[i,i])*100:.4f}%\n"

    # 지평별 변동성 비교 테이블
    report += f"\n지평별 변동성 비교:\n"
    header = f"  {'자산':>5s}"
    for hz in hz_names:
        T = horizons_results[hz]["T"]
        label = f"σ_{hz}({T}d)"
        header += f"  {label:>12s}"
    report += header + "\n"

    for i, asset in enumerate(assets):
        row = f"  {asset:>5s}"
        for hz in hz_names:
            vol = np.sqrt(horizons_results[hz]["Sigma"][i, i])
            row += f"  {vol*100:11.2f}%"
        report += row + "\n"

    # 상관관계 행렬 (primary)
    corr = compute_correlation_matrix(Sigma_primary)
    report += f"""
상관관계 행렬 ({primary}):
"""
    header = "         " + "  ".join(f"{a:>7s}" for a in assets)
    report += header + "\n"
    for i, a in enumerate(assets):
        row = f"  {a:>5s}  " + "  ".join(f"{corr[i,j]:7.3f}" for j in range(len(assets)))
        report += row + "\n"

    report += """
--------------------------------------------------------------------------------
3. 뷰 설정 (views.yaml, 멀티 지평)
--------------------------------------------------------------------------------
"""
    for v in views_data["views"]:
        report += f"  [{v['id']}] {v['name']}\n"
        report += f"       P={v['P']}\n"
        for hz in hz_names:
            q_key = f"Q_{hz}"
            c_key = f"conf_{hz}"
            q_val = v.get(q_key, v.get("Q_3m", "N/A"))
            c_val = v.get(c_key, v.get("conf", "N/A"))
            marker = " ◀ primary" if hz == primary else ""
            if isinstance(q_val, (int, float)) and isinstance(c_val, (int, float)):
                report += f"       {hz}: Q={q_val*100:.4f}%, conf={c_val*100:.0f}%{marker}\n"
            else:
                report += f"       {hz}: Q={q_val}, conf={c_val}{marker}\n"

    report += """
뷰 변경사항 (vs 이전):
"""
    for change in view_changes:
        report += f"  {change}\n"

    # --- 지평별 BL 기대수익 비교 ---
    report += f"""
--------------------------------------------------------------------------------
4. Black-Litterman 결과 (멀티 지평 비교)
--------------------------------------------------------------------------------
"""
    # 4a: π 비교
    report += f"시장균형 기대수익 π (지평별):\n"
    header = f"  {'자산':>5s}"
    for hz in hz_names:
        marker = "*" if hz == primary else " "
        header += f"  {'π_'+hz+marker:>10s}"
    report += header + "\n"
    for i, asset in enumerate(assets):
        row = f"  {asset:>5s}"
        for hz in hz_names:
            row += f"  {horizons_results[hz]['pi'][i]*100:9.4f}%"
        report += row + "\n"

    # 4b: μ_BL 비교
    report += f"\nBL 사후 기대수익 μ_BL (지평별):\n"
    header = f"  {'자산':>5s}"
    for hz in hz_names:
        marker = "*" if hz == primary else " "
        header += f"  {'μ_'+hz+marker:>10s}"
    report += header + "\n"
    for i, asset in enumerate(assets):
        row = f"  {asset:>5s}"
        for hz in hz_names:
            row += f"  {horizons_results[hz]['mu_bl'][i]*100:9.4f}%"
        report += row + "\n"

    report += f"\n  (* = primary horizon: {primary})\n"

    # 4c: π vs μ_BL 괴리 분석 (primary)
    if divergence:
        report += f"""
π vs μ_BL 괴리 분석 ({primary}):
"""
        for i, asset in enumerate(assets):
            d = divergence[i]
            report += f"  {asset}: π={d['pi']*100:+.4f}% → μ_BL={d['mu_bl']*100:+.4f}%  "
            report += f"괴리={d['diff_pct']:+.4f}%p  z={d['z_score']:+.2f} ({d['direction']} {d['strength']})\n"

    # --- 지평별 w* 비교 ---
    report += f"""
--------------------------------------------------------------------------------
5. 최적화 결과 (QP, 멀티 지평 비교)
--------------------------------------------------------------------------------
"""
    header = f"  {'자산':>5s}"
    for hz in hz_names:
        marker = "*" if hz == primary else " "
        header += f"  {'w*_'+hz+marker:>9s}"
    report += header + "\n"
    for i, asset in enumerate(assets):
        row = f"  {asset:>5s}"
        for hz in hz_names:
            row += f"  {horizons_results[hz]['w_star'][i]*100:8.2f}%"
        report += row + "\n"

    # 합계
    row = f"  {'합계':>5s}"
    for hz in hz_names:
        row += f"  {horizons_results[hz]['w_star'].sum()*100:8.2f}%"
    report += row + "\n"

    report += f"""
--------------------------------------------------------------------------------
6. 민스키 오버레이 (primary: {primary})
--------------------------------------------------------------------------------
현재 상태: {minsky_state}
GLD 타겟: {config['minsky']['gld_targets'][minsky_state]*100:.0f}%

w_final (최종 비중, {primary} 기준):
"""
    for i, asset in enumerate(assets):
        report += f"  {asset}: {w_final[i]*100:.2f}%\n"
    report += f"  합계: {w_final.sum()*100:.2f}%\n"

    # --- 확장 포트폴리오 진단 ---
    report += f"""
--------------------------------------------------------------------------------
7. 포트폴리오 진단 (확장, {primary} 기준)
--------------------------------------------------------------------------------
예상 {primary} 변동성: {port_vol_primary*100:.2f}%
예상 연간 변동성:  {port_vol_annual*100:.2f}%  (√(252/{T_primary}) 연환산)
"""

    # 7a: VaR/CVaR
    if var_cvar:
        report += f"""
  [7a] VaR / CVaR ({primary}, 파라메트릭)
"""
        report += f"  포트폴리오 기대수익: {var_cvar['mu_p']*100:.4f}%\n"
        for cl_key in ["95%", "99%"]:
            if cl_key in var_cvar:
                v = var_cvar[cl_key]
                report += f"  {cl_key} VaR:  {v['VaR_pct']:.2f}%  |  CVaR: {v['CVaR_pct']:.2f}%\n"

    # 7b: 위험기여도
    if risk_contrib:
        report += """
  [7b] 위험기여도 (Risk Contribution)
"""
        report += f"  {'자산':>5s}  {'비중':>7s}  {'한계위험':>8s}  {'위험기여':>8s}  {'기여비율':>8s}\n"
        for i, asset in enumerate(assets):
            report += f"  {asset:>5s}  {w_final[i]*100:6.2f}%  "
            report += f"{risk_contrib['marginal_risk'][i]*100:7.4f}%  "
            report += f"{risk_contrib['risk_contribution'][i]*100:7.4f}%  "
            report += f"{risk_contrib['risk_pct'][i]*100:7.2f}%\n"
        report += f"  HHI (집중도): {risk_contrib['hhi']:.4f}"
        if risk_contrib['hhi'] > 0.35:
            report += "  ⚠ 위험 집중도 높음"
        report += "\n"

    # 7c: 제약조건 근접도
    if constraint_prox:
        report += """
  [7c] 제약조건 근접도
"""
        report += f"  {'자산':>5s}  {'비중':>7s}  {'하한':>6s}  {'상한':>6s}  {'활용도':>6s}  {'상태':>15s}\n"
        for i, asset in enumerate(assets):
            cp_i = constraint_prox[i]
            report += f"  {asset:>5s}  {cp_i['weight']*100:6.2f}%  "
            report += f"{cp_i['lb']*100:5.1f}%  {cp_i['ub']*100:5.1f}%  "
            report += f"{cp_i['utilization']*100:5.1f}%  {cp_i['status']}\n"

    # 7d: 전회 비중 대비 변화
    prev_w = None
    if prev_weights and "weights" in prev_weights:
        prev_w = prev_weights["weights"].get("w_final")

    if prev_w:
        report += """
  [7d] 전회 대비 비중 변화
"""
        report += f"  {'자산':>5s}  {'전회':>7s}  {'현재':>7s}  {'변화':>8s}  {'판단'}\n"
        for i, asset in enumerate(assets):
            pw = prev_w.get(asset, 0)
            cw = w_final[i]
            diff_val = cw - pw
            judgment = ""
            if abs(diff_val) > 0.03:
                judgment = "⚠ ±3% 초과"
            elif abs(diff_val) > 0.02:
                judgment = "주의"
            report += f"  {asset:>5s}  {pw*100:6.2f}%  {cw*100:6.2f}%  {diff_val*100:+7.2f}%p  {judgment}\n"
    else:
        report += """
  [7d] 전회 대비 비중 변화: (이전 JSON 리포트 없음 — 다음 실행부터 추적)
"""

    # 8: 비중 변화 (w* → w_final)
    report += f"""
--------------------------------------------------------------------------------
8. 비중 변화 (w*_{primary} → w_final, 민스키 오버레이 효과)
--------------------------------------------------------------------------------
"""
    for i, asset in enumerate(assets):
        diff_val = (w_final[i] - w_star[i]) * 100
        sign = "+" if diff_val >= 0 else ""
        report += f"  {asset}: {sign}{diff_val:.2f}%p\n"

    report += """
================================================================================
"""
    return report


def save_report(report: str, output_dir: str = "reports"):
    """리포트 파일 저장"""
    Path(output_dir).mkdir(exist_ok=True)

    kst = ZoneInfo("Asia/Seoul")
    now = datetime.now(kst)
    filename = now.strftime("%Y%m%d_%H%M%S_bl_report.txt")
    filepath = Path(output_dir) / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    return filepath


# ============================================================
# 메인 실행
# ============================================================
def main():
    print("\n" + "="*60)
    print("  BL + 민스키 오버레이 포트폴리오 최적화 (v5.0-MH)")
    print("  멀티 지평(Multi-Horizon) 지원")
    print("="*60 + "\n")

    # 1. 설정 로드
    print("[1/9] 설정 로드...")
    config = load_config()
    views_data = load_views()

    assets = config["assets"]
    lam = config["ewma"]["lambda"]
    init_window = config["ewma"]["init_window"]
    tau = config["bl"]["tau"]
    delta = config["bl"]["delta"]
    horizons = config["bl"]["horizons"]         # {"1m": 21, "3m": 63, "6m": 126}
    primary_horizon = config["bl"]["primary_horizon"]  # "3m"

    w_mkt = np.array(config["w_mkt"])
    lb = np.array([config["constraints"][a][0] for a in assets])
    ub = np.array([config["constraints"][a][1] for a in assets])

    minsky_state = config["minsky"]["current_state"]
    gld_target = config["minsky"]["gld_targets"][minsky_state]

    print(f"  자산: {assets}")
    print(f"  뷰 수: {len(views_data['views'])}개")
    print(f"  지평: {horizons} (primary: {primary_horizon})")
    print(f"  민스키 상태: {minsky_state} (GLD 타겟: {gld_target*100:.0f}%)")

    # 2. 뷰 변경사항 확인
    print("\n[2/9] 뷰 변경사항 확인...")
    prev_views = load_previous_views()
    view_changes = diff_views(prev_views, views_data, horizons=list(horizons.keys()))
    for change in view_changes:
        print(f"  {change}")

    # 3. 데이터 수집
    print("\n[3/9] 가격 데이터 수집...")
    prices = fetch_prices(config)

    # 4. 수익률 계산
    print("\n[4/9] 로그수익률 계산...")
    returns = compute_returns(prices)
    print(f"  수익률 데이터: {len(returns)}일")

    # 5. EWMA 공분산
    print("\n[5/9] EWMA 공분산 계산...")
    Sigma_daily = compute_ewma_cov(returns, lam, init_window)
    print(f"  λ={lam}, 초기윈도우={init_window}일")

    # 6. 멀티 지평 BL 계산
    print("\n[6/9] Black-Litterman 계산 (멀티 지평)...")
    horizons_results = {}
    w_final = None

    for hz_name, T in horizons.items():
        Sigma_T = T * Sigma_daily
        P, Q, conf, Omega = build_view_matrices(views_data, Sigma_T, horizon=hz_name)
        mu_bl, pi_hz = black_litterman(Sigma_T, w_mkt, P, Q, Omega, tau, delta)

        # 7. QP 최적화
        w_star = optimize_qp(mu_bl, Sigma_T, delta, lb, ub)

        horizons_results[hz_name] = {
            "T": T,
            "Sigma": Sigma_T,
            "pi": pi_hz,
            "mu_bl": mu_bl,
            "w_star": w_star,
            "P": P,
            "Q": Q,
            "conf": conf,
        }

        marker = " ◀ PRIMARY" if hz_name == primary_horizon else ""
        print(f"  [{hz_name}] T={T}d, τ={tau}, δ={delta}{marker}")

    # 8. 민스키 오버레이 (primary만)
    print(f"\n[7/9] QP 최적화 완료 (전 지평)")

    print(f"\n[8/9] 민스키 오버레이 적용 ({primary_horizon})...")
    pr = horizons_results[primary_horizon]
    w_final = apply_minsky_overlay(pr["w_star"], gld_target, lb, ub)

    # 9. 확장 진단 (primary 기준)
    print("\n[9/9] 확장 진단 계산...")
    Sigma_primary = pr["Sigma"]
    divergence = compute_pi_bl_divergence(pr["pi"], pr["mu_bl"], Sigma_primary)
    risk_contrib = compute_risk_contribution(w_final, Sigma_primary)
    constraint_prox = compute_constraint_proximity(w_final, lb, ub)
    var_cvar = compute_var_cvar(w_final, pr["mu_bl"], Sigma_primary)
    prev_weights = load_previous_weights()
    print("  π vs μ_BL 괴리, 위험기여도, 제약근접도, VaR/CVaR 완료")

    # 뷰 스냅샷 저장
    print("\n[+] 뷰 스냅샷 저장...")
    snapshot_path = save_views_snapshot()
    print(f"  저장: {snapshot_path}")

    # 리포트 생성 (멀티 지평 확장판)
    print("\n[+] 리포트 생성...")
    report = generate_report(
        config, views_data, prices, Sigma_daily, horizons_results,
        primary_horizon, w_final, minsky_state, view_changes,
        divergence=divergence,
        risk_contrib=risk_contrib,
        constraint_prox=constraint_prox,
        var_cvar=var_cvar,
        prev_weights=prev_weights
    )

    print(report)

    filepath = save_report(report)
    print(f"리포트 저장: {filepath}")

    # JSON 구조화 출력 (에이전트 파싱용)
    print("\n[+] JSON 출력 생성...")
    json_data = build_json_output(
        config, views_data, prices, Sigma_daily, horizons_results,
        primary_horizon, w_final, minsky_state, view_changes,
        divergence, risk_contrib, constraint_prox, var_cvar, prev_weights
    )
    json_path = save_json_output(json_data)
    print(f"JSON 저장: {json_path}")

    return {
        "prices": prices,
        "Sigma_daily": Sigma_daily,
        "horizons_results": horizons_results,
        "primary_horizon": primary_horizon,
        "w_final": w_final,
        "minsky_state": minsky_state,
        "view_changes": view_changes,
        "divergence": divergence,
        "risk_contrib": risk_contrib,
        "constraint_prox": constraint_prox,
        "var_cvar": var_cvar,
        "json_data": json_data
    }


if __name__ == "__main__":
    main()
