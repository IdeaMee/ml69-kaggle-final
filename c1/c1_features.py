"""C1 fraud: label-free features (over train+test inputs) and out-of-fold label features."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find common.py at the repo root

import numpy as np
import pandas as pd

from common import DATA

BIG = 1e9  # group stride in the combined (group, time) sort key; t < 1e7 seconds


def load_c1():
    tr = pd.read_csv(DATA / "c1" / "train.csv")
    te = pd.read_csv(DATA / "c1" / "test.csv")
    df = pd.concat([tr.assign(is_train=1), te.assign(is_train=0, is_fraud=np.nan)], ignore_index=True)
    ts = pd.to_datetime(df["timestamp"])
    df["t"] = (ts - ts.min()).dt.total_seconds()
    df["hour"] = ts.dt.hour + ts.dt.minute / 60
    df["dow"] = ts.dt.dayofweek
    df["night"] = ((df["hour"] < 6) | (df["hour"] >= 23)).astype(int)
    df["day"] = (df["t"] // 86400).astype(int)
    for name, gap in [("s1", 1800), ("s2", 7200)]:
        o = df.sort_values(["user_id", "t"])
        newsess = o["user_id"].ne(o["user_id"].shift()) | (o["t"].diff() > gap)
        df[name] = newsess.cumsum().reindex(df.index).astype(int)
    return df


def _unsort(order, vals):
    out = np.empty(len(order), dtype=float)
    out[order] = vals
    return out


def window_feats(g, t, amt, prefix, prev_w=(), sym_w=(), sum_prev_w=()):
    """Counts/sums of OTHER rows of the same group within time windows (label-free)."""
    key = g.astype(np.float64) * BIG + t
    order = np.argsort(key, kind="stable")
    ks = key[order]
    n = len(ks)
    idx = np.arange(n)
    csum = np.concatenate([[0.0], np.cumsum(amt[order])])
    out = {}
    for w in prev_w:
        lo = np.searchsorted(ks, ks - w, "left")
        out[f"{prefix}_prev_{w}s"] = _unsort(order, idx - lo)
    for w in sum_prev_w:
        lo = np.searchsorted(ks, ks - w, "left")
        out[f"{prefix}_sumprev_{w}s"] = _unsort(order, csum[idx] - csum[lo])
    for w in sym_w:
        lo = np.searchsorted(ks, ks - w, "left")
        hi = np.searchsorted(ks, ks + w, "right")
        out[f"{prefix}_sym_{w}s"] = _unsort(order, hi - lo - 1)
    gs = g[order]
    same = gs[1:] == gs[:-1]
    d = np.diff(ks)
    prev_gap = np.r_[np.nan, np.where(same, d, np.nan)]
    next_gap = np.r_[np.where(same, d, np.nan), np.nan]
    out[f"{prefix}_gap_prev"] = _unsort(order, prev_gap)
    out[f"{prefix}_gap_next"] = _unsort(order, next_gap)
    return out


def label_free_features(df):
    f = pd.DataFrame(index=df.index)
    for c in ["hour", "dow", "night", "day", "account_age_days", "ip_risk_score", "promo_applied"]:
        f[c] = df[c]
    f["log_amount"] = np.log1p(df["amount"])
    f["ip_nan"] = df["ip_risk_score"].isna().astype(int)
    for c in ["channel", "merchant_category", "device_os", "province", "customer_segment", "app_version"]:
        f[c] = df[c].astype("category").cat.codes
    u = df["user_id"].astype("category").cat.codes.to_numpy()
    d = df["device_id"].astype("category").cat.codes.to_numpy()
    t = df["t"].to_numpy()
    amt = df["amount"].to_numpy(float)
    la = f["log_amount"]
    grp = df.groupby("user_id")
    f["u_n"] = grp["amount"].transform("size")
    f["u_med"] = grp["amount"].transform("median")
    f["u_mean_log"] = la.groupby(df["user_id"]).transform("mean")
    f["u_std_log"] = la.groupby(df["user_id"]).transform("std")
    f["amt_ratio_med"] = df["amount"] / f["u_med"]
    f["amt_z_log"] = (la - f["u_mean_log"]) / (f["u_std_log"] + 1e-6)
    f["amt_rank_user"] = grp["amount"].rank(pct=True)
    f["amt_max_user_ratio"] = df["amount"] / grp["amount"].transform("max")
    f["dev_nusers"] = df.groupby("device_id")["user_id"].transform("nunique")
    f["dev_n"] = df.groupby("device_id")["amount"].transform("size")
    f["u_ndev"] = grp["device_id"].transform("nunique")
    f["u_nprov"] = grp["province"].transform("nunique")
    cat_mean = la.groupby(df["merchant_category"]).transform("mean")
    f["amt_vs_cat"] = la - cat_mean
    ch_mean = la.groupby(df["channel"]).transform("mean")
    f["amt_vs_channel"] = la - ch_mean
    # first-time flags (chronological over train+test)
    srt = df.sort_values("t")
    for col, name in [("device_id", "dev"), ("province", "prov"), ("merchant_category", "merch"), ("channel", "chan"), ("device_os", "os")]:
        first = ~srt.duplicated(["user_id", col])
        f[f"first_{name}"] = first.reindex(df.index).astype(int)
    f["u_seq"] = srt.groupby("user_id").cumcount().reindex(df.index)
    f["u_seq_frac"] = f["u_seq"] / f["u_n"]
    for k, v in window_feats(u, t, amt, "u", prev_w=(600, 3600, 86400, 604800), sym_w=(600, 3600, 86400), sum_prev_w=(3600, 86400)).items():
        f[k] = v
    for k, v in window_feats(d, t, amt, "d", sym_w=(3600, 86400)).items():
        f[k] = v
    f["amt_vs_u_recent24"] = df["amount"] / (f["u_sumprev_86400s"] / np.maximum(f["u_prev_86400s"], 1) + 1)
    for name in ["s1", "s2"]:
        sg = df.groupby(name)
        f[f"{name}_n"] = sg["amount"].transform("size")
        f[f"{name}_dur"] = sg["t"].transform("max") - sg["t"].transform("min")
        f[f"{name}_pos"] = sg["t"].rank(method="first") / f[f"{name}_n"]
        f[f"{name}_maxratio"] = f["amt_ratio_med"].groupby(df[name]).transform("max")
        f[f"{name}_meanratio"] = f["amt_ratio_med"].groupby(df[name]).transform("mean")
        f[f"{name}_ndev"] = sg["device_id"].transform("nunique")
        f[f"{name}_firstdev"] = f["first_dev"].groupby(df[name]).transform("max")
        f[f"{name}_night"] = f["night"].groupby(df[name]).transform("max")
    # chronological (past-only) amount context per user, plus integer-amount flag
    srt2 = df.assign(_la=la).sort_values(["user_id", "t"])
    gp = srt2.groupby("user_id")["amount"]
    prev_max = gp.transform(lambda s: s.shift().cummax())
    prev_mean = srt2.groupby("user_id")["_la"].transform(lambda s: s.shift().expanding().mean())
    f["amt_vs_prevmax"] = (srt2["amount"] / prev_max).reindex(df.index)
    f["la_vs_prevmean"] = (srt2["_la"] - prev_mean).reindex(df.index)
    f["amt_is_int"] = (df["amount"] % 1 == 0).astype(int)
    f["u_rate_1h_vs_week"] = f["u_prev_3600s"] / (f["u_prev_604800s"] / 168 + 0.1)
    return f


def lab_feats(g, t, y, known, prefix, wins=(3600, 86400)):
    """Features from the labels of OTHER rows of the same group (self excluded). known: bool mask of rows whose
    label may be used. Everything is vectorised with a combined (group, time) sort key."""
    g = g.astype(np.int64)
    n = len(g)
    key = g.astype(np.float64) * BIG + t
    src = np.where(known)[0]
    o = src[np.argsort(key[src], kind="stable")]
    ks = key[o]
    ys = np.nan_to_num(y[o]).astype(float)
    ns = len(ks)
    cumf = np.concatenate([[0.0], np.cumsum(ys)])
    ng = int(g.max()) + 1
    tot_k = np.bincount(g[src], minlength=ng).astype(float)
    tot_f = np.bincount(g[src], weights=ys, minlength=ng)
    self_k = known.astype(float)
    self_f = np.where(known, np.nan_to_num(y), 0.0)
    out = {}
    out[f"{prefix}_nk"] = tot_k[g] - self_k
    out[f"{prefix}_nf"] = tot_f[g] - self_f
    out[f"{prefix}_frate"] = out[f"{prefix}_nf"] / np.maximum(out[f"{prefix}_nk"], 1)
    for w in wins:
        lo = np.searchsorted(ks, key - w, "left")
        hi = np.searchsorted(ks, key + w, "right")
        out[f"{prefix}_fr_{w}s"] = cumf[hi] - cumf[lo] - self_f
        out[f"{prefix}_kn_{w}s"] = (hi - lo) - self_k
    gk = g.astype(np.float64) * BIG
    p = np.searchsorted(ks, key, "left") - 1
    ok = (p >= 0)
    pc = np.clip(p, 0, ns - 1)
    ok &= ks[pc] >= gk
    out[f"{prefix}_prev_y"] = np.where(ok, ys[pc], np.nan)
    out[f"{prefix}_prev_gap"] = np.where(ok, key - ks[pc], np.nan)
    q = np.searchsorted(ks, key, "right")
    ok = q < ns
    qc = np.clip(q, 0, ns - 1)
    ok &= ks[qc] < gk + BIG
    out[f"{prefix}_next_y"] = np.where(ok, ys[qc], np.nan)
    out[f"{prefix}_next_gap"] = np.where(ok, ks[qc] - key, np.nan)
    fk = ks[ys == 1]
    nf = len(fk)
    if nf:
        p = np.searchsorted(fk, key, "left") - 1
        ok = p >= 0
        pc = np.clip(p, 0, nf - 1)
        ok &= fk[pc] >= gk
        out[f"{prefix}_pf_gap"] = np.where(ok, key - fk[pc], np.nan)
        q = np.searchsorted(fk, key, "right")
        ok = q < nf
        qc = np.clip(q, 0, nf - 1)
        ok &= fk[qc] < gk + BIG
        out[f"{prefix}_nf_gap"] = np.where(ok, fk[qc] - key, np.nan)
    else:
        out[f"{prefix}_pf_gap"] = np.full(n, np.nan)
        out[f"{prefix}_nf_gap"] = np.full(n, np.nan)
    return out


def label_features(df, known):
    """All label-derived features for every row, using labels of rows where `known` is True (self excluded)."""
    t = df["t"].to_numpy()
    y = df["is_fraud"].to_numpy(float)
    u = df["user_id"].astype("category").cat.codes.to_numpy()
    d = df["device_id"].astype("category").cat.codes.to_numpy()
    out = {}
    out.update(lab_feats(u, t, y, known, "lu", wins=(600, 3600, 86400)))
    out.update(lab_feats(d, t, y, known, "ld", wins=(3600, 86400)))
    out.update(lab_feats(df["s1"].to_numpy(), t, y, known, "ls1", wins=(1800,)))
    out.update(lab_feats(df["s2"].to_numpy(), t, y, known, "ls2", wins=(1800,)))
    return pd.DataFrame(out, index=df.index)
