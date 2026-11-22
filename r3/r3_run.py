"""R3 solar: two validation views (new houses / new days of known houses), LightGBM L1 on energy/base with weight=base.

usage: python r3_run.py <tag> [--seeds N] [--no-house]
Writes outputs/r3/oof_A_<tag>.npy, oof_B_<tag>.npy, test_<tag>.npy, sub_<tag>.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find common.py at the repo root

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common import DATA, OUT, mae, validate_submission
from r3_features import build_features, load_r3

K_SHRINK = 20.0
W_B, W_A = 0.765, 0.235  # share of test rows from known houses / new houses


def house_factor(h, d, e, b, known, glob):
    """Shrunk energy/base ratio of the house from `known` daytime rows, leave-one-(house,day)-out for rows of days in `known`."""
    nh, nd = h.max() + 1, d.max() + 1
    Se = np.bincount(h[known], weights=e[known], minlength=nh)
    Sb = np.bincount(h[known], weights=b[known], minlength=nh)
    key = h.astype(np.int64) * nd + d
    Sed = np.bincount(key[known], weights=e[known], minlength=nh * nd)
    Sbd = np.bincount(key[known], weights=b[known], minlength=nh * nd)
    se = Se[h] - Sed[key]
    sb = Sb[h] - Sbd[key]
    fac = (se + K_SHRINK * glob) / (sb + K_SHRINK)
    return np.where(Sb[h] > 0, fac, np.nan)


SPECS = [("hr10", "hour", (6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19)), ("hr", "hour", (6, 9, 12, 15, 19)), ("dc", "days_clean", (-1, 30, 90, 1e9)), ("cl", "cloud_cover", (-1, 0.3, 0.7, 2.0))]


def add_bin_factors(X, F, h, d, e, b, known, fac_house, specs):
    for name, col, edges in specs:
        hb = house_factor_bins(h, d, F[col].to_numpy(), e, b, known, fac_house, edges)
        for j in range(hb.shape[1]):
            X[f"hf_{name}{j}"] = hb[:, j]
            X[f"hfr_{name}{j}"] = hb[:, j] / fac_house
    return X


def bin_cols(specs):
    return [f"{p}_{n}{j}" for n, c, e in specs for j in range(len(e) - 1) for p in ("hf", "hfr")]


def house_factor_bins(h, d, hour, e, b, known, fac_house, edges=(6, 9, 12, 15, 19)):
    """Per-house, per-hour-bin factor (leave-one-day-out), shrunk toward the house's overall factor. Returns (n, nbins)."""
    nb = len(edges) - 1
    bidx = np.clip(np.digitize(hour, edges) - 1, 0, nb - 1)
    out = np.full((len(h), nb), np.nan)
    for j in range(nb):
        kj = known & (bidx == j)
        nh, nd = h.max() + 1, d.max() + 1
        Se = np.bincount(h[kj], weights=e[kj], minlength=nh)
        Sb = np.bincount(h[kj], weights=b[kj], minlength=nh)
        key = h.astype(np.int64) * nd + d
        Sed = np.bincount(key[kj], weights=e[kj], minlength=nh * nd)
        Sbd = np.bincount(key[kj], weights=b[kj], minlength=nh * nd)
        se, sb = Se[h] - Sed[key], Sb[h] - Sbd[key]
        out[:, j] = (se + 10.0 * fac_house) / (sb + 10.0)
    return out


def lgb_params(seed):
    return dict(objective="l1", learning_rate=0.05, num_leaves=63, min_child_samples=40, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, n_jobs=10, seed=seed, metric="l1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--no-house", action="store_true")
    ap.add_argument("--bins", action="store_true")
    ap.add_argument("--specs", default="hr10")
    ap.add_argument("--obj-a", default="l1")
    ap.add_argument("--obj-b", default="l2")
    ap.add_argument("--mask", type=float, default=0.25)
    a = ap.parse_args()
    t0 = time.time()
    SP = [sp for sp in SPECS if sp[0] in a.specs.split(",")]
    df = load_r3()
    F, sg = build_features(df)
    tr = np.where(df.is_train.to_numpy() == 1)[0]
    te = np.where(df.is_train.to_numpy() == 0)[0]
    e = df["energy_kwh"].to_numpy(float)
    b = F["base_phys"].to_numpy(float)
    up = F["sun_up"].to_numpy() == 1
    h = pd.factorize(df["house_id"])[0]
    d = pd.factorize(df["date"])[0]
    elev = F["elev"].to_numpy()
    day_rows = up & (elev > 15) & (b > 0.05)
    glob = np.nansum(e[tr][day_rows[tr]]) / np.nansum(b[tr][day_rows[tr]])
    print(f"features {F.shape[1]} cols, global ratio {glob:.4f}; base geometry ok ({time.time()-t0:.1f}s)", flush=True)
    y_tr = e[tr]
    print("naive MAE (global ratio x base, night 0):", round(mae(y_tr, np.where(up[tr], glob * b[tr], 0.0)), 4))
    print("night rows nonzero in train:", int(((~up[tr]) & (y_tr > 0)).sum()))
    rng = np.random.RandomState(0)

    def design(src_mask, known_houses_mask=None, mask_frac=0.0, seed=0):
        """Feature matrix with house factor computed from rows in src_mask (daytime only)."""
        X = F.copy()
        if not a.no_house:
            src = src_mask & day_rows
            fac = house_factor(h, d, np.nan_to_num(e), b, src, glob)
            if mask_frac > 0:
                hs = np.unique(h[src_mask])
                r = np.random.RandomState(seed)
                drop = r.choice(hs, int(len(hs) * mask_frac), replace=False)
                fac = np.where(np.isin(h, drop), np.nan, fac)
            X["house_fac"] = fac
        return X

    def fit_predict(Xtr, ytr, wtr, Xva_list, rounds=None, vset=None, seed=0):
        dtr = lgb.Dataset(Xtr, ytr, weight=wtr)
        if rounds is None:
            dva = lgb.Dataset(vset[0], vset[1], weight=vset[2], reference=dtr)
            m = lgb.train(lgb_params(seed), dtr, 5000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
        else:
            m = lgb.train(lgb_params(seed), dtr, rounds)
        return m, [m.predict(X, num_iteration=m.best_iteration if rounds is None else None) for X in Xva_list]

    # ---------------- two validation views, two model types ----------------
    # A: houses unseen -> model WITHOUT house factor, GroupKFold by house
    # B: known houses, unseen days -> model WITH house factor, GroupKFold by date
    def cols(with_house):
        c = list(F.columns)
        if with_house:
            c += ["house_fac"]
            if a.bins:
                c += bin_cols(SP)
        return c

    def fit(Xt, yt, wt, vset, seed, rounds=None, obj="l1"):
        if obj == "l2":
            wt = wt ** 2
            if vset is not None:
                vset = (vset[0], vset[1], vset[2] ** 2)
        dtr = lgb.Dataset(Xt, yt, weight=wt)
        p = lgb_params(seed)
        p["objective"] = obj
        if rounds is None:
            dva = lgb.Dataset(vset[0], vset[1], weight=vset[2], reference=dtr)
            return lgb.train(p, dtr, 6000, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
        return lgb.train(p, dtr, rounds)

    oofs, iters = {}, {}
    for view in ["A", "B"]:
        oof = np.zeros(len(tr))
        groups = h[tr] if view == "A" else d[tr]
        its = []
        for k, (itr, iva) in enumerate(GroupKFold(5).split(tr, groups=groups)):
            src_mask = np.zeros(len(df), bool)
            src_mask[tr[itr]] = True
            X = F.copy()
            if view == "B":
                X["house_fac"] = house_factor(h, d, np.nan_to_num(e), b, src_mask & day_rows, glob)
                if a.bins:
                    X = add_bin_factors(X, F, h, d, np.nan_to_num(e), b, src_mask & day_rows, X["house_fac"].to_numpy(), SP)
            tm = up[tr[itr]] & (b[tr[itr]] > 1e-6)
            vm = up[tr[iva]] & (b[tr[iva]] > 1e-6)
            c = cols(view == "B")
            Xt, Xv = X.iloc[tr[itr]][tm][c], X.iloc[tr[iva]][vm][c]
            yt, wt = e[tr[itr]][tm] / b[tr[itr]][tm], b[tr[itr]][tm]
            yv, wv = e[tr[iva]][vm] / b[tr[iva]][vm], b[tr[iva]][vm]
            if k == 0: print(f"view {view}: {len(c)} features, e.g. {c[-4:]}", flush=True)
            m = fit(Xt, yt, wt, (Xv, yv, wv), seed=k, obj=(a.obj_a if view == "A" else a.obj_b))
            pred = np.zeros(len(iva))
            pred[vm] = m.predict(Xv, num_iteration=m.best_iteration) * b[tr[iva]][vm]
            oof[iva] = pred
            its.append(m.best_iteration)
            print(f"view {view} fold{k}: iters={m.best_iteration} MAE={mae(e[tr[iva]], pred):.5f} ({time.time()-t0:.0f}s)", flush=True)
        oofs[view] = oof
        iters[view] = int(np.mean(its))
        print(f"== view {view}: MAE={mae(y_tr, oof):.5f}", flush=True)
    score = W_B * mae(y_tr, oofs["B"]) + W_A * mae(y_tr, oofs["A"])
    print(f"weighted CV MAE = {W_B}*B + {W_A}*A = {score:.5f}  (A={mae(y_tr, oofs['A']):.5f}, B={mae(y_tr, oofs['B']):.5f})")

    # ---------------- final fit: known-house model (B) and new-house model (A) on all train rows ----------------
    src_all = np.zeros(len(df), bool)
    src_all[tr] = True
    fac_all = house_factor(h, d, np.nan_to_num(e), b, src_all & day_rows, glob)  # LODO for train rows, full for test rows
    X = F.copy()
    X["house_fac"] = fac_all
    if a.bins:
        X = add_bin_factors(X, F, h, d, np.nan_to_num(e), b, src_all & day_rows, fac_all, SP)
    known_house_test = np.isin(h[te], np.unique(h[tr]))
    tm = up[tr] & (b[tr] > 1e-6)
    pred_te = np.zeros(len(te))
    for view, sel in [("B", known_house_test), ("A", ~known_house_test)]:
        c = cols(view == "B")
        rounds = int(1.1 * iters[view])
        pv = np.zeros(len(te))
        for s_ in range(a.seeds):
            m = fit(X.iloc[tr][tm][c], e[tr][tm] / b[tr][tm], b[tr][tm], None, seed=100 + s_, rounds=rounds, obj=(a.obj_a if view == "A" else a.obj_b))
            pv += m.predict(X.iloc[te][c]) * b[te] * up[te]
        pred_te[sel] = (pv / a.seeds)[sel]
        if view == "B":
            imp = pd.Series(m.feature_importance("gain"), index=c)
    pred_te = np.clip(pred_te, 0, None)
    print("top features (known-house model):")
    print((imp / imp.sum()).sort_values(ascending=False).head(12).round(4).to_string())
    print(f"test rows: known-house {known_house_test.sum()}, new-house {(~known_house_test).sum()} ({(~known_house_test).mean():.3f})")
    print(f"test pred: mean {pred_te.mean():.4f} (train mean {y_tr.mean():.4f}); zeros {np.mean(pred_te == 0):.3f} (train {np.mean(y_tr == 0):.3f})")
    out = OUT / "r3"
    np.save(out / f"oof_A_{a.tag}.npy", oofs["A"])
    np.save(out / f"oof_B_{a.tag}.npy", oofs["B"])
    np.save(out / f"test_{a.tag}.npy", pred_te)
    np.save(out / "y_train.npy", y_tr)
    sample = pd.read_csv(DATA / "r3" / "sample_submission.csv")
    sub = pd.DataFrame({"id": df.loc[te, "id"].to_numpy(), "energy_kwh": pred_te}).set_index("id").loc[sample["id"]].reset_index()
    validate_submission(sub, sample, "r3")
    sub.to_csv(out / f"sub_{a.tag}.csv", index=False)
    print(f"saved sub_{a.tag}.csv  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
