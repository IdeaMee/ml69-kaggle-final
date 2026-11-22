"""C1 fraud: repeated stratified K-fold with per-fold out-of-fold label features, LightGBM, pooled-OOF threshold.

usage: python c1_run.py <tag> [--repeats N] [--no-label] [--seed S]
Writes outputs/c1/oof_<tag>.npy (mean over repeats), test_<tag>.npy (probabilities), sub_<tag>.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find common.py at the repo root

import argparse
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import DATA, OUT, best_threshold, f1_binary, validate_submission
from c1_features import label_features, label_free_features, load_c1

CATS = ["channel", "merchant_category", "device_os", "province", "customer_segment", "app_version"]


def lgb_params(seed):
    return dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=20, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=10, seed=seed,
                metric="average_precision")


def partner_max(sess, p, mask):
    """For every row in `mask` (unlabeled in this fold): max of predictions of the OTHER masked rows of its session (NaN if none)."""
    d = pd.DataFrame({"s": sess[mask], "p": p[mask]})
    d["r"] = d.groupby("s")["p"].rank(method="first", ascending=False)
    top1 = d["p"].where(d["r"] == 1).groupby(d["s"]).transform("max")
    top2 = d["p"].where(d["r"] == 2).groupby(d["s"]).transform("max")
    pm = np.where(d["r"] == 1, top2, top1)
    out = np.full(len(sess), np.nan)
    out[np.where(mask)[0]] = pm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--no-label", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    df = load_c1()
    base = label_free_features(df)
    tr_idx = np.where(df["is_train"].to_numpy() == 1)[0]
    te_idx = np.where(df["is_train"].to_numpy() == 0)[0]
    y = df["is_fraud"].to_numpy(float)
    ytr = y[tr_idx].astype(int)
    print(f"features label-free: {base.shape[1]} cols, build {time.time()-t0:.1f}s", flush=True)

    oof = np.zeros((a.repeats, len(tr_idx)))
    pm_oof = np.full((a.repeats, len(tr_idx)), np.nan)
    pm_test = np.zeros(len(te_idx))
    sess1 = df["s1"].to_numpy()
    test_pred = np.zeros(len(te_idx))
    n_models = 0
    fold_info = []
    imp = None
    for rep in range(a.repeats):
        skf = StratifiedKFold(a.folds, shuffle=True, random_state=a.seed + rep)
        for k, (itr, iva) in enumerate(skf.split(tr_idx, ytr)):
            if a.no_label:
                X = base
            else:
                # validation/test rows: label features from ALL rows of this training fold.
                # training rows: inner K-fold (a row never sees its own label nor the labels of its inner fold),
                # which avoids the leave-one-out leakage (fraud rows would otherwise always be 1 lower).
                known = np.zeros(len(df), bool)
                known[tr_idx[itr]] = True
                L = label_features(df, known)
                inner = StratifiedKFold(a.inner, shuffle=True, random_state=a.seed + 7 * k + rep)
                for ia, ib in inner.split(itr, ytr[itr]):
                    kn = np.zeros(len(df), bool)
                    kn[tr_idx[itr[ia]]] = True
                    Li = label_features(df, kn)
                    L.iloc[tr_idx[itr[ib]]] = Li.iloc[tr_idx[itr[ib]]].to_numpy()
                X = pd.concat([base, L], axis=1)
            cat = [c for c in CATS if c in X.columns]
            dtr = lgb.Dataset(X.iloc[tr_idx[itr]], ytr[itr], categorical_feature=cat, free_raw_data=False)
            dva = lgb.Dataset(X.iloc[tr_idx[iva]], ytr[iva], categorical_feature=cat, reference=dtr)
            if a.rounds > 0:  # fixed rounds: no use of validation labels, avoids the noisy early stopping on ~90 positives
                m = lgb.train(lgb_params(a.seed + 100 * rep + k), dtr, a.rounds)
                m.best_iteration = a.rounds
            else:
                m = lgb.train(lgb_params(a.seed + 100 * rep + k), dtr, 3000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(150, verbose=False)])
            pv = m.predict(X.iloc[tr_idx[iva]], num_iteration=m.best_iteration)
            pt = m.predict(X.iloc[te_idx], num_iteration=m.best_iteration)
            oof[rep, iva] = pv
            test_pred += pt
            pu = np.full(len(df), np.nan)
            pu[tr_idx[iva]] = pv
            pu[te_idx] = pt
            pmx = partner_max(sess1, pu, ~np.isnan(pu))
            pm_oof[rep, iva] = pmx[tr_idx[iva]]
            pm_test += np.nan_to_num(pmx[te_idx], nan=0.0)
            n_models += 1
            ap_k = average_precision_score(ytr[iva], oof[rep, iva])
            fold_info.append((rep, k, m.best_iteration, ap_k))
            g = pd.Series(m.feature_importance("gain"), index=X.columns)
            imp = g if imp is None else imp + g
            print(f"rep{rep} fold{k}: best_iter={m.best_iteration} AP={ap_k:.4f} ({time.time()-t0:.0f}s)", flush=True)
    test_pred /= n_models
    pm_test /= n_models
    oof_mean = oof.mean(0)
    pm_mean = np.nan_to_num(pm_oof, nan=0.0).mean(0)
    best_w, best_f = None, -1
    print("partner propagation q = max(p, w * max_partner_p):")
    for w in [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]:
        q = np.maximum(oof_mean, w * pm_mean)
        thr_w, f_w = best_threshold(ytr, q, smooth=0.003)
        print(f"  w={w}: F1={f_w:.4f} thr={thr_w:.3f}")
        if f_w > best_f + 1e-9:
            best_w, best_f = w, f_w
    f_w0 = best_threshold(ytr, oof_mean, smooth=0.003)[1]
    if best_f - f_w0 < 0.003:  # gain within noise -> keep the plain model
        best_w = 0.0
    print("best w:", best_w, "(0.0 = propagation not used; gain must exceed 0.003)")
    thr, f1 = best_threshold(ytr, oof_mean)
    thr_s, f1_s = best_threshold(ytr, oof_mean, smooth=0.003)
    print(f"\nOOF: AP={average_precision_score(ytr, oof_mean):.4f} AUC={roc_auc_score(ytr, oof_mean):.4f}")
    print(f"best F1={f1:.4f} @thr={thr:.4f} | plateau-centre thr={thr_s:.4f} F1={f1_s:.4f} | F1@0.5={f1_binary(ytr, oof_mean >= 0.5):.4f}")
    for rep in range(a.repeats):
        print(f"  repeat {rep}: best F1={best_threshold(ytr, oof[rep])[1]:.4f}")
    # per-fold F1 at the pooled threshold (noise estimate)
    skf = StratifiedKFold(a.folds, shuffle=True, random_state=a.seed)
    fs = [f1_binary(ytr[iva], oof[0, iva] >= thr_s) for _, iva in skf.split(tr_idx, ytr)]
    print("per-fold F1 at plateau thr (repeat0):", np.round(fs, 4), "mean", round(float(np.mean(fs)), 4), "std", round(float(np.std(fs)), 4))
    np.save(out_dir_placeholder := OUT / "c1" / f"pm_oof_{a.tag}.npy", pm_mean)
    np.save(OUT / "c1" / f"pm_test_{a.tag}.npy", pm_test)
    q_test = np.maximum(test_pred, best_w * pm_test)
    q_oof = np.maximum(oof_mean, best_w * pm_mean)
    thr_q, f1_q = best_threshold(ytr, q_oof, smooth=0.003)
    print(f"with propagation (w={best_w}): OOF F1={f1_q:.4f} thr={thr_q:.4f}; test positives {(q_test >= thr_q).sum()}")
    pred = (q_test >= thr_q).astype(int)
    print(f"test predicted positives: {pred.sum()} (expected ~386); OOF positives at thr: {(oof_mean >= thr_s).sum()} of {ytr.sum()} true")
    print("top features:\n", (imp / imp.sum()).sort_values(ascending=False).head(25).round(4).to_string())

    out = OUT / "c1"
    np.save(out / f"oof_{a.tag}.npy", oof_mean)
    np.save(out / f"test_{a.tag}.npy", test_pred)
    np.save(out / "y_train.npy", ytr)
    sample = pd.read_csv(DATA / "c1" / "sample_submission.csv")
    sub = pd.DataFrame({"id": df.loc[te_idx, "id"].to_numpy(), "is_fraud": pred}).set_index("id").loc[sample["id"]].reset_index()
    validate_submission(sub, sample, "c1")
    sub.to_csv(out / f"sub_{a.tag}.csv", index=False)
    pd.Series({"thr_plateau": thr_s, "thr_best": thr, "f1_oof": f1_s}).to_json(out / f"meta_{a.tag}.json")
    print(f"saved sub_{a.tag}.csv  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
