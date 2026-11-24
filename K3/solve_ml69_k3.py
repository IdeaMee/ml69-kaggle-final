"""
ML69_K3 Smart Meter Daily Routine Clustering Pipeline (v5-Fast)
- Fully Vectorized (K*S) E-step & M-step (no nested Python loops over clusters/shifts)
- Coarse-to-Fine Shift Schedule: fast 0.5-step convergence -> 0.125-step fine polishing
- Finishes in ~3-5 seconds instead of minutes!
"""

import os
import time
import zipfile
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.special import logsumexp
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG
# ==============================================================================
DATA_ZIP = "ml69-k3.zip"
DATA_CSV = "data.csv"
SUB_CSV = "sample_submission.csv"
OUT_CSV = "submission_ml69_k3_v5.csv"

K_CLUSTERS = 6
SPECTRAL_MAX_SHIFT = 2
SPECTRAL_KNN = 25
SPECTRAL_KSIG = 7
SPIKE_THRESH = 1.2

EM_MAX_SHIFT = 4.0
STUDENT_NU = 3.0
SPIKE_Z_CUTOFF = 3.5
SEED = 42


# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
def load_dataset():
    if os.path.exists(DATA_CSV) and os.path.exists(SUB_CSV):
        df = pd.read_csv(DATA_CSV)
        sub = pd.read_csv(SUB_CSV)
    elif os.path.exists(os.path.join("data", DATA_CSV)):
        df = pd.read_csv(os.path.join("data", DATA_CSV))
        sub = pd.read_csv(os.path.join("data", SUB_CSV))
    elif os.path.exists(DATA_ZIP):
        with zipfile.ZipFile(DATA_ZIP, "r") as z:
            df = pd.read_csv(z.open("data.csv"))
            sub = pd.read_csv(z.open("sample_submission.csv"))
    else:
        raise FileNotFoundError("ไม่พบ data.csv หรือ ml69-k3.zip ในโฟลเดอร์ปัจจุบัน")

    time_cols = [c for c in df.columns if c.startswith("t") and "_" in c]
    assert len(time_cols) == 48, f"Expected 48 time columns, got {len(time_cols)}"
    return df, sub, time_cols


# ==============================================================================
# 2. VECTORIZED CIRCULAR LINEAR INTERPOLATION SHIFT
# ==============================================================================
def circular_shift_batch(arr_2d, shifts_1d):
    """
    เลื่อน arr_2d (K, T) ด้วย shifts_1d (S,) แบบ Vectorized ทีเดียว
    คืนค่า shape (K, S, T)
    """
    K, T = arr_2d.shape
    S = len(shifts_1d)
    # idx shape: (S, T)
    idx = (np.arange(T, dtype=np.float64)[None, :] - shifts_1d[:, None]) % T
    i0 = np.floor(idx).astype(int) % T
    i1 = (i0 + 1) % T
    w = (idx - np.floor(idx))[None, :, :]  # (1, S, T)
    return (1.0 - w) * arr_2d[:, i0] + w * arr_2d[:, i1]


def circular_shift_1d(arr, shift):
    T = arr.shape[-1]
    idx = (np.arange(T, dtype=np.float64) - shift) % T
    i0 = np.floor(idx).astype(int) % T
    i1 = (i0 + 1) % T
    w = idx - np.floor(idx)
    return (1.0 - w) * arr[i0] + w * arr[i1]


# ==============================================================================
# 3. STAGE 1: SPECTRAL INITIALIZATION
# ==============================================================================
def spectral_init_chunked(X_log, K=6, max_shift=2, k_nn=25, k_sig=7,
                          spike_thresh=1.2, chunk=500, seed=42):
    t0 = time.time()
    N, T = X_log.shape

    nbrs = np.stack([np.roll(X_log, s, axis=1) for s in [-2, -1, 1, 2]], axis=0)
    resid_local = X_log - np.nanmedian(nbrs, axis=0)
    X_clean = np.where(resid_local > spike_thresh, np.nan, X_log)

    M = (~np.isnan(X_clean)).astype(np.float32)
    X0 = np.where(np.isnan(X_clean), 0.0, X_clean).astype(np.float32)
    X02 = X0 ** 2

    shifts = list(range(-max_shift, max_shift + 1))
    Xr_list = [np.roll(X0, s, axis=1) for s in shifts]
    Xr2_list = [np.roll(X02, s, axis=1) for s in shifts]
    Mr_list = [np.roll(M, s, axis=1) for s in shifts]

    knn_idx = np.zeros((N, k_nn), dtype=np.int32)
    knn_dist = np.zeros((N, k_nn), dtype=np.float32)

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        Xi, Xi2, Mi = X0[start:end], X02[start:end], M[start:end]
        best = np.full((end - start, N), np.inf, dtype=np.float32)

        for Xr, Xr2, Mr in zip(Xr_list, Xr2_list, Mr_list):
            cnt = np.maximum(Mi @ Mr.T, 1.0)
            sx = Xi @ Mr.T
            sy = Mi @ Xr.T
            sx2 = Xi2 @ Mr.T
            sy2 = Mi @ Xr2.T
            sxy = Xi @ Xr.T
            var_diff = (sx2 + sy2 - 2.0 * sxy) / cnt - ((sx - sy) / cnt) ** 2
            d = np.sqrt(np.maximum(var_diff, 0.0))
            d = np.where(cnt >= 10, d, np.inf)
            np.minimum(best, d, out=best)

        for i in range(end - start):
            best[i, start + i] = np.inf

        idx_top = np.argpartition(best, k_nn, axis=1)[:, :k_nn]
        row_ar = np.arange(end - start)[:, None]
        d_top = best[row_ar, idx_top]
        ord_top = np.argsort(d_top, axis=1)
        knn_idx[start:end] = idx_top[row_ar, ord_top]
        knn_dist[start:end] = d_top[row_ar, ord_top]

    sig = np.maximum(knn_dist[:, k_sig - 1], 1e-6)
    rows = np.repeat(np.arange(N), k_nn)
    cols = knn_idx.ravel()
    dists = knn_dist.ravel()
    weights = np.exp(-(dists ** 2) / (sig[rows] * sig[cols]))

    W = csr_matrix((weights, (rows, cols)), shape=(N, N))
    W = 0.5 * (W + W.T)
    deg = np.asarray(W.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D_inv = csr_matrix((d_inv_sqrt, (np.arange(N), np.arange(N))), shape=(N, N))
    A = D_inv @ W @ D_inv

    vals, vecs = eigsh(A, k=min(12, N - 2), which="LA", tol=1e-6)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]

    print(f"[Stage 1] Spectral Graph + Eigsh finished in {time.time() - t0:.2f}s")
    U = vecs[:, :K]
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    return KMeans(n_clusters=K, n_init=100, random_state=seed).fit_predict(U)


# ==============================================================================
# 4. STAGE 2: VECTORIZED COARSE-TO-FINE STUDENT-T SHIFT-EM
# ==============================================================================
def run_em_stage(X_log, M, X0, templates, sigma2_t, pi, tau2,
                 shifts, nu, n_iter, irls_steps, trim_z=None, smooth_var=True):
    N, T = X_log.shape
    K = templates.shape[0]
    S = len(shifts)
    KS = K * S
    cnt_total = np.maximum(np.sum(M, axis=1), 1.0)  # (N,)

    # แบ่งเป็น batch ย่อยของบ้าน (เช่น 500 หลัง) เพื่อคุม RAM และให้ CPU Cache วิ่งเต็มสปีด
    batch_size = 500
    prev_ll = -np.inf
    prev_hard = None

    for it in range(n_iter):
        # เตรียม shifted templates & variances ทั้งหมดพร้อมกัน: (K, S, T) -> (KS, T)
        T_all = circular_shift_batch(templates, shifts).reshape(KS, T)
        V_all = np.maximum(circular_shift_batch(sigma2_t, shifts).reshape(KS, T), 0.008)
        log_V_all = np.log(V_all)  # (KS, T)

        # Prior log p(k) + log p(s|k) -> (KS,)
        lp_s = -0.5 * (shifts[None, :] ** 2) / tau2[:, None]
        lp_s -= logsumexp(lp_s, axis=1, keepdims=True)
        log_prior = (np.log(pi[:, None] + 1e-12) + lp_s).reshape(KS)

        gamma_k = np.empty((N, K), dtype=np.float64)
        gamma_ks_sum = np.zeros((K, S), dtype=np.float64)
        num_t_ks = np.zeros((K, S, T), dtype=np.float64)
        den_t_ks = np.zeros((K, S, T), dtype=np.float64)
        var_num_ks = np.zeros((K, S, T), dtype=np.float64)
        var_den_ks = np.zeros((K, S, T), dtype=np.float64)
        total_ll = 0.0

        # ---- VECTORIZED E-STEP + ACCUMULATED M-STEP BY BATCH ----
        for b0 in range(0, N, batch_size):
            b1 = min(b0 + batch_size, N)
            B = b1 - b0
            Xb = X_log[b0:b1]       # (B, T)
            X0b = X0[b0:b1]         # (B, T)
            Mb = M[b0:b1]           # (B, T)
            Mb3 = Mb[:, None, :]    # (B, 1, T)

            # diff: (B, KS, T)
            diff = Xb[:, None, :] - T_all[None, :, :]
            c = np.nanmedian(diff, axis=2)  # (B, KS)
            diff0 = np.where(Mb3 > 0.5, diff, 0.0)

            for _ in range(irls_steps):
                r = diff0 - c[:, :, None]
                w = Mb3 * ((nu + 1.0) / (nu * V_all[None, :, :] + r * r))
                c = np.sum(w * diff0, axis=2) / np.maximum(np.sum(w, axis=2), 1e-12)

            r = np.where(Mb3 > 0.5, diff0 - c[:, :, None], 0.0)
            z2 = (r * r) / V_all[None, :, :]
            ll_pts = -0.5 * log_V_all[None, :, :] - 0.5 * (nu + 1.0) * np.log1p(z2 / nu)

            if trim_z is not None:
                valid_m = Mb3 * (z2 <= (trim_z ** 2))
                cnt_v = np.maximum(np.sum(valid_m, axis=2), 1.0)
                ll_b = np.sum(valid_m * ll_pts, axis=2) * (cnt_total[b0:b1, None] / cnt_v)
            else:
                ll_b = np.sum(Mb3 * ll_pts, axis=2)

            log_joint_b = log_prior[None, :] + ll_b  # (B, KS)
            log_norm_b = logsumexp(log_joint_b, axis=1)
            total_ll += float(np.sum(log_norm_b))

            g_b = np.exp(log_joint_b - log_norm_b[:, None]).reshape(B, K, S)
            gamma_k[b0:b1] = np.sum(g_b, axis=2)
            gamma_ks_sum += np.sum(g_b, axis=0)

            # สะสมค่าสถิติสำหรับ M-Step ทันทีใน Batch (ไม่ต้องเก็บ array ขนาดยักษ์ไว้ใน RAM)
            g_flat = g_b.reshape(B, KS, 1)
            y_cent = X0b[:, None, :] - c[:, :, None]
            u_it = Mb3 * ((nu + 1.0) / (nu + z2))
            w_eff = g_flat * u_it

            num_t_ks += np.sum(w_eff * y_cent, axis=0).reshape(K, S, T)
            den_t_ks += np.sum(w_eff, axis=0).reshape(K, S, T)
            var_num_ks += np.sum(w_eff * (r * r), axis=0).reshape(K, S, T)
            var_den_ks += np.sum(g_flat * Mb3, axis=0).reshape(K, S, T)

        hard_labels = np.argmax(gamma_k, axis=1)
        changed = N if prev_hard is None else int(np.sum(hard_labels != prev_hard))
        prev_hard = hard_labels.copy()

        # ---- M-STEP ----
        for k in range(K):
            nk = float(np.sum(gamma_k[:, k]))
            if nk < 5.0:
                continue
            pi[k] = nk / N

            num_t = np.zeros(T, dtype=np.float64)
            den_t = np.zeros(T, dtype=np.float64)
            var_num = np.zeros(T, dtype=np.float64)
            var_den = np.zeros(T, dtype=np.float64)

            for s_idx, s in enumerate(shifts):
                if gamma_ks_sum[k, s_idx] < 1e-8:
                    continue
                num_t += circular_shift_1d(num_t_ks[k, s_idx], -s)
                den_t += circular_shift_1d(den_t_ks[k, s_idx], -s)
                var_num += circular_shift_1d(var_num_ks[k, s_idx], -s)
                var_den += circular_shift_1d(var_den_ks[k, s_idx], -s)

            t_new = num_t / np.maximum(den_t, 1e-8)
            var_new = var_num / np.maximum(var_den, 1e-8)
            if smooth_var:
                var_new = 0.15 * np.roll(var_new, -1) + 0.70 * var_new + 0.15 * np.roll(var_new, 1)
            var_new = np.clip(var_new, 0.008, 1.5)

            s_weights = gamma_ks_sum[k] / np.maximum(np.sum(gamma_ks_sum[k]), 1e-12)
            mean_s = float(np.sum(s_weights * shifts))
            var_s = float(np.sum(s_weights * ((shifts - mean_s) ** 2)))

            t_new = circular_shift_1d(t_new, mean_s)
            var_new = circular_shift_1d(var_new, mean_s)
            t_new -= np.mean(t_new)

            templates[k] = t_new
            sigma2_t[k] = var_new
            tau2[k] = np.clip(var_s, 0.25, 6.0)

        delta_ll = total_ll - prev_ll
        if it % 5 == 0 or changed == 0 or it == n_iter - 1:
            max_prob = np.max(gamma_k, axis=1)
            uncertain = int(np.sum(max_prob < 0.90))
            print(
                f"  Iter {it:2d} | LL={total_ll:11.2f} (dLL={delta_ll:+8.2f}) | "
                f"changed={changed:4d} | low_conf(<0.9)={uncertain:3d} ({uncertain/N*100:.2f}%)"
            )
        if it >= 8 and changed == 0 and abs(delta_ll) < 0.2:
            break
        prev_ll = total_ll

    return hard_labels, gamma_k, templates, sigma2_t, pi, tau2


def fit_student_t_shift_em_fast(X_log, init_labels, K=6, max_shift=4.0,
                                nu=3.0, z_cutoff=3.5):
    t0 = time.time()
    N, T = X_log.shape
    M = (~np.isnan(X_log)).astype(np.float64)
    X0 = np.where(M > 0.5, X_log, 0.0)

    templates = np.zeros((K, T), dtype=np.float64)
    sigma2_t = np.full((K, T), 0.12, dtype=np.float64)
    pi = np.zeros(K, dtype=np.float64)
    tau2 = np.full(K, 1.5, dtype=np.float64)

    row_med = np.nanmedian(X_log, axis=1, keepdims=True)
    X_centered = X_log - row_med

    for k in range(K):
        idx = init_labels == k
        pi[k] = max(idx.mean(), 1e-3)
        t_k = np.nanmedian(X_centered[idx], axis=0)
        t_k -= np.mean(t_k)
        templates[k] = t_k
        r_k = X_centered[idx] - t_k[None, :]
        mad = np.nanmedian(np.abs(r_k), axis=0) * 1.4826
        sigma2_t[k] = np.clip(mad ** 2, 0.02, 1.0)

    # Phase 1: Coarse Grid (step=0.5 -> 17 shifts) ลู่เข้าที่เร็วมาก
    print("[Stage 2 - Phase 1] Coarse Grid (step=0.50)...")
    shifts_coarse = np.arange(-max_shift, max_shift + 1e-9, 0.5)
    labels, probs, templates, sigma2_t, pi, tau2 = run_em_stage(
        X_log, M, X0, templates, sigma2_t, pi, tau2,
        shifts=shifts_coarse, nu=nu, n_iter=30, irls_steps=3,
        trim_z=None, smooth_var=True
    )

    # Phase 2: Fine Grid (step=0.125 -> 65 shifts) + Spike Trimming + Sharp Variance
    print("[Stage 2 - Phase 2] Fine Grid (step=0.125) + Spike Trimming...")
    shifts_fine = np.arange(-max_shift, max_shift + 1e-9, 0.125)
    labels, probs, templates, sigma2_t, pi, tau2 = run_em_stage(
        X_log, M, X0, templates, sigma2_t, pi, tau2,
        shifts=shifts_fine, nu=nu, n_iter=15, irls_steps=4,
        trim_z=z_cutoff, smooth_var=False
    )

    print(f"[Stage 2] Total EM finished in {time.time() - t0:.2f}s")
    return labels, probs, templates, sigma2_t, tau2


# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
def main():
    df, sub, time_cols = load_dataset()
    X = df[time_cols].values.astype(np.float64)
    X_log = np.log(np.where(X > 0, X, np.nan))
    N, T = X_log.shape
    print(f"Loaded {N} houses x {T} time slots ({time_cols[0]} .. {time_cols[-1]})")

    init_labels = spectral_init_chunked(
        X_log,
        K=K_CLUSTERS,
        max_shift=SPECTRAL_MAX_SHIFT,
        k_nn=SPECTRAL_KNN,
        k_sig=SPECTRAL_KSIG,
        spike_thresh=SPIKE_THRESH,
        seed=SEED,
    )

    labels, probs, templates, sigma2_t, tau2 = fit_student_t_shift_em_fast(
        X_log,
        init_labels=init_labels,
        K=K_CLUSTERS,
        max_shift=EM_MAX_SHIFT,
        nu=STUDENT_NU,
        z_cutoff=SPIKE_Z_CUTOFF,
    )

    print("\n=== Final Cluster Diagnostics ===")
    max_p = np.max(probs, axis=1)
    print(f"Mean posterior confidence: {max_p.mean():.5f}")
    print(f"Houses with prob >= 0.99: {(max_p >= 0.99).sum()}/{N} ({(max_p >= 0.99).mean()*100:.2f}%)")
    print(f"Houses with prob >= 0.95: {(max_p >= 0.95).sum()}/{N} ({(max_p >= 0.95).mean()*100:.2f}%)")
    print(f"Houses with prob <  0.90: {(max_p <  0.90).sum()}/{N} ({(max_p <  0.90).mean()*100:.2f}%)")

    for k in range(K_CLUSTERS):
        m = labels == k
        prof_2h = np.round(np.exp(templates[k, ::4]), 2)
        print(
            f"Cluster {k} | n={m.sum():4d} | med_sigma={np.median(np.sqrt(sigma2_t[k])):.3f} | "
            f"shift_std={np.sqrt(tau2[k]):.2f} | 2h_profile={prof_2h.tolist()}"
        )

    out = pd.DataFrame({"id": df["id"].values, "cluster": labels.astype(int)})
    out = sub[["id"]].merge(out, on="id", how="left")
    assert len(out) == len(sub), "Row count mismatch!"
    assert out["cluster"].isnull().sum() == 0, "NaN cluster found!"
    assert (out["id"].values == sub["id"].values).all(), "ID ordering mismatch!"

    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV} successfully! Shape={out.shape}")


if __name__ == "__main__":
    main()