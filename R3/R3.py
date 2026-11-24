"""
Solar rooftop hourly energy prediction  ->  submission.csv  (id,energy_kwh)

Run:  python solar_submission.py [--data_dir DIR] [--out submission.csv] [--validate]

Approach
--------
1. Physics features: solar position (computed for Thai time, UTC+7, at the middle of
   each hour), plane-of-array geometry (angle of incidence from tilt/azimuth),
   a clear-sky irradiance proxy on the tilted plane.
2. Weather cleaning: missing temp/cloud/humidity filled by time-interpolation within
   the house-day, then by neighbouring houses at the same hour; plus smoothed /
   neighbour-averaged cloud features (cloud cover is a noisy irradiance proxy).
3. Target = specific yield (kWh per kW installed) -> model is scale-free in capacity
   and generalises to new houses. Prediction is multiplied back by capacity_kw.
4. LightGBM (MAE-friendly: L1 objective blend) with several seeds. Per-house
   behaviour for known houses is captured with out-of-fold house residual encoding;
   new houses simply get the neutral value.
"""
import argparse
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

P = argparse.ArgumentParser()
P.add_argument("--data_dir", default="/mnt/user-data/uploads")
P.add_argument("--out", default="/mnt/user-data/outputs/submission.csv")
P.add_argument("--validate", action="store_true", help="run holdout validation (houses + dates)")
P.add_argument("--seeds", type=int, default=3)
args = P.parse_args()

tr = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
te = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
tr["is_train"] = 1
te["is_train"] = 0
te["energy_kwh"] = np.nan
df = pd.concat([tr, te], ignore_index=True)
df["dt"] = pd.to_datetime(df["datetime"])
df["date"] = df["dt"].dt.normalize()
df["hour"] = df["dt"].dt.hour
df["doy"] = df["dt"].dt.dayofyear
df["month"] = df["dt"].dt.month
WEATHER = ["temp_c", "cloud_cover", "humidity"]

# ------------------------------------------------------------------ weather fill
df = df.sort_values(["house_id", "dt"]).reset_index(drop=True)
for c in WEATHER:
    df[c + "_missing"] = df[c].isna().astype(int)
    df[c] = df.groupby(["house_id", "date"])[c].transform(lambda s: s.interpolate(limit_direction="both"))

# neighbour (same datetime, nearest houses by lat/lon) averages -- uses all rows, no target
lat = df["lat"].values
lon = df["lon"].values
nb = {c: np.full(len(df), np.nan) for c in WEATHER}
nb_cloud_far = np.full(len(df), np.nan)
for _, idx in df.groupby("datetime").indices.items():
    if len(idx) < 2:
        continue
    d = np.hypot(lat[idx][:, None] - lat[idx][None, :], lon[idx][:, None] - lon[idx][None, :])
    np.fill_diagonal(d, np.inf)
    k = min(5, len(idx) - 1)
    order = np.argsort(d, axis=1)[:, :k]
    w = 1.0 / (np.take_along_axis(d, order, 1) + 0.3)
    for c in WEATHER:
        v = df[c].values[idx]
        vv = v[order]
        m = ~np.isnan(vv)
        num = np.nansum(vv * w * m, 1)
        den = np.sum(w * m, 1)
        nb[c][idx] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    nb_cloud_far[idx] = np.nanmean(df["cloud_cover"].values[idx])
for c in WEATHER:
    df["nb_" + c] = nb[c]
    df[c] = df[c].fillna(df["nb_" + c])
    df[c] = df[c].fillna(df.groupby("datetime")[c].transform("mean"))
    df[c] = df[c].fillna(df[c].mean())
df["nb_cloud_far"] = nb_cloud_far

# temporal context of cloud inside the house-day
g = df.groupby(["house_id", "date"])["cloud_cover"]
df["cloud_prev"] = g.shift(1)
df["cloud_next"] = g.shift(-1)
df["cloud_roll3"] = g.transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
df["cloud_daymean"] = g.transform("mean")
df["cloud_midday"] = df.assign(_c=df["cloud_cover"].where(df["hour"].between(9, 15))).groupby(
    ["house_id", "date"])["_c"].transform("mean")
df["nb_cloud_day"] = df.groupby(["house_id", "date"])["nb_cloud_cover"].transform("mean")
df["cloud_blend"] = 0.5 * df["cloud_cover"] + 0.5 * df["nb_cloud_cover"].fillna(df["cloud_cover"])
df["hum_daymean"] = df.groupby(["house_id", "date"])["humidity"].transform("mean")
df["temp_daymean"] = df.groupby(["house_id", "date"])["temp_c"].transform("mean")
df["temp_range"] = df.groupby(["house_id", "date"])["temp_c"].transform(lambda s: s.max() - s.min())

# ------------------------------------------------------------------ solar geometry
def solar_geometry(dt_local, lat_deg, lon_deg, tz=7.0):
    """Return zenith(rad), azimuth(rad, from north clockwise) for local-time timestamps."""
    doy = dt_local.dt.dayofyear.values
    frac_h = (dt_local.dt.hour + dt_local.dt.minute / 60.0).values
    gam = 2 * np.pi / 365.0 * (doy - 1 + (frac_h - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(gam) - 0.032077 * np.sin(gam)
                    - 0.014615 * np.cos(2 * gam) - 0.040849 * np.sin(2 * gam))
    decl = (0.006918 - 0.399912 * np.cos(gam) + 0.070257 * np.sin(gam)
            - 0.006758 * np.cos(2 * gam) + 0.000907 * np.sin(2 * gam)
            - 0.002697 * np.cos(3 * gam) + 0.00148 * np.sin(3 * gam))
    tst = frac_h * 60 + eqt + 4 * lon_deg - 60 * tz           # true solar time, minutes
    ha = np.deg2rad(tst / 4 - 180)
    la = np.deg2rad(lat_deg)
    cosz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, -1, 1)
    zen = np.arccos(cosz)
    sinz = np.sqrt(1 - cosz ** 2) + 1e-9
    cos_az = (np.sin(decl) - cosz * np.sin(la)) / (sinz * np.cos(la))
    az = np.arccos(np.clip(cos_az, -1, 1))
    az = np.where(ha > 0, 2 * np.pi - az, az)
    return zen, az, ha, decl


tilt = np.deg2rad(df["tilt_deg"].values)
pan_az = np.deg2rad(df["azimuth_deg"].values)
acc_irr = np.zeros(len(df))
acc_cosz = np.zeros(len(df))
acc_aoi = np.zeros(len(df))
acc_dni = np.zeros(len(df))
OFFS = [10, 30, 50]  # minutes inside the hour -> averages over HH:00-HH:59
for m in OFFS:
    t = df["dt"] + pd.Timedelta(minutes=m)
    zen, saz, ha, decl = solar_geometry(t, df["lat"].values, df["lon"].values)
    cosz = np.cos(zen)
    cos_aoi = np.cos(zen) * np.cos(tilt) + np.sin(zen) * np.sin(tilt) * np.cos(saz - pan_az)
    up = (cosz > 0.02).astype(float)
    # Haurwitz-style clear-sky GHI and a simple beam/diffuse split on the tilted plane
    ghi = 1098 * cosz * np.exp(-0.057 / np.maximum(cosz, 0.02)) * up
    dni = 900 * np.exp(-0.2 / np.maximum(cosz, 0.05)) * up
    dhi = np.maximum(ghi - dni * cosz, 0)
    poa = (dni * np.maximum(cos_aoi, 0) + dhi * (1 + np.cos(tilt)) / 2 + 0.2 * ghi * (1 - np.cos(tilt)) / 2) * up
    acc_irr += poa / len(OFFS)
    acc_cosz += np.maximum(cosz, 0) / len(OFFS)
    acc_aoi += np.maximum(cos_aoi, 0) * up / len(OFFS)
    acc_dni += dni * np.maximum(cos_aoi, 0) / len(OFFS)
    if m == 30:
        df["sun_zen"] = np.rad2deg(zen)
        df["sun_az"] = np.rad2deg(saz)
        df["hour_angle"] = np.rad2deg(ha)
        df["decl"] = np.rad2deg(decl)
        df["rel_az"] = (np.rad2deg(saz) - df["azimuth_deg"] + 180) % 360 - 180
df["poa_clear"] = acc_irr
df["cosz"] = acc_cosz
df["cos_aoi"] = acc_aoi
df["beam_poa"] = acc_dni
df["ghi_clear_proxy"] = acc_cosz * 1000
df["poa_clear_ratio"] = df["poa_clear"] / (df["ghi_clear_proxy"] + 1)
df["daylight"] = (df["cosz"] > 0).astype(int)
df["az_south_dev"] = np.abs(df["azimuth_deg"] - 180)
df["az_ew"] = np.sin(np.deg2rad(df["azimuth_deg"]))      # +1 west, -1 east
# peak/daylight hours of the day (identical for the whole day, helps with partial-hour edges)
df["day_poa_sum"] = df.groupby(["house_id", "date"])["poa_clear"].transform("sum")
df["poa_share"] = df["poa_clear"] / (df["day_poa_sum"] + 1e-6)

# weather x geometry interactions
df["poa_x_clear"] = df["poa_clear"] * (1 - df["cloud_cover"])
df["poa_x_clear_blend"] = df["poa_clear"] * (1 - df["cloud_blend"])
df["poa_x_clear_sq"] = df["poa_clear"] * (1 - 0.75 * df["cloud_blend"] ** 3.4)  # Kasten-Czeplak
df["poa_x_clear_roll"] = df["poa_clear"] * (1 - df["cloud_roll3"])
df["temp_loss"] = 1 - 0.0035 * np.maximum(df["temp_c"] + 20 * (1 - df["cloud_blend"]) * (df["poa_clear"] > 0) - 25, 0)
df["soil_days"] = df["days_since_cleaning"]
df["soil_log"] = np.log1p(df["days_since_cleaning"])
df["soil_x_poa"] = df["days_since_cleaning"] * df["poa_x_clear"] / 1000
df["cap_x_poa_clear"] = df["capacity_kw"] * df["poa_clear"] / 1000
df["cap_x_poa_cloud"] = df["capacity_kw"] * df["poa_x_clear_sq"] / 1000

# house-level behaviour descriptors available for ALL houses (train+test, no target):
# how often/long they go without cleaning
hs = df.groupby("house_id")["days_since_cleaning"].agg(
    house_soil_mean="mean", house_soil_median="median", house_soil_max="max", house_soil_p90=lambda s: s.quantile(0.9))
df = df.merge(hs, left_on="house_id", right_index=True, how="left")
df["soil_rel_house"] = df["days_since_cleaning"] / (df["house_soil_mean"] + 1)

# ------------------------------------------------------------------ target
df["yield"] = df["energy_kwh"] / df["capacity_kw"]

FEATURES = [
    "lat", "lon", "capacity_kw", "tilt_deg", "azimuth_deg", "hour", "doy", "month",
    "temp_c", "cloud_cover", "humidity", "days_since_cleaning",
    "temp_c_missing", "cloud_cover_missing", "humidity_missing",
    "nb_temp_c", "nb_cloud_cover", "nb_humidity", "nb_cloud_far",
    "cloud_prev", "cloud_next", "cloud_roll3", "cloud_daymean", "cloud_midday", "nb_cloud_day",
    "cloud_blend", "hum_daymean", "temp_daymean", "temp_range",
    "sun_zen", "sun_az", "hour_angle", "decl", "rel_az",
    "poa_clear", "cosz", "cos_aoi", "beam_poa", "ghi_clear_proxy", "poa_clear_ratio",
    "az_south_dev", "az_ew", "day_poa_sum", "poa_share",
    "poa_x_clear", "poa_x_clear_blend", "poa_x_clear_sq", "poa_x_clear_roll", "temp_loss",
    "soil_log", "soil_x_poa", "cap_x_poa_clear", "cap_x_poa_cloud",
    "house_soil_mean", "house_soil_median", "house_soil_max", "house_soil_p90", "soil_rel_house",
]

PARAMS = dict(
    objective="regression_l1", learning_rate=0.06, num_leaves=63, min_child_samples=40,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    n_estimators=800, verbose=-1, n_jobs=-1,
)

from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

# ฟีเจอร์ที่ "ไม่ระบุตัวบ้าน" ใช้สร้าง baseline เท่านั้น
BASE = ['hour','doy','month','temp_c','cloud_cover','humidity','days_since_cleaning',
        'temp_c_missing','cloud_cover_missing','humidity_missing','nb_temp_c','nb_cloud_cover',
        'nb_humidity','nb_cloud_far','cloud_prev','cloud_next','cloud_roll3','cloud_daymean',
        'cloud_midday','nb_cloud_day','cloud_blend','hum_daymean','temp_daymean','temp_range',
        'sun_zen','sun_az','hour_angle','decl','cosz','ghi_clear_proxy','temp_loss','soil_log',
        'poa_x_clear_sq','poa_clear']

HF = ['hs_rat','hs_res','hs_w','hh_rat','hh_rat_m1','hh_rat_p1','h_rat_all']
PF = ['pf_cur','pf_m1','pf_p1','pf_mid','pf_rel','pf_rel_m1','pf_rel_p1',
      'pf_onset','pf_since','pf_min_aft','pf_morn','pf_n']
SF = ['ht_rat','ht_w','hd_rat','hd_w','ht_rat_near','hz_rat']
HOUSE_COLS = HF + PF + SF


def oof_base(S):
    out = np.zeros(len(S)); dates = S["date"].values; u = np.unique(dates)
    prm = dict(PARAMS, objective="regression", n_estimators=300, learning_rate=0.1)
    for a, b in KFold(5, shuffle=True, random_state=0).split(u):
        h = np.isin(dates, u[b])
        m = lgb.LGBMRegressor(**prm).fit(S.loc[~h, BASE], S.loc[~h, "yield"])
        out[h] = m.predict(S.loc[h, BASE])
    return out


def house_feats(S, T):
    """ฟีเจอร์ระดับบ้านจากวันอื่นของบ้านเดียวกัน (ไม่รวมวันเดียวกับแถวที่ทำนาย)"""
    out = pd.DataFrame(np.nan, index=T.index, columns=HF)
    Sv = S[(S.base > 0.08) & (S.daylight == 1)].reset_index(drop=True)
    Sg = Sv.groupby("house_id").indices
    for h, ti in T.groupby("house_id").indices.items():
        if h not in Sg: continue
        s = Sv.iloc[Sg[h]]; t = T.iloc[ti]
        same = t.date.values[:, None] == s.date.values[None, :]
        W = np.exp(-0.5 * (((t.sun_az.values[:, None] - s.sun_az.values[None, :]) / 5.0) ** 2
                           + ((t.sun_zen.values[:, None] - s.sun_zen.values[None, :]) / 4.0) ** 2))
        W[same] = 0; sw = W.sum(1)
        out.loc[t.index, "hs_rat"] = np.where(sw > 0.05, (W @ s.rat.values) / sw, np.nan)
        out.loc[t.index, "hs_res"] = np.where(sw > 0.05, (W @ s.res.values) / sw, np.nan)
        out.loc[t.index, "hs_w"] = sw
        for name, sh in (("hh_rat", 0), ("hh_rat_m1", -1), ("hh_rat_p1", 1)):
            M = ((t.hour.values[:, None] + sh) == s.hour.values[None, :]) & (~same)
            c = M.sum(1)
            out.loc[t.index, name] = np.where(c > 0, (M @ s.rat.values) / np.maximum(c, 1), np.nan)
        M = (~same) & (s.poa_clear.values[None, :] > 200); c = M.sum(1)
        out.loc[t.index, "h_rat_all"] = np.where(c > 0, (M @ s.rat.values) / np.maximum(c, 1), np.nan)
    return out


def profile_feats(S, T):
    """โปรไฟล์ 'ผลผลิตเทียบ baseline' รายชั่วโมงของแต่ละบ้าน -> จับเงาบังช่วงบ่าย"""
    out = pd.DataFrame(np.nan, index=T.index, columns=PF)
    Sv = S[(S.base > 0.08) & (S.daylight == 1) & (S.cloud_cover < 0.75)]
    Sg = Sv.groupby("house_id").indices
    for h, ti in T.groupby("house_id").indices.items():
        if h not in Sg: continue
        s = Sv.iloc[Sg[h]]
        ds = np.array(sorted(s.date.unique())); di = {d: i for i, d in enumerate(ds)}
        R = np.full((len(ds), 24), np.nan)
        R[s.date.map(di).values, s.hour.values] = s.rat.values
        t = T.iloc[ti]; cache = {}
        for idx, d, hr in zip(t.index, t.date.values, t.hour.values):
            if d not in cache:
                Rm = np.delete(R, di[d], axis=0) if d in di else R
                prof = np.nanmedian(Rm, axis=0); n = (~np.isnan(Rm)).sum(0)
                mid = np.nanmedian(prof[10:14]); rel = prof / mid
                ons = np.nan
                for hh in range(12, 19):
                    if rel[hh] < 0.75: ons = hh; break
                aft = np.nanmin(rel[14:18]) if np.isfinite(rel[14:18]).any() else np.nan
                cache[d] = (prof, mid, rel, ons, aft, np.nanmean(rel[7:9]), n)
            prof, mid, rel, ons, aft, morn, n = cache[d]
            out.loc[idx] = [prof[hr], prof[max(hr - 1, 0)], prof[min(hr + 1, 23)], mid, rel[hr],
                            rel[max(hr - 1, 0)], rel[min(hr + 1, 23)], ons,
                            (hr - ons) if ons == ons else np.nan, aft, morn, n[hr]]
    return out


def soil_feats(S, T):
    """ประสิทธิภาพของบ้านในช่วงเวลา/ระดับคราบสกปรกใกล้เคียงกัน (ใช้ช่วงกลางวัน 9-13 น.)"""
    out = pd.DataFrame(np.nan, index=T.index, columns=SF)
    Sv = S[(S.base > 0.08) & S.hour.between(9, 13) & (S.cloud_cover < 0.8)]
    Sg = Sv.groupby("house_id").indices
    for h, ti in T.groupby("house_id").indices.items():
        if h not in Sg: continue
        s = Sv.iloc[Sg[h]]; t = T.iloc[ti]
        same = t.date.values[:, None] == s.date.values[None, :]
        dd = (t.date.values[:, None] - s.date.values[None, :]) / np.timedelta64(1, "D")
        def kern(D, bw):
            W = np.exp(-0.5 * (D / bw) ** 2); W[same] = 0; sw = W.sum(1)
            return np.where(sw > 0.05, (W @ s.rat.values) / np.maximum(sw, 1e-9), np.nan), sw
        out.loc[t.index, "ht_rat"], out.loc[t.index, "ht_w"] = kern(dd, 12.0)
        out.loc[t.index, "ht_rat_near"], _ = kern(dd, 3.0)
        out.loc[t.index, "hd_rat"], out.loc[t.index, "hd_w"] = kern(
            t.days_since_cleaning.values[:, None] - s.days_since_cleaning.values[None, :], 6.0)
        M = ~same; c = M.sum(1)
        out.loc[t.index, "hz_rat"] = np.where(c > 0, (M @ s.rat.values) / np.maximum(c, 1), np.nan)
    return out


MODELS = [
    dict(objective="huber", alpha=0.15, n_estimators=500, learning_rate=0.08,
         num_leaves=63, min_child_samples=40),
    dict(objective="regression", n_estimators=1200, learning_rate=0.03,
         num_leaves=31, min_child_samples=60),
]


def fit_predict(train_df, test_df, seeds=2):
    S = train_df.reset_index(drop=True).copy()
    T = test_df.reset_index(drop=True).copy()
    S["base"] = oof_base(S)
    S["res"] = S["yield"] - S["base"]
    S["rat"] = (S["yield"] / S["base"].clip(lower=0.08)).clip(0, 3)

    def allf(X):
        return pd.concat([X, house_feats(S, X), profile_feats(S, X), soil_feats(S, X)], axis=1)
    Tr, Te = allf(S), allf(T)

    # จำลองบ้านใหม่: ซ่อนฟีเจอร์ระดับบ้านของ 15% ของบ้านตอนเทรน
    rng = np.random.RandomState(1)
    mh = rng.choice(Tr.house_id.unique(), int(0.15 * Tr.house_id.nunique()), replace=False)
    Tr.loc[Tr.house_id.isin(mh), HOUSE_COLS] = np.nan

    feats = FEATURES + HOUSE_COLS
    preds, k = np.zeros(len(Te)), 0
    for cfg in MODELS:
        for s in range(seeds):
            m = lgb.LGBMRegressor(**{**PARAMS, **cfg}, random_state=s)
            m.fit(Tr[feats], Tr["yield"], sample_weight=Tr["capacity_kw"].values)
            preds += m.predict(Te[feats]); k += 1
    preds = np.maximum(preds / k, 0) * Te["capacity_kw"].values
    preds[Te["daylight"].values == 0] = 0.0
    return preds

def add_house_residual(train_df, apply_df, oof_folds=5, seed=0):
    """
    House-level multiplicative performance factor (shading, degradation, wiring ...).
    Computed as sum(actual)/sum(expected) with a simple expected-yield baseline model,
    shrunk towards 1. For training rows it is computed out-of-fold by DAY so that the
    target of a row never leaks into its own feature.
    """
    base_feat = ["poa_x_clear_sq", "poa_clear", "soil_log", "temp_loss", "cloud_blend", "hour", "doy"]
    prm = dict(PARAMS, n_estimators=300, learning_rate=0.08, num_leaves=31)

    def factor(fit_rows, eval_rows):
        m = lgb.LGBMRegressor(**prm, random_state=seed)
        m.fit(fit_rows[base_feat], fit_rows["yield"])
        pred_fit = np.maximum(m.predict(fit_rows[base_feat]), 0)
        f = fit_rows.assign(pred=pred_fit).groupby("house_id").agg(a=("yield", "sum"), p=("pred", "sum"), n=("yield", "size"))
        k = 200.0  # shrinkage (rows)
        f["fac"] = (f["a"] + k * f["p"] / np.maximum(f["p"], 1e-6) * 1.0) / (f["p"] + k * 1.0 + 1e-9)
        f["fac"] = (f["a"] + 0.0) / np.maximum(f["p"], 1e-6)
        w = f["n"] / (f["n"] + k)
        f["fac"] = 1 + w * (f["fac"] - 1)
        return eval_rows["house_id"].map(f["fac"]).fillna(1.0).values, m

    out_train = np.ones(len(train_df))
    dates = train_df["date"].values
    uniq = np.unique(dates)
    kf = KFold(oof_folds, shuffle=True, random_state=seed)
    for fi, (a, b) in enumerate(kf.split(uniq)):
        hold = np.isin(dates, uniq[b])
        out_train[hold], _ = factor(train_df[~hold], train_df[hold])
    out_apply, _ = factor(train_df, apply_df)
    return out_train, out_apply


def fit_predict(train_df, test_df, seeds, use_house_fac=True):
    train_df = train_df.copy()
    test_df = test_df.copy()
    feats = list(FEATURES)
    if use_house_fac:
        ftr, fte = add_house_residual(train_df, test_df)
        train_df["house_fac"] = ftr
        test_df["house_fac"] = fte
        feats.append("house_fac")
    preds = np.zeros(len(test_df))
    for s in range(seeds):
        m = lgb.LGBMRegressor(**PARAMS, random_state=s)
        m.fit(train_df[feats], train_df["yield"], sample_weight=train_df["capacity_kw"].values)
        preds += m.predict(test_df[feats]) / seeds
    preds = np.maximum(preds, 0) * test_df["capacity_kw"].values
    preds[test_df["daylight"].values == 0] = 0.0
    return preds


train = df[df.is_train == 1].reset_index(drop=True)
test = df[df.is_train == 0].reset_index(drop=True)

# ------------------------------------------------------------------ optional validation
if args.validate:
    rng = np.random.RandomState(42)
    houses = train["house_id"].unique()
    new_h = rng.choice(houses, size=int(0.15 * len(houses)), replace=False)
    days = train["date"].unique()
    new_d = rng.choice(days, size=int(0.15 * len(days)), replace=False)
    is_new_house = train["house_id"].isin(new_h)
    is_new_day = train["date"].isin(new_d)
    fit_mask = ~is_new_house & ~is_new_day
    val_mask = is_new_house | is_new_day
    pr = fit_predict(train[fit_mask], train[val_mask], seeds=1)
    y = train.loc[val_mask, "energy_kwh"].values
    nh_mask = is_new_house[val_mask].values
    print(f"holdout MAE={np.abs(pr-y).mean():.4f}  new={np.abs(pr-y)[nh_mask].mean():.4f}  seen={np.abs(pr-y)[~nh_mask].mean():.4f}")

# ------------------------------------------------------------------ final fit + submission
test_pred = fit_predict(train, test, seeds=args.seeds)
sub = pd.DataFrame({"id": test["id"].astype(int), "energy_kwh": test_pred})
sub = sub.set_index("id").loc[te["id"].values].reset_index()   # same order as test.csv
assert len(sub) == len(te) and sub["energy_kwh"].notna().all()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
sub.to_csv(args.out, index=False)
print("saved", args.out, sub.shape)
print(sub.describe().T)
