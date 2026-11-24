import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def haversine_matrix(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """คำนวณระยะทางเป็นกิโลเมตรระหว่างพิกัดชุดที่ 1 (N,) และชุดที่ 2 (M,)"""
    r = 6371.0
    lat1_r = np.radians(lat1)[:, None]
    lon1_r = np.radians(lon1)[:, None]
    lat2_r = np.radians(lat2)[None, :]
    lon2_r = np.radians(lon2)[None, :]

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def preprocess_and_engineer(train_df: pd.DataFrame, test_df: pd.DataFrame, stations_df: pd.DataFrame):
    """ทำความสะอาดข้อมูลที่นายหน้ากรอกผิดพลาด และสร้างฟีเจอร์เชิงพื้นที่และตัวห้อง"""
    df = pd.concat(
        [train_df.assign(_is_train=1), test_df.assign(_is_train=0)],
        ignore_index=True
    )

    # 1. แก้ไขข้อมูล area_sqm ที่กรอกจุดทศนิยมผิด (คูณ 10 เกินจริง) แยกตามจำนวนห้องนอน
    area_thresholds = {0: 100.0, 1: 150.0, 2: 250.0, 3: 400.0, 4: 600.0}
    df["area_typo_fixed"] = 0
    for br, th in area_thresholds.items():
        mask = (df["bedrooms"] == br) & (df["area_sqm"] >= th)
        df.loc[mask, "area_sqm"] = df.loc[mask, "area_sqm"] / 10.0
        df.loc[mask, "area_typo_fixed"] = 1

    # 2. แปลงเวลาที่ลงประกาศ
    df["year"] = df["listing_month"].str[:4].astype(int)
    df["month"] = df["listing_month"].str[5:7].astype(int)
    df["time_idx"] = (df["year"] - 2023) * 12 + df["month"]

    # 3. ลด Noise ระดับโครงการ (พิกัดและอายุอาคารแท้จริงของโครงการ)
    df["prj_lat"] = df.groupby("project_id")["lat"].transform("median")
    df["prj_lon"] = df.groupby("project_id")["lon"].transform("median")
    df["lat_sq"] = df["prj_lat"] ** 2
    df["lon_sq"] = df["prj_lon"] ** 2
    df["lat_lon"] = df["prj_lat"] * df["prj_lon"]

    # คำนวณอายุอาคาร ณ ปี 2023 ของแต่ละโครงการ แล้วบวกตามปีที่ลงประกาศจริง
    df["_age_base_2023"] = df["building_age"] - (df["year"] - 2023)
    df["age_2023"] = df.groupby("project_id")["_age_base_2023"].transform("median")
    df["true_age"] = df["age_2023"] + (df["year"] - 2023)
    df["true_age_sq"] = df["true_age"] ** 2
    df.drop(columns=["_age_base_2023"], inplace=True)

    df["prj_listing_count"] = df.groupby("project_id")["id"].transform("count")

    # 4. คำนวณระยะทางไปยังสถานีรถไฟฟ้าทั้งหมดใน stations.csv
    dist_mat = haversine_matrix(
        df["prj_lat"].values,
        df["prj_lon"].values,
        stations_df["lat"].values,
        stations_df["lon"].values,
    )
    sorted_idx = np.argsort(dist_mat, axis=1)
    sorted_dists = np.take_along_axis(dist_mat, sorted_idx, axis=1)

    extra_cols = {
        "dist_st_1": sorted_dists[:, 0],
        "dist_st_2": sorted_dists[:, 1],
        "dist_st_3": sorted_dists[:, 2],
        "dist_st_mean3": sorted_dists[:, :3].mean(axis=1),
        "dist_st_1_sq": sorted_dists[:, 0] ** 2,
        "log_dist_st_1": np.log1p(sorted_dists[:, 0]),
        "inv_dist_st_1": 1.0 / (sorted_dists[:, 0] + 0.1),
        "exp_dist_300m": np.exp(-sorted_dists[:, 0] / 0.3),
        "exp_dist_500m": np.exp(-sorted_dists[:, 0] / 0.5),
        "exp_dist_1km": np.exp(-sorted_dists[:, 0] / 1.0),
        "is_highrise": (df["total_floors"] > 8).astype(int),
        "st_within_500m": (dist_mat <= 0.5).sum(axis=1),
        "st_within_1km": (dist_mat <= 1.0).sum(axis=1),
        "st_within_2km": (dist_mat <= 2.0).sum(axis=1),
        "nearest_station": stations_df["station_code"].values[sorted_idx[:, 0]],
        "nearest_line": stations_df["line"].values[sorted_idx[:, 0]],
    }

    for idx_s, s_row in stations_df.iterrows():
        extra_cols[f"d_st_{s_row['station_code']}"] = dist_mat[:, idx_s]

    for line_code in sorted(stations_df["line"].unique()):
        line_cols = np.where(stations_df["line"].values == line_code)[0]
        extra_cols[f"dist_line_{line_code}"] = dist_mat[:, line_cols].min(axis=1)

    hub_coords = np.array([
        [13.7450, 100.5400],
        [13.7320, 100.5380],
        [13.7710, 100.5920],
        [13.7920, 100.5340],
        [13.8400, 100.5600],
    ])
    hub_dists = haversine_matrix(
        df["prj_lat"].values, df["prj_lon"].values, hub_coords[:, 0], hub_coords[:, 1]
    )
    for i in range(hub_dists.shape[1]):
        extra_cols[f"dist_hub_{i}"] = hub_dists[:, i]
    extra_cols["min_hub_dist"] = hub_dists.min(axis=1)

    df = pd.concat([df, pd.DataFrame(extra_cols, index=df.index)], axis=1)

    # 5. ฟีเจอร์ระดับห้องชุดและค่า Missing (รวมเทอมกำลังสองของ floor_ratio และ log_area)
    df["log_area"] = np.log(df["area_sqm"])
    df["log_area_sq"] = df["log_area"] ** 2
    df["floor_missing"] = df["floor"].isna().astype(int)
    df["floor_filled"] = df["floor"].fillna(df["total_floors"] * 0.5)
    df["floor_ratio"] = df["floor_filled"] / df["total_floors"]
    df["floor_ratio_sq"] = df["floor_ratio"] ** 2
    df["is_top_floor"] = (df["floor"] == df["total_floors"]).astype(int)
    df["is_low_floor"] = (df["floor"] <= 4).astype(int)

    df["bath_missing"] = df["bathrooms"].isna().astype(int)
    df["bathrooms_filled"] = df["bathrooms"].fillna(df["bedrooms"].clip(lower=1))
    df["area_per_bedroom"] = df["area_sqm"] / (df["bedrooms"] + 1.0)
    df["bath_per_bedroom"] = df["bathrooms_filled"] / (df["bedrooms"] + 1.0)

    # 6. แปลงค่าหมวดหมู่ที่มีลำดับและเติมค่าว่าง
    furn_map = {"none": 0, "partly": 1, "fully": 2}
    view_map = {"blocked": 0, "garden": 1, "pool": 2, "city": 3, "river": 4}
    df["furnished_ord"] = df["furnished"].map(furn_map)
    df["view_ord"] = df["view_type"].map(view_map)

    df["furnished"] = df["furnished"].fillna("missing")
    df["view_type"] = df["view_type"].fillna("missing")

    # ฟีเจอร์ค่าเฉลี่ยระดับโครงการ
    df["prj_mean_area"] = df.groupby("project_id")["area_sqm"].transform("mean")
    df["prj_max_area"] = df.groupby("project_id")["area_sqm"].transform("max")
    df["prj_mean_furn"] = df.groupby("project_id")["furnished_ord"].transform("mean")
    df["area_vs_prj_mean"] = df["area_sqm"] / df["prj_mean_area"]

    train_clean = df[df["_is_train"] == 1].drop(columns=["_is_train"]).reset_index(drop=True)
    test_clean = df[df["_is_train"] == 0].drop(columns=["_is_train", "price"]).reset_index(drop=True)
    return train_clean, test_clean


def fit_predict_within_eb(tr_df: pd.DataFrame, y_tr: np.ndarray, target_df: pd.DataFrame, lam: float = 0.14):
    """
    1) ประมาณค่าน้ำหนักตัวห้องด้วย Within-Project Fixed Effects (Demeaning) เพื่อตัด Omitted Variable Bias
    2) ทำนายค่าพรีเมียมโครงการใหม่ด้วย Ridge + RandomForest + HistGBM + Spatial Kriging (GP)
    3) รวมค่าจริงของโครงการเข้ากับ Spatial Prior ด้วย Empirical Bayes Shrinkage
    """
    unit_num = [
        "log_area", "log_area_sq", "bedrooms", "bathrooms_filled", "bath_missing",
        "floor_ratio", "floor_ratio_sq", "floor_missing", "is_top_floor", "is_low_floor",
        "true_age", "true_age_sq", "time_idx", "area_per_bedroom"
    ]
    unit_cat = ["furnished", "view_type"]

    u_enc = ColumnTransformer([
        ("num", "passthrough", unit_num),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), unit_cat)
    ])
    X_u_tr = u_enc.fit_transform(tr_df)
    if hasattr(X_u_tr, "toarray"):
        X_u_tr = X_u_tr.toarray()
    X_u_tgt = u_enc.transform(target_df)
    if hasattr(X_u_tgt, "toarray"):
        X_u_tgt = X_u_tgt.toarray()

    # หักค่าเฉลี่ยรายโครงการออก (Within-Project Demeaning) เพื่อให้ได้ความชันสเปกห้องที่ไม่ปนเปื้อนโครงการ
    prj_ids = tr_df["project_id"].values
    df_xu = pd.DataFrame(X_u_tr)
    df_xu["_prj"] = prj_ids
    df_xu["_y"] = y_tr

    grp_means = df_xu.groupby("_prj").transform("mean")
    X_demeaned = df_xu.drop(columns=["_prj", "_y"]).values - grp_means.drop(columns=["_y"]).values
    y_demeaned = y_tr - grp_means["_y"].values

    u_reg = Ridge(alpha=1.0, fit_intercept=False)
    u_reg.fit(X_demeaned, y_demeaned)

    u_tr = X_u_tr @ u_reg.coef_
    u_target = X_u_tgt @ u_reg.coef_

    # ส่วนต่างที่เหลือคือค่าพรีเมียมแท้จริงของแต่ละโครงการ
    res_tr = y_tr - u_tr
    stats = pd.DataFrame({"project_id": prj_ids, "res": res_tr}).groupby("project_id")["res"].agg(["count", "mean"])

    # สร้างชุดข้อมูลระดับโครงการสำหรับทำนายโครงการใหม่
    st_dist_cols = [
        c for c in tr_df.columns
        if c.startswith("d_st_") or c.startswith("dist_line_") or c.startswith("dist_hub_")
    ]
    prj_num_cols = [
        "prj_lat", "prj_lon", "lat_sq", "lon_sq", "lat_lon",
        "total_floors", "is_highrise", "has_pool", "has_gym", "age_2023",
        "dist_st_1", "dist_st_2", "dist_st_3", "dist_st_1_sq", "log_dist_st_1",
        "exp_dist_300m", "exp_dist_500m", "exp_dist_1km",
        "st_within_500m", "st_within_1km", "st_within_2km", "min_hub_dist",
        "prj_mean_area", "prj_max_area"
    ] + st_dist_cols
    prj_cat_cols = ["nearest_station", "nearest_line"]
    all_prj_cols = prj_num_cols + prj_cat_cols

    prj_tr = tr_df.groupby("project_id")[all_prj_cols].first()
    prj_tr["mean_res"] = stats["mean"]
    w_tr = stats["count"].values / (stats["count"].values + lam)

    prj_tgt = target_df.groupby("project_id")[all_prj_cols].first()

    # 2A: Ridge ระดับโครงการ (ถ่วงน้ำหนักตามจำนวนประกาศในโครงการ)
    prj_prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), prj_num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), prj_cat_cols)
    ])
    prj_ridge = Pipeline([("prep", prj_prep), ("m", Ridge(alpha=25.0))])
    prj_ridge.fit(prj_tr[all_prj_cols], prj_tr["mean_res"].values, m__sample_weight=w_tr)
    pred_p_ridge_tr = prj_ridge.predict(prj_tr[all_prj_cols])
    pred_p_ridge = prj_ridge.predict(prj_tgt[all_prj_cols])

    # 2B: RandomForest ระดับโครงการ
    X_tr_rf = pd.get_dummies(prj_tr[all_prj_cols])
    X_tgt_rf = pd.get_dummies(prj_tgt[all_prj_cols]).reindex(columns=X_tr_rf.columns, fill_value=0)
    prj_rf = RandomForestRegressor(n_estimators=250, max_depth=10, min_samples_leaf=3, n_jobs=-1, random_state=42)
    prj_rf.fit(X_tr_rf, prj_tr["mean_res"].values, sample_weight=w_tr)
    pred_p_rf = prj_rf.predict(X_tgt_rf)

    # 2C: HistGBM ระดับโครงการ
    X_tr_hg = prj_tr[all_prj_cols].copy()
    X_tgt_hg = prj_tgt[all_prj_cols].copy()
    for c in prj_cat_cols:
        X_tr_hg[c] = X_tr_hg[c].astype("category")
        X_tgt_hg[c] = X_tgt_hg[c].astype("category")
    prj_hg = HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.03, max_iter=250, min_samples_leaf=12,
        l2_regularization=5.0, categorical_features=prj_cat_cols, random_state=42
    )
    prj_hg.fit(X_tr_hg, prj_tr["mean_res"].values, sample_weight=w_tr)
    pred_p_hg = prj_hg.predict(X_tgt_hg)

    # 2D: Spatial Kriging (Gaussian Process) บนพิกัด (lat, lon) เพื่อเก็บความต่อเนื่องเชิงพื้นที่
    gp_resid_tr = prj_tr["mean_res"].values - pred_p_ridge_tr
    coords_tr = np.column_stack([prj_tr["prj_lat"].values * 111.0, prj_tr["prj_lon"].values * 107.8])
    coords_tgt = np.column_stack([prj_tgt["prj_lat"].values * 111.0, prj_tgt["prj_lon"].values * 107.8])
    kernel = (
        ConstantKernel(0.01, (1e-5, 0.2))
        * Matern(length_scale=1.5, length_scale_bounds=(0.1, 15.0), nu=1.5)
        + WhiteKernel(0.04, (1e-4, 0.2))
    )
    gp = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=0)
    gp.fit(coords_tr, gp_resid_tr)
    kriging_tgt = gp.predict(coords_tgt)

    # รวม Prior ระดับโครงการ
    base_prior = 0.50 * (pred_p_ridge + kriging_tgt) + 0.30 * pred_p_rf + 0.20 * pred_p_hg
    prj_prior_series = pd.Series(base_prior, index=prj_tgt.index)
    sp_prior_target = target_df["project_id"].map(prj_prior_series).values

    # ขั้นที่ 3: Empirical Bayes Shrinkage
    t_prjs = target_df["project_id"]
    cnt = t_prjs.map(stats["count"]).fillna(0).values
    mn = t_prjs.map(stats["mean"]).fillna(0).values
    eb_premium = (cnt * mn + lam * sp_prior_target) / (cnt + lam)

    return u_target, sp_prior_target, eb_premium


def main():
    print("1. กำลังโหลดข้อมูล...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    stations_df = pd.read_csv("stations.csv")

    print("2. กำลังทำความสะอาดข้อมูลและสร้างฟีเจอร์เชิงลึก...")
    train_fe, test_fe = preprocess_and_engineer(train_df, test_df, stations_df)
    y_log = np.log1p(train_fe["price"].values)

    hgb_cat_cols = ["furnished", "view_type", "nearest_station", "nearest_line"]
    drop_cols = ["id", "price", "listing_month", "agent_code", "project_id"]
    base_feat_cols = [c for c in train_fe.columns if c not in drop_cols]
    num_cols = [c for c in base_feat_cols if c not in hgb_cat_cols]

    n_splits = 10
    print(f"3. กำลังฝึกสอนโมเดลด้วย {n_splits}-Fold Cross-Validation (Within-Project EB + Spatial Kriging + Ensemble)...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_eb = np.zeros(len(train_fe))
    pred_eb = np.zeros(len(test_fe))

    oof_hgb = np.zeros(len(train_fe))
    pred_hgb = np.zeros(len(test_fe))

    oof_et = np.zeros(len(train_fe))
    pred_et = np.zeros(len(test_fe))

    oof_ridge = np.zeros(len(train_fe))
    pred_ridge = np.zeros(len(test_fe))

    oof_lgb = np.zeros(len(train_fe))
    pred_lgb = np.zeros(len(test_fe))

    _, sp_test_full, _ = fit_predict_within_eb(train_fe, y_log, test_fe, lam=0.14)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_fe), 1):
        tr_fold = train_fe.iloc[tr_idx].reset_index(drop=True)
        val_fold = train_fe.iloc[val_idx].reset_index(drop=True)
        y_tr_fold = y_log[tr_idx]
        y_val_fold = y_log[val_idx]

        # Nested OOF EB features ภายใน tr_fold เพื่อป้องกัน Data Leakage
        inner_kf = KFold(n_splits=5, shuffle=True, random_state=100 + fold)
        in_unit = np.zeros(len(tr_fold))
        in_sp = np.zeros(len(tr_fold))
        in_eb = np.zeros(len(tr_fold))

        for itr, ival in inner_kf.split(tr_fold):
            u_iv, sp_iv, eb_iv = fit_predict_within_eb(
                tr_fold.iloc[itr], y_tr_fold[itr], tr_fold.iloc[ival], lam=0.14
            )
            in_unit[ival] = u_iv
            in_sp[ival] = sp_iv
            in_eb[ival] = eb_iv

        u_val, sp_val, eb_val = fit_predict_within_eb(tr_fold, y_tr_fold, val_fold, lam=0.14)
        u_tst, sp_tst, eb_tst = fit_predict_within_eb(tr_fold, y_tr_fold, test_fe, lam=0.14)

        oof_eb[val_idx] = u_val + eb_val
        pred_eb += (u_tst + eb_tst) / n_splits

        # เตรียมชุดฟีเจอร์สำหรับ Tree Models
        X_tr_tree = tr_fold[base_feat_cols].copy()
        X_tr_tree["unit_base_pred"] = in_unit
        X_tr_tree["spatial_prior"] = in_sp
        X_tr_tree["prj_eb_premium"] = in_eb
        X_tr_tree["eb_log_price"] = in_unit + in_eb

        X_val_tree = val_fold[base_feat_cols].copy()
        X_val_tree["unit_base_pred"] = u_val
        X_val_tree["spatial_prior"] = sp_val
        X_val_tree["prj_eb_premium"] = eb_val
        X_val_tree["eb_log_price"] = u_val + eb_val

        X_tst_tree = test_fe[base_feat_cols].copy()
        X_tst_tree["unit_base_pred"] = u_tst
        X_tst_tree["spatial_prior"] = sp_tst
        X_tst_tree["prj_eb_premium"] = eb_tst
        X_tst_tree["eb_log_price"] = u_tst + eb_tst

        # --- Model 1: HistGradientBoostingRegressor ---
        X_tr_hgb = X_tr_tree.copy()
        X_val_hgb = X_val_tree.copy()
        X_tst_hgb = X_tst_tree.copy()
        for c in hgb_cat_cols:
            X_tr_hgb[c] = X_tr_hgb[c].astype("category")
            X_val_hgb[c] = X_val_hgb[c].astype("category")
            X_tst_hgb[c] = X_tst_hgb[c].astype("category")

        hgb = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.03,
            max_iter=1500,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=2.0,
            categorical_features=hgb_cat_cols,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=42 + fold,
        )
        hgb.fit(X_tr_hgb, y_tr_fold)
        oof_hgb[val_idx] = hgb.predict(X_val_hgb)
        pred_hgb += hgb.predict(X_tst_hgb) / n_splits

        # --- Model 2: ExtraTreesRegressor ---
        et_cols = [c for c in X_tr_tree.columns if c not in hgb_cat_cols]
        et_imp = SimpleImputer(strategy="median")
        X_tr_et = et_imp.fit_transform(X_tr_tree[et_cols])
        X_val_et = et_imp.transform(X_val_tree[et_cols])
        X_tst_et = et_imp.transform(X_tst_tree[et_cols])

        et = ExtraTreesRegressor(
            n_estimators=250,
            max_depth=16,
            min_samples_leaf=5,
            max_features=0.7,
            n_jobs=-1,
            random_state=42 + fold,
        )
        et.fit(X_tr_et, y_tr_fold)
        oof_et[val_idx] = et.predict(X_val_et)
        pred_et += et.predict(X_tst_et) / n_splits

        # --- Model 3: Full Ridge Regression (OneHot project_id + agent_code + spatial_prior) ---
        full_ridge_prep = ColumnTransformer([
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]), num_cols + ["spatial_prior"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [
                "project_id", "furnished", "view_type", "nearest_station", "nearest_line", "agent_code"
            ])
        ])

        tr_r = tr_fold.copy()
        tr_r["spatial_prior"] = in_sp
        val_r = val_fold.copy()
        val_r["spatial_prior"] = sp_val
        tst_r = test_fe.copy()
        tst_r["spatial_prior"] = sp_test_full

        ridge_pipe = Pipeline([("prep", full_ridge_prep), ("model", Ridge(alpha=3.0))])
        ridge_pipe.fit(tr_r, y_tr_fold)
        oof_ridge[val_idx] = ridge_pipe.predict(val_r)
        pred_ridge += ridge_pipe.predict(tst_r) / n_splits

        # --- Model 4: LightGBM (ถ้ามีในเครื่อง) ---
        if HAS_LGB:
            X_tr_lgb = X_tr_hgb.copy()
            X_tr_lgb["project_id"] = tr_fold["project_id"].astype("category")
            X_val_lgb = X_val_hgb.copy()
            X_val_lgb["project_id"] = val_fold["project_id"].astype("category")
            X_tst_lgb = X_tst_hgb.copy()
            X_tst_lgb["project_id"] = test_fe["project_id"].astype("category")

            lgb_model = lgb.LGBMRegressor(
                objective="regression",
                metric="rmse",
                learning_rate=0.03,
                n_estimators=2500,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42 + fold,
                n_jobs=-1,
                verbose=-1,
            )
            lgb_model.fit(
                X_tr_lgb,
                y_tr_fold,
                eval_set=[(X_val_lgb, y_val_fold)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            oof_lgb[val_idx] = lgb_model.predict(X_val_lgb)
            pred_lgb += lgb_model.predict(X_tst_lgb) / n_splits

        fold_eb = root_mean_squared_error(y_val_fold, oof_eb[val_idx])
        fold_hgb = root_mean_squared_error(y_val_fold, oof_hgb[val_idx])
        print(f"   Fold {fold:2d} | Within-EB RMSLE: {fold_eb:.5f} | HistGBM RMSLE: {fold_hgb:.5f}")

    # รวมผลทำนาย (Ensemble Blending)
    if HAS_LGB:
        oof_blend = 0.45 * oof_eb + 0.20 * oof_lgb + 0.15 * oof_hgb + 0.10 * oof_et + 0.10 * oof_ridge
        pred_blend = 0.45 * pred_eb + 0.20 * pred_lgb + 0.15 * pred_hgb + 0.10 * pred_et + 0.10 * pred_ridge
    else:
        oof_blend = 0.55 * oof_eb + 0.20 * oof_hgb + 0.12 * oof_et + 0.13 * oof_ridge
        pred_blend = 0.55 * pred_eb + 0.20 * pred_hgb + 0.12 * pred_et + 0.13 * pred_ridge

    cv_rmsle = root_mean_squared_error(y_log, oof_blend)
    print(f"\n>>> Overall 10-Fold CV RMSLE (Ensemble): {cv_rmsle:.5f} <<<")

    final_prices = np.expm1(pred_blend)
    final_prices = np.maximum(final_prices, 0.0)

    submission = pd.DataFrame({
        "id": test_df["id"].astype(int),
        "price": np.round(final_prices, 2)
    })
    output_path = "submission.csv"
    submission.to_csv(output_path, index=False)

    print(f"4. สร้างไฟล์ '{output_path}' สำเร็จ! (จำนวน {len(submission):,} แถว)")
    print(submission.head(10))


if __name__ == "__main__":
    main()