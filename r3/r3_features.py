"""R3 solar: own NOAA solar geometry (pvlib is blocked on this machine), physical expected energy, weather imputation,
cloud lags and house-efficiency factors."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find common.py at the repo root

import numpy as np
import pandas as pd

from common import DATA


def solar_geometry(dt_local, lat, lon, tilt_deg, az_deg, shift_min=30):
    """NOAA formulas at mid-hour (datetime + 30 min, Thailand UTC+7). Returns dict of arrays."""
    t = pd.to_datetime(dt_local) + pd.to_timedelta(shift_min, unit="m")
    doy = t.dt.day_of_year.to_numpy()
    hr = (t.dt.hour + t.dt.minute / 60).to_numpy()
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
           - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hr * 60 + eqt + 4 * lon - 60 * 7
    ha = np.deg2rad(tst / 4 - 180)
    la = np.deg2rad(lat)
    cz = np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)
    sz = np.sqrt(np.clip(1 - cz ** 2, 0, 1))
    saz = np.mod(np.arctan2(-np.sin(ha) * np.cos(dec), np.sin(dec) * np.cos(la) - np.cos(dec) * np.sin(la) * np.cos(ha)), 2 * np.pi)
    tilt = np.deg2rad(tilt_deg)
    paz = np.deg2rad(az_deg)
    caoi = cz * np.cos(tilt) + sz * np.sin(tilt) * np.cos(saz - paz)
    e0 = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)  # earth-sun distance factor
    czp = np.clip(cz, 0.0, None)
    ghi_cs = np.where(cz > 0.0, 1098.0 * czp * np.exp(-0.059 / np.maximum(czp, 1e-3)) * e0, 0.0)  # Haurwitz clear-sky GHI
    beam_poa = ghi_cs * np.clip(caoi, 0, None) / np.maximum(czp, 0.1)  # crude beam on plane of array
    diff_poa = 0.15 * ghi_cs * (1 + np.cos(tilt)) / 2
    poa = 0.85 * beam_poa + diff_poa
    return dict(cz=cz, saz=saz, caoi=caoi, ghi_cs=ghi_cs, poa=poa, doy=doy, hour=t.dt.hour.to_numpy(), e0=e0,
                sun_up=(cz > 0).astype(int), elev=np.rad2deg(np.arcsin(np.clip(cz, -1, 1))))


def load_r3():
    tr = pd.read_csv(DATA / "r3" / "train.csv")
    te = pd.read_csv(DATA / "r3" / "test.csv")
    df = pd.concat([tr.assign(is_train=1), te.assign(is_train=0, energy_kwh=np.nan)], ignore_index=True)
    df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df["hod"] = pd.to_datetime(df["datetime"]).dt.hour
    return df


def build_features(df):
    f = pd.DataFrame(index=df.index)
    sg = solar_geometry(df["datetime"], df["lat"].to_numpy(), df["lon"].to_numpy(), df["tilt_deg"].to_numpy(), df["azimuth_deg"].to_numpy())
    for k in ["cz", "saz", "caoi", "ghi_cs", "poa", "doy", "elev", "e0"]:
        f[k] = sg[k]
    f["sun_up"] = sg["sun_up"]
    f["hour"] = df["hod"]
    cap = df["capacity_kw"].to_numpy()
    f["cap"] = cap
    f["tilt"] = df["tilt_deg"]
    f["azim"] = df["azimuth_deg"]
    f["lat"] = df["lat"]
    f["lon"] = df["lon"]
    f["base"] = cap * f["poa"] / 1000.0  # physical expected energy (kWh) before weather/soiling/house effects
    f["days_clean"] = df["days_since_cleaning"]
    # weather imputation inside the house-day: linear interpolation along the 24 h, then same-timestamp mean of the
    # nearest-latitude houses is NOT used; fall back to the daily/house median
    w = df[["house_id", "date", "hod", "temp_c", "cloud_cover", "humidity"]].copy()
    w = w.sort_values(["house_id", "date", "hod"])
    for c in ["temp_c", "cloud_cover", "humidity"]:
        miss = w[c].isna()
        w[c + "_nan"] = miss.astype(int)
        w[c] = w.groupby(["house_id", "date"])[c].transform(lambda s: s.interpolate(limit_direction="both"))
        w[c] = w[c].fillna(w.groupby("hod")[c].transform("median"))
    w = w.reindex(df.index)
    for c in ["temp_c", "cloud_cover", "humidity"]:
        f[c] = w[c]
        f[c + "_nan"] = w[c + "_nan"]
    g = w.groupby([df["house_id"], df["date"]])
    for c in ["cloud_cover", "humidity"]:
        f[c + "_prev"] = g[c].shift(1)
        f[c + "_next"] = g[c].shift(-1)
        f[c + "_day_mean"] = g[c].transform("mean")
    f["cloud_roll3"] = g["cloud_cover"].transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
    f["temp_day_max"] = g["temp_c"].transform("max")
    f["cloud_x_clean"] = f["cloud_cover"] * f["days_clean"]
    f["base_x_cloudclear"] = f["base"] * (1 - f["cloud_cover"])
    # explicit multiplicative physics terms
    cc = f["cloud_cover"].clip(0, 1)
    f["kc"] = 1 - 0.75 * cc ** 3.4  # Kasten-Czeplak cloud transmittance
    f["soil"] = 1 - 0.164 * (1 - np.exp(-f["days_clean"] / 45.7))  # saturating soiling curve measured within houses
    f["soil_lin"] = np.clip(1 - 0.0015 * f["days_clean"], 0.3, 1)
    cell = f["temp_c"] + 0.025 * f["poa"] * f["kc"]
    f["cell_t"] = cell
    f["derate"] = 1 - 0.004 * (cell - 25)
    f["base_phys"] = f["base"] * f["kc"] * f["soil"] * f["derate"]
    f["base_kc"] = f["base"] * f["kc"]
    hg = df.groupby("house_id")["days_since_cleaning"]
    f["house_dc_mean"] = hg.transform("mean")
    f["house_dc_max"] = hg.transform("max")
    f["house_dc_min"] = hg.transform("min")
    f["house_dc_std"] = hg.transform("std")
    f["dc_vs_house"] = f["days_clean"] - f["house_dc_mean"]
    f["kc_next"] = g["cloud_cover"].shift(-1).clip(0, 1).pipe(lambda x: 1 - 0.75 * x ** 3.4)
    f["kc_prev"] = g["cloud_cover"].shift(1).clip(0, 1).pipe(lambda x: 1 - 0.75 * x ** 3.4)
    return f, sg
