# ml69-kaggle-final

Team **IdeaMee** — ML69 Kaggle competitions (host: Transfinity), course *Machine Learning (1322308)*.
ทีม IdeaMee — งาน Kaggle ของวิชา Machine Learning: repo นี้เก็บเฉพาะ **โค้ดโมเดล** และ **ไฟล์คำตอบ (.csv) ที่ส่งจริง**

## Branches
| Branch | Content |
|---|---|
| `main` | this README only |
| `ice` | code + submitted answers for **C1** (fraud) and **R3** (solar), by `ice` |
| `<your-name>` | each member pushes their own work to a branch named after themselves |

Please do not push to `main` directly; keep your work on your own branch.
กรุณาอย่า push เข้า `main` โดยตรง ให้ push ไปที่ branch ชื่อตัวเอง

## Layout of branch `ice`
```
common.py                 shared helpers: paths, F1 / threshold search, MAE, submission-format validator
c1/                       C1 fraud transaction detection (metric: F1 of the fraud class)
  c1_features.py          features (label-free + out-of-fold label features)
  c1_run.py               training, CV, threshold, writes the submission
  sub_fix200_k10r4.csv    submitted answer
r3/                       R3 hourly rooftop-solar energy (metric: MAE)
  r3_features.py          own NOAA solar geometry + physical multipliers + weather features
  r3_run.py               training, two validation views, writes the submission
  sub_soil2.csv           submitted answer
```
Competition data are **not** in this repo (Kaggle rules). Download them after joining the competition and put them in `data/raw/<id>/`, e.g. `data/raw/c1/train.csv`.

## Results (Public leaderboard, ~30% of test; final ranking uses the hidden 70%)
| Competition | Metric | Our CV estimate | Public score |
|---|---|---|---|
| C1 fraud | F1 (higher is better) | 0.881 | 0.88990 |
| R3 solar | MAE (lower is better) | 0.0411 | 0.03885 |

## Run
Python 3.13. Install: `pip install numpy pandas scikit-learn scipy lightgbm`.
```
python c1/c1_run.py fix200_k10r4 --folds 10 --repeats 4 --seed 7 --rounds 200
python r3/r3_run.py soil2 --bins --seeds 3
```
Outputs (predictions, OOF arrays, submission csv) are written to `outputs/c1/` and `outputs/r3/` (not tracked). Re-running with these commands reproduces the committed csv files exactly.

## Method in one paragraph each
- **C1:** fraud appears as short sessions (same user, gaps ≤ 30 min) that are 99.6% pure. LightGBM on label-free features (device sharing, amount vs the user's own history, time gaps, session statistics) plus out-of-fold label features (labels of other rows of the same user / device / session, computed with an inner K-fold to avoid leave-one-out leakage). Threshold = centre of the F1 plateau of the pooled out-of-fold predictions. About 20% of frauds are isolated single transactions that look like normal large purchases, which limits F1 to roughly 0.88–0.89.
- **R3:** sun position computed at mid-hour (Thailand UTC+7, +30 min) with NOAA formulas; physical base energy = capacity × plane-of-array irradiance × cloud transmittance × saturating soiling × temperature derate. LightGBM predicts the ratio energy / base. Two models: known houses (per-house efficiency factors from other days, overall and per hour) and new houses (no house factor). Night hours are exactly 0.

## Notes
- Keep this repository **private** while the competition is running (Kaggle rules: no private code sharing outside the team).
- Libraries blocked on the author's machine (Windows Application Control) were avoided: no pvlib, pyvrp, ortools, `sklearn.neighbors`.
