"""Anchor one virtual patient to each recorded STN hemisphere.

Each OFF-medication resting recording gives a beta peak frequency and height
(twin.spectra.features on the 3-45 Hz Welch spectrum). The model library with
the recording background of twin.anchor (exponent chi = median aperiodic
exponent of all OFF recordings) gives the same two features for every
(delay scale c, severity s); the hemisphere is assigned the (c, s_off) with
the nearest features. The ON recording of the same hemisphere is assigned
s_on on the same delay scale by its peak height.
Madrid: dataset 2 recordings (OFF and ON) where available, otherwise the
dataset 1 OFF recording. Dusseldorf: rest block, adjacent bipolar pair with
the largest OFF beta peak, the same pair for ON.
Usage: python scripts/anchor_patients.py
Writes results/patients/anchor.csv (one row per hemisphere),
recording_features.csv, recording_spectra.npz (3-45 Hz Welch spectra) and
anchor_background.json."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from twin.anchor import FREQS, feature_table, fit_features  # noqa: E402
from twin.spectra import psd, preprocess, features  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results" / "patients"


def spectrum(x, fs):
    y, fs2 = preprocess(x, fs)
    f, P = psd(y, fs2)
    return np.interp(FREQS, f, P)


def madrid():
    base = DATA / "madrid"
    rates = {int(m.group(1)): float(m.group(2)) for m in
             (re.match(r"^(\d+)//(\d+)", ln.strip()) for ln in (base / "README.txt").read_text().splitlines()) if m}
    files = {}
    for ds in ("dataset_1", "dataset_2"):
        for f in sorted((base / ds).glob("*.mat")):
            m = re.match(r"P(\d+)_(\d)_(OFF|ON)_(MASTN|LASTN)\.mat", f.name)
            files[(int(m.group(1)), m.group(4), m.group(3), int(m.group(2)))] = f
    out = []
    for (pat, side, med, d), f in sorted(files.items()):
        if d == 1 and (pat, side, "OFF", 2) in files:
            continue
        x = next(v for k, v in loadmat(f).items() if not k.startswith("__")).ravel().astype(float)
        out.append(dict(cohort="madrid", patient=f"M{pat:02d}", hemisphere=side, med=med, duration_s=len(x) / rates[pat],
                        P=spectrum(x, rates[pat])))
    return out


def dusseldorf():
    from twin.dusseldorf import rest_recordings
    return rest_recordings(DATA / "ds004998" / "rest", spectrum)


def main():
    recs = madrid()
    if (DATA / "ds004998" / "rest").exists():
        recs += dusseldorf()
    for r in recs:
        r.update(features(FREQS, r["P"])[0])
    chi = float(np.median([r["exponent"] for r in recs if r["med"] == "OFF"]))
    lib = dict(np.load(OUT / "model_library.npz"))
    table = feature_table(lib, chi)
    rows = []
    hemis = {}
    for r in recs:
        hemis.setdefault((r["cohort"], r["patient"], r["hemisphere"]), {})[r["med"]] = r
    for (cohort, pat, side), meds in sorted(hemis.items()):
        off = meds["OFF"]
        fo = fit_features(off["peak_hz"], off["peak_db"], lib, table)
        row = dict(cohort=cohort, patient=pat, hemisphere=side, c=fo["c"], s_off=fo["s"],
                   peak_hz_off=off["peak_hz"], peak_db_off=off["peak_db"],
                   model_peak_hz_off=fo["model_peak_hz"], model_peak_db_off=fo["model_peak_db"])
        if "ON" in meds:
            on = meds["ON"]
            fn = fit_features(on["peak_hz"], on["peak_db"], lib, table, c_fixed=fo["c"])
            row.update(s_on=fn["s"], peak_hz_on=on["peak_hz"], peak_db_on=on["peak_db"],
                       model_peak_db_on=fn["model_peak_db"])
        rows.append(row)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "anchor.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k != "P"} for r in recs]).to_csv(OUT / "recording_features.csv", index=False)
    (OUT / "anchor_background.json").write_text(json.dumps(dict(chi=chi), indent=1))
    np.savez_compressed(OUT / "recording_spectra.npz", f=FREQS,
                        **{f"{r['cohort']}|{r['patient']}|{r['hemisphere']}|{r['med']}": r["P"] for r in recs})
    pd.set_option("display.width", 250)
    print(f"chi = {chi:.3f}")
    print(df.round(2).to_string())


if __name__ == "__main__":
    main()
