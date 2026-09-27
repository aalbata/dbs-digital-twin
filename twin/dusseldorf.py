"""Resting STN recordings of the Dusseldorf cohort (OpenNeuro ds004998,
Rassoulou et al., Sci. Data 2024).

For every subject and medication state, the leading part of one run is read
with twin.fifread. The rest block is the first 'rest' event of the run's
events file; intervals marked 'bad_lfp' are excluded. STN contacts are mapped
through the subject's montage file, contacts marked bad in the channels file
are dropped, and adjacent contacts on the same side form bipolar pairs. For
each hemisphere the pair with the largest OFF-medication beta peak is used in
both medication states. Spectra are averaged over the good stretches of the
rest block that are at least 2 s long."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .fifread import read_partial_fif
from .spectra import features, preprocess, psd

MIN_SEG_S = 2.0


def _good_intervals(events, t_end):
    ev = pd.read_csv(events, sep="\t")
    rest = ev[ev.trial_type == "rest"].iloc[0]
    a, b = float(rest.onset), min(float(rest.onset + rest.duration), t_end)
    bad = ev[ev.trial_type == "bad_lfp"][["onset", "duration"]].values
    cuts = [(a, b)]
    for o, d in bad:
        new = []
        for s, e in cuts:
            if o + d <= s or o >= e:
                new.append((s, e))
            else:
                if o > s:
                    new.append((s, o))
                if o + d < e:
                    new.append((o + d, e))
        cuts = new
    return [(s, e) for s, e in cuts if e - s >= MIN_SEG_S], b - a


def _spectrum(x, fs, intervals, spectrum_fn):
    """Length-weighted mean spectrum over the good intervals."""
    num, den = 0.0, 0.0
    for s, e in intervals:
        seg = x[int(round(s * fs)):int(round(e * fs))]
        num = num + spectrum_fn(seg, fs) * (e - s)
        den += e - s
    return num / den


def rest_recordings(folder, spectrum_fn):
    folder = Path(folder)
    out = []
    for fif in sorted(folder.glob("*_meg.fif")):
        m = re.match(r"(sub-\w+?)_ses-PeriOp_task-(\w+?)_acq-(Med\w+?)_run-(\d)(_split-\d+)?_meg\.fif", fif.name)
        sub, med = m.group(1), m.group(3)
        stem = fif.name[:-len("_meg.fif")]
        events = folder / f"{stem}_events.tsv"
        chans = pd.read_csv(folder / f"{stem.split('_split-')[0]}_channels.tsv", sep="\t") \
            if not (folder / f"{stem}_channels.tsv").exists() else pd.read_csv(folder / f"{stem}_channels.tsv", sep="\t")
        bad = set(chans.loc[chans.status == "bad", "name"])
        mont = pd.read_csv(folder / f"{sub}_ses-PeriOp_montage.tsv", sep="\t")
        fs, names, x, _ = read_partial_fif(fif, lambda n: n.startswith("EEG"))
        t_end = x.shape[1] / fs
        intervals, rest_len = _good_intervals(events, t_end)
        idx = {n: k for k, n in enumerate(names)}
        for side in ("right", "left"):
            contacts = [c for c in mont[f"{side}_contacts_old"].dropna() if c in idx]
            for a, b in zip(contacts[:-1], contacts[1:]):
                if a in bad or b in bad:
                    continue
                sig = x[idx[a]] - x[idx[b]]
                P = _spectrum(sig, fs, intervals, spectrum_fn)
                out.append(dict(cohort="dusseldorf", patient=sub, hemisphere=side, pair=f"{a}-{b}", med=med[3:].upper(),
                                duration_s=sum(e - s for s, e in intervals), rest_s=rest_len, P=P))
    # keep, per hemisphere, the pair with the largest OFF beta peak
    from .anchor import FREQS
    keep = []
    for (sub, side), grp in pd.DataFrame(out).groupby(["patient", "hemisphere"]):
        offs = grp[grp.med == "OFF"]
        if offs.empty:
            continue
        best = max(offs.index, key=lambda k: features(FREQS, out[k]["P"])[0]["peak_db"])
        pair = out[best]["pair"]
        keep += [out[k] for k in grp.index if out[k]["pair"] == pair]
    return keep
