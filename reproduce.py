"""Reproduce the results, statistics and figures.

    python reproduce.py            fast path: statistics, reported values and
                                   figures from the shipped result files
                                   (about 7 minutes)
    python reproduce.py --full     full path: download the recordings, run
                                   every simulation and optimization, then the
                                   fast path (see README for the run time)

The full path runs the steps below in order; steps within one stage are
independent and run in parallel worker processes (--workers, default 4)."""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

FULL = [
    [["download_data.py"]],
    [["model_verification.py"], ["step_check.py"], ["model_library.py"]],
    [["anchor_patients.py"]],
    [["calibrate_twins.py", "dev"], ["calibrate_twins.py", "test"], ["calibrate_twins.py", "synth"],
     ["calibrate_twins.py", "stress"], ["calibrate_twins.py", "test", "16", "30"],
     ["calibrate_twins.py", "test", "16", "120"]],
    [["personalize.py", "dev", "population", f] for f in ("continuous", "single", "dual")]
    + [["personalize.py", c, "twin", f] for c in ("dev", "test", "synth", "stress")
       for f in ("continuous", "single", "dual")]
    + [["personalize.py", "test", "direct", f] for f in ("continuous", "single", "dual")]
    + [["personalize.py", c, "direct", "dual"] for c in ("synth", "stress")]
    + [["personalize.py", c, "twin", "dual"] for c in ("test_rest30", "test_rest120")],
    [["timing.py"]],
    [["evaluate.py", c] for c in ("dev", "test", "synth", "stress", "test_rest30", "test_rest120")]
    + [["drift.py", c] for c in ("dev", "test")],
]

FAST = [
    ["twin_accuracy.py"],
    ["analyze.py", "dev"], ["analyze.py", "test"], ["analyze.py", "synth"], ["analyze.py", "stress"],
    ["analyze.py", "test_rest30"], ["analyze.py", "test_rest120"],
    ["fig_overview.py"], ["fig_anchor.py"], ["fig_twin_id.py"], ["fig_traces.py"], ["fig_outcomes.py"],
    ["fig_single_ramp.py"], ["fig_sensitivity.py"], ["report_values.py"],
]


def run(args, log_dir):
    name = "_".join(a.replace(".py", "") for a in args)
    t0 = time.time()
    with open(log_dir / f"{name}.txt", "w") as fh:
        r = subprocess.run([PY, "-u", str(ROOT / "scripts" / args[0])] + args[1:], cwd=ROOT, stdout=fh,
                           stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{' '.join(args)} failed, see {log_dir / (name + '.txt')}")
    print(f"{' '.join(args)}: {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stages = FULL if a.full else []
    for stage in stages:
        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(lambda s: run(s, log_dir), stage))
    for step in FAST:
        run(step, log_dir)


if __name__ == "__main__":
    main()
