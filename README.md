# Calibration-based digital twins for personalizing deep brain stimulation policies

Code and result files for the manuscript "Calibration-Based Digital Twins for Personalizing Deep Brain Stimulation Policies in Virtual Parkinsonian Patients".

The package simulates virtual patients with the mean-field basal ganglia-thalamocortical model of van Albada and colleagues (J. Theor. Biol. 257, 2009), anchors them to resting subthalamic recordings from two public datasets, identifies a digital twin of each patient from a short calibration protocol, personalizes continuous, single-threshold and dual-threshold stimulation policies by Gaussian-process Bayesian optimization on the twin, and evaluates the policies on the patients against population-tuned policies and direct per-patient optimization.

## Environment

Results were produced on Windows 11 (Intel Core Ultra 7 155U) with Python 3.12.10 and the package versions pinned in `requirements.txt`.

```
python -m pip install -r requirements.txt
```

## Reproduction

```
python reproduce.py            # fast path: statistics, reported values and figures from the shipped result files (about 7 minutes)
python reproduce.py --full     # full path: download the recordings (about 35 GiB), then every simulation and optimization
```

The fast path starts from the shipped files in `results/` and rewrites `results/analysis/` (including `reported_values.json`, every value reported in the article), `results/model/single_ramp.csv`, `results/patients/virtual_vs_recorded.csv`, `results/test/traces.csv` and `figures/`, and writes the output of each step to `results/logs/`. The full path first runs `scripts/download_data.py`, which fetches the Madrid recordings from Zenodo (record 10078352, CC BY 4.0) and the first 900 MiB of 40 runs of the Dusseldorf dataset, which hold their first rest blocks (only the leading part of the longest blocks), from OpenNeuro (ds004998, version 1.2.2, CC0) into `data/`; the recordings themselves are not shipped. The computations behind the shipped results took about 17 hours on the machine above with several processes in parallel. The full path runs up to four steps at a time (`--workers`), the model-library and evaluation steps each start their own pool of worker processes, and a run time of the same order should be expected. `results/model/timing.json` depends on the machine and is not expected to match.

## Layout

| Path | Content |
|---|---|
| `twin/model.py` | mean-field model, Heun integration with delays, published parameter sets |
| `twin/linear.py` | fixed points, linear spectra and stability of the model |
| `twin/cohort.py` | severity axis between the published healthy, parkinsonian and 20-Hz states |
| `twin/spectra.py`, `twin/fifread.py`, `twin/dusseldorf.py` | spectral features, FIF reader, Dusseldorf rest-block extraction |
| `twin/anchor.py`, `twin/patients.py` | anchoring to recordings, virtual patients and hidden traits |
| `twin/closedloop.py` | stimulation, recording and sensing model, medication cycle, policies, cost |
| `twin/bo.py` | batched Gaussian-process Bayesian optimization |
| `twin/study.py` | calibration protocol, twin identification, policy optimization |
| `twin/stats.py`, `twin/plotstyle.py` | paired statistics, figure style |
| `scripts/` | one script per step; `reproduce.py` lists the order |
| `data/ds004998_selection.json` | the 40 Dusseldorf runs used (one per subject and medication state) |
| `results/model` | model verification, step-size check, single-threshold ramp check, timing of one medication cycle |
| `results/patients` | model library, recorded features and spectra, anchoring |
| `results/<cohort>` | patients, calibration, twins, policies, evaluation (cohorts `dev`, `test`, `synth`, `stress`, `test_rest30`, `test_rest120`) |
| `results/analysis` | summary statistics, paired comparisons, budget tables, twin accuracy |

## Notation

| Article | Code |
|---|---|
| STN population $\zeta$, relay nuclei $t$ | `z`, `s` (for example `vzp2`, `vsp1`) |
| severity $s$, off/on medication | `s`, `s_off`, `s_on` |
| delay scale $c$ | `c` (`ct_scale` in the model) |
| stimulation efficacy $g$, command $u$ | `g`, `u` |
| background level $\kappa$, noise level $\sigma_n$ | `kappa`, `noise` |
| continuous amplitude $A$ | `a` |
| single threshold $\eta$, $A$, ramp rate $R$ | `thr`, `a`, `ramp` |
| dual threshold $\eta_\mathrm{lo}$, $\delta$, $q$, $A_{\max}$, $R$ | `lo`, `gap`, `floor`, `a_max`, `ramp` |
| cost $J = B + \lambda U + w D$ | `J`, `B`, `U`, `V` (command variation) |

## Figures and tables

| Output | Script |
|---|---|
| Fig. 1 | `scripts/fig_overview.py` |
| Fig. 2 | `scripts/fig_anchor.py` |
| Fig. 3, Table I | `scripts/fig_twin_id.py`, `scripts/twin_accuracy.py`, `scripts/report_values.py` |
| Fig. 4 | `scripts/fig_traces.py` |
| Fig. 5, Table II | `scripts/fig_outcomes.py`, `scripts/analyze.py`, `scripts/report_values.py` |
| Fig. 6 | `scripts/fig_sensitivity.py` |
| Fig. S1 | `scripts/fig_single_ramp.py` |
| Tables S1 and S2 | `scripts/report_values.py` |
| All numbers in the text and tables | `scripts/report_values.py` (`results/analysis/reported_values.json`) |

## Main results

Madrid test cohort (34 virtual patients), medication-cycle deployment:

| Family | Configuration | J | B | U | better |
|---|---|---|---|---|---|
| none | no stimulation | 0.954 | 0.954 | 0.000 | |
| continuous | population | 0.415 | 0.176 | 0.477 |  |
| continuous | digital twin | 0.396 | 0.226 | 0.269 | 71% |
| continuous | checked twin | 0.396 | 0.205 | 0.455 | 71% |
| continuous | direct, 20 trials (reference) | 0.389 | 0.215 | 0.293 | 94% |
| single | population | 0.394 | 0.221 | 0.341 |  |
| single | digital twin | 0.453 | 0.299 | 0.220 | 59% |
| single | checked twin | 0.394 | 0.237 | 0.341 | 50% |
| single | direct, 40 trials (reference) | 0.391 | 0.258 | 0.236 | 91% |
| dual | population | 0.421 | 0.171 | 0.500 |  |
| dual | digital twin | 0.392 | 0.274 | 0.299 | 68% |
| dual | checked twin | 0.392 | 0.196 | 0.478 | 68% |
| dual | direct, 60 trials (reference) | 0.390 | 0.222 | 0.290 | 94% |

Columns: `J`, cohort median of the medication-cycle cost (beta burden plus 0.5 times stimulation energy plus 0.05 times command variation, lower is better); `B`, cohort median of the beta burden relative to the development-cohort median of the mean beta envelope in the off-medication rest recording of the calibration; `U`, cohort median of the stimulation energy (mean squared amplitude relative to 10 mV squared); `better`, share of patients with a lower cost than the population policy of the same family (patients with equal costs, mostly those for whom the checked twin fell back on the population policy, are not counted as better).

## License

MIT (see `LICENSE`). The recordings used for anchoring are not redistributed; `scripts/download_data.py` fetches them from their public repositories under their own licenses. The spectra and spectral features in `results/patients/` are derived from the Dusseldorf dataset of Rassoulou et al. (OpenNeuro ds004998, https://doi.org/10.18112/openneuro.ds004998.v1.2.2, CC0) and the Madrid dataset of Pardo-Valencia et al. (Zenodo, https://doi.org/10.5281/zenodo.10078352, CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).
