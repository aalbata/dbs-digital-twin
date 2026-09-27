"""Spectral features used identically for patient recordings and model
signals: Welch power spectrum, aperiodic fit (a line in log-log coordinates
fitted over 3-45 Hz outside the beta range), and the beta peak (frequency and
height above the aperiodic fit, in dB, within 12-35 Hz)."""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, decimate

FIT_BAND = (3.0, 45.0)
BETA_SEARCH = (12.0, 35.0)
EXCLUDE = (11.0, 36.0)


def psd(x, fs, seg_s=2.0):
    """Welch spectrum with 2-s Hann segments and 50% overlap."""
    n = int(round(seg_s * fs))
    return welch(x, fs=fs, nperseg=n, noverlap=n // 2, axis=-1)


def features(f, P):
    """Beta peak frequency (Hz), peak height above the aperiodic fit (dB),
    aperiodic exponent and the relative 13-30 Hz power above the fit.
    P may be (n_freq,) or (n_signals, n_freq)."""
    P = np.atleast_2d(P)
    fit = (f >= FIT_BAND[0]) & (f <= FIT_BAND[1])
    bg = fit & ((f < EXCLUDE[0]) | (f > EXCLUDE[1]))
    lf = np.log10(f[bg])
    out = []
    for p in P:
        slope, icpt = np.polyfit(lf, np.log10(p[bg]), 1)
        resid = 10 * (np.log10(p) - (icpt + slope * np.log10(np.maximum(f, 1e-9))))
        s = (f >= BETA_SEARCH[0]) & (f <= BETA_SEARCH[1])
        k = np.argmax(np.where(s, resid, -np.inf))
        beta = (f >= 13) & (f <= 30)
        excess = np.clip(p[beta] - 10 ** (icpt + slope * np.log10(f[beta])), 0, None).sum() / p[fit].sum()
        out.append(dict(peak_hz=float(f[k]), peak_db=float(resid[k]), exponent=float(-slope),
                        beta_excess=float(excess)))
    return out


def preprocess(x, fs, target_fs=500.0, hp=1.0):
    """High-pass (zero phase) and resample to target_fs for spectral analysis."""
    sos = butter(2, hp, btype="high", fs=fs, output="sos")
    y = sosfiltfilt(sos, x, axis=-1)
    q = int(round(fs / target_fs))
    if q > 1 and abs(fs / q - target_fs) < 1e-6 * target_fs:
        return decimate(y, q, ftype="fir", zero_phase=True, axis=-1), target_fs
    from scipy.signal import resample_poly
    from fractions import Fraction
    fr = Fraction(target_fs / fs).limit_denominator(1000)
    return resample_poly(y, fr.numerator, fr.denominator, axis=-1), target_fs


def residual_db(f, P):
    """Spectrum in dB above its aperiodic fit (same fit as features)."""
    fit = (f >= FIT_BAND[0]) & (f <= FIT_BAND[1])
    bg = fit & ((f < EXCLUDE[0]) | (f > EXCLUDE[1]))
    slope, icpt = np.polyfit(np.log10(f[bg]), np.log10(P[bg]), 1)
    return 10 * (np.log10(P) - (icpt + slope * np.log10(f)))
