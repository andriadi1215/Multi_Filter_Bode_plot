import numpy as np
from scipy import signal


def _check_frequency(frequency):
    f = np.asarray(frequency, dtype=float)
    if np.any(f <= 0):
        raise ValueError("Frekuensi harus lebih besar dari 0")
    return f


def _response(num, den, frequency):
    """Evaluasi H(s) = num/den di s = jw."""
    f = _check_frequency(frequency)
    _, H = signal.freqs(num, den, worN=2 * np.pi * f)
    magnitude = 20 * np.log10(np.maximum(np.abs(H), 1e-12))
    phase = np.angle(H, deg=True)
    return H, magnitude, phase


def low_pass_filter(frequency, R, C):
    if R <= 0 or C <= 0:
        raise ValueError("R dan C harus lebih besar dari 0")
    tau = R * C
    return _response([1], [tau, 1], frequency)


def high_pass_filter(frequency, R, C):
    if R <= 0 or C <= 0:
        raise ValueError("R dan C harus lebih besar dari 0")
    tau = R * C
    return _response([tau, 0], [tau, 1], frequency)


def band_pass_filter(frequency, R1, C1, R2, C2):
    if min(R1, C1, R2, C2) <= 0:
        raise ValueError("Semua komponen harus lebih besar dari 0")
    if cutoff_rc(R1, C1) >= cutoff_rc(R2, C2):
        raise ValueError("f_low (HPF) harus lebih kecil dari f_high (LPF)")
    t1, t2 = R1 * C1, R2 * C2
    num = [t1, 0]
    den = np.polymul([t1, 1], [t2, 1])
    return _response(num, den, frequency)


def band_stop_filter(frequency, f0, Q):
    if f0 <= 0 or Q <= 0:
        raise ValueError("f0 dan Q harus lebih besar dari 0")
    w0 = 2 * np.pi * f0
    return _response([1, 0, w0**2], [1, w0 / Q, w0**2], frequency)


# ---------- Frekuensi cutoff (untuk penanda di grafik) ----------

def cutoff_rc(R, C):
    """fc filter RC orde 1 (Hz)."""
    return 1 / (2 * np.pi * R * C)


def band_edges(f0, Q):
    """Frekuensi -3 dB bawah dan atas untuk filter orde 2 (BPF/BSF)."""
    half = 1 / (2 * Q)
    root = np.sqrt(1 + half**2)
    return f0 * (root - half), f0 * (root + half)


def center_and_q(f_low, f_high):
    """Dari f_low dan f_high ke f0 dan Q (bandwidth = f_high - f_low)."""
    f0 = np.sqrt(f_low * f_high)
    return f0, f0 / (f_high - f_low)