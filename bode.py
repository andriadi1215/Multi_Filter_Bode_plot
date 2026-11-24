import numpy as np
import matplotlib.pyplot as plt

from filters import (
    low_pass_filter,
    high_pass_filter,
    band_pass_filter,
    band_stop_filter,
    cutoff_rc,
    band_edges,
)

COLORS = {
    "LPF": "tab:blue",
    "HPF": "tab:orange",
    "BPF": "tab:green",
    "BSF": "tab:red",
}


def plot_all_filters(
    frequency,
    R=1000,
    C=100e-9,
    bpf=(1000, 159.15e-9, 1000, 31.83e-9),
    bsf=(2000, 2),
):
    """Plot Bode LPF, HPF, BPF, BSF dalam satu figure + penanda cutoff."""
    R1, C1, R2, C2 = bpf
    f0, Q = bsf

    responses = {
        "LPF": low_pass_filter(frequency, R, C),
        "HPF": high_pass_filter(frequency, R, C),
        "BPF": band_pass_filter(frequency, R1, C1, R2, C2),
        "BSF": band_stop_filter(frequency, f0, Q),
    }

    fc_rc = cutoff_rc(R, C)
    f_low, f_high = cutoff_rc(R1, C1), cutoff_rc(R2, C2)
    s_low, s_high = band_edges(f0, Q)

    cutoffs = {
        "LPF": [fc_rc],
        "HPF": [fc_rc],
        "BPF": [f_low, f_high],
        "BSF": [s_low, s_high],
    }
    labels = {
        "LPF": f"LPF (fc = {fc_rc:.0f} Hz)",
        "HPF": f"HPF (fc = {fc_rc:.0f} Hz)",
        "BPF": f"BPF ({f_low:.0f} - {f_high:.0f} Hz)",
        "BSF": f"BSF ({s_low:.0f} - {s_high:.0f} Hz)",
    }

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for name, (_, mag, phase) in responses.items():
        color = COLORS[name]
        ax[0].semilogx(frequency, mag, color=color, label=labels[name])
        ax[1].semilogx(frequency, phase, color=color, label=labels[name])

        # Penanda frekuensi cutoff
        for fc in cutoffs[name]:
            ax[0].axvline(fc, color=color, ls=":", alpha=0.6)
            ax[1].axvline(fc, color=color, ls=":", alpha=0.6)

    # Garis -3 dB
    ax[0].axhline(-3, color="gray", ls="--", lw=1, label="-3 dB")

    ax[0].set_title("Bode Plot - Magnitude Response")
    ax[0].set_ylabel("Magnitude (dB)")
    ax[0].set_ylim(-60, 5)

    ax[1].set_title("Bode Plot - Phase Response")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Phase (Degrees)")
    ax[1].set_yticks(range(-180, 181, 45))

    for a in ax:
        a.grid(True, which="both", alpha=0.4)
        a.legend(loc="best", fontsize=8)
        a.set_xlim(frequency.min(), frequency.max())

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    frequency = np.logspace(1, 5, 1000)
    fig = plot_all_filters(frequency)
    plt.show()