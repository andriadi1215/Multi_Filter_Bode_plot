import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from filters import (
    low_pass_filter,
    high_pass_filter,
    band_pass_filter,
    band_stop_filter,
    cutoff_rc,
    band_edges,
)
from bode import plot_all_filters, COLORS

# ------------------------------------------------------------------
# Konfigurasi halaman
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Filter Bode Plotter",
    page_icon="📊",
    layout="wide",
)

st.title("Multi-Filter Bode Plotter")
st.write(
    "Aplikasi analisis karakteristik filter LPF, HPF, BPF, dan BSF "
    "melalui Bode plot (magnitude dan fase)."
)
st.divider()

LABEL_TO_NAME = {
    "Low-Pass Filter (LPF)": "LPF",
    "High-Pass Filter (HPF)": "HPF",
    "Band-Pass Filter (BPF)": "BPF",
    "Band-Stop Filter (BSF)": "BSF",
}
MODE_SATU = "Satu filter"
MODE_SEMUA = "Semua filter (perbandingan)"

# ------------------------------------------------------------------
# Pilihan mode dan jenis filter
# ------------------------------------------------------------------
mode = st.radio("Mode Tampilan", [MODE_SATU, MODE_SEMUA], horizontal=True)
satu = mode == MODE_SATU

if satu:
    label = st.selectbox("Pilih Jenis Filter", list(LABEL_TO_NAME))
    daftar = [LABEL_TO_NAME[label]]
else:
    daftar = ["LPF", "HPF", "BPF", "BSF"]

# ------------------------------------------------------------------
# Parameter filter
# ------------------------------------------------------------------
st.subheader("Parameter Filter")
p = {}

if "LPF" in daftar or "HPF" in daftar:
    st.markdown("**" + " / ".join(n for n in ("LPF", "HPF") if n in daftar)
                + " (rangkaian RC)**")
    c1, c2 = st.columns(2)
    p["R"] = c1.number_input("Resistor (Ohm)", min_value=1.0, value=1000.0)
    p["C"] = c2.number_input(
        "Kapasitor (nF)", min_value=0.001, value=100.0, format="%.3f"
    ) * 1e-9

if "BPF" in daftar:
    st.markdown("**BPF (HPF dikaskade dengan LPF)**")
    c1, c2, c3, c4 = st.columns(4)
    p["R1"] = c1.number_input("Resistor HPF (Ohm)", min_value=1.0, value=1000.0)
    p["C1"] = c2.number_input(
        "Kapasitor HPF (nF)", min_value=0.001, value=159.15, format="%.3f"
    ) * 1e-9
    p["R2"] = c3.number_input("Resistor LPF (Ohm)", min_value=1.0, value=1000.0)
    p["C2"] = c4.number_input(
        "Kapasitor LPF (nF)", min_value=0.001, value=31.83, format="%.3f"
    ) * 1e-9

if "BSF" in daftar:
    st.markdown("**BSF (notch)**")
    c1, c2 = st.columns(2)
    p["f0"] = c1.number_input("Center Frequency (Hz)", min_value=1.0, value=2000.0)
    p["Q"] = c2.number_input("Quality Factor", min_value=0.1, value=2.0)

# ------------------------------------------------------------------
# Pengaturan frekuensi
# ------------------------------------------------------------------
st.subheader("Pengaturan Frekuensi")
c1, c2, c3 = st.columns(3)
f_min = c1.number_input("Frekuensi Minimum (Hz)", min_value=1.0, value=10.0)
f_max = c2.number_input("Frekuensi Maksimum (Hz)", min_value=2.0, value=100000.0)
f_uji = c3.number_input("Frekuensi yang dianalisis (Hz)", min_value=1.0, value=1000.0)

if f_max <= f_min:
    st.error("Frekuensi maksimum harus lebih besar dari minimum.")
    st.stop()

frequency = np.logspace(np.log10(f_min), np.log10(f_max), 1000)


# ------------------------------------------------------------------
# Fungsi bantu
# ------------------------------------------------------------------
def hitung(nama, freq):
    if nama == "LPF":
        return low_pass_filter(freq, p["R"], p["C"])
    if nama == "HPF":
        return high_pass_filter(freq, p["R"], p["C"])
    if nama == "BPF":
        return band_pass_filter(freq, p["R1"], p["C1"], p["R2"], p["C2"])
    return band_stop_filter(freq, p["f0"], p["Q"])


def frekuensi_cutoff(nama):
    if nama in ("LPF", "HPF"):
        return [cutoff_rc(p["R"], p["C"])]
    if nama == "BPF":
        return [cutoff_rc(p["R1"], p["C1"]), cutoff_rc(p["R2"], p["C2"])]
    return list(band_edges(p["f0"], p["Q"]))


def plot_single(nama, freq, mag, phase, cutoff):
    warna = COLORS[nama]
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax[0].semilogx(freq, mag, color=warna)
    ax[1].semilogx(freq, phase, color=warna)
    ax[0].axhline(-3, color="gray", ls="--", lw=1, label="-3 dB")

    for fc in cutoff:
        for a in ax:
            a.axvline(fc, color=warna, ls=":", alpha=0.7)
        ax[0].text(
            fc, 0.02, f"{fc:.0f} Hz",
            transform=ax[0].get_xaxis_transform(),
            rotation=90, va="bottom", ha="right", fontsize=8,
        )

    if nama == "BSF":
        ax[0].set_ylim(-60, 5)

    ax[0].set_title(f"Magnitude Response - {nama}")
    ax[0].set_ylabel("Magnitude (dB)")
    ax[1].set_title(f"Phase Response - {nama}")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Phase (Degrees)")

    for a in ax:
        a.grid(True, which="both", alpha=0.4)
        a.set_xlim(freq.min(), freq.max())
    ax[0].legend(loc="best", fontsize=8)

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Menghitung respons filter
# ------------------------------------------------------------------
try:
    hasil = {nama: hitung(nama, frequency) for nama in daftar}
except ValueError as e:
    st.error(str(e))
    st.stop()


# ------------------------------------------------------------------
# Hasil analisis
# ------------------------------------------------------------------
def tampilkan_analisis(nama):
    _, mag, _ = hasil[nama]
    cutoff = frekuensi_cutoff(nama)

    # Hasil di frekuensi yang dipilih pengguna
    _, m_uji, ph_uji = hitung(nama, np.array([f_uji]))
    c1, c2, c3 = st.columns(3)
    c1.metric("Magnitude", f"{m_uji[0]:.2f} dB")
    c2.metric("Phase", f"{ph_uji[0]:.2f}°")
    c3.metric("Frequency", f"{f_uji:.2f} Hz")

    st.markdown("**Parameter turunan**")

    if nama in ("LPF", "HPF"):
        fc = cutoff[0]
        _, m_fc, ph_fc = hitung(nama, np.array([fc]))
        a, b, c = st.columns(3)
        a.metric("Frekuensi cutoff (fc)", f"{fc:.2f} Hz")
        b.metric("Magnitude di fc", f"{m_fc[0]:.2f} dB")
        c.metric("Fase di fc", f"{ph_fc[0]:.2f}°")

    elif nama == "BPF":
        f_low, f_high = cutoff
        f_tengah = np.sqrt(f_low * f_high)
        a, b, c = st.columns(3)
        a.metric("f_low (sudut HPF)", f"{f_low:.2f} Hz")
        b.metric("f_high (sudut LPF)", f"{f_high:.2f} Hz")
        c.metric("Frekuensi tengah (f0)", f"{f_tengah:.2f} Hz")

        # Tepi -3 dB sebenarnya, dibaca dari kurva
        puncak = mag.max()
        idx = np.where(mag >= puncak - 3)[0]
        bawah = frequency[idx[0]] if idx[0] > 0 else None
        atas = frequency[idx[-1]] if idx[-1] < len(frequency) - 1 else None

        d, e, f, g = st.columns(4)
        d.metric("Puncak magnitude", f"{puncak:.2f} dB")
        e.metric("Tepi -3 dB bawah", f"{bawah:.1f} Hz" if bawah else "di luar rentang")
        f.metric("Tepi -3 dB atas", f"{atas:.1f} Hz" if atas else "di luar rentang")
        if bawah and atas:
            bw = atas - bawah
            g.metric("Bandwidth / Q", f"{bw:.0f} Hz / {np.sqrt(bawah * atas) / bw:.2f}")
        else:
            g.metric("Bandwidth / Q", "-")
        st.caption(
            "f_low dan f_high adalah frekuensi sudut masing-masing RC. "
            "Karena dua filter orde 1 dikaskade, puncak BPF bisa di bawah 0 dB "
            "dan tepi -3 dB sebenarnya bergeser dari f_low dan f_high."
        )

    else:  # BSF
        f_low, f_high = cutoff
        a, b, c, d = st.columns(4)
        a.metric("Frekuensi tengah (f0)", f"{p['f0']:.2f} Hz")
        b.metric("Tepi -3 dB bawah", f"{f_low:.2f} Hz")
        c.metric("Tepi -3 dB atas", f"{f_high:.2f} Hz")
        d.metric("Bandwidth / Q", f"{p['f0'] / p['Q']:.2f} Hz / {p['Q']:.2f}")


st.divider()
st.subheader("Hasil Analisis")

if satu:
    tampilkan_analisis(daftar[0])
else:
    for tab, nama in zip(st.tabs(daftar), daftar):
        with tab:
            tampilkan_analisis(nama)

# ------------------------------------------------------------------
# Tabel hasil perhitungan
# ------------------------------------------------------------------
st.subheader("Data Hasil Perhitungan")

n_titik = st.slider("Jumlah titik pada tabel", 5, 50, 10)
idx = np.linspace(0, len(frequency) - 1, n_titik, dtype=int)

data = {"Frequency (Hz)": frequency[idx]}
for nama in daftar:
    H, mag, phase = hasil[nama]
    ket = "" if satu else f"{nama} "
    if satu:
        data["Transfer Function"] = [
            f"{h.real:.4f} {h.imag:+.4f}j" for h in H[idx]
        ]
    data[f"{ket}Magnitude (dB)"] = mag[idx]
    data[f"{ket}Phase (Degrees)"] = phase[idx]

st.dataframe(pd.DataFrame(data).round(4))

# ------------------------------------------------------------------
# Grafik Bode
# ------------------------------------------------------------------
st.divider()
st.subheader("Bode Plot")

if satu:
    nama = daftar[0]
    _, mag, phase = hasil[nama]
    fig = plot_single(nama, frequency, mag, phase, frekuensi_cutoff(nama))
else:
    fig = plot_all_filters(
        frequency,
        R=p["R"], C=p["C"],
        bpf=(p["R1"], p["C1"], p["R2"], p["C2"]),
        bsf=(p["f0"], p["Q"]),
    )

st.pyplot(fig)

# ------------------------------------------------------------------
# Download
# ------------------------------------------------------------------
csv_data = {"Frequency (Hz)": frequency}
for nama in daftar:
    _, mag, phase = hasil[nama]
    csv_data[f"{nama} Magnitude (dB)"] = mag
    csv_data[f"{nama} Phase (Degrees)"] = phase
csv_bytes = pd.DataFrame(csv_data).to_csv(index=False).encode("utf-8")

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200)
plt.close(fig)

d1, d2 = st.columns(2)
d1.download_button("Download data (CSV)", csv_bytes, "bode_data.csv", "text/csv")
d2.download_button("Download grafik (PNG)", buf.getvalue(), "bode_plot.png", "image/png")

# ------------------------------------------------------------------
# Rumus
# ------------------------------------------------------------------
with st.expander("Rumus yang dipakai"):
    st.latex(r"f_c = \frac{1}{2\pi RC}")
    st.latex(r"H_{LPF}(s)=\frac{1}{RCs+1},\qquad H_{HPF}(s)=\frac{RCs}{RCs+1}")
    st.latex(r"H_{BPF}(s)=H_{HPF}(s)\,H_{LPF}(s)")
    st.latex(r"H_{BSF}(s)=\frac{s^2+\omega_0^2}{s^2+\frac{\omega_0}{Q}s+\omega_0^2}")
    st.latex(r"|H|_{dB}=20\log_{10}|H(j\omega)|,\qquad \angle H(j\omega)")

st.success("Analisis filter berhasil dilakukan!")