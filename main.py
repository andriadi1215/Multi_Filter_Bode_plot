import numpy as np

from filters import (
    low_pass_filter,
    high_pass_filter,
    band_pass_filter,
    band_stop_filter,
    cutoff_rc,
    band_edges,
)

# ---------------- Parameter pengujian ----------------
R, C = 1000, 100e-9                           # LPF dan HPF
R1, C1, R2, C2 = 1000, 159.15e-9, 1000, 31.83e-9   # BPF
F0, Q = 2000, 2                               # BSF

hasil_cek = []


def judul(teks):
    print("\n" + "=" * 44)
    print(f"  {teks}")
    print("=" * 44)


def cek(keterangan, nilai, harapan, toleransi=0.05):
    """Bandingkan hasil program dengan nilai teori."""
    ok = abs(nilai - harapan) <= toleransi
    hasil_cek.append(ok)
    status = "OK" if ok else "GAGAL"
    print(f"  [{status:5}] {keterangan}: {nilai:.2f} (teori {harapan:.2f})")


def uji(fungsi, f, *args):
    """Hitung respons di satu frekuensi, kembalikan (magnitude, fase)."""
    _, mag, phase = fungsi(np.array([f]), *args)
    return mag[0], phase[0]


# ---------------- LPF ----------------
judul("LOW-PASS FILTER")
fc = cutoff_rc(R, C)
print(f"  R = {R} Ohm, C = {C * 1e9:.0f} nF, fc = {fc:.2f} Hz")
mag, ph = uji(low_pass_filter, fc, R, C)
cek("Magnitude di fc (dB)", mag, -3.01)
cek("Fase di fc (derajat)", ph, -45.0)

# ---------------- HPF ----------------
judul("HIGH-PASS FILTER")
print(f"  R = {R} Ohm, C = {C * 1e9:.0f} nF, fc = {fc:.2f} Hz")
mag, ph = uji(high_pass_filter, fc, R, C)
cek("Magnitude di fc (dB)", mag, -3.01)
cek("Fase di fc (derajat)", ph, 45.0)

# ---------------- BPF ----------------
judul("BAND-PASS FILTER")
f_low, f_high = cutoff_rc(R1, C1), cutoff_rc(R2, C2)
f_tengah = np.sqrt(f_low * f_high)
print(f"  f_low = {f_low:.2f} Hz, f_high = {f_high:.2f} Hz, "
      f"f0 = {f_tengah:.2f} Hz")
mag, ph = uji(band_pass_filter, f_tengah, R1, C1, R2, C2)
cek("Magnitude di f0 (dB)", mag, -1.58)   # puncak BPF, bukan 0 dB
cek("Fase di f0 (derajat)", ph, 0.0)

# ---------------- BSF ----------------
judul("BAND-STOP FILTER")
f_bawah, f_atas = band_edges(F0, Q)
print(f"  f0 = {F0} Hz, Q = {Q}, "
      f"tepi -3 dB = {f_bawah:.2f} Hz dan {f_atas:.2f} Hz")
for nama, f in (("tepi bawah", f_bawah), ("tepi atas", f_atas)):
    mag, _ = uji(band_stop_filter, f, F0, Q)
    cek(f"Magnitude di {nama} (dB)", mag, -3.01)
mag, _ = uji(band_stop_filter, F0, F0, Q)
hasil_cek.append(mag < -60)
print(f"  [{'OK' if mag < -60 else 'GAGAL':5}] Redaman di f0 sangat dalam (< -60 dB)")

print("\n  Tabel respons BSF")
print("  " + "-" * 40)
frequency = np.array([100, 1000, 2000, 5000, 10000])
H, magnitude, phase = band_stop_filter(frequency, F0, Q)
for i in range(len(frequency)):
    print(f"  {frequency[i]:>7} Hz | {magnitude[i]:>9.2f} dB | {phase[i]:>8.2f} deg")

# ---------------- Ringkasan ----------------
judul("RINGKASAN")
print(f"  {sum(hasil_cek)} dari {len(hasil_cek)} pengujian lolos")
print("\nPengujian filter selesai!")