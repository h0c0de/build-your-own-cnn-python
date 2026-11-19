"""Uji Irisan 1 (CNN) — Conv3x3 forward.

Jangkar panduan Part 1: wilayah pertama contoh 4x4
panduan dikali filter Sobel vertikal menjumlah tepat
29 (tabel perkalian panduan: 62 - 33 = 29). Ditambah
konvolusi referensi independen (loop naif) di uji ini,
dan bentuk keluaran MNIST panduan (26, 26, 8).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from konv import Conv3x3

LULUS = 0


def ok(nama):
    global LULUS
    LULUS += 1
    print("  ok %2d. %s" % (LULUS, nama))


def konvolusi_naif(citra, filters):
    """Referensi independen: loop eksplisit per piksel,
    per filter, per elemen — tanpa broadcasting."""
    h, w = citra.shape
    nf = filters.shape[0]
    out = np.zeros((h - 2, w - 2, nf))
    for i in range(h - 2):
        for j in range(w - 2):
            for f in range(nf):
                total = 0.0
                for x in range(3):
                    for y in range(3):
                        total += citra[i + x, j + y] * filters[f, x, y]
                out[i, j, f] = total
    return out


# Wilayah pertama & filter dari tabel panduan persis
wilayah = np.array([[0, 50, 0],
                    [0, 80, 31],
                    [33, 90, 0]], dtype=float)
sobel_vertikal = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=float)
assert np.sum(wilayah * sobel_vertikal) == 29
ok("jangkar panduan: jumlah perkalian wilayah pertama = 29 (62 - 33)")

# Citra 4x4 yang wilayah kiri-atasnya persis wilayah panduan
citra = np.array([[0, 50, 0, 10],
                  [0, 80, 31, 20],
                  [33, 90, 0, 30],
                  [5, 5, 5, 5]], dtype=float)
conv = Conv3x3(1)
conv.filters = sobel_vertikal[np.newaxis]
out = conv.forward(citra)
assert out.shape == (2, 2, 1)
assert out[0, 0, 0] == 29
ok("forward pada citra 4x4 panduan: piksel pertama = 29, bentuk (2, 2, 1)")
assert np.allclose(out[:, :, 0],
                   konvolusi_naif(citra, conv.filters)[:, :, 0])
ok("seluruh keluaran 2x2 == referensi naif independen")

rng = np.random.default_rng(7)
citra_acak = rng.normal(size=(9, 11))
conv3 = Conv3x3(3)
out_acak = conv3.forward(citra_acak)
assert out_acak.shape == (7, 9, 3)
assert np.allclose(out_acak, konvolusi_naif(citra_acak, conv3.filters))
ok("citra acak 9x11 x 3 filter == referensi naif (broadcasting benar)")

wilayah_list = list(conv3.iterate_regions(citra_acak))
assert len(wilayah_list) == 7 * 9
assert wilayah_list[0][1:] == (0, 0) and wilayah_list[-1][1:] == (6, 8)
ok("iterate_regions menghasilkan (h-2)*(w-2) wilayah dengan koordinat benar")

# Bentuk MNIST panduan: 28x28 -> (26, 26, 8)
data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "aset", "mnist.npz"))
citra_mnist = data["x_test"][0].astype(float)
conv8 = Conv3x3(8)
out_mnist = conv8.forward(citra_mnist)
assert out_mnist.shape == (26, 26, 8)
ok("citra MNIST asli: forward Conv3x3(8) -> (26, 26, 8) persis panduan")

print("\nIRISAN 1 HIJAU: %d pemeriksaan" % LULUS)
