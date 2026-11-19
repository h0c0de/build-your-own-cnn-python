"""Uji Irisan 3 (CNN) — backprop tiga lapis, wasit numerik.

Pola build 11 seri ini: gradien analitik panduan
dibandingkan dengan gradien beda-hingga (finite
difference) dari loss yang sama. Softmax: gradien input
dikembalikan langsung; gradien bobot dibaca dari delta
SGD dengan lr=1 (w_baru = w - 1 * grad). Conv: hal yang
sama untuk gradien filter. MaxPool: perutean gradien
diperiksa dengan tangan, termasuk keanehan seri
panduan.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from konv import Conv3x3
from pool import MaxPool2
from softmax import Softmax

LULUS = 0
EPS = 1e-5


def ok(nama):
    global LULUS
    LULUS += 1
    print("  ok %2d. %s" % (LULUS, nama))


def grad_numerik(fungsi_loss, arr):
    """Beda hingga tengah untuk setiap elemen arr;
    fungsi_loss dipanggil sesudah arr diubah di tempat."""
    grad = np.zeros_like(arr, dtype=float)
    it = np.nditer(arr, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        asal = arr[idx]
        arr[idx] = asal + EPS
        lp = fungsi_loss()
        arr[idx] = asal - EPS
        lm = fungsi_loss()
        arr[idx] = asal
        grad[idx] = (lp - lm) / (2 * EPS)
        it.iternext()
    return grad


# ---------- Softmax ----------
np.random.seed(3)
sm = Softmax(6, 4)
x = np.random.randn(6)
label = 2


def loss_sm():
    out = sm.forward(x)
    return -np.log(out[label])


out = sm.forward(x)
grad_awal = np.zeros(4)
grad_awal[label] = -1 / out[label]

num_w = grad_numerik(loss_sm, sm.weights)
num_x = grad_numerik(loss_sm, x)
num_b = grad_numerik(loss_sm, sm.biases)

sm.forward(x)  # pulihkan cache utk x asli
w_sebelum = sm.weights.copy()
b_sebelum = sm.biases.copy()
d_inputs = sm.backprop(grad_awal, learn_rate=1.0)
ana_w = w_sebelum - sm.weights
ana_b = b_sebelum - sm.biases

assert np.allclose(ana_w, num_w, atol=1e-6), np.max(np.abs(ana_w - num_w))
ok("Softmax.backprop: gradien bobot == numerik (selisih maks %.2e)"
   % np.max(np.abs(ana_w - num_w)))
assert np.allclose(d_inputs, num_x, atol=1e-6)
ok("Softmax.backprop: gradien input yang dikembalikan == numerik")
assert np.allclose(ana_b, num_b, atol=1e-6)
ok("Softmax.backprop: gradien bias == numerik")

# ---------- MaxPool2 ----------
pool = MaxPool2()
volume = np.zeros((4, 4, 1))
volume[:, :, 0] = [[1, 3, 2, 0],
                   [5, 2, 1, 4],
                   [0, 1, 9, 2],
                   [3, 2, 1, 1]]
pool.forward(volume)
grad_out = np.array([[[10.0], [20.0]], [[30.0], [40.0]]])
d_in = pool.backprop(grad_out)
harap = np.zeros((4, 4, 1))
harap[1, 0, 0] = 10.0   # maks wilayah kiri-atas = 5 di (1,0)
harap[1, 3, 0] = 20.0   # maks kanan-atas = 4 di (1,3)
harap[3, 0, 0] = 30.0   # maks kiri-bawah = 3 di (3,0)
harap[2, 2, 0] = 40.0   # maks kanan-bawah = 9 di (2,2)
assert np.array_equal(d_in, harap)
ok("MaxPool2.backprop: gradien hanya mendarat di posisi argmax")

pool2 = MaxPool2()
# kasus seri dibangun langsung: wilayah 2x2 kiri-atas
# memuat dua nilai maksimum yang sama (7 dan 7)
v = np.zeros((4, 4, 1))
v[:, :, 0] = [[7, 7, 0, 0],
              [1, 2, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]
pool2.forward(v)
d_seri = pool2.backprop(np.ones((2, 2, 1)))
assert d_seri[0, 0, 0] == 1.0 and d_seri[0, 1, 0] == 1.0
assert d_seri[1, 0, 0] == 0.0 and d_seri[1, 1, 0] == 0.0
ok("MaxPool2.backprop: keanehan seri panduan — SEMUA posisi seri menerima gradien")

# ---------- Conv3x3 ----------
np.random.seed(5)
conv = Conv3x3(2)
citra = np.random.randn(5, 5)
G = np.random.randn(3, 3, 2)  # dL/dout tetap


def loss_conv():
    return float(np.sum(conv.forward(citra) * G))


num_f = grad_numerik(loss_conv, conv.filters)
conv.forward(citra)  # pulihkan cache
f_sebelum = conv.filters.copy()
hasil = conv.backprop(G, learn_rate=1.0)
ana_f = f_sebelum - conv.filters
assert hasil is None
ok("Conv3x3.backprop mengembalikan None (gradien input dilewati panduan — tercatat)")
assert np.allclose(ana_f, num_f, atol=1e-6), np.max(np.abs(ana_f - num_f))
ok("Conv3x3.backprop: gradien filter == numerik (selisih maks %.2e)"
   % np.max(np.abs(ana_f - num_f)))

print("== regresi irisan 1-2 ==")
import subprocess
for n in (1, 2):
    r = subprocess.run([sys.executable, "uji_irisan%d.py" % n],
                       cwd=os.path.dirname(os.path.abspath(__file__)),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    ok("uji_irisan%d hijau penuh" % n)

print("\nIRISAN 3 HIJAU: %d pemeriksaan" % LULUS)
