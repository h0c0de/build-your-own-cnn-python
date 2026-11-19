"""Uji Irisan 2 (CNN) — MaxPool2 + Softmax + forward utuh.

Jangkar panduan: bentuk (13, 13, 8) sesudah pool;
probabilitas softmax berjumlah 1; jaringan TAK terlatih
di 1000 citra uji MNIST memberi loss rata-rata ~= 2.302
(= -ln 0.1) dan akurasi di pita tebakan acak — persis
angka yang dilaporkan Part 1 panduan.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from konv import Conv3x3
from pool import MaxPool2
from softmax import Softmax

LULUS = 0


def ok(nama):
    global LULUS
    LULUS += 1
    print("  ok %2d. %s" % (LULUS, nama))


# --- MaxPool2: contoh hitung tangan ---
volume = np.zeros((4, 4, 1))
volume[:, :, 0] = [[1, 3, 2, 0],
                   [5, 2, 1, 4],
                   [0, 1, 9, 2],
                   [3, 2, 1, 1]]
pool = MaxPool2()
out_pool = pool.forward(volume)
assert out_pool.shape == (2, 2, 1)
assert np.array_equal(out_pool[:, :, 0], [[5, 4], [3, 9]])
ok("MaxPool2 hitung tangan 4x4x1 -> [[5,4],[3,9]]")

volume2 = np.stack([volume[:, :, 0], volume[:, :, 0] * 2], axis=2)
out_pool2 = pool.forward(volume2)
assert out_pool2.shape == (2, 2, 2)
assert np.array_equal(out_pool2[:, :, 1], [[10, 8], [6, 18]])
ok("MaxPool2 bekerja per-filter secara independen")

# --- Softmax: sifat probabilitas ---
np.random.seed(0)
sm = Softmax(5, 3)
probs = sm.forward(np.array([1.0, -2.0, 0.5, 3.0, -1.0]))
assert probs.shape == (3,)
assert abs(np.sum(probs) - 1.0) < 1e-12
assert np.all(probs > 0)
totals_manual = np.dot(np.array([1.0, -2.0, 0.5, 3.0, -1.0]),
                       sm.weights) + sm.biases
exp_manual = np.exp(totals_manual)
assert np.allclose(probs, exp_manual / np.sum(exp_manual))
ok("Softmax: probabilitas positif, berjumlah 1, cocok dgn rumus manual")

# --- Forward utuh tak terlatih di MNIST (angka panduan) ---
data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "aset", "mnist.npz"))
x_test = data["x_test"][:1000].astype(float)
y_test = data["y_test"][:1000]

np.random.seed(42)
conv = Conv3x3(8)                   # 28x28 -> 26x26x8
pool = MaxPool2()                   # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10


def forward(image, label):
    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0
    return out, loss, acc


out_satu, _, _ = forward(x_test[0], y_test[0])
assert out_satu.shape == (10,)
ok("forward utuh satu citra -> 10 probabilitas")

total_loss = 0.0
benar = 0
for im, lb in zip(x_test, y_test):
    _, l, a = forward(im, lb)
    total_loss += l
    benar += a
rata_loss = total_loss / 1000
akurasi = benar / 1000
print("    jaringan tak terlatih: loss %.3f | akurasi %.3f"
      % (rata_loss, akurasi))
assert abs(rata_loss - 2.302) < 0.1, rata_loss
ok("loss tak terlatih ~= 2.302 (= -ln 0.1) persis klaim panduan")
assert 0.03 <= akurasi <= 0.20, akurasi
ok("akurasi tak terlatih di pita tebakan acak (%.1f%%)" % (akurasi * 100))

print("\nIRISAN 2 HIJAU: %d pemeriksaan" % LULUS)
