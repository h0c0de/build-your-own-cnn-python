"""Uji Irisan 4 (CNN) — pelatihan penuh protokol panduan.

1000 citra latih, 3 epoch (shuffle per epoch), lr 0.005,
seed tetap 42 — lalu 1000 citra uji. Panduan melaporkan
test loss 0.598 / akurasi 0.78 pada protokol ini; di sini
yang dikunci adalah arah & orde besarnya: loss runtuh
dari ~= 2.302 dan akurasi uji jauh di atas pita acak.
Angka persis run ini dicetak untuk CATATAN.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

np.random.seed(42)
import cnn  # lapis dibuat saat import -> seed di atas yang berlaku

LULUS = 0


def ok(nama):
    global LULUS
    LULUS += 1
    print("  ok %2d. %s" % (LULUS, nama))


train_images, train_labels, test_images, test_labels = cnn.muat_data()
assert len(train_images) == 1000 and len(test_images) == 1000
ok("data: 1000 latih + 1000 uji (protokol panduan)")

loss_per_epoch = []
for epoch in range(3):
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]
    total_loss = 0.0
    benar = 0
    for im, label in zip(train_images, train_labels):
        l, acc = cnn.train(im, label)
        total_loss += l
        benar += acc
    rata = total_loss / len(train_images)
    loss_per_epoch.append(rata)
    print("    epoch %d: loss latih %.3f | akurasi latih %.3f"
          % (epoch + 1, rata, benar / len(train_images)))

assert loss_per_epoch[0] < 2.302
ok("epoch 1: loss latih %.3f sudah di bawah 2.302" % loss_per_epoch[0])
assert loss_per_epoch[2] < loss_per_epoch[0] * 0.5
ok("epoch 3: loss latih %.3f < separuh epoch 1 (%.3f)"
   % (loss_per_epoch[2], loss_per_epoch[0]))

test_loss = 0.0
benar = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = cnn.forward(im, label)
    test_loss += l
    benar += acc
test_loss /= len(test_images)
test_acc = benar / len(test_images)
print("    UJI: loss %.3f | akurasi %.3f" % (test_loss, test_acc))

assert test_acc >= 0.60, test_acc
ok("akurasi uji %.1f%% >= 60%% (panduan: 78%%; acak: 10%%)"
   % (test_acc * 100))
assert test_loss < 1.2, test_loss
ok("loss uji %.3f < 1.2 (dari 2.302 tak terlatih)" % test_loss)

print("== regresi irisan 1-3 ==")
import subprocess
for n in (1, 2, 3):
    r = subprocess.run([sys.executable, "uji_irisan%d.py" % n],
                       cwd=os.path.dirname(os.path.abspath(__file__)),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    ok("uji_irisan%d hijau penuh" % n)

print("\nIRISAN 4 HIJAU: %d pemeriksaan" % LULUS)
