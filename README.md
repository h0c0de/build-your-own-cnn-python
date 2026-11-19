# Build Your Own CNN — Convolutional Neural Network dari Nol

Implementasi ulang CNN MNIST dari seri dua bagian
**Victor Zhou**, "CNNs, Part 1 & 2" (victorzhou.com),
hanya dengan numpy: `Conv3x3(8) -> MaxPool2 ->
Softmax(1352, 10)`, lengkap dengan backprop yang
diturunkan dan dibuktikan per lapis.

Hasil pelatihan protokol panduan (1000 citra latih,
3 epoch, lr 0.005): **akurasi uji 83.0%, loss 0.521** —
dari garis dasar jaringan tak terlatih loss 2.302 /
akurasi ~10%. (Panduan melaporkan 78% / 0.598.)

## Isi

- `konv.py` — lapis konvolusi 3x3 (forward + backprop
  gradien filter).
- `pool.py` — max pooling 2x2 (forward + backprop
  perutean argmax).
- `softmax.py` — lapis softmax (forward + backprop
  penuh: bobot, bias, input).
- `cnn.py` — perakitan jaringan, `forward()` /
  `train()`, dan demo pelatihan (`python3 cnn.py`).
- `uji_irisan1.py` … `uji_irisan4.py` — baterai uji per
  irisan (total 29 pemeriksaan): jangkar konvolusi
  panduan (29), angka tak-terlatih panduan (2.302),
  cek gradien numerik untuk softmax & conv, dan
  pelatihan penuh.

## Menjalankan

```bash
python3 uji_irisan1.py   # dst. sampai uji_irisan4.py
python3 cnn.py           # demo latih + uji (~2 menit)
```

Butuh numpy dan `aset/mnist.npz` (MNIST dalam format
npz: x_train/y_train/x_test/y_test) — berkas data tidak
ikut diterbitkan ke repo.

## Atribusi

Struktur, derivasi gradien, dan protokol mengikuti
"CNNs, Part 1" dan "Part 2" oleh Victor Zhou
(https://victorzhou.com/blog/intro-to-cnns-part-1/ dan
https://victorzhou.com/blog/intro-to-cnns-part-2/),
diimplementasikan ulang sebagai latihan belajar.
