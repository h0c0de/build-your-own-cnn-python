# RENCANA — Build 19: CNN dari Nol (Victor Zhou)

Panduan: Victor Zhou, "CNNs, Part 1: An Introduction to
Convolutional Neural Networks" + "Part 2: Training a
Convolutional Neural Network" —
https://victorzhou.com/blog/intro-to-cnns-part-1/ dan
https://victorzhou.com/blog/intro-to-cnns-part-2/
Dibaca penuh sebelum mulai (20 Nov 2026). Penulis yang
sama dengan build 11 seri ini (Neural Network).

Arsitektur panduan persis: Conv3x3 (8 filter) ->
MaxPool2 -> Softmax (13*13*8 = 1352 -> 10) atas MNIST,
input dinormalkan (x/255 - 0.5), loss cross-entropy
-ln(p[label]), SGD lr 0.005. Dataset: aset/mnist.npz
yang sama dengan build 11 (sha256 731c5ac6...).

Jangkar panduan yang diverifikasi sebelum kode:
- contoh konvolusi 4x4 panduan: wilayah pertama x
  filter Sobel vertikal menjumlah tepat 29 (tabel
  perkalian panduan: 62 - 33);
- bentuk: conv 28x28 -> (26, 26, 8); pool -> (13, 13, 8);
- jaringan tak terlatih: loss rata-rata ~= 2.302
  (= -ln 0.1) dan akurasi ~= 10%;
- hasil akhir panduan (1000 latih, 3 epoch, 1000 uji):
  test accuracy 0.78, test loss 0.598.

## Irisan

- [x] Irisan 1 — Conv3x3 forward: inisialisasi filter
  randn/9, iterate_regions, konvolusi lewat broadcasting.
  Uji: jangkar 29 panduan; konvolusi kecil dihitung
  tangan vs implementasi referensi independen di uji;
  bentuk keluaran MNIST (26, 26, 8).
- [x] Irisan 2 — MaxPool2 + Softmax forward + forward
  utuh: pool mengambil maks per wilayah 2x2 (bentuk
  (13, 13, 8)); softmax meratakan input, probabilitas
  berjumlah 1; jaringan tak terlatih di 1000 citra uji:
  loss ~= 2.302, akurasi di pita tebakan acak.
- [x] Irisan 3 — Backprop tiga lapis: gradien awal
  -1/p[label]; Softmax.backprop (turunan out thd totals
  dua kasus k=c / k!=c, SGD di dalam); MaxPool2.backprop
  (gradien hanya ke posisi argmax — termasuk keanehan
  seri panduan: SEMUA posisi yang sama dengan maks
  menerima gradien); Conv3x3.backprop (dL/dfilter =
  jumlah dL/dout * wilayah citra; gradien input sengaja
  tidak dihitung panduan — dicatat, bukan disembunyikan).
  Uji: cek gradien numerik (beda hingga) untuk softmax
  dan conv — wasit yang sama dengan build 11.
- [x] Irisan 4 — Latih + finale: protokol panduan
  (1000 citra latih, 3 epoch, shuffle per epoch,
  lr 0.005, seed tetap) lalu uji 1000 citra: akurasi
  akhir harus jauh di atas pita acak dan loss turun
  tegas dari ~= 2.302; angka persisnya dilaporkan apa
  adanya (panduan: 0.78). README + terbit.

## Pagar selesai

- [x] `uji_irisan1.py` … `uji_irisan4.py` semua hijau.
- [x] Gradien analitik softmax & conv cocok dengan
  gradien numerik dalam toleransi ketat.
- [x] `CATATAN.md` satu halaman per irisan: konsep inti + temuan.
- [x] Repo GitHub publik `build-your-own-cnn-python`
  terbit; blob SHA lokal == remote terverifikasi.
