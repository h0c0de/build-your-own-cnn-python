# CATATAN — CNN dari Nol (Victor Zhou)

Panduan: Victor Zhou, "CNNs, Part 1: An Introduction to
Convolutional Neural Networks" dan "Part 2: Training a
Convolutional Neural Network" (victorzhou.com). Penulis
yang sama dengan build 11 seri ini; wasit gradiennya
juga sama: beda hingga numerik.

## Irisan 1 — Conv3x3 forward

Konvolusi di sini sebenarnya korelasi-silang (panduan
mengakuinya): filter 3x3 digeser dengan padding valid,
tiap piksel keluaran = jumlah perkalian elemen wilayah
x filter. Satu baris numpy menyelesaikan semua filter
sekaligus lewat broadcasting:
`np.sum(im_region * self.filters, axis=(1, 2))`.
Jangkar panduan terbukti persis: wilayah pertama contoh
4x4-nya dikali Sobel vertikal menjumlah 29 (62 - 33
dari tabel perkaliannya). Inisialisasi filter
`randn / 9` — pembagian itu menahan varians awal agar
pelatihan tidak mati di tempat.

## Irisan 2 — Pooling, softmax, dan angka "belum belajar"

MaxPool2 membagi volume jadi wilayah 2x2 tak tumpang
tindih dan mengambil maks per filter: 26x26x8 ->
13x13x8. Softmax adalah lapis fully-connected biasa
(input diratakan, totals = input . W + b) yang
keluarannya dinormalkan menjadi probabilitas; loss
cross-entropy `-ln(p[label])`. Angka khas panduan
terbukti di data asli: jaringan tak terlatih pada 1000
citra uji memberi loss rata-rata tepat 2.302
(= -ln 0.1, tebakan acak seragam) dan akurasi 12.2% —
di pita acak 10%. Angka ini penting: ia adalah garis
dasar yang harus dikalahkan irisan 4.

## Irisan 3 — Backprop tiga lapis, dibuktikan numerik

Pola panduan: forward menyimpan cache (input terakhir,
totals), backward menerima dL/dout dan mengembalikan
dL/din. Gradien awal: vektor nol dengan
`-1 / out[label]` di posisi label.

- Softmax: hanya elemen label yang tak nol, jadi
  turunan out(c) thd totals diturunkan dua kasus —
  k = c memakai aturan hasil-bagi, k != c memakai
  aturan rantai — lalu dL/dw = input^T @ dL/dt,
  dL/db = dL/dt, dL/din = W @ dL/dt, dan SGD diterapkan
  di dalam backprop. Ketiganya cocok dengan gradien
  numerik sampai ~2e-11.
- MaxPool2: gradien hanya mengalir ke piksel yang
  menjadi maksimum wilayahnya (yang lain marginalnya
  nol). Keanehan panduan yang dipertahankan apa
  adanya: perbandingan `==` membuat SEMUA posisi yang
  seri dengan maks menerima gradien penuh yang sama —
  terbukti di uji seri (dua posisi 7 sama-sama menerima).
- Conv3x3: d out(i,j) / d filter(x,y) = piksel citra
  padanannya, jadi dL/dfilter adalah jumlah
  dL/dout(i,j,f) * wilayah(i,j) — cocok numerik ~3e-11.
  Panduan SENGAJA tidak menghitung gradien input conv
  ("left as an exercise") dan backprop-nya
  mengembalikan None; di sini itu dicatat dan diuji
  sebagai perilaku, bukan ditambal diam-diam.

## Irisan 4 — Pelatihan: mengalahkan angka panduan

Protokol panduan persis: 1000 citra latih, 3 epoch,
shuffle tiap epoch, lr 0.005, seed tetap 42. Hasil run
ini (deterministik dengan seed itu):

- loss latih per epoch: 1.221 -> 0.533 -> 0.422
- akurasi latih epoch 3: 87.7%
- **uji: loss 0.521, akurasi 83.0%**

Panduan melaporkan loss uji 0.598 dan akurasi 0.78 —
run ini sedikit lebih baik di kedua angka (perbedaan
benih acak dan versi MNIST; orde dan arahnya sama).
Satu langkah latih ~= 28 ms: seluruh protokol 3000
langkah selesai dalam ~2 menit di numpy murni —
panduan sendiri membatasi 1000 citra "in the interest
of time", dan versi Keras arsitektur yang sama di
artikelnya mencapai 97.4% pada data penuh.
