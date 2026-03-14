import numpy as np
import time
from collections import Counter
from tensorflow.keras.datasets import mnist

# ============================================================
# Hata Analizi v99 — Confusion Matrix + Karışan Çiftler
# Kendi makine'nde calistir: python error_analysis.py
# ============================================================

def build_gabor_kernels(freqs=(0.1,0.2,0.3,0.4), n_thetas=6,
                        sigma=2.0, gamma=0.5, ksize=9):
    half = ksize // 2
    thetas = np.linspace(0, np.pi, n_thetas, endpoint=False)
    yg, xg = np.mgrid[-half:half+1, -half:half+1].astype(np.float32)
    kernels = []
    for freq in freqs:
        lam = 1.0 / freq
        for theta in thetas:
            ct, st = np.cos(theta), np.sin(theta)
            xr =  xg*ct + yg*st
            yr = -xg*st + yg*ct
            k = (np.exp(-(xr**2 + gamma**2*yr**2) / (2*sigma**2))
                 * np.cos(2*np.pi*xr/lam)).astype(np.float32)
            k -= k.mean()
            kernels.append(k.ravel())
    return np.stack(kernels)

def gabor_features(images_flat, K, img_h=28, img_w=28,
                   ksize=9, cell_size=4, batch_size=1000):
    N = len(images_flat)
    half = ksize // 2
    n_filters = len(K)
    nch = img_h // cell_size
    ncw = img_w // cell_size
    feat_dim = nch * ncw * n_filters
    out = np.zeros((N, feat_dim), dtype=np.float32)
    for start in range(0, N, batch_size):
        xb = images_flat[start:start+batch_size].reshape(-1, img_h, img_w)
        nb = len(xb)
        xb_pad = np.pad(xb, ((0,0),(half,half),(half,half)), mode='reflect')
        s = xb_pad.strides
        patches = np.lib.stride_tricks.as_strided(
            xb_pad,
            shape=(nb, img_h, img_w, ksize, ksize),
            strides=(s[0], s[1], s[2], s[1], s[2])
        )
        p2d = np.ascontiguousarray(patches.reshape(nb, img_h*img_w, ksize*ksize))
        resp = np.abs(p2d @ K.T).reshape(nb, img_h, img_w, n_filters)
        resp = resp[:, :nch*cell_size, :ncw*cell_size, :]
        resp = resp.reshape(nb, nch, cell_size, ncw, cell_size, n_filters)
        out[start:start+nb] = resp.mean(axis=(2,4)).reshape(nb, feat_dim)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / (norms + 1e-8)

def fit_pca(x, n):
    mean = x.mean(axis=0); xc = x - mean
    vals, vecs = np.linalg.eigh((xc.T @ xc) / len(x))
    idx = np.argsort(vals)[::-1][:n]
    return mean, vecs[:, idx].astype(np.float32)

def apply_pca(x, mean, vecs):
    return (x - mean) @ vecs

def knn_predict(x_tr, y_tr, x_te, k=10, bs=500):
    preds = np.zeros(len(x_te), dtype=int)
    sq = np.sum(x_tr**2, axis=1)
    oh = np.eye(10, dtype=np.float32)[y_tr]
    for i in range(0, len(x_te), bs):
        xb = x_te[i:i+bs]
        d = np.sum(xb**2, axis=1, keepdims=True) + sq - 2*(xb @ x_tr.T)
        np.maximum(d, 0, out=d)
        ti = np.argpartition(d, k, axis=1)[:, :k]
        db = np.take_along_axis(d, ti, axis=1)
        w  = 1.0 / (db + 1e-10)
        oh2 = oh[ti]
        preds[i:i+len(xb)] = (w[:,:,None]*oh2).sum(axis=1).argmax(axis=1)
    return preds

# ── Veri ─────────────────────────────────────────────────────
print("Veri yukleniyor...")
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
x_raw = x_test.copy()  # gorsel icin sakla
x_train_raw = x_train_full.copy()

x_train_full = x_train_full.reshape(60000, -1).astype(np.float32) / 255.0
x_test       = x_test.reshape(10000, -1).astype(np.float32) / 255.0
y_te = y_test.astype(int)

# ── Pipeline (60k full train) ────────────────────────────────
print("Gabor...", flush=True); t0 = time.time()
K = build_gabor_kernels()
g_train = gabor_features(x_train_full, K)
g_test  = gabor_features(x_test, K)
print("  {:.1f}sn".format(time.time()-t0))

print("PCA(256)...", flush=True); t0 = time.time()
pm, pv = fit_pca(g_train, 256)
xp_train = apply_pca(g_train, pm, pv).astype(np.float32)
xp_test  = apply_pca(g_test,  pm, pv).astype(np.float32)
print("  {:.1f}sn".format(time.time()-t0))

print("kNN (k=10)...", flush=True); t0 = time.time()
preds = knn_predict(xp_train, y_train_full.astype(int), xp_test, k=10)
acc = np.mean(preds == y_te) * 100
print("  {:.1f}sn  -> {:.2f}%".format(time.time()-t0, acc))

# ══════════════════════════════════════════════════════════════
# ANALIZ BASLADI
# ══════════════════════════════════════════════════════════════
wrong = preds != y_te
n_wrong = wrong.sum()
print("\n" + "="*65)
print("  TEST HATA ANALIZI — {}/{} yanlis ({:.2f}%)".format(
    n_wrong, len(y_te), acc))
print("="*65)

# ── 1. Full Confusion Matrix ─────────────────────────────────
cm = np.zeros((10, 10), dtype=int)
for i in range(len(y_te)):
    cm[y_te[i], preds[i]] += 1

print("\n[1] CONFUSION MATRIX (satir=gercek, sutun=tahmin)")
print("       ", "  ".join("{:>5d}".format(i) for i in range(10)))
print("       ", "  ".join("-----" for _ in range(10)))
for i in range(10):
    row = "  ".join("{:>5d}".format(cm[i,j]) for j in range(10))
    wrong_i = cm[i].sum() - cm[i,i]
    print("  [{}]  {}   | {} yanlis".format(i, row, wrong_i))

# ── 2. Hatali ciftler (yonlu) ────────────────────────────────
print("\n[2] EN COK KARISAN CIFTLER (yonlu: gercek -> tahmin)")
errors = Counter()
for i in np.where(wrong)[0]:
    errors[(y_te[i], preds[i])] += 1

for (true_c, pred_c), cnt in errors.most_common(25):
    bar = "█" * cnt
    print("  {} -> {}  : {:>2d}  {}".format(true_c, pred_c, cnt, bar))

# ── 3. Simetrik cift analizi ─────────────────────────────────
print("\n[3] SIMETRIK CIFTLER (iki yonlu toplam)")
pair_sym = Counter()
for (a, b), cnt in errors.items():
    pair_sym[(min(a,b), max(a,b))] += cnt

for (a, b), cnt in pair_sym.most_common(15):
    bar = "█" * cnt
    a_to_b = errors.get((a,b), 0)
    b_to_a = errors.get((b,a), 0)
    print("  {} <-> {}  : {:>2d}  ({}->{}={}, {}->{}={})  {}".format(
        a, b, cnt, a, b, a_to_b, b, a, b_to_a, bar))

# ── 4. Sinif bazli hata orani ────────────────────────────────
print("\n[4] SINIF BAZLI HATA ORANI")
class_stats = []
for c in range(10):
    mask = y_te == c
    n_total = mask.sum()
    n_wrong_c = (preds[mask] != c).sum()
    rate = n_wrong_c / n_total * 100
    class_stats.append((c, n_wrong_c, n_total, rate))
    bar = "█" * int(rate * 5)
    print("  Sinif {} :  {:>2d}/{:>4d} yanlis  ({:.2f}%)  {}".format(
        c, n_wrong_c, n_total, rate, bar))

# En zor siniflar
class_stats.sort(key=lambda x: -x[3])
print("\n  En zor siniflar: ", end="")
for c, nw, nt, rate in class_stats[:3]:
    print("{}({:.2f}%) ".format(c, rate), end="")
print()

# ── 5. Yanlis orneklerin detayi ───────────────────────────────
print("\n[5] TUM YANLIS ORNEKLER (index | gercek | tahmin)")
wrong_idx = np.where(wrong)[0]
for wi in wrong_idx:
    print("  #{:>5d}  gercek={}  tahmin={}".format(wi, y_te[wi], preds[wi]))

# ── 6. Matplotlib gorsellestime ──────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 6a. Confusion matrix heatmap
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Sadece off-diagonal (hatalar)
    cm_err = cm.copy()
    np.fill_diagonal(cm_err, 0)

    ax = axes[0]
    im = ax.imshow(cm_err, cmap='Reds', interpolation='nearest')
    ax.set_title('Confusion Matrix (sadece hatalar)', fontsize=14)
    ax.set_xlabel('Tahmin')
    ax.set_ylabel('Gercek')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            if i != j and cm_err[i,j] > 0:
                ax.text(j, i, str(cm_err[i,j]), ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='white' if cm_err[i,j] > cm_err.max()*0.5 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 6b. Sinif bazli hata orani
    ax2 = axes[1]
    classes = range(10)
    rates = [100 - cm[c,c]/cm[c].sum()*100 for c in classes]
    colors = ['#e74c3c' if r > np.mean(rates) else '#3498db' for r in rates]
    ax2.bar(classes, rates, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_title('Sinif Bazli Hata Orani (%)', fontsize=14)
    ax2.set_xlabel('Sinif')
    ax2.set_ylabel('Hata %')
    ax2.set_xticks(range(10))
    ax2.axhline(y=np.mean(rates), color='gray', linestyle='--', alpha=0.7,
                label='Ortalama: {:.2f}%'.format(np.mean(rates)))
    ax2.legend()
    for i, r in enumerate(rates):
        ax2.text(i, r + 0.05, '{:.1f}%'.format(r), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('error_analysis_charts.png', dpi=150, bbox_inches='tight')
    print("\n[Chart] error_analysis_charts.png kaydedildi")

    # 6c. Yanlis orneklerin goruntusu (en cok karisan 3 cift)
    top_pairs = pair_sym.most_common(3)
    fig2, axes2 = plt.subplots(3, 10, figsize=(20, 7))
    fig2.suptitle('En Cok Karisan Ciftlerin Yanlis Ornekleri', fontsize=14)

    for row, ((ca, cb), cnt) in enumerate(top_pairs):
        mask = wrong & np.isin(y_te, [ca, cb]) & np.isin(preds, [ca, cb])
        indices = np.where(mask)[0][:10]
        for col in range(10):
            ax = axes2[row, col]
            ax.axis('off')
            if col < len(indices):
                idx = indices[col]
                ax.imshow(x_raw[idx], cmap='gray')
                color = 'red'
                ax.set_title('G:{} T:{}'.format(y_te[idx], preds[idx]),
                            fontsize=8, color=color)
            if col == 0:
                ax.set_ylabel('{}<->{} ({})'.format(ca, cb, cnt),
                             fontsize=10, rotation=0, labelpad=60)

    plt.tight_layout()
    plt.savefig('error_examples.png', dpi=150, bbox_inches='tight')
    print("[Chart] error_examples.png kaydedildi")

except ImportError:
    print("\n[!] matplotlib bulunamadi, gorsellestime atlandi.")

print("\n" + "="*65)
print("Analiz tamamlandi!")
print("="*65)
