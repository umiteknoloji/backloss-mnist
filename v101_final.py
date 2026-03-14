import numpy as np
import time

# ============================================================
# BackLoss v101 — 99.11%
# Gabor(7freq×8theta, gamma=0.4) → PCA(512) → kNN(k=5)
# GPU: pip install cupy-cuda12x   (veya cupy-cuda11x)
# CPU: otomatik fallback
# ============================================================

# ── GPU / CPU otomatik secim ─────────────────────────────────
try:
    import cupy as cp
    xp = cp
    GPU = True
    print("GPU modu (CuPy)")
except ImportError:
    xp = np
    GPU = False
    print("CPU modu (NumPy)")

def to_np(a):
    return cp.asnumpy(a) if GPU else a

def to_xp(a):
    return xp.asarray(a)

# ── Gabor Kernels ────────────────────────────────────────────
def build_gabor_kernels():
    freqs    = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    n_thetas = 8
    sigma    = 2.0
    gamma    = 0.4
    ksize    = 9
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
    return np.stack(kernels)  # (n_filters, ksize*ksize)

# ── Gabor Feature Extraction ────────────────────────────────
def gabor_features(images_flat, K_np, batch_size=1000):
    img_h, img_w, ksize, cell_size = 28, 28, 9, 4
    half = ksize // 2
    n_filters = len(K_np)
    nch = img_h // cell_size   # 7
    ncw = img_w // cell_size   # 7
    feat_dim = nch * ncw * n_filters
    N = len(images_flat)

    K = to_xp(K_np)
    out = np.zeros((N, feat_dim), dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xb = to_xp(images_flat[start:end]).reshape(-1, img_h, img_w)
        nb = len(xb)

        xb_pad = xp.pad(xb, ((0,0),(half,half),(half,half)), mode='reflect')
        s = xb_pad.strides
        patches = xp.lib.stride_tricks.as_strided(
            xb_pad,
            shape=(nb, img_h, img_w, ksize, ksize),
            strides=(s[0], s[1], s[2], s[1], s[2])
        )
        p2d = xp.ascontiguousarray(patches.reshape(nb, img_h*img_w, ksize*ksize))
        resp = xp.abs(p2d @ K.T).reshape(nb, img_h, img_w, n_filters)
        resp = resp[:, :nch*cell_size, :ncw*cell_size, :]
        resp = resp.reshape(nb, nch, cell_size, ncw, cell_size, n_filters)
        cell_feats = resp.mean(axis=(2,4)).reshape(nb, feat_dim)
        out[start:end] = to_np(cell_feats)

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / (norms + 1e-8)

# ── PCA ──────────────────────────────────────────────────────
def fit_pca(x, n_components=512):
    mean = x.mean(axis=0)
    xc = x - mean
    cov = (xc.T @ xc) / len(x)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1][:n_components]
    return mean, vecs[:, idx].astype(np.float32)

def apply_pca(x, mean, vecs):
    return ((x - mean) @ vecs).astype(np.float32)

# ── kNN (GPU destekli) ───────────────────────────────────────
def knn_predict(x_tr, y_tr, x_te, k=5, bs=500):
    x_tr_g = to_xp(x_tr)
    y_tr_g = to_xp(y_tr)
    oh = to_xp(np.eye(10, dtype=np.float32))[y_tr_g]
    sq_tr = xp.sum(x_tr_g**2, axis=1)

    preds = np.zeros(len(x_te), dtype=int)
    for i in range(0, len(x_te), bs):
        xb = to_xp(x_te[i:i+bs])
        d = xp.sum(xb**2, axis=1, keepdims=True) + sq_tr - 2*(xb @ x_tr_g.T)
        xp.maximum(d, 0, out=d)
        ti = xp.argpartition(d, k, axis=1)[:, :k]
        db = xp.take_along_axis(d, ti, axis=1)
        w  = 1.0 / (db + 1e-10)
        oh2 = oh[ti]
        batch_preds = to_np((w[:,:,None]*oh2).sum(axis=1).argmax(axis=1))
        preds[i:i+len(batch_preds)] = batch_preds
    return preds

# ══════════════════════════════════════════════════════════════
# ANA PIPELINE
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist

    t_total = time.time()

    # Veri
    print("Veri yukleniyor...", flush=True)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, -1).astype(np.float32) / 255.0
    x_test  = x_test.reshape(10000, -1).astype(np.float32) / 255.0
    y_train = y_train.astype(int)
    y_test  = y_test.astype(int)

    # 1) Gabor
    print("Gabor kernels (7×8=56 filters)...", flush=True)
    K = build_gabor_kernels()
    print("  {} filters, ksize=9".format(len(K)))

    print("Gabor features (train)...", end=" ", flush=True); t0 = time.time()
    g_train = gabor_features(x_train, K)
    print("{:.1f}sn  dim={}".format(time.time()-t0, g_train.shape[1]))

    print("Gabor features (test)...", end=" ", flush=True); t0 = time.time()
    g_test = gabor_features(x_test, K)
    print("{:.1f}sn".format(time.time()-t0))

    # 2) PCA(512)
    print("PCA(512)...", end=" ", flush=True); t0 = time.time()
    pca_mean, pca_vecs = fit_pca(g_train, 512)
    xp_train = apply_pca(g_train, pca_mean, pca_vecs)
    xp_test  = apply_pca(g_test,  pca_mean, pca_vecs)
    print("{:.1f}sn".format(time.time()-t0))

    # 3) kNN(k=5)
    print("kNN (k=5)...", end=" ", flush=True); t0 = time.time()
    preds = knn_predict(xp_train, y_train, xp_test, k=5)
    print("{:.1f}sn".format(time.time()-t0))

    # Sonuc
    acc = np.mean(preds == y_test) * 100
    n_wrong = (preds != y_test).sum()

    print()
    print("="*50)
    print("  Dogruluk : {:.2f}%".format(acc))
    print("  Yanlis   : {} / 10000".format(n_wrong))
    print("  Toplam   : {:.1f}sn".format(time.time()-t_total))
    print("="*50)

    # Detayli analiz
    target_pairs = [(2,7),(4,9),(8,9),(7,9),(3,5)]
    wrong = preds != y_test
    print("\n  Hedef cift hatalari:")
    for (a,b) in target_pairs:
        ab = (wrong & (y_test==a) & (preds==b)).sum()
        ba = (wrong & (y_test==b) & (preds==a)).sum()
        print("    {} <-> {} : {}".format(a, b, ab+ba))

    print("\n  Sinif bazli:")
    for c in range(10):
        mask = y_test == c
        nw = (preds[mask] != c).sum()
        print("    {} : {}/{}  ({:.2f}%)".format(
            c, nw, mask.sum(), nw/mask.sum()*100))
