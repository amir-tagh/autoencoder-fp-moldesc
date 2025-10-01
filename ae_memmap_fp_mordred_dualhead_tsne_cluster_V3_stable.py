#!/usr/bin/env python3
"""
Stable dual-head autoencoder for massive SMILES corpora using memmaps.
- FP head: BCEWithLogits + per-bit pos_weight from empirical frequencies
- Mordred head: MSE or Huber (SmoothL1)
- Training stability: LR warm-up + cosine decay, weight decay, grad clipping
- Optional Mordred standardization using *_kept_stats.npz (mean/std)
- Rich evaluation: per-feature errors, hist/CDF, PCA/t-SNE, clustering, barh plots

Run example (20M rows, z=256):
python ae_memmap_fp_mordred_dualhead_tsne_cluster_V3_stable.py train \
  --fp_memmap      "$SCRATCH/fp_22M.uint8" \
  --mordred_memmap "$SCRATCH/mordred_5M.f32" \
  --nrows 20000000 \
  --fp_bits 512 --latent_dim 256 \
  --epochs 40 --batch_size 4096 \
  --chunk_rows 100000 --num_workers 8 --ncpus 32 \
  --lambda_fp 0.7 --lambda_md 1.0 --use_huber \
  --lr 3e-4 --weight_decay 1e-4 --grad_clip 1.0 \
  --warmup_epochs 3 --eta_min 1e-5 \
  --eval_tsne_n 100000 --cluster_method kmeans --cluster_k 50 \
  --outdir AE_OUT_20M_z256
"""

import os, json, math, argparse, time, warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.use_deterministic_algorithms(False)

# ------------------------- utils -------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def set_num_threads(n):
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(n)
    try:
        torch.set_num_threads(n)
    except Exception:
        pass

# ------------------------- dataset -------------------------
class ConcatMemmapIterable(IterableDataset):
    """Streams batches from FP and Mordred memmaps, with optional Mordred standardization."""
    def __init__(self, fp_path, mordred_path, nrows, fp_bits, mordred_dim,
                 chunk_rows=100_000, batch_size=4096, seed=42,
                 md_mean=None, md_std=None):
        super().__init__()
        self.fp_path, self.mordred_path = fp_path, mordred_path
        self.nrows, self.fp_bits, self.mordred_dim = nrows, fp_bits, mordred_dim
        self.chunk_rows, self.batch_size, self.seed = chunk_rows, batch_size, seed
        self.md_mean = None if md_mean is None else np.asarray(md_mean, dtype=np.float32)
        self.md_std  = None if md_std  is None else np.asarray(md_std,  dtype=np.float32)

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid, wnum = (worker.id, worker.num_workers) if worker else (0,1)

        Xfp = np.memmap(self.fp_path, mode='r', dtype='uint8', shape=(self.nrows, self.fp_bits))
        Xm  = np.memmap(self.mordred_path, mode='r', dtype='float32', shape=(self.nrows, self.mordred_dim))

        starts = list(range(0, self.nrows, self.chunk_rows))
        rng = np.random.default_rng(self.seed + wid)
        rng.shuffle(starts); starts = starts[wid::wnum]

        for s in starts:
            e = min(s + self.chunk_rows, self.nrows)
            fp_blk = Xfp[s:e].astype(np.float32, copy=False)
            md_blk = Xm[s:e]
            if self.md_mean is not None and self.md_std is not None:
                md_blk = (md_blk - self.md_mean) / (self.md_std + 1e-8)
            idx = rng.permutation(e - s)
            for i in range(0, len(idx), self.batch_size):
                sel = idx[i:i+self.batch_size]
                xb = np.concatenate([fp_blk[sel], md_blk[sel]], axis=1)
                yield torch.from_numpy(xb)

# ------------------------- model (dual head) -------------------------
class DualHeadAE(nn.Module):
    def __init__(self, fp_bits, md_dim, latent_dim):
        super().__init__()
        self.fp_bits, self.md_dim = fp_bits, md_dim
        input_dim = fp_bits + md_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        self.dec_fp  = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(), nn.Linear(512, fp_bits)
        )  # logits for BCE
        self.dec_md  = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(), nn.Linear(512, md_dim)
        )  # continuous for MSE/Huber
    def forward(self, x):
        x = x.float()
        z = self.encoder(x)
        return self.dec_fp(z), self.dec_md(z), z

# ------------------------- training -------------------------
def compute_fp_bit_stats(fp_memmap_path, nrows, fp_bits, out_npz):
    Xfp = np.memmap(fp_memmap_path, mode='r', dtype='uint8', shape=(nrows, fp_bits))
    freq = Xfp.mean(axis=0).astype(np.float64)          # p(bit=1)
    pos_w = (1.0 - freq) / np.clip(freq, 1e-8, None)    # for BCE pos_weight
    np.savez(out_npz, freq=freq, pos_weight=pos_w)
    return freq, pos_w


def train_from_memmaps(
    fp_memmap, mordred_memmap, nrows, fp_bits, mordred_dim, outdir,
    latent_dim=128, epochs=20, batch_size=4096, ncpus=32,
    chunk_rows=100_000, num_workers=8, val_sample=100_000,
    lambda_fp=1.0, lambda_md=1.0, use_huber=False,
    lr=3e-4, weight_decay=1e-4, grad_clip=1.0, warmup_epochs=3, eta_min=1e-5,
    md_mean=None, md_std=None
):
    device = torch.device("cpu")
    set_num_threads(ncpus)
    model = DualHeadAE(fp_bits, mordred_dim, latent_dim).to(device)

    # Optimizer with weight decay
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # LR schedule helpers (epoch-wise warm-up + cosine)
    def _set_lr(optimizer, value):
        for g in optimizer.param_groups:
            g["lr"] = float(value)
    def _lr_for_epoch(ep):  # ep is 0-based
        if warmup_epochs > 0 and ep < warmup_epochs:
            return lr * (ep + 1) / float(warmup_epochs)
        t = max(0, ep - warmup_epochs)
        T = max(1, epochs - warmup_epochs)
        return eta_min + 0.5*(lr - eta_min)*(1 + math.cos(math.pi * t / T))

    # FP pos_weight
    stats_npz = os.path.join(outdir, "fp_bit_stats.npz")
    if not os.path.exists(stats_npz):
        ensure_dir(outdir)
        compute_fp_bit_stats(fp_memmap, nrows, fp_bits, stats_npz)
    Z = np.load(stats_npz)
    pos_weight = torch.tensor(Z["pos_weight"], dtype=torch.float32, device=device)
    fp_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    md_crit = nn.SmoothL1Loss() if use_huber else nn.MSELoss()

    # Streaming dataset
    ds = ConcatMemmapIterable(
        fp_memmap, mordred_memmap, nrows, fp_bits, mordred_dim,
        chunk_rows=chunk_rows, batch_size=batch_size,
        md_mean=md_mean, md_std=md_std
    )
    dl = DataLoader(
        ds, batch_size=None, num_workers=num_workers,
        persistent_workers=(num_workers>0), prefetch_factor=2
    )

    # Fixed validation subset
    rng = np.random.default_rng(42)
    vsz = min(val_sample, nrows)
    vidx = np.sort(rng.choice(nrows, size=vsz, replace=False))
    np.save(os.path.join(outdir, "val_indices.npy"), vidx)

    Xfp_v = np.memmap(fp_memmap, mode='r', dtype='uint8', shape=(nrows, fp_bits))[vidx].astype(np.float32)
    Xmd_v = np.memmap(mordred_memmap, mode='r', dtype='float32', shape=(nrows, mordred_dim))[vidx]
    if md_mean is not None and md_std is not None:
        Xmd_v = (Xmd_v - md_mean) / (md_std + 1e-8)
    Xmd_v = Xmd_v.astype(np.float32, copy=False)

    Xval = np.concatenate([Xfp_v, Xmd_v], axis=1).astype(np.float32)
    Xval = torch.from_numpy(np.concatenate([Xfp_v, Xmd_v], axis=1))
    dl_val = DataLoader(TensorDataset(Xval), batch_size=batch_size, shuffle=False, num_workers=0)

    tr_hist, va_hist, lr_hist = [], [], []
    for ep in range(epochs):
        # set LR for this epoch
        cur_lr = _lr_for_epoch(ep)
        _set_lr(opt, cur_lr)
        lr_hist.append(cur_lr)

        model.train(); s,n=0.0,0
        with tqdm(desc=f"Epoch {ep+1}/{epochs} [train]", unit="batch") as p:
            for xb in dl:
                xb = xb.to(device)
                x_fp, x_md = xb[:, :fp_bits], xb[:, fp_bits:]
                fp_logits, md_out, _ = model(xb)
                loss = lambda_fp*fp_crit(fp_logits, x_fp) + lambda_md*md_crit(md_out, x_md)
                opt.zero_grad(); loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                s += loss.item()*xb.size(0); n += xb.size(0); p.update(1)
        tr = s / max(1,n); tr_hist.append(tr)

        model.eval(); s=0.0
        with torch.no_grad(), tqdm(total=len(dl_val), desc=f"Epoch {ep+1}/{epochs} [val]", unit="batch") as p:
            for (xb,) in dl_val:
                xb = xb.to(device)
                x_fp, x_md = xb[:, :fp_bits], xb[:, fp_bits:]
                fp_logits, md_out, _ = model(xb)
                s += (lambda_fp*fp_crit(fp_logits, x_fp) + lambda_md*md_crit(md_out, x_md)).item() * xb.size(0)
                p.update(1)
        va = s / len(Xval); va_hist.append(va)

        # plots
        plt.figure(); plt.plot(tr_hist,label="train"); plt.plot(va_hist,label="val")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Autoencoder Training Loss")
        plt.tight_layout(); plt.savefig(os.path.join(outdir,"training_loss_plot.png")); plt.close()

        # CSV metrics with LR
        pd.DataFrame({
            "epoch": np.arange(1, len(tr_hist)+1),
            "train_loss": tr_hist,
            "val_loss": va_hist,
            "lr": lr_hist
        }).to_csv(os.path.join(outdir, "training_metrics.csv"), index=False)

        # LR plot
        plt.figure(); plt.plot(np.arange(1, len(lr_hist)+1), lr_hist)
        plt.xlabel("Epoch"); plt.ylabel("LR"); plt.title("Learning rate (warm-up + cosine)")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "lr_schedule_plot.png")); plt.close()

    torch.save(model.state_dict(), os.path.join(outdir, "autoencoder_model.pt"))
    return model, tr_hist, va_hist

# ------------------------- evaluation + clustering -------------------------
def evaluate_and_cluster(
    model, outdir, fp_memmap, mordred_memmap, nrows, fp_bits, mordred_dim,
    batch_size=4096, eval_tsne_n=10000,
    cluster_method="kmeans", cluster_k=8, dbscan_eps=3.0, dbscan_min_samples=20,
    hdbscan_min_cluster_size=50,
    eval_topk_features=6, eval_bar_top=50, md_names=None,
    md_mean=None, md_std=None
):    
    device = torch.device("cpu"); model.eval()

    # Build validation tensors (matching train-time dtype and standardization)
    vidx = np.load(os.path.join(outdir, "val_indices.npy"))

    Xfp_v = np.memmap(fp_memmap, mode='r', dtype='uint8',
                  shape=(nrows, fp_bits))[vidx].astype(np.float32, copy=False)

    Xmd_v = np.memmap(mordred_memmap, mode='r', dtype='float32',
                  shape=(nrows, mordred_dim))[vidx]
    if md_mean is not None and md_std is not None:
        Xmd_v = (Xmd_v - md_mean) / (md_std + 1e-8)
    Xmd_v = Xmd_v.astype(np.float32, copy=False)

    Xval_np = np.concatenate([Xfp_v, Xmd_v], axis=1).astype(np.float32, copy=False)
    Xval = torch.from_numpy(Xval_np)
    dl_val = DataLoader(TensorDataset(Xval), batch_size=batch_size,
                    shuffle=False, num_workers=0)


    per_sample_mse, per_sample_tanimoto = [], []
    feat_sse = np.zeros(fp_bits + mordred_dim, dtype=np.float64)
    n_total = 0

    keep_n = min(eval_tsne_n, len(Xval))
    stride = max(1, len(Xval)//keep_n)
    latents, lat_sel_idx = [], []

    for bidx, (xb,) in enumerate(tqdm(dl_val, desc="Eval", unit="batch")):
        xb = xb.to(device)
        x_fp = xb[:, :fp_bits]
        with torch.no_grad():
            fp_logits, md_out, z = model(xb)
            probs = torch.sigmoid(fp_logits)
            recon = torch.cat([probs, md_out], dim=1)
        diff = (recon - xb).cpu().numpy()
        mse = (diff*diff).mean(axis=1); per_sample_mse.append(mse)
        n_total += xb.size(0); feat_sse += (diff*diff).sum(axis=0)

        orig = x_fp.cpu().numpy().astype(np.uint8)
        pred = (probs.cpu().numpy() >= 0.5).astype(np.uint8)
        inter = (orig & pred).sum(axis=1)
        union = np.maximum((orig | pred).sum(axis=1), 1)
        tani = inter/union
        per_sample_tanimoto.append(tani)

        z_np = z.cpu().numpy()
        start = bidx*batch_size
        for i in range(z_np.shape[0]):
            gi = start + i
            if gi % stride == 0 and len(latents) < keep_n:
                latents.append(z_np[i]); lat_sel_idx.append(vidx[gi])

    per_sample_mse = np.concatenate(per_sample_mse)
    per_sample_tanimoto = np.concatenate(per_sample_tanimoto)
    per_feature_mse = feat_sse / max(1, n_total)

    # per-sample csv
    pd.DataFrame({"idx": vidx, "mse": per_sample_mse, "tanimoto": per_sample_tanimoto}) \
      .to_csv(os.path.join(outdir, "val_per_sample_metrics.csv"), index=False)

    # hist + cdf
    plt.figure(figsize=(6,4)); plt.hist(per_sample_mse, bins=50)
    plt.xlabel("Per-sample MSE"); plt.ylabel("Count")
    plt.title("Validation per-sample reconstruction error"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "recon_error_hist.png")); plt.close()

    plt.figure(figsize=(6,4))
    xs = np.linspace(0,100,len(per_sample_mse)); plt.plot(xs, np.sort(per_sample_mse))
    plt.xlabel("Percentile"); plt.ylabel("MSE"); plt.title("Reconstruction error CDF (validation)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "recon_error_cdf.png")); plt.close()

    # FP vs Mordred mean feature-wise error
    fp_mean = per_feature_mse[:fp_bits].mean() if fp_bits>0 else 0.0
    md_mean = per_feature_mse[fp_bits:].mean() if mordred_dim>0 else 0.0
    plt.figure(figsize=(4,4)); plt.bar(["FP bits","Mordred"], [fp_mean, md_mean])
    plt.ylabel("Mean per-feature MSE"); plt.title("FP vs Mordred reconstruction error")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fp_vs_mordred_mse.png")); plt.close()

    # bit freq vs error
    fp_stats = np.load(os.path.join(outdir, "fp_bit_stats.npz"))
    bit_mse = per_feature_mse[:fp_bits]
    plt.figure(figsize=(5,4))
    plt.scatter(fp_stats["freq"], bit_mse, s=8)
    plt.xlabel("Bit frequency p(bit=1)"); plt.ylabel("Per-bit MSE")
    plt.title("FP bit frequency vs reconstruction error")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "fp_freq_vs_error.png")); plt.close()

    # --------- Top-N features barh ---------
    if eval_bar_top and eval_bar_top > 0:
        order = np.argsort(-per_feature_mse)[:eval_bar_top]
        labels = []
        for i in order:
            if i < fp_bits: labels.append(f"fp_{i}")
            else:
                j = i - fp_bits
                labels.append((md_names[j] if md_names is not None and j < len(md_names) else f"md_{j}"))
        plt.figure(figsize=(8, max(4, 0.25*len(order))))
        plt.barh(range(len(order)), per_feature_mse[order][::-1])
        plt.yticks(range(len(order)), [str(l)[:60] for l in labels[::-1]])
        plt.xlabel("Per-feature MSE"); plt.title(f"Top {len(order)} features by reconstruction error")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"per_feature_mse_top{len(order)}_barh.png")); plt.close()

    # --------- Overlay hist for worst Mordred K ---------
    if eval_topk_features and eval_topk_features > 0 and mordred_dim > 0:
        k = min(eval_topk_features, mordred_dim)
        md_errs = per_feature_mse[fp_bits:]
        worst_local = np.argsort(-md_errs)[:k]
        worst_global = fp_bits + worst_local
        worst_names = [(md_names[i] if md_names is not None else f"md_{i}") for i in worst_local]

        orig_list = [ [] for _ in range(k) ]
        recon_list = [ [] for _ in range(k) ]
        for (xb,) in DataLoader(TensorDataset(Xval), batch_size=batch_size, shuffle=False):
            xb = xb.to(device)
            with torch.no_grad():
                fp_logits, md_out, _ = model(xb)
                probs = torch.sigmoid(fp_logits)
                recon = torch.cat([probs, md_out], dim=1).cpu().numpy()
            xb_np = xb.cpu().numpy()
            for j, gidx in enumerate(worst_global):
                orig_list[j].append(xb_np[:, gidx]); recon_list[j].append(recon[:, gidx])

        cols = 3; rows = int(math.ceil(k/cols))
        plt.figure(figsize=(cols*4.0, rows*3.0))
        for j in range(k):
            o = np.concatenate(orig_list[j]); r = np.concatenate(recon_list[j])
            ax = plt.subplot(rows, cols, j+1)
            ax.hist(o, bins=40, alpha=0.6, label="orig", density=True)
            ax.hist(r, bins=40, alpha=0.6, label="recon", density=True)
            ax.set_title(str(worst_names[j])[:50])
            if j % cols == 0: ax.set_ylabel("density")
            if j // cols == rows-1: ax.set_xlabel("value")
            ax.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"mordred_top{k}_orig_vs_recon_hist.png")); plt.close()

    # --------- t-SNE + clustering ---------
    if len(latents) >= 10:
        Z = np.vstack(latents).astype(np.float64)
        np.save(os.path.join(outdir, "z_val_subset.npy"), Z)

        pca = PCA(n_components=2, random_state=0)
        Zp = pca.fit_transform(Z); ev = pca.explained_variance_ratio_
        plt.figure(figsize=(5,4)); plt.scatter(Zp[:,0], Zp[:,1], s=3, alpha=0.5)
        plt.title(f"Latent PCA (2D), EVR={ev[0]:.2f}/{ev[1]:.2f}")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "latent_pca.png")); plt.close()

        tsne = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=30, random_state=0)
        Zt = tsne.fit_transform(Z)
        np.save(os.path.join(outdir, "latent_tsne.npy"), Zt)

        # color by error
        map_idx_to_pos = {idx:i for i,idx in enumerate(vidx)}
        err_for_subset = np.array([per_sample_mse[map_idx_to_pos[i]] for i in lat_sel_idx])
        tani_for_subset = np.array([per_sample_tanimoto[map_idx_to_pos[i]] for i in lat_sel_idx])

        plt.figure(figsize=(5,4))
        sc = plt.scatter(Zt[:,0], Zt[:,1], s=4, c=err_for_subset, cmap="viridis")
        plt.colorbar(sc, label="Per-sample MSE")
        plt.title("Latent t-SNE colored by recon error")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "latent_tsne_colored_by_error.png")); plt.close()

        # clustering
        labels = None; sil = None
        if cluster_method == "kmeans":
            km = KMeans(n_clusters=cluster_k, n_init=10, random_state=0)
            labels = km.fit_predict(Zt)
            if cluster_k > 1: sil = silhouette_score(Zt, labels)
        elif cluster_method == "dbscan":
            db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            labels = db.fit_predict(Zt)
            ok = (labels != -1).sum()
            if ok > 1 and len(np.unique(labels[labels!=-1]))>1:
                sil = silhouette_score(Zt[labels!=-1], labels[labels!=-1])
        elif cluster_method == "hdbscan":
            try:
                import hdbscan
                hd = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size)
                labels = hd.fit_predict(Zt)
                ok = (labels != -1).sum()
                if ok > 1 and len(np.unique(labels[labels!=-1]))>1:
                    sil = silhouette_score(Zt[labels!=-1], labels[labels!=-1])
            except Exception:
                print("[warn] hdbscan not available, falling back to kmeans")
                km = KMeans(n_clusters=cluster_k, n_init=10, random_state=0)
                labels = km.fit_predict(Zt)
                if cluster_k > 1: sil = silhouette_score(Zt, labels)
        else:
            raise ValueError("cluster_method must be kmeans|dbscan|hdbscan")

        plt.figure(figsize=(6,5))
        sc = plt.scatter(Zt[:,0], Zt[:,1], s=5, c=labels, cmap="tab20")
        ttl = f"Latent t-SNE clusters ({cluster_method})" + (f", silhouette={sil:.3f}" if sil is not None else "")
        plt.title(ttl)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "latent_tsne_clusters.png")); plt.close()

        dfc = pd.DataFrame({
            "val_idx": lat_sel_idx,
            "tsne_x": Zt[:,0], "tsne_y": Zt[:,1],
            "cluster": labels,
            "mse": err_for_subset,
            "tanimoto": tani_for_subset
        })
        dfc.to_csv(os.path.join(outdir, "tsne_clusters_per_sample.csv"), index=False)

        agg = dfc.groupby("cluster").agg(
            n=("cluster","size"),
            mse_mean=("mse","mean"),
            mse_median=("mse","median"),
            tanimoto_mean=("tanimoto","mean")
        ).sort_values("n", ascending=False)
        agg.to_csv(os.path.join(outdir, "cluster_stats.csv"))

        top_clusters = agg.head(10).index.tolist()
        plt.figure(figsize=(7,4))
        plt.boxplot([dfc[dfc.cluster==c].mse.values for c in top_clusters], labels=list(map(str, top_clusters)))
        plt.xlabel("cluster"); plt.ylabel("MSE"); plt.title("Recon error by cluster (top 10)")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "cluster_mse_boxplot.png")); plt.close()

    with open(os.path.join(outdir, "fp_mean_tanimoto.txt"), "w") as f:
        f.write(f"{float(per_sample_tanimoto.mean()):.4f}\n")

# ------------------------- latent tuning -------------------------
def _num_params(model): return int(sum(p.numel() for p in model.parameters()))

def choose_by_pareto(rows, delta=0.02):
    best = min(r["val_loss"] for r in rows); cutoff = best*(1.0+float(delta))
    return sorted([r for r in rows if r["val_loss"]<=cutoff], key=lambda r:r["latent_dim"])[0], best, cutoff

def train_short(fp_memmap, mordred_memmap, nrows, fp_bits, md_dim, outdir, z, epochs, batch_size, chunk_rows, num_workers, ncpus):
    model, tr, va = train_from_memmaps(fp_memmap, mordred_memmap, nrows, fp_bits, md_dim, outdir,
                                       latent_dim=z, epochs=epochs, batch_size=batch_size,
                                       ncpus=ncpus, chunk_rows=chunk_rows, num_workers=num_workers)
    return float(va[-1]), float(tr[-1]), _num_params(model)

def tune_latent_dim(fp_memmap, mordred_memmap, nrows, fp_bits, md_dim, outdir, grid, epochs=8, batch_size=4096,
                    chunk_rows=100_000, num_workers=8, ncpus=32, delta=0.02, repeats=1):
    ensure_dir(outdir)
    rows=[]
    for d in grid:
        best_val,best_tr,best_params,best_time=None,None,None,None
        for rep in range(repeats):
            t0=time.time()
            val,tr,params = train_short(fp_memmap,mordred_memmap,nrows,fp_bits,md_dim,outdir,int(d),
                                        epochs,batch_size,chunk_rows,num_workers,ncpus)
            tt=time.time()-t0
            if best_val is None or val<best_val:
                best_val,best_tr,best_params,best_time = val,tr,params,tt
        rows.append(dict(latent_dim=int(d), val_loss=best_val, train_loss=best_tr,
                         params=best_params, time_s=best_time))
        print(f"[tune] z={d:4d}  val={best_val:.6f}  params={best_params:,}  time={best_time:.1f}s")
    chosen,best,cutoff = choose_by_pareto(rows,delta)
    print(f"[tune] best val={best:.6f}; cutoff={cutoff:.6f} (delta={delta*100:.1f}%)")
    print(f"[tune] chosen z={chosen['latent_dim']} (val={chosen['val_loss']:.6f}, params={chosen['params']:,})")
    df=pd.DataFrame(rows).sort_values("latent_dim")
    df.to_csv(os.path.join(outdir,"latent_tuning_results.csv"), index=False)
    plt.figure(figsize=(6,4)); plt.plot(df["latent_dim"],df["val_loss"],marker="o")
    plt.scatter([chosen["latent_dim"]],[chosen["val_loss"]],s=80,label=f"z={chosen['latent_dim']}")
    plt.xlabel("latent_dim"); plt.ylabel("val MSE"); plt.title("Latent size tuning"); plt.xscale("log",base=2); plt.grid(True,ls="--",alpha=.4); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"latent_tuning_plot.png")); plt.close()
    return int(chosen["latent_dim"])

# ------------------------- CLI -------------------------
def main():
    ap = argparse.ArgumentParser(description="FP + Mordred AE (dual-head) with evaluation, t-SNE & clustering [stable]")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # train
    ap_tr = sub.add_parser("train")
    ap_tr.add_argument("--fp_memmap", required=True)
    ap_tr.add_argument("--mordred_memmap", required=True)
    ap_tr.add_argument("--nrows", type=int, required=True)
    ap_tr.add_argument("--fp_bits", type=int, default=2048)
    ap_tr.add_argument("--outdir", required=True)
    ap_tr.add_argument("--latent_dim", type=int, default=128)
    ap_tr.add_argument("--epochs", type=int, default=20)
    ap_tr.add_argument("--batch_size", type=int, default=4096)
    ap_tr.add_argument("--chunk_rows", type=int, default=100_000)
    ap_tr.add_argument("--num_workers", type=int, default=8)
    ap_tr.add_argument("--ncpus", type=int, default=os.cpu_count() or 1)
    # metadata (logged only)
    ap_tr.add_argument("--fp_radius", type=int, default=None, help="(optional metadata) radius used to create FP memmap")
    # losses
    ap_tr.add_argument("--lambda_fp", type=float, default=1.0)
    ap_tr.add_argument("--lambda_md", type=float, default=1.0)
    ap_tr.add_argument("--use_huber", action="store_true")
    # stability knobs
    ap_tr.add_argument("--lr", type=float, default=3e-4)
    ap_tr.add_argument("--weight_decay", type=float, default=1e-4)
    ap_tr.add_argument("--grad_clip", type=float, default=1.0, help="0 to disable")
    ap_tr.add_argument("--warmup_epochs", type=int, default=3)
    ap_tr.add_argument("--eta_min", type=float, default=1e-5, help="min LR for cosine schedule")
    # eval
    ap_tr.add_argument("--no_eval", action="store_true")
    ap_tr.add_argument("--eval_tsne_n", type=int, default=10000)
    ap_tr.add_argument("--cluster_method", choices=["kmeans","dbscan","hdbscan"], default="kmeans")
    ap_tr.add_argument("--cluster_k", type=int, default=8)
    ap_tr.add_argument("--dbscan_eps", type=float, default=3.0)
    ap_tr.add_argument("--dbscan_min_samples", type=int, default=20)
    ap_tr.add_argument("--hdbscan_min_cluster_size", type=int, default=50)
    ap_tr.add_argument("--eval_topk_features", type=int, default=6, help="Top-K Mordred descriptors for orig vs recon overlays")
    ap_tr.add_argument("--eval_bar_top", type=int, default=50, help="Top-N features for bar plot by per-feature MSE")

    # tune
    ap_tu = sub.add_parser("tune")
    ap_tu.add_argument("--fp_memmap", required=True)
    ap_tu.add_argument("--mordred_memmap", required=True)
    ap_tu.add_argument("--nrows", type=int, required=True)
    ap_tu.add_argument("--fp_bits", type=int, default=2048)
    ap_tu.add_argument("--outdir", required=True)
    ap_tu.add_argument("--latent_grid", type=str, default="32,64,128,256,512")
    ap_tu.add_argument("--epochs_tune", type=int, default=8)
    ap_tu.add_argument("--batch_size", type=int, default=4096)
    ap_tu.add_argument("--chunk_rows", type=int, default=100_000)
    ap_tu.add_argument("--num_workers", type=int, default=8)
    ap_tu.add_argument("--ncpus", type=int, default=os.cpu_count() or 1)
    ap_tu.add_argument("--delta", type=float, default=0.02)
    ap_tu.add_argument("--repeats", type=int, default=1)

    args = ap.parse_args()

    if args.cmd == "train":
        ensure_dir(args.outdir)

        # Mordred names + (optional) mean/std from *_kept_stats.npz
        base = os.path.splitext(args.mordred_memmap)[0]
        kept_stats = base + "_kept_stats.npz"
        if not os.path.exists(kept_stats):
            cand = [os.path.join(os.path.dirname(args.mordred_memmap), p)
                    for p in os.listdir(os.path.dirname(args.mordred_memmap)) if p.endswith("_kept_stats.npz")]
            if not cand:
                raise FileNotFoundError("Could not find *_kept_stats.npz for Mordred memmap.")
            kept_stats = cand[0]
        Zs = np.load(kept_stats, allow_pickle=True)
        md_names = list(Zs["names"]); md_dim = len(md_names)
        # Try to parse mean/std keys that may vary across pipelines
        md_mean = None; md_std = None
        for k in ("mean","means","mu"):
            if k in Zs: md_mean = Zs[k]
        for k in ("std","stds","sigma"):
            if k in Zs: md_std = Zs[k]

        # Train
        model, tr_hist, va_hist = train_from_memmaps(
            fp_memmap=args.fp_memmap, mordred_memmap=args.mordred_memmap, nrows=args.nrows,
            fp_bits=args.fp_bits, mordred_dim=md_dim, outdir=args.outdir,
            latent_dim=args.latent_dim, epochs=args.epochs, batch_size=args.batch_size,
            ncpus=args.ncpus, chunk_rows=args.chunk_rows, num_workers=args.num_workers,
            lambda_fp=args.lambda_fp, lambda_md=args.lambda_md, use_huber=args.use_huber,
            lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
            warmup_epochs=args.warmup_epochs, eta_min=args.eta_min,
            md_mean=md_mean, md_std=md_std
        )

        # Log metadata incl. fp_radius if provided
        meta = {
            "fp_bits": int(args.fp_bits), "latent_dim": int(args.latent_dim),
            "mordred_dim": int(md_dim),
            "fp_radius": (None if args.fp_radius is None else int(args.fp_radius)),
            "lr": float(args.lr), "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip), "warmup_epochs": int(args.warmup_epochs),
            "eta_min": float(args.eta_min)
        }
        with open(os.path.join(args.outdir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if not args.no_eval:
            evaluate_and_cluster(
                model=model, outdir=args.outdir, fp_memmap=args.fp_memmap, mordred_memmap=args.mordred_memmap,
                nrows=args.nrows, fp_bits=args.fp_bits, mordred_dim=md_dim, batch_size=args.batch_size,
                eval_tsne_n=args.eval_tsne_n, cluster_method=args.cluster_method, cluster_k=args.cluster_k,
                dbscan_eps=args.dbscan_eps, dbscan_min_samples=args.dbscan_min_samples,
                hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
                eval_topk_features=args.eval_topk_features, eval_bar_top=args.eval_bar_top, md_names=md_names,md_mean=md_mean, md_std=md_std
            )
        print(f"[train] Done. Outputs in {args.outdir}")

    elif args.cmd == "tune":
        base = os.path.splitext(args.mordred_memmap)[0]
        kept_stats = base + "_kept_stats.npz"
        Zs = np.load(kept_stats, allow_pickle=True); md_dim = len(list(Zs["names"]))
        grid = [int(x) for x in args.latent_grid.split(",") if x.strip()]
        z = tune_latent_dim(fp_memmap=args.fp_memmap, mordred_memmap=args.mordred_memmap,
                            nrows=args.nrows, fp_bits=args.fp_bits, md_dim=md_dim, outdir=args.outdir,
                            grid=grid, epochs=args.epochs_tune, batch_size=args.batch_size,
                            chunk_rows=args.chunk_rows, num_workers=args.num_workers,
                            ncpus=args.ncpus, delta=args.delta, repeats=args.repeats)
        print(f"[tune] recommended latent_dim = {z}")

if __name__ == "__main__":
    main()

