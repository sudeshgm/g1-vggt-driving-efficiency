"""Metrics. All pose metrics take camera-to-world 4x4 matrices."""
import numpy as np

RANGE_BINS_M = ((0, 15), (15, 40), (40, 80), (80, np.inf))

def evaluation_subset(n_frames=32, max_stride=4):
    """Frames present in every config (N, N/2, N/4 uniform subsampling from index 0). Score only these."""
    return np.arange(0, n_frames, max_stride)

def subsample_indices(n_frames, stride):
    return np.arange(0, n_frames, stride)

def median_scale(pred, gt, mask):
    return np.median(gt[mask]) / np.median(pred[mask])

def depth_metrics_binned(pred, gt, bins=RANGE_BINS_M, align="median"):
    """pred, gt: arrays (any shape, same). gt==0 -> no LiDAR. Returns per-bin AbsRel, delta<1.25, n, coverage."""
    valid = (gt > 0) & np.isfinite(pred) & (pred > 0)
    s = median_scale(pred, gt, valid) if align == "median" else 1.0
    p = pred * s
    out = {"scale": float(s)}
    n_valid = valid.sum()
    for lo, hi in list(bins) + [(0, np.inf)]:
        m = valid & (gt >= lo) & (gt < hi)
        key = "all" if (lo, hi) == (0, np.inf) else f"{lo}-{hi if np.isfinite(hi) else 'inf'}"
        if m.sum() == 0:
            out[key] = dict(absrel=np.nan, d125=np.nan, n=0, coverage=0.0); continue
        r = np.abs(p[m] - gt[m]) / gt[m]
        d = np.maximum(p[m] / gt[m], gt[m] / p[m])
        out[key] = dict(absrel=float(r.mean()), d125=float((d < 1.25).mean()), n=int(m.sum()),
                        coverage=float(m.sum() / max(n_valid, 1)))
    return out

def umeyama(src, dst, with_scale=True):
    """Find s,R,t minimizing ||dst - (s R src + t)||. src,dst (N,3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    C = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(C)
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: E[2, 2] = -1
    R = U @ E @ Vt
    s = (S * np.diag(E)).sum() / (xs ** 2).sum(1).mean() if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t

def ate(pred_c2w, gt_c2w):
    """ATE RMSE (m) of camera centres after sim(3) Umeyama alignment."""
    ps, gs = pred_c2w[:, :3, 3], gt_c2w[:, :3, 3]
    s, R, t = umeyama(ps, gs)
    al = (s * (R @ ps.T)).T + t
    return float(np.sqrt(((al - gs) ** 2).sum(1).mean())), (s, R, t)

def _rot_err_deg(R1, R2):
    c = (np.trace(R1.T @ R2) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

def _ang_deg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9: return np.nan  # undefined for zero baseline
    c = np.dot(a, b) / (na * nb)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

def pairwise_pose_errors(pred_c2w, gt_c2w):
    """VGGT/RelPose convention: for all pairs i<j, rotation error and translation-direction angle of relative pose."""
    n = len(pred_c2w); re, te = [], []
    for i in range(n):
        for j in range(i + 1, n):
            Rp = np.linalg.inv(pred_c2w[j]) @ pred_c2w[i]
            Rg = np.linalg.inv(gt_c2w[j]) @ gt_c2w[i]
            re.append(_rot_err_deg(Rp[:3, :3], Rg[:3, :3]))
            # translation direction error is sign-ambiguous-free here (we know direction)
            te.append(_ang_deg(Rp[:3, 3], Rg[:3, 3]))
    return np.array(re), np.array(te)

def pose_auc(re, te, max_th=30):
    """AUC of max(rot,trans) error over thresholds 1..max_th deg (VGGT eval style). NaN trans -> counted as rot only."""
    err = np.where(np.isnan(te), re, np.maximum(re, te))
    ths = np.arange(1, max_th + 1)
    acc = np.array([(err < t).mean() for t in ths])
    return float(acc.mean())

def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0, stat=np.mean):
    v = np.asarray([x for x in values if np.isfinite(x)])
    if len(v) == 0: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)])
    return float(stat(v)), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))

def pointmap_error(pred_pts, gt_pts, gt_depth, bins=RANGE_BINS_M):
    """Point-map error: Umeyama sim(3) of predicted 3D points (pred depth+K+pose) onto GT LiDAR world points,
    per range band: mean relative 3D error ||p-g||/d_gt and median absolute error (m, GT scale)."""
    s, R, t = umeyama(pred_pts, gt_pts)
    e = np.linalg.norm((s * (R @ pred_pts.T)).T + t - gt_pts, axis=1); out = {}
    for lo, hi in list(bins) + [(0, np.inf)]:
        m = (gt_depth >= lo) & (gt_depth < hi)
        key = "all" if (lo, hi) == (0, np.inf) else f"{lo}-{hi if np.isfinite(hi) else 'inf'}"
        out[key] = dict(rel=float((e[m] / gt_depth[m]).mean()) if m.any() else np.nan,
                        med_m=float(np.median(e[m])) if m.any() else np.nan)
    return out
