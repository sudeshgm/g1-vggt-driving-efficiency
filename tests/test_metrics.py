import numpy as np
from g1.metrics import *
from g1.geometry import make_T, quat_to_R

def test_eval_subset_in_all_configs():
    ev = set(evaluation_subset(32, 4))
    for s in (1, 2, 4): assert ev <= set(subsample_indices(32, s))
    assert len(ev) == 8

def test_depth_bins_perfect_and_scaled():
    gt = np.array([5., 12, 30, 60, 100, 0]); pred = gt * 3 + (gt == 0)
    m = depth_metrics_binned(pred, gt)
    assert np.isclose(m["scale"], 1/3) and m["all"]["absrel"] < 1e-9 and m["all"]["d125"] == 1
    assert m["0-15"]["n"] == 2 and np.isclose(sum(m[k]["coverage"] for k in ["0-15","15-40","40-80","80-inf"]), 1)

def test_depth_bin_isolated_error():
    gt = np.array([5., 5, 5, 100]); pred = np.array([5., 5, 5, 150])
    m = depth_metrics_binned(pred, gt)
    assert m["0-15"]["absrel"] == 0 and np.isclose(m["80-inf"]["absrel"], 0.5)

def test_umeyama_recovers_sim3():
    rng = np.random.default_rng(0); X = rng.normal(size=(50, 3))
    R = quat_to_R([0.9, 0.1, 0.3, -0.2]); Y = 2.5 * X @ R.T + [1, 2, 3]
    s, R2, t = umeyama(X, Y)
    assert np.isclose(s, 2.5) and np.allclose(R2, R) and np.allclose(t, [1, 2, 3])

def _traj(n=8):
    return np.stack([make_T(quat_to_R([1, 0, 0, 0.05*i]), [i*1.0, 0.1*i*i, 0]) for i in range(n)])

def test_ate_zero_under_similarity():
    g = _traj(); S = make_T(quat_to_R([0.8, 0.2, 0.1, 0.3]), [5, -1, 2])
    p = np.stack([S @ T for T in g]); p[:, :3, 3] *= 1  # rigid
    p2 = p.copy(); p2[:, :3, 3] = 0.3 * p[:, :3, 3]
    assert ate(p2, g)[0] < 1e-6

def test_pose_auc_perfect_and_bad():
    g = _traj(); re, te = pairwise_pose_errors(g, g)
    assert pose_auc(re, te) == 1.0
    noisy = g.copy(); noisy[:, :3, 3] += np.random.default_rng(1).normal(scale=3, size=(8, 3))
    assert pose_auc(*pairwise_pose_errors(noisy, g)) < 0.9

def test_bootstrap_ci_brackets_mean():
    m, lo, hi = bootstrap_ci(np.arange(30.), n_boot=2000)
    assert lo < m < hi and np.isclose(m, 14.5)

def test_pointmap_error_zero_under_sim3():
    rng = np.random.default_rng(2); g = rng.uniform(1, 60, size=(200, 3)); d = g[:, 2]
    R = quat_to_R([0.9, 0.2, -0.1, 0.3]); p = 0.4 * (g @ R.T) + [3, 1, -2]
    o = pointmap_error(p, g, d); assert o["all"]["rel"] < 1e-9 and set(o) >= {"0-15", "40-80", "all"}
