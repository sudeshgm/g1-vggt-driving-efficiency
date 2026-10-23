"""End-to-end: fake predictions == GT (scaled) must score AUC=1, ATE=0, AbsRel=0; and score only eval frames."""
import json, os, subprocess, sys, numpy as np, csv
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def test_oracle_predictions(tmp_path):
    W = json.load(open(f"{R}/manifests/windows.json"))["windows"][:3]
    for stride in (1, 4):
        d = tmp_path / f"oracle_s{stride}"; d.mkdir()
        for w in W:
            idx = np.arange(0, 32, stride); H, Wd = 294, 518
            T0 = np.array(w["frames"][0]["cam_to_world"])
            ext = np.stack([np.linalg.inv(np.linalg.inv(T0) @ np.array(w["frames"][i]["cam_to_world"]))[:3] for i in idx])
            ext[:, :, 3] *= 0.37  # unknown scale
            dep = np.zeros((len(idx), H, Wd), np.float16)
            for k, i in enumerate(idx):
                f = w["frames"][i]
                if "depth_gt" in f:
                    g = np.load(f"{R}/" + f["depth_gt"]); u = ((g["u"] + .5) * Wd / 1600).astype(int); v = ((g["v"] + .5) * H / 900).astype(int)
                    dep[k, v, u] = g["d"] * 0.37
            K = np.stack([np.diag([Wd / 1600, H / 900, 1.]) @ np.array(w["frames"][i]["K"]) for i in idx])
            np.savez(d / f"{w['window_id']}.npz", idx=idx, intrinsic=K, extrinsic=ext, depth=dep, input_hw=np.array([H, Wd]), secs=1.0, peak_mem=0)
    man = tmp_path / "m.json"; json.dump({"windows": W}, open(man, "w"))
    subprocess.check_call([sys.executable, "scripts/score.py", str(tmp_path / "oracle_s1"), str(tmp_path / "oracle_s4"),
                           "--out", str(tmp_path / "res"), "--manifest", str(man)], cwd=R)
    rows = list(csv.DictReader(open(tmp_path / "res/per_window.csv")))
    assert len(rows) == 6
    for r in rows:
        assert float(r["auc30"]) == 1.0 and float(r["ate_m"]) < 1e-3
        # collisions at 294x518 make z-buffer ties; allow tiny error
        assert float(r["absrel_all"]) < 0.02 and float(r["pm_rel_all"]) < 0.03
