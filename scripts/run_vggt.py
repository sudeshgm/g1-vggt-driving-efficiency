"""GPU driver (DRAFT, not run on CPU box). Stock VGGT-1B controls. fp16 autocast (T4 sm75 has no bf16),
attention via torch SDPA (VGGT's fused_attn path) -> on sm75 PyTorch picks mem-efficient/math kernels; no FA2.
Usage: python scripts/run_vggt.py --bundle bundle --stride 1 --scale 1.0 --out preds/dense_N32
Writes per-window npz: idx, extrinsic (world->cam 3x4, VGGT convention), depth (N,H,W), depth_conf, input_hw, secs, peak_mem."""
import argparse, json, os, time, numpy as np, torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import torch.nn.functional as F
ap = argparse.ArgumentParser(); ap.add_argument("--bundle", default="bundle"); ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--out", required=True)
ap.add_argument("--manifest", default="manifests/windows.json"); ap.add_argument("--img_root", default="bundle/images")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval().half()  # non-commercial weights, research use only
W = json.load(open(a.manifest))["windows"]
for w in W:
    p = f"{a.out}/{w['window_id']}.npz"
    if os.path.exists(p): continue
    idx = list(range(0, w["n"], a.stride)); files = [os.path.join(a.img_root, w["frames"][i]["file"]) for i in idx]
    imgs = load_and_preprocess_images(files).cuda()             # (S,3,294,518)
    if a.scale != 1.0:
        H = int(round(imgs.shape[2] * a.scale / 14)) * 14; Wd = int(round(imgs.shape[3] * a.scale / 14)) * 14
        imgs = F.interpolate(imgs, size=(H, Wd), mode="bilinear", antialias=True, align_corners=False)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = model(imgs.half())
    torch.cuda.synchronize(); dt = time.time() - t0
    ext, intr = pose_encoding_to_extri_intri(out["pose_enc"], imgs.shape[-2:])
    np.savez_compressed(p, idx=np.array(idx), extrinsic=ext[0].float().cpu().numpy(), intrinsic=intr[0].float().cpu().numpy(),
        depth=out["depth"][0, ..., 0].float().cpu().numpy().astype(np.float16), depth_conf=out["depth_conf"][0].float().cpu().numpy().astype(np.float16),
        input_hw=np.array(imgs.shape[-2:]), secs=dt, peak_mem=torch.cuda.max_memory_allocated())
    print(w["window_id"], len(idx), tuple(imgs.shape[-2:]), f"{dt:.2f}s", torch.cuda.max_memory_allocated() / 2**30, "GiB")
