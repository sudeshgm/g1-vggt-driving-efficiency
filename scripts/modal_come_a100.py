"""Co-Me fused-kernel timing check on ONE Modal A100 (budget < 16 min wall, hard timeout 15.5 min).
Why: Kaggle T4 (sm75) can't run Co-Me's sm80+ fused kernels, so Block-1 Co-Me s/window is INVALID.
This measures dense VGGT vs Co-Me (fused) vs Co-Me (pure PyTorch, the path we ran on T4) on the SAME A100, so we
get the fused/unfused speed ratio (the accuracy is already measured on T4 - math-equivalent). Random input, Co-Me's own
bench (32 frames; bench.py hard-resizes to 504x504), at thresholds 0.5 (default) and our b0.5 0.447. No data or preds are uploaded.
Run:  modal run scripts/modal_come_a100.py      (results printed + saved to the 'g1-come' volume)
"""
import modal
COME_SHA = "5c7cfbc14da37f5dbb991e3a6f9e2c4a5efb8833"
image = (modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.03-py3")  # = Co-Me's cu128 docker base
         .apt_install("git")
         .run_commands(f"git clone https://github.com/co-me-tokens/CoMe /come",
                       f"cd /come && git checkout {COME_SHA} && git submodule update --init --depth 1 src/cuda_extension/cutlass",
                       "pip install -r /come/docker/requirements.txt"))
vol = modal.Volume.from_name("g1-come", create_if_missing=True)
app = modal.App("g1-come-a100", image=image)

@app.function(gpu="A100-40GB", timeout=930, volumes={"/out": vol})
def check():
    import subprocess, time, json, os
    t0 = time.time(); log = {}
    def sh(cmd, limit):
        print("+", cmd, flush=True); s = time.time()
        r = subprocess.run(cmd, shell=True, cwd="/come", timeout=limit, capture_output=True, text=True)
        print(r.stdout[-4000:], r.stderr[-3000:], flush=True); log[cmd] = dict(rc=r.returncode, secs=time.time() - s, tail=r.stdout[-2000:])
        return r
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv && python -c 'import torch;print(torch.__version__, torch.cuda.get_device_capability())'", 60)
    # kernel builds: single arch (sm80) via the repo's own gencode; ~6-9 min budget total
    sh("MAX_JOBS=12 bash src/cuda_extension/co_me/install.sh", 240)
    sh("bash src/cuda_extension/flash_attn/install.sh", 420)
    left = 930 - (time.time() - t0) - 30
    CK = ""  # default ckpt path output/confidence_distill/vggt/20260324_172742/step_02000 ships in the repo
    # 1) dense VGGT vs FUSED Co-Me (sm80 kernels), 32 frames, 10 samples, our b0.5 threshold
    sh(f"python -m src.bench first_n=10 {CK} acc.token_merger.threshold=0.447", max(60, left * 0.5))
    left = 930 - (time.time() - t0) - 30
    # 2) dense VGGT vs DIRECT (pure-PyTorch; the path we timed on T4), same threshold -> fused/unfused ratio
    sh(f"python -m src.bench first_n=6 model@acc=co_me_direct_vggt_3x3 {CK} acc.token_merger.threshold=0.447", max(60, left))
    log["total_secs"] = time.time() - t0
    json.dump(log, open("/out/come_a100_check.json", "w"), indent=1); vol.commit(); print("TOTAL", log["total_secs"])
    return log

@app.local_entrypoint()
def main():
    r = check.remote(); print({k: (v["rc"], round(v["secs"])) if isinstance(v, dict) else v for k, v in r.items()})
