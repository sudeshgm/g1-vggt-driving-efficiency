# Vendored third-party code (pinned; research/non-commercial use)
| method | repo | commit | vendored | license | patches |
|---|---|---|---|---|---|
| FastVGGT | github.com/mystorm16/FastVGGT | 6526e275a29572653a034762bb3c6c9ce280ff55 | `FastVGGT/{vggt,merging}` (full package, run from source via PYTHONPATH) | see FastVGGT/LICENSE.txt | `FastVGGT.g1.patch`: torch/torchvision pins stripped; hard-coded bf16 -> `FASTVGGT_DTYPE` env (fp16 T4 / fp32 CPU; default bf16 = upstream); `device="cuda"` -> model device; cuda sync/empty_cache guarded; similarity matmul in input dtype instead of bf16 |
| GoToHunt | github.com/zsh2000/gotohunt | 6c92c4f0c747847d85d78fadd3a426e7046e45dc | official `layers/attention.py` (used verbatim, loaded by path) + `aggregator.reference.py` (routing replicated in g1/gotohunt.py) | Apache-2.0 header / see LICENSE | none |
| Co-Me | github.com/co-me-tokens/CoMe | 5c7cfbc14da37f5dbb991e3a6f9e2c4a5efb8833 | distilled VGGT confidence predictor ckpt `ckpt/vggt/20260324_172742/step_02000` + reference sources in `ref/` | CC BY-NC 4.0 (LICENSE file; README says BY-NC-SA) | none to vendored files; g1/come.py is a pure-PyTorch port of `direct_accelerate` (naive merge verbatim, index_select split, SDPA with log-weight per-key bias replaces sm80+ flash-attn kernel) |
| AVGGT | — | — | g1/avggt.py (our reimplementation) | — | — |
