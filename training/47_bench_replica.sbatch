#!/bin/bash
#SBATCH --job-name=qp_bench
#SBATCH --partition=avg
#SBATCH --account=avg
# Pin the card type: `avg` also contains Tesla T4s (ai09/ai10), and an unpinned
# request landed there first time -- 17.6 s/batch, which reads as a broken model
# rather than a slow GPU. Benchmark on what training will actually use.
#SBATCH --gres=gpu:lovelace_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench-%j.out
#SBATCH --error=logs/bench-%j.err
# Paths (interpreters, dataset, outputs) come from env.sh at the repo root.
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/env.sh"
set -euo pipefail
cd "$PTP_REPO"; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
${PY_TRACKON} -u - <<'PYEOF'
import sys, time, torch
sys.path.insert(0, '.')
from droid_qp.action_decoder import ActionMemory
dev='cuda'
print(f"\n{'slots':>6s} {'B':>3s} {'ckpt':>5s} {'s/batch':>8s} {'peak GB':>8s} {'params':>8s}")
for slots in (24, 8, 1):
    for B in (8, 16):
        for ck in (True,):
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                m = ActionMemory(D=256, heads=4, layers=3, M=24, scene_slots=slots,
                                 n_obj=100, checkpoint=ck).to(dev).train()
                Tw, M, N = 48, 24, 128
                L = (M-1)+Tw
                r=torch.randn(B,L,N,256,device=dev); s=torch.randn(B,L,100,256,device=dev)
                h=torch.ones(B,L,dtype=torch.bool,device=dev)
                pm=torch.zeros(B,N,dtype=torch.bool,device=dev); pm[:,110:]=True
                for _ in range(2):   # warm up cudnn/alloc
                    o=m(r,s,h,pm,Tw); o.float().pow(2).mean().backward(); m.zero_grad()
                torch.cuda.synchronize(); t0=time.time()
                o=m(r,s,h,pm,Tw); o.float().pow(2).mean().backward()
                torch.cuda.synchronize(); dt=time.time()-t0
                n=sum(p.numel() for p in m.parameters())
                print(f"{slots:6d} {B:3d} {str(ck):>5s} {dt:8.2f} "
                      f"{torch.cuda.max_memory_allocated()/1e9:8.1f} {n/1e6:7.2f}M")
                del m,r,s,o
            except torch.OutOfMemoryError:
                print(f"{slots:6d} {B:3d} {str(ck):>5s} {'OOM':>8s}")
                torch.cuda.empty_cache()
PYEOF
date; echo done
