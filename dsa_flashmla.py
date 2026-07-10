"""
Benchmark: FlashMLA sparse prefill with GLM-5 parameters.

GLM-5 config (https://www.modelscope.cn/models/ZhipuAI/GLM-5/file/view/master/config.json):
  num_attention_heads=64, kv_lora_rank=512, qk_rope_head_dim=64, index_topk=2048

After matrix absorption:
  h_q=64, h_kv=1, d_qk=576, d_v=512, topk=2048

Scenario:
  Sparse prefill; s_q and s_kv are configured independently and swept as a
  cartesian product (no KV cache hit-rate modeling).
  - s_q  ∈ SQ_LIST   (query tokens, independent)
  - s_kv ∈ SKV_LIST  (KV cache length, independent)
  - topk = 2048 (fixed)
  Each query gathers topk keys from the s_kv cache.

API:
  flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v)
    q:       [s_q, h_q, d_qk], bfloat16
    kv:      [s_kv, h_kv, d_qk], bfloat16
    indices: [s_q, h_kv, topk], int32
    sm_scale: float
    d_v:     int (512)
  Returns: (output [s_q, h_q, d_v], max_logits [s_q, h_q], lse [s_q, h_q])
"""

import time
import os
import torch

from sgl_kernel.flash_mla import flash_mla_sparse_fwd

# ── GLM-5 model parameters (after matrix absorption) ──
H_Q = 64
H_KV = 1
D_QK = 576       # kv_lora_rank(512) + qk_rope_head_dim(64)
D_V = 512        # kv_lora_rank
TOPK = 2048      # index_topk
SM_SCALE = D_QK ** -0.5

SQ_LIST = [16384, 32768, 65536, 131072]
SKV_LIST = [16384, 32768, 65536, 131072]

NUM_WARMUP = 5
NUM_RUNS = 20


def make_tensors(s_q: int, s_kv: int, topk: int, device: torch.device):
    q = torch.randn(s_q, H_Q, D_QK, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, H_KV, D_QK, dtype=torch.bfloat16, device=device)

    topk_actual = min(topk, s_kv)
    indices = torch.stack([
        torch.randperm(s_kv, device=device)[:topk_actual] for _ in range(s_q * H_KV)
    ]).view(s_q, H_KV, topk_actual).to(torch.int32)

    return q, kv, indices


def bench_one(s_q: int, s_kv: int, topk: int, device: torch.device):
    q, kv, indices = make_tensors(s_q, s_kv, topk, device)

    def run():
        flash_mla_sparse_fwd(q, kv, indices, SM_SCALE, D_V)

    # Warmup (also primes any lazy init before graph capture)
    torch.cuda.synchronize()
    for _ in range(NUM_WARMUP):
        run()
    torch.cuda.synchronize()

    # Capture a 1-iteration CUDA graph (flash_mla_sparse_fwd is graph-capturable),
    # then time NUM_RUNS replays with per-iteration events. This removes per-call
    # launch overhead while keeping per-iteration avg/min/max.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize()

    for _ in range(NUM_WARMUP):
        graph.replay()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_RUNS)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_RUNS)]

    for i in range(NUM_RUNS):
        start_events[i].record()
        graph.replay()
        end_events[i].record()

    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    del graph

    # FLOPs (sparse MLA, follows the theoretical formula with s_k -> topk):
    #   score (Q @ K^T): 2 * h_q * s_q * topk * d_k
    #   output (S @ V):  2 * h_q * s_q * topk * d_v
    #   Total: 2 * h_q * s_q * topk * (d_k + d_v)
    topk_actual = min(topk, s_kv)
    flops = 2.0 * H_Q * s_q * topk_actual * (D_QK + D_V)
    tflops = flops / (avg_ms * 1e-3) / 1e12

    # Memory access volume (bf16, follows the theoretical formula):
    #   q:      h_q * s_q * d_k        (read)
    #   k:      topk * s_q * d_k       (gathered per q token; capped at s_kv * d_k)
    #   v:      topk * s_q * d_v       (gathered per q token; capped at s_kv * d_v)
    #   output: h_q * s_q * d_v        (write)
    # In MLA, K and V share the same kv tensor (V = kv[..., :d_v]), so the kv read
    # volume is bounded by min(topk * s_q, s_kv) * d_k (not d_k + d_v).
    kv_tokens = min(topk_actual * s_q, s_kv)
    mem_bytes = 2 * (H_Q * s_q * D_QK            # q read
                     + kv_tokens * D_QK           # kv gather read (covers both K and V via shared latent)
                     + H_Q * s_q * D_V)           # output write
    tbps = mem_bytes / (avg_ms * 1e-3) / 1e12

    # Compute-memory ratio (FLOPs / byte) — kernel is compute-bound when this >> GPU's ratio
    flops_per_byte = flops / mem_bytes

    return avg_ms, min_ms, max_ms, tflops, tbps, flops_per_byte


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    num_runs = int(os.environ.get("NUM_RUNS", NUM_RUNS))
    num_warmup = int(os.environ.get("NUM_WARMUP", NUM_WARMUP))

    print("=" * 90)
    print("GLM-5 FlashMLA Sparse Prefill Benchmark")
    print("=" * 90)
    print(f"Model:    h_q={H_Q}, h_kv={H_KV}, d_qk={D_QK}, d_v={D_V}, topk={TOPK}")
    print(f"SQ list:  {SQ_LIST}")
    print(f"SKV list: {SKV_LIST}")
    print(f"Bench:    {num_warmup} warmup + {num_runs} runs per case")
    print(f"Logic:    s_q ∈ SQ_LIST × s_kv ∈ SKV_LIST, topk={TOPK} (fixed)")
    print("=" * 90)

    header = f"{'s_q':>8s} {'s_kv':>8s} {'topk':>6s} {'avg(ms)':>10s} {'min(ms)':>10s} {'max(ms)':>10s} {'TFlops':>10s} {'TB/s':>10s} {'F/B':>8s}"
    print(header)
    print("-" * len(header))

    results = []

    for s_kv in SKV_LIST:
        for s_q in SQ_LIST:
            topk = TOPK

            torch.cuda.empty_cache()

            try:
                avg_ms, min_ms, max_ms, tflops, tbps, fpb = bench_one(s_q, s_kv, topk, device)
                print(f"{s_q:>8d} {s_kv:>8d} {topk:>6d} {avg_ms:>10.3f} {min_ms:>10.3f} {max_ms:>10.3f} {tflops:>10.1f} {tbps:>10.3f} {fpb:>8.1f}")
                results.append({
                    "s_q": s_q,
                    "s_kv": s_kv,
                    "topk": topk,
                    "avg_ms": avg_ms,
                    "min_ms": min_ms,
                    "max_ms": max_ms,
                    "tflops": tflops,
                    "tbps": tbps,
                    "flops_per_byte": fpb,
                })
            except Exception as e:
                print(f"{s_q:>8d} {s_kv:>8d} {topk:>6d}   FAILED: {e}")
                results.append({
                    "s_q": s_q,
                    "s_kv": s_kv,
                    "topk": topk,
                    "avg_ms": 0,
                    "min_ms": 0,
                    "max_ms": 0,
                    "tflops": 0,
                    "tbps": 0,
                    "flops_per_byte": 0,
                    "error": str(e),
                })

            time.sleep(0.3)

    print("=" * len(header))

    # CSV output
    csv_path = "glm5_sparse_prefill_perf.csv"
    with open(csv_path, "w") as f:
        f.write("s_q,s_kv,topk,h_q,d_qk,d_v,avg_ms,min_ms,max_ms,tflops,tbps,flops_per_byte\n")
        for r in results:
            f.write(f"{r['s_q']},{r['s_kv']},{r['topk']},{H_Q},{D_QK},{D_V},"
                    f"{r['avg_ms']:.4f},{r['min_ms']:.4f},{r['max_ms']:.4f},{r['tflops']:.2f},{r['tbps']:.4f},{r['flops_per_byte']:.2f}\n")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
