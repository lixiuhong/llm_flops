"""
Benchmark: FlashMLA dense prefill (traditional MLA) with GLM-5 parameters.

This is the dense (non-sparse) counterpart of dsa_flashmla.py: the sparsity
(topk / indices) is removed, so every query token attends to ALL s_kv keys.

GLM-5 config (https://www.modelscope.cn/models/ZhipuAI/GLM-5/file/view/master/config.json):
  num_attention_heads=64, kv_lora_rank=512, qk_rope_head_dim=64

After matrix absorption:
  h_q=64, h_kv=1, d_qk=576, d_v=512

Scenario (identical to dsa_flashmla.py except topk is dropped):
  Dense prefill; s_q and s_kv are configured independently and swept as a
  cartesian product (no topk, no hit-rate modeling).
  - s_q  ∈ SQ_LIST   (query tokens, independent)
  - s_kv ∈ SKV_LIST  (KV cache length, independent)
  Dense attention: each query attends to all s_kv keys.

API:
  flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v,
                         tile_scheduler_metadata, num_splits, causal=False)
    q:           [1, s_q, h_q, d_qk], bfloat16
    k_cache:     [num_blocks, page_block_size, h_kv, d_qk], bfloat16
    block_table: [1, num_blocks_per_seq], int32
    cache_seqlens: [1], int32
    Returns: (output [1, s_q, h_q, d_v], softmax_lse [1, h_q, s_q])
"""

import time
import os
import torch

from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata

# ── GLM-5 model parameters (after matrix absorption) ──
H_Q = 64
H_KV = 1
D_QK = 576       # kv_lora_rank(512) + qk_rope_head_dim(64)
D_V = 512        # kv_lora_rank
BLOCK_SIZE = 64  # page block size (matches sglang paged KV cache)
SM_SCALE = D_QK ** -0.5

SQ_LIST = [16384, 32768, 65536, 131072]
SKV_LIST = [16384, 32768, 65536, 131072]

NUM_WARMUP = 5
NUM_RUNS = 20


def make_tensors(s_q: int, s_kv: int, device: torch.device):
    num_blocks_per_seq = (s_kv + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = num_blocks_per_seq  # single request (batch=1)

    # q: [1, s_q, h_q, d_qk]
    q = torch.randn(1, s_q, H_Q, D_QK, dtype=torch.bfloat16, device=device)
    # k_cache: [total_blocks, page_block_size, h_kv, d_qk]
    k_cache = torch.randn(total_blocks, BLOCK_SIZE, H_KV, D_QK, dtype=torch.bfloat16, device=device)
    # block_table: [1, num_blocks_per_seq]
    block_table = torch.arange(total_blocks, dtype=torch.int32, device=device).view(1, num_blocks_per_seq)
    # cache_seqlens: [1]
    cache_seqlens = torch.tensor([s_kv], dtype=torch.int32, device=device)

    return q, k_cache, block_table, cache_seqlens


def bench_one(s_q: int, s_kv: int, device: torch.device):
    q, k_cache, block_table, cache_seqlens = make_tensors(s_q, s_kv, device)
    # sched_meta must be created per (s_q, s_kv) shape; the config is cached on first call.
    tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * H_Q // H_KV, H_KV)
    torch.cuda.synchronize()

    def run():
        flash_mla_with_kvcache(
            q, k_cache, block_table, cache_seqlens, D_V,
            tile_scheduler_metadata, num_splits, causal=False,
            softmax_scale=SM_SCALE,
        )

    # Warmup (initializes sched_meta on first call, before graph capture)
    torch.cuda.synchronize()
    for _ in range(NUM_WARMUP):
        run()
    torch.cuda.synchronize()

    # Capture a 1-iteration CUDA graph (flash_mla_with_kvcache is graph-capturable
    # once sched_meta is initialized), then time NUM_RUNS replays with per-iteration
    # events. Removes per-call launch overhead while keeping per-iteration avg/min/max.
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

    # FLOPs (dense MLA, follows the theoretical formula with s_k -> s_kv):
    #   score (Q @ K^T): 2 * h_q * s_q * s_kv * d_k
    #   output (S @ V):  2 * h_q * s_q * s_kv * d_v
    #   Total: 2 * h_q * s_q * s_kv * (d_k + d_v)
    flops = 2.0 * H_Q * s_q * s_kv * (D_QK + D_V)
    tflops = flops / (avg_ms * 1e-3) / 1e12

    # Memory access volume (bf16, follows the theoretical formula):
    #   q:      h_q * s_q * d_k        (read)
    #   kv:     s_kv * d_k             (every query attends to ALL s_kv keys, but the
    #                                  same KV cache is loaded once and reused across
    #                                  query tiles -> bounded by s_kv * d_k, not s_q * s_kv)
    #   output: h_q * s_q * d_v        (write)
    # In MLA, K and V share the same k_cache tensor (V = k_cache[..., :d_v]), so the kv
    # read volume is bounded by s_kv * d_k (not d_k + d_v).
    kv_tokens = min(s_kv * s_q, s_kv)   # = s_kv: dense dedup bound (all queries share the cache)
    mem_bytes = 2 * (H_Q * s_q * D_QK    # q read
                    + kv_tokens * D_QK  # kv read (covers both K and V via shared latent)
                    + H_Q * s_q * D_V)  # output write
    tbps = mem_bytes / (avg_ms * 1e-3) / 1e12

    # Compute-memory ratio (FLOPs / byte) — kernel is compute-bound when this >> GPU's ratio
    flops_per_byte = flops / mem_bytes

    del graph, q, k_cache, block_table, cache_seqlens, tile_scheduler_metadata, num_splits
    return avg_ms, min_ms, max_ms, tflops, tbps, flops_per_byte


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    num_runs = int(os.environ.get("NUM_RUNS", NUM_RUNS))
    num_warmup = int(os.environ.get("NUM_WARMUP", NUM_WARMUP))

    print("=" * 90)
    print("GLM-5 FlashMLA Dense Prefill Benchmark (traditional MLA)")
    print("=" * 90)
    print(f"Model:    h_q={H_Q}, h_kv={H_KV}, d_qk={D_QK}, d_v={D_V}")
    print(f"SQ list:  {SQ_LIST}")
    print(f"SKV list: {SKV_LIST}")
    print(f"Bench:    {num_warmup} warmup + {num_runs} runs per case")
    print(f"Logic:    s_q ∈ SQ_LIST × s_kv ∈ SKV_LIST, dense (no topk)")
    print("=" * 90)

    header = f"{'s_q':>8s} {'s_kv':>8s} {'avg(ms)':>10s} {'min(ms)':>10s} {'max(ms)':>10s} {'TFlops':>10s} {'TB/s':>10s} {'F/B':>8s}"
    print(header)
    print("-" * len(header))

    results = []

    for s_kv in SKV_LIST:
        for s_q in SQ_LIST:
            torch.cuda.empty_cache()

            try:
                avg_ms, min_ms, max_ms, tflops, tbps, fpb = bench_one(s_q, s_kv, device)
                print(f"{s_q:>8d} {s_kv:>8d} {avg_ms:>10.3f} {min_ms:>10.3f} {max_ms:>10.3f} {tflops:>10.1f} {tbps:>10.3f} {fpb:>8.1f}")
                results.append({
                    "s_q": s_q,
                    "s_kv": s_kv,
                    "avg_ms": avg_ms,
                    "min_ms": min_ms,
                    "max_ms": max_ms,
                    "tflops": tflops,
                    "tbps": tbps,
                    "flops_per_byte": fpb,
                })
            except Exception as e:
                print(f"{s_q:>8d} {s_kv:>8d}   FAILED: {e}")
                results.append({
                    "s_q": s_q,
                    "s_kv": s_kv,
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
    csv_path = "glm5_dense_prefill_perf.csv"
    with open(csv_path, "w") as f:
        f.write("s_q,s_kv,h_q,d_qk,d_v,avg_ms,min_ms,max_ms,tflops,tbps,flops_per_byte\n")
        for r in results:
            f.write(f"{r['s_q']},{r['s_kv']},{H_Q},{D_QK},{D_V},"
                    f"{r['avg_ms']:.4f},{r['min_ms']:.4f},{r['max_ms']:.4f},{r['tflops']:.2f},{r['tbps']:.4f},{r['flops_per_byte']:.2f}\n")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
