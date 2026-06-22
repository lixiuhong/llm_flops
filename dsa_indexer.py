"""
Benchmark: DSA Indexer GEMM (FP8) with GLM-5 parameters using cuBLAS FP8 (torch._scaled_mm).

GLM-5 DSA Indexer config:
  hidden_size      = 6144
  q_lora_rank      = 2048
  index_n_heads    = 32
  index_head_dim   = 128

DSA Indexer GEMM breakdown (all FP8 via cuBLAS):
  7a. index_k_proj:        [S, 6144] × [6144, 128]      K downsample (varies with S)
  7b. index_q_upproj:      [M, 2048] × [2048, 4096]     Q upsample   (varies with M)
  7c. index_weights_proj:  [M, 6144] × [6144, 32]       score weights (varies with M)
  7d. index_score:         [32*M, 128] × [128, S]        Q@K^T score  (varies with M and S)

Timing: CUDA Graph capture + replay.
"""

import time
import os

import torch

# ── GLM-5 DSA Indexer parameters ──
HIDDEN_SIZE = 6144
Q_LORA_RANK = 2048
INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128

M_LIST = [16, 256, 512, 1024]
S_LIST = [65536, 65536*2, 65536*4]

NUM_WARMUP = 5
NUM_RUNS = 20


def cast_to_fp8_per_tensor(x: torch.Tensor):
    """Per-tensor FP8 quantization for cuBLAS _scaled_mm."""
    amax = x.abs().float().amax()
    scale = (amax / torch.finfo(torch.float8_e4m3fn).max).float().clamp(min=1e-12)
    x_fp8 = (x.float() / scale).to(torch.float8_e4m3fn)
    # _scaled_mm expects scale as a scalar tensor on the same device
    return x_fp8, scale.view(1).to(x.device)


def make_fp8_gemm_tensors(M: int, K: int, N: int, device: torch.device):
    """Create FP8 tensors for cuBLAS: A [M, K] @ B^T [N, K] -> [M, N]."""
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    a_fp8, a_scale = cast_to_fp8_per_tensor(a_bf16)
    b_fp8, b_scale = cast_to_fp8_per_tensor(b_bf16)
    del a_bf16, b_bf16
    return a_fp8, a_scale, b_fp8, b_scale


def bench_cublas_fp8(M: int, K: int, N: int, device: torch.device):
    """Benchmark cuBLAS FP8 GEMM using CUDA Graph."""
    a_fp8, a_scale, b_fp8, b_scale = make_fp8_gemm_tensors(M, K, N, device)

    def run():
        torch._scaled_mm(
            a_fp8, b_fp8.t(),
            scale_a=a_scale, scale_b=b_scale,
            out_dtype=torch.bfloat16,
        )

    # Warmup
    torch.cuda.synchronize()
    for _ in range(NUM_WARMUP):
        run()
    torch.cuda.synchronize()

    # Capture CUDA Graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(NUM_RUNS):
            run()
    torch.cuda.synchronize()

    # Warmup graph
    for _ in range(NUM_WARMUP):
        graph.replay()
    torch.cuda.synchronize()

    # Time graph replay
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / NUM_RUNS

    flops = 2.0 * M * N * K
    tflops = flops / (avg_ms * 1e-3) / 1e12
    # FP8 = 1 byte
    mem_bytes = M * K * 1 + N * K * 1 + M * N * 2
    tbps = mem_bytes / (avg_ms * 1e-3) / 1e12
    fpb = flops / mem_bytes if mem_bytes > 0 else 0

    del graph, a_fp8, a_scale, b_fp8, b_scale
    return avg_ms, tflops, tbps, fpb


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    VRAM_LIMIT_BYTES = 60 * (1024 ** 3)

    print("=" * 120)
    print("GLM-5 DSA Indexer GEMM (cuBLAS FP8) Benchmark  [CUDA Graph timing]")
    print("=" * 120)
    print(f"Params:  hidden={HIDDEN_SIZE}, q_lora_rank={Q_LORA_RANK}, "
          f"index_n_heads={INDEX_N_HEADS}, index_head_dim={INDEX_HEAD_DIM}")
    print(f"M list:  {M_LIST}")
    print(f"S list:  {S_LIST}")
    print(f"Bench:   {NUM_WARMUP} warmup + {NUM_RUNS} graph replays")
    print("=" * 120)

    # GEMM specs: (name, gemm_M_fn, K, gemm_N_fn)
    gemm_specs = [
        ("index_k_proj",       lambda M, S: S,          HIDDEN_SIZE, lambda M, S: INDEX_HEAD_DIM),
        ("index_q_upproj",     lambda M, S: M,          Q_LORA_RANK, lambda M, S: INDEX_N_HEADS * INDEX_HEAD_DIM),
        ("index_weights_proj", lambda M, S: M,          HIDDEN_SIZE, lambda M, S: INDEX_N_HEADS),
        ("index_score",        lambda M, S: INDEX_N_HEADS * M, INDEX_HEAD_DIM, lambda M, S: S),
    ]

    all_results = []

    for M in M_LIST:
        for S in S_LIST:
            print(f"\n{'='*100}")
            print(f"  M={M}, S={S}")
            print(f"{'='*100}")
            print(f"  {'name':<22s} {'shape':>30s} {'avg(ms)':>12s} {'TFlops':>12s} {'TB/s':>12s} {'FLOP/B':>12s}")
            print(f"  {'-'*98}")

            for spec_name, m_fn, K, n_fn in gemm_specs:
                gemm_M = m_fn(M, S)
                gemm_N = n_fn(M, S)

                # For index_score: check VRAM and fall back to per-head if needed
                n_heads_run = 1
                label = ""
                if spec_name == "index_score":
                    out_bytes = gemm_M * gemm_N * 2
                    if out_bytes > VRAM_LIMIT_BYTES:
                        gemm_M = M
                        n_heads_run = INDEX_N_HEADS
                        label = f" (per-head x{INDEX_N_HEADS})"

                torch.cuda.empty_cache()
                shape_str = f"[{gemm_M}, {K}]×[{K}, {gemm_N}]"

                try:
                    avg_ms_one, tflops_one, tbps_one, fpb = bench_cublas_fp8(gemm_M, K, gemm_N, device)

                    if n_heads_run > 1:
                        gemm_M_full = INDEX_N_HEADS * M
                        avg_ms = avg_ms_one * n_heads_run
                        flops_full = 2.0 * gemm_M_full * gemm_N * K
                        mem_full = gemm_M_full * K * 1 + gemm_N * K * 1 + gemm_M_full * gemm_N * 2
                        tflops = flops_full / (avg_ms * 1e-3) / 1e12
                        tbps = mem_full / (avg_ms * 1e-3) / 1e12
                        shape_str = f"[{gemm_M_full}, {K}]×[{K}, {gemm_N}]"
                    else:
                        avg_ms = avg_ms_one
                        tflops = tflops_one
                        tbps = tbps_one

                    print(f"  {spec_name + label:<22s} {shape_str:>30s} {avg_ms:>12.4f} {tflops:>12.1f} {tbps:>12.3f} {fpb:>12.1f}")
                    all_results.append({
                        "name": spec_name, "M": M, "S": S,
                        "gemm_M": m_fn(M, S), "K": K, "gemm_N": n_fn(M, S),
                        "avg_ms": avg_ms, "tflops": tflops, "tbps": tbps, "fpb": fpb,
                    })
                except Exception as e:
                    print(f"  {spec_name:<22s} {shape_str:>30s}   FAILED: {e}")
                    all_results.append({
                        "name": spec_name, "M": M, "S": S,
                        "gemm_M": m_fn(M, S), "K": K, "gemm_N": n_fn(M, S),
                        "avg_ms": 0, "tflops": 0, "tbps": 0, "fpb": 0, "error": str(e),
                    })

    # CSV
    csv_path = "glm5_dsa_indexer_perf.csv"
    with open(csv_path, "w") as f:
        f.write("name,M,S,gemm_M,K,gemm_N,avg_ms,tflops,tbps,flops_per_byte\n")
        for r in all_results:
            f.write(f"{r['name']},{r['M']},{r['S']},{r['gemm_M']},{r['K']},{r['gemm_N']},"
                    f"{r['avg_ms']:.4f},{r['tflops']:.2f},{r['tbps']:.4f},{r['fpb']:.2f}\n")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()

