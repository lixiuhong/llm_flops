"""
Benchmark: GLM-5 Attention GEMM/BMM (FP8) using DeepGEMM + CUDA Graph.

GLM-5 Attention config (after matrix absorption):
  hidden_size      = 6144   (d)
  q_lora_rank      = 2048   (d_ql)
  kv_lora_rank     = 512    (d_c)
  qk_nope_head_dim = 192    (d_qn)
  qk_rope_head_dim = 64     (d_r)
  qk_head_dim      = 256    (d_qk = d_qn + d_r)
  v_head_dim       = 256    (d_v)
  num_attention_heads = 64   (h)

Attention GEMM/BMM breakdown:
  1. q_a_proj:       GEMM  [M, 6144] × [6144, 2048]         Q low-rank down projection
  2. q_b_proj:       GEMM  [M, 2048] × [2048, 16384]        Q up projection (h*qk_head_dim = 64*256)
  3. absorbed_W_UK:  BMM   batch=64, [M, 192] × [192, 512]  per-head: q_nope -> kv_lora_rank
  4. kv_a_proj:      GEMM  [M, 6144] × [6144, 576]          KV joint low-rank down projection
  5. absorbed_W_UV:  BMM   batch=64, [M, 512] × [512, 256]  per-head: attn_out -> v_head_dim
  6. o_proj:         GEMM  [M, 16384] × [16384, 6144]       output projection

Timing: CUDA Graph capture + replay.
GEMM uses deep_gemm.fp8_gemm_nt; BMM uses deep_gemm.bmm_fp8 (or torch.bmm with FP8).
"""

import time
import os

import torch
import deep_gemm
from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor

# ── GLM-5 Attention parameters ──
HIDDEN_SIZE = 6144
Q_LORA_RANK = 2048
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = 256       # = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = 256
NUM_HEADS = 64

M_LIST = [1024, 4096, 16384, 65536]

NUM_WARMUP = 5
NUM_RUNS = 20


def per_token_cast_to_fp8(x: torch.Tensor):
    assert x.dim() == 2
    m, k = x.shape
    assert k % 128 == 0
    x_view = x.view(m, k // 128, 128)
    x_amax = x_view.abs().float().amax(dim=-1)
    x_scale = (x_amax / torch.finfo(torch.float8_e4m3fn).max).float().clamp(min=1e-12)
    x_fp8 = (x_view.float() / x_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return x_fp8.view(m, k), x_scale


def per_block_cast_to_fp8(w: torch.Tensor):
    assert w.dim() == 2
    n, k = w.shape
    assert k % 128 == 0
    n_ceil = (n + 127) // 128 * 128
    if n < n_ceil:
        w_padded = torch.zeros(n_ceil, k, dtype=w.dtype, device=w.device)
        w_padded[:n] = w
    else:
        w_padded = w
    w_view = w_padded.view(n_ceil // 128, 128, k // 128, 128)
    w_amax = w_view.abs().float().amax(dim=(1, 3))
    w_scale = (w_amax / torch.finfo(torch.float8_e4m3fn).max).float().clamp(min=1e-12)
    w_fp8 = (w_view.float() / w_scale[:, None, :, None]).to(torch.float8_e4m3fn)
    return w_fp8.view(n_ceil, k)[:n].contiguous(), w_scale


def cast_to_fp8_per_tensor(x: torch.Tensor):
    amax = x.abs().float().amax()
    scale = (amax / torch.finfo(torch.float8_e4m3fn).max).float().clamp(min=1e-12)
    x_fp8 = (x.float() / scale).to(torch.float8_e4m3fn)
    return x_fp8, scale.view(1).to(x.device)


# ── GEMM benchmark (DeepGEMM FP8) ──

def bench_gemm(M: int, K: int, N: int, device: torch.device):
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    x_fp8, x_scale = per_token_cast_to_fp8(x_bf16)
    x_scale = get_mn_major_tma_aligned_tensor(x_scale)
    del x_bf16

    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    w_fp8, w_scale = per_block_cast_to_fp8(w_bf16)
    del w_bf16

    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)

    def run():
        deep_gemm.fp8_gemm_nt((x_fp8, x_scale), (w_fp8, w_scale), out)

    torch.cuda.synchronize()
    for _ in range(NUM_WARMUP):
        run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(NUM_RUNS):
            run()
    torch.cuda.synchronize()

    for _ in range(NUM_WARMUP):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / NUM_RUNS
    flops = 2.0 * M * N * K
    tflops = flops / (avg_ms * 1e-3) / 1e12
    mem_bytes = M * K * 1 + N * K * 1 + M * N * 2
    tbps = mem_bytes / (avg_ms * 1e-3) / 1e12
    fpb = flops / mem_bytes if mem_bytes > 0 else 0

    del graph, x_fp8, x_scale, w_fp8, w_scale, out
    return avg_ms, tflops, tbps, fpb


# ── BMM benchmark (cuBLAS FP8 via torch._scaled_mm, loop over heads) ──

def bench_bmm(batch: int, M: int, K: int, N: int, device: torch.device):
    """Benchmark BMM by running single-head FP8 GEMM in a loop inside CUDA Graph.
    batch=num_heads, each head: [M, K] × [K, N].
    """
    # Per-head weight: batch separate [N, K] matrices
    w_list = []
    w_scale_list = []
    for _ in range(batch):
        w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        w_fp8, w_s = cast_to_fp8_per_tensor(w_bf16)
        w_list.append(w_fp8)
        w_scale_list.append(w_s)
        del w_bf16

    # Input: [M, K] shared across heads (same c_q or attn_out for all heads)
    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    x_fp8, x_scale = cast_to_fp8_per_tensor(x_bf16)
    del x_bf16

    def run():
        for h in range(batch):
            torch._scaled_mm(
                x_fp8, w_list[h].t(),
                scale_a=x_scale, scale_b=w_scale_list[h],
                out_dtype=torch.bfloat16,
            )

    torch.cuda.synchronize()
    for _ in range(NUM_WARMUP):
        run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(NUM_RUNS):
            run()
    torch.cuda.synchronize()

    for _ in range(NUM_WARMUP):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()

    avg_ms = start.elapsed_time(end) / NUM_RUNS

    # Total FLOPs/mem across all heads
    flops = 2.0 * batch * M * N * K
    tflops = flops / (avg_ms * 1e-3) / 1e12
    # Read: batch * (M*K + N*K) * 1 byte (FP8), Write: batch * M*N * 2 bytes (bf16)
    mem_bytes = batch * (M * K * 1 + N * K * 1 + M * N * 2)
    tbps = mem_bytes / (avg_ms * 1e-3) / 1e12
    fpb = flops / mem_bytes if mem_bytes > 0 else 0

    del graph, x_fp8, x_scale, w_list, w_scale_list
    return avg_ms, tflops, tbps, fpb


# ── Attention GEMM/BMM specs ──

ATTN_SPECS = [
    # (name, type, params_fn)
    # params_fn(M) -> (gemm_M, K, N) for GEMM, or (batch, gemm_M, K, N) for BMM
    ("q_a_proj",       "gemm", lambda M: (M, HIDDEN_SIZE, Q_LORA_RANK)),
    ("q_b_proj",       "gemm", lambda M: (M, Q_LORA_RANK, NUM_HEADS * QK_HEAD_DIM)),
    ("absorbed_W_UK",  "bmm",  lambda M: (NUM_HEADS, M, QK_NOPE_HEAD_DIM, KV_LORA_RANK)),
    ("kv_a_proj",      "gemm", lambda M: (M, HIDDEN_SIZE, KV_LORA_RANK + QK_ROPE_HEAD_DIM)),
    ("absorbed_W_UV",  "bmm",  lambda M: (NUM_HEADS, M, KV_LORA_RANK, V_HEAD_DIM)),
    ("o_proj",         "gemm", lambda M: (M, NUM_HEADS * V_HEAD_DIM, HIDDEN_SIZE)),
]


def main():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("=" * 120)
    print("GLM-5 Attention GEMM/BMM (FP8) Benchmark  [CUDA Graph timing]")
    print("=" * 120)
    print(f"Params:  hidden={HIDDEN_SIZE}, q_lora_rank={Q_LORA_RANK}, kv_lora_rank={KV_LORA_RANK}")
    print(f"         qk_nope={QK_NOPE_HEAD_DIM}, qk_rope={QK_ROPE_HEAD_DIM}, v_head={V_HEAD_DIM}, heads={NUM_HEADS}")
    print(f"M list:  {M_LIST}")
    print(f"Bench:   {NUM_WARMUP} warmup + {NUM_RUNS} graph replays")
    print("=" * 120)

    all_results = []

    for M in M_LIST:
        print(f"\n{'='*110}")
        print(f"  M = {M}")
        print(f"{'='*110}")
        print(f"  {'name':<20s} {'type':<6s} {'shape':>40s} {'avg(ms)':>12s} {'TFlops':>12s} {'TB/s':>12s} {'FLOP/B':>12s}")
        print(f"  {'-'*108}")

        for spec_name, spec_type, params_fn in ATTN_SPECS:
            torch.cuda.empty_cache()

            if spec_type == "gemm":
                gM, K, N = params_fn(M)
                shape_str = f"[{gM}, {K}] × [{K}, {N}]"
                try:
                    avg_ms, tflops, tbps, fpb = bench_gemm(gM, K, N, device)
                except Exception as e:
                    print(f"  {spec_name:<20s} {'GEMM':<6s} {shape_str:>40s}   FAILED: {e}")
                    all_results.append({"name": spec_name, "type": "GEMM", "M": M,
                                         "avg_ms": 0, "tflops": 0, "tbps": 0, "fpb": 0, "error": str(e)})
                    continue

            elif spec_type == "bmm":
                batch, gM, K, N = params_fn(M)
                shape_str = f"batch={batch}, [{gM}, {K}] × [{K}, {N}]"
                try:
                    avg_ms, tflops, tbps, fpb = bench_bmm(batch, gM, K, N, device)
                except Exception as e:
                    print(f"  {spec_name:<20s} {'BMM':<6s} {shape_str:>40s}   FAILED: {e}")
                    all_results.append({"name": spec_name, "type": "BMM", "M": M,
                                         "avg_ms": 0, "tflops": 0, "tbps": 0, "fpb": 0, "error": str(e)})
                    continue

            type_label = "GEMM" if spec_type == "gemm" else "BMM"
            print(f"  {spec_name:<20s} {type_label:<6s} {shape_str:>40s} {avg_ms:>12.4f} {tflops:>12.1f} {tbps:>12.3f} {fpb:>12.1f}")
            all_results.append({
                "name": spec_name, "type": type_label, "M": M,
                "shape": shape_str,
                "avg_ms": avg_ms, "tflops": tflops, "tbps": tbps, "fpb": fpb,
            })

    # ── Summary table: each GEMM/BMM as a row, M values as columns ──
    print(f"\n{'='*120}")
    print("  Summary: avg_ms per operator across M values")
    print(f"{'='*120}")
    col_w = 14
    header = f"  {'name':<22s} {'type':<6s}"
    for M in M_LIST:
        header += f"{'M=' + str(M):>{col_w}s}"
    print(header)
    print(f"  {'-'*(22 + 6 + col_w * len(M_LIST))}")

    for spec_name, spec_type, _ in ATTN_SPECS:
        type_label = "GEMM" if spec_type == "gemm" else "BMM"
        row = f"  {spec_name:<22s} {type_label:<6s}"
        for M in M_LIST:
            match = [r for r in all_results if r["name"] == spec_name and r["M"] == M]
            if match and match[0]["avg_ms"] > 0:
                row += f"{match[0]['avg_ms']:>{col_w}.4f}"
            else:
                row += f"{'FAIL':>{col_w}s}"
        print(row)

    # CSV
    csv_path = "glm5_attention_gemm_perf.csv"
    with open(csv_path, "w") as f:
        f.write("name,type,M,shape,avg_ms,tflops,tbps,flops_per_byte\n")
        for r in all_results:
            shape = r.get("shape", "")
            f.write(f"{r['name']},{r['type']},{r['M']},\"{shape}\","
                    f"{r['avg_ms']:.4f},{r['tflops']:.2f},{r['tbps']:.4f},{r['fpb']:.2f}\n")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()

