#!/usr/bin/env julia
# Minimal repro: Zygote DimensionMismatch when using cross-attention with two
# RoPE slices of different lengths (Q length ≠ K length). Test 1: two separate
# RoPEs → backward OK. Test 2: one RoPE sliced two ways → DimensionMismatch.
#
# Run: julia --project=. examples/zygote_rope_cross_attention_minimal.jl
#
# No project deps except Onion, Flux, Zygote (same as main project).
# No wrapper fix — raw RoPE slices only.

using Flux
using Zygote
using Random
using Onion: TransformerBlock, RoPE

Random.seed!(42)

# Small sizes
dim = 32
n_heads = 4
head_dim = dim ÷ n_heads
Lq = 3   # decoder (Q) length — rope is applied to Q, so rope must have Lq positions
Lk = 5   # encoder (K/V) length — krope is applied to K, so krope must have Lk positions
@assert Lq != Lk "Need different lengths to trigger the bug"

block = TransformerBlock(dim, n_heads, n_heads; norm_eps=1f-5)
xq = randn(Float32, dim, Lq, 1)   # Q: (dim, Lq, batch)
xk = randn(Float32, dim, Lk, 1)   # K,V: (dim, Lk, batch)
xv = randn(Float32, dim, Lk, 1)

# Onion: apply_rope_qk(rope, krope, q, k) → rope(q), krope(k). So rope length = Lq, krope length = Lk.
function forward(block, xq, xk, xv, rope_q, rope_k)
    block(xq, xk, xv; rope=rope_q, krope=rope_k)
end 

function materialize_rope(r::RoPE)
    RoPE(copy(r.cos), copy(r.sin))
end

# ---------------------------------------------------------------------------
# Test -: Two independent RoPEs (no shared object) — backward fixed.
# ---------------------------------------------------------------------------
rope_for_q = RoPE(head_dim, 64)   # used only for Q
rope_for_k = RoPE(head_dim, 64)   # used only for K

# Materialize slices before passing to gradient
rope_q = materialize_rope(rope_for_q[1:Lq])
rope_k = materialize_rope(rope_for_k[1:Lk])

out1 = forward(block, xq, xk, xv, rope_q, rope_k)
@assert size(out1) == (dim, Lq, 1)
println("Forward OK (two RoPEs): out size = ", size(out1))

println("Backward with two separate RoPEs (gradient w.r.t. both)...")
try
    # No slicing inside gradient
    g = Zygote.gradient(() -> sum(forward(block, xq, xk, xv, rope_q, rope_k)))
    println("Backward OK: two RoPEs work (gradients are separate).")
catch e
    println("ERROR (unexpected with two RoPEs): ", e)
end

# ---------------------------------------------------------------------------
# Test 1: Two independent RoPEs (no shared object) — backward should succeed.
# ---------------------------------------------------------------------------
rope_for_q = RoPE(head_dim, 64)   # used only for Q
rope_for_k = RoPE(head_dim, 64)   # used only for K
rope_q = materialize_rope(rope_for_q[1:Lq])
rope_k = materialize_rope(rope_for_k[1:Lk])

out1 = forward(block, xq, xk, xv, rope_q, rope_k)
@assert size(out1) == (dim, Lq, 1)
println("Forward OK (two RoPEs): out size = ", size(out1))

println("Backward with two separate RoPEs (gradient w.r.t. both)...")
try
    g = Zygote.gradient((rq, rk) -> sum(forward(block, xq, xk, xv, rq[1:Lq], rk[1:Lk])),
                        rope_for_q, rope_for_k)
    println("Backward OK: two RoPEs work (gradients are separate).")
catch e
    println("ERROR (unexpected with two RoPEs): ", e)
end

# ---------------------------------------------------------------------------
# Test 2: One RoPE sliced two ways (shared) — backward triggers DimensionMismatch.
# ---------------------------------------------------------------------------
rope_full = RoPE(head_dim, 64)
rope_q = rope_full[1:Lq]   # length Lq for Q
rope_k = rope_full[1:Lk]   # length Lk for K (not swapped)
@assert size(rope_q.cos, 2) == Lq && size(rope_k.cos, 2) == Lk "Q/K lengths: rope_q→Lq=$Lq, rope_k→Lk=$Lk"

out2 = forward(block, xq, xk, xv, rope_q, rope_k)
@assert size(out2) == (dim, Lq, 1)
println("\nForward OK (one RoPE, two slices): out size = ", size(out2))

# Zygote accumulates gradients for rope_q and rope_k into the same RoPE → DimensionMismatch
# (broadcast of tangents with axes OneTo(5) and OneTo(3)).
println("Backward with one RoPE sliced two ways (rope_q length $Lq, rope_k length $Lk)...")
try
    g = Zygote.gradient(rope_full) do r
        sum(forward(block, xq, xk, xv, r[1:Lq], r[1:Lk]))
    end
    println("Backward OK (unexpected).")
catch e
    println("ERROR (expected): ", e)
end
