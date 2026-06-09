from __future__ import annotations


PREFERRED_RUNTIME_OPTIMIZATIONS = [
    "Highest priority: use F.scaled_dot_product_attention instead of manual attention math when supported. Prefill may use causal mode when query/key lengths align; one-token decode over the full valid KV cache must use non-causal SDPA or a correct offset-aware mask. Keep a correct fallback.",
    "Apply final norm and lm_head only to hidden[:, -1, :] because prefill/decode return one logits row per request; transformer layers still process all required tokens.",
    "Fuse Q/K/V weights once at Engine init, use one F.linear per layer, split with config-derived sizes, and avoid duplicate unfused GPU weights.",
    "Fuse MLP gate/up weights once at Engine init, then split and compute SiLU(gate) * up before down projection.",
    "Bundle each layer's weights into compact tuple/object references to avoid hot-loop string construction and dictionary lookups.",
    "Precompute or amortized-grow RoPE cos/sin tables, then index them by absolute prefill/decode positions.",
    "Also retain useful options such as equal-length grouping, cache preallocation, inference_mode, dtype-aware weights, fewer Python loops, and benchmarked compile/Triton paths.",
    "Lower priority: try shared packed KV only if the above are already present and profiling still shows KV stack/concat or movement as a bottleneck; preserve the correct generic KV path, heterogeneous-state fallback, and request order.",
]


PREFERRED_RUNTIME_STRATEGY = "\n".join(
    f"{index}. {instruction}"
    for index, instruction in enumerate(PREFERRED_RUNTIME_OPTIMIZATIONS, start=1)
)
