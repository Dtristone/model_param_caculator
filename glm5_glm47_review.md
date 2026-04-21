
# Review of the GLM-5 / GLM-4.7 FLOPs + memory calculator

## Scope

I reviewed the uploaded README and the calculator modules:

- `base.py`
- `linear.py`
- `attention.py`
- `mla.py`
- `dsa.py`
- `ffn.py`
- `moe.py`
- `kv_cache.py`
- `model.py`

I **could not** review `plan.md` or `config_parser.py` because they were not included.

I also compared the calculator against the public GLM-family configs and the Hugging Face reference implementations for:

- **GLM-5** (`zai-org/GLM-5`, model type `glm_moe_dsa`)
- **GLM-4.7** (`zai-org/GLM-4.7`, model type `glm4_moe`)
- **GLM-4.7-Flash** (`zai-org/GLM-4.7-Flash`, model type `glm4_moe_lite`) because your MLA path matches this family much more directly than base GLM-4.7.

---

## Executive summary

### Verdict

The calculator is **not yet mathematically correct or complete** for GLM-5 and GLM-4.7 end-to-end.

What is already good:

- Standard linear-layer math is mostly fine.
- Standard attention FLOPs are fine at the dense-kernel level.
- The project correctly recognizes that MLA / DSA / MoE need dedicated formulas.
- The GLM-5 parameterization in the README is close to the right *architecture family*.

What is not correct enough yet:

1. **Dense-vs-MoE layer pattern is missing.**
   - GLM-5: first **3** MLP layers are dense, the rest are sparse MoE.
   - GLM-4.7: first **3** MLP layers are dense, the rest are sparse MoE.
   - GLM-4.7-Flash: first **1** MLP layer is dense, the rest are sparse MoE.

2. **Shared experts are missing from the MoE math.**
   - Both GLM-5 and GLM-4.7 have `n_shared_experts = 1`.

3. **GLM-4.7 base attention width is modeled incorrectly.**
   - In real GLM-4.7, `q_proj` outputs `num_attention_heads * head_dim = 96 * 128 = 12288`, while `hidden_size = 5120`.
   - Therefore `o_proj` is **12288 -> 5120**, not **5120 -> 5120**.

4. **GLM-4.7 `q_norm` / `k_norm` is missing.**
   - `use_qk_norm = true` in the released GLM-4.7 config.

5. **GLM-5 / GLM-4.7-Flash MLA layernorms are missing.**
   - Real DeepSeek/GLM MLA has:
     - `q_a_layernorm`
     - `kv_a_layernorm`

6. **The DSA indexer math is wrong.**
   - Your code models a “top-k softmax”.
   - The reference GLM-5 indexer does **ReLU + weighted sum + top-k**, not softmax.

7. **Attention uses one `seq_len` for both query length and KV length.**
   - This is only correct for full-prefill with no cache.
   - It is wrong for decode / prefix-cache / long-context generation.

8. **MLA / DSA memory depends on runtime mode, not config alone.**
   - Your current code assumes a compressed-cache / absorbed runtime.
   - The current HF reference implementations for DeepSeek-V3 / GLM-4.7-Flash / GLM-5 use **expanded K/V cache** in the main attention path.

9. **`num_nextn_predict_layers` (MTP / next-n prediction) is not modeled.**
   - The configs contain it.
   - Your calculator ignores it.
   - This likely matters when you compare against public “headline parameter count” numbers.

### The two most important consequences

- **GLM-5 total params are materially overcounted** if all 78 layers are treated as MoE.
- **GLM-4.7 can look “correct” by total params only by accident** because multiple mistakes partially cancel.

---

## Cross-check numbers from the real configs

These numbers are useful sanity checks for your future code changes.

### 1) GLM-5

Config highlights:

- `hidden_size = 6144`
- `num_hidden_layers = 78`
- `num_attention_heads = 64`
- `kv_lora_rank = 512`
- `q_lora_rank = 2048`
- `qk_nope_head_dim = 192`
- `qk_rope_head_dim = 64`
- `v_head_dim = 256`
- `n_routed_experts = 256`
- `n_shared_experts = 1`
- `num_experts_per_tok = 8`
- `first_k_dense_replace = 3`

With the **correct dense/sparse layer pattern** and **shared experts**, the parameter count works out to:

- **GLM-5 total params (no MTP modeled)** ≈ **743.91B**
- **decoder active params per token** ≈ **39.88B**

That is very close to the published “744B total / 40B active” headline and is the right direction for your calculator.

By contrast, the current calculator logic (all layers MoE, no shared expert, no MLA layernorms) would land near:

- **GLM-5 current-code-like total** ≈ **769.40B**

So the current design is off by roughly:

- **+25.5B params**

The biggest reason is simple:

- sparse-MoE MLP params per GLM-5 sparse layer ≈ **9.703B**
- dense MLP params per GLM-5 dense layer ≈ **0.226B**

So mis-modeling the first 3 dense layers as MoE overcounts by about:

- **3 × (9.703B - 0.226B) = 28.43B**

Shared-expert omission pulls that back down slightly, but not nearly enough.

### 2) GLM-4.7 (base, not Flash)

Config highlights:

- `hidden_size = 5120`
- `num_hidden_layers = 92`
- `num_attention_heads = 96`
- `num_key_value_heads = 8`
- `head_dim = 128`
- `n_routed_experts = 160`
- `n_shared_experts = 1`
- `num_experts_per_tok = 8`
- `first_k_dense_replace = 3`
- `attention_bias = true`
- `use_qk_norm = true`

With the **correct transformer-layer math** and **no MTP modeled**, I get:

- **GLM-4.7 total params (no MTP modeled)** ≈ **352.80B**
- **decoder active params per token** ≈ **32.08B**

The **32.08B active** figure is exactly the kind of number you want: it lines up with the public “32B active” description.

However, your **current code-like logic** can accidentally land near:

- **GLM-4.7 current-code-like total** ≈ **358.08B**

which looks close to the public 358B headline even though the layer math is wrong.

That accidental match happens because these errors partially cancel:

- overcount: treating the first 3 dense layers as MoE  
  → about **+10.83B**
- undercount: missing shared experts  
  → about **-2.10B**
- undercount: wrong `o_proj` size (`5120 -> 5120` instead of `12288 -> 5120`)  
  → about **-3.38B**
- undercount: missing `q_norm` / `k_norm`  
  → small

So **do not use only total params as the validation target for GLM-4.7**. It can pass for the wrong reason.

### 3) GLM-4.7-Flash

This is important because your MLA code is much closer to this model than to base GLM-4.7.

With the released config and no MTP modeled, I get:

- **GLM-4.7-Flash total params (no MTP modeled)** ≈ **29.94B**

The public model card headline is about **31B**, so the remaining gap is likely in optional components such as the omitted next-n prediction / MTP block.

---

## Detailed findings and fixes

## A. Layer pattern is missing (`first_k_dense_replace` / `mlp_layer_types`)

### Problem

`model.py` computes one representative layer and multiplies it by `N`.

That is wrong for GLM-5 / GLM-4.7 / GLM-4.7-Flash because these models are **not homogeneous across layers**:

- GLM-5: first 3 MLP layers dense, rest sparse
- GLM-4.7: first 3 dense, rest sparse
- GLM-4.7-Flash: first 1 dense, rest sparse

### Why it matters

This directly changes:

- parameter count
- weight bytes
- FLOPs
- HBM reads for expert weights
- per-layer breakdown

### Fix

Do **not** scale one layer by `N`.

Instead compute at least two layer templates:

- `dense_layer_stats`
- `sparse_layer_stats`

and aggregate by the real pattern:

```python
for layer_idx in range(num_hidden_layers):
    if mlp_layer_types is not None:
        is_sparse = (mlp_layer_types[layer_idx] == "sparse")
    else:
        is_sparse = layer_idx >= first_k_dense_replace
```

For GLM-5 and GLM-4.7, this is a must-fix.

---

## B. Shared experts are missing from MoE

### Problem

`moe.py` models only:

- router
- routed experts

but real DeepSeek/GLM MoE also contains:

- `shared_experts`

### Correct parameter math

For one sparse MoE layer:

- router params  
  `= h * n_routed_experts`
- routed expert params  
  `= n_routed_experts * FFN_params(h, moe_intermediate_size)`
- shared expert params  
  `= FFN_params(h, moe_intermediate_size * n_shared_experts)`

For GLM-5 and GLM-4.7 with `n_shared_experts = 1`, that means one always-on extra expert MLP per sparse layer.

### Correct active-FLOPs math

Per token:

- routed active experts  
  `= num_experts_per_tok * FFN_FLOPs(h, moe_intermediate_size)`
- shared experts  
  `= 1 * FFN_FLOPs(h, moe_intermediate_size * n_shared_experts)`

The shared expert is **always active** and should not be multiplied by `top_k`.

### Fix

Change `moe_stats(...)` to accept:

- `n_routed_experts`
- `n_shared_experts`

and add a separate shared-expert FFN child.

---

## C. GLM-4.7 base attention output shape is wrong

### Problem

`output_proj_stats(...)` assumes:

- `in_features = hidden_size`
- `out_features = hidden_size`

That is not true for GLM-4.7 base.

### Real GLM-4.7 base attention shapes

- `q_proj`: `hidden_size -> num_attention_heads * head_dim`
- `k_proj`: `hidden_size -> num_key_value_heads * head_dim`
- `v_proj`: `hidden_size -> num_key_value_heads * head_dim`
- `o_proj`: `num_attention_heads * head_dim -> hidden_size`

For GLM-4.7:

- `hidden_size = 5120`
- `num_attention_heads * head_dim = 96 * 128 = 12288`

So:

- correct `o_proj` is **12288 -> 5120**
- current code models **5120 -> 5120**

### Impact

This undercounts, per layer:

- `o_proj` params
- `o_proj` FLOPs
- `o_proj` HBM reads
- `o_proj` activation bytes

The per-layer parameter difference is:

- correct `o_proj` params = `12288 * 5120 = 62,914,560`
- current code-like `o_proj` params = `5120 * 5120 + 5120 bias = 26,219,520`

difference:

- **36,695,040 params per layer**

Across 92 layers, that is about:

- **3.38B params**

### Fix

Redesign `output_proj_stats` to accept explicit input width:

```python
def output_proj_stats(
    in_features: int,
    out_features: int,
    ...
)
```

For standard attention:

- `in_features = num_attention_heads * head_dim`
- `out_features = hidden_size`

For MLA:

- `in_features = num_attention_heads * v_head_dim`
- `out_features = hidden_size`

---

## D. GLM-4.7 base Q/O tensor widths are wrong in `attention.py`

### Problem

`attention.py` uses `hidden_size` as the size of the Q tensor and the pre-`o_proj` output tensor for HBM accounting.

That is only correct when:

- `num_attention_heads * head_dim == hidden_size`

but GLM-4.7 base does **not** satisfy that.

### Correct shape variables

For standard attention, keep these widths separate:

- `q_width = num_attention_heads * head_dim`
- `kv_width = num_key_value_heads * head_dim`
- `attn_out_width = num_attention_heads * head_dim`

Then use:

- Q read: `B * q_len * q_width`
- K read: `B * kv_len * kv_width`
- V read: `B * kv_len * kv_width`
- attention output write: `B * q_len * attn_out_width`

### Fix

Refactor `attention_stats(...)` to take:

- `q_len`
- `kv_len`
- `q_width`
- `kv_width`
- `attn_out_width`

Do not derive Q/O widths from `hidden_size`.

---

## E. GLM-4.7 `use_qk_norm = true` is missing

### Problem

Base GLM-4.7 applies `q_norm` and `k_norm` over `head_dim`.

Your calculator does not model these at all.

### Correct param math

If `use_qk_norm` is true:

- add `RMSNorm(head_dim)` for Q
- add `RMSNorm(head_dim)` for K

So the extra per-layer params are:

- `2 * head_dim`

For GLM-4.7:

- `2 * 128 = 256 params/layer`

### Correct FLOPs (approximate)

Per token, per layer:

- Q norm: `~4 * num_attention_heads * head_dim`
- K norm: `~4 * num_key_value_heads * head_dim`

So:

- `~4 * (a + k) * d * B * q_or_k_len`

### Fix

Add optional Q/K norm stats in the standard-attention layer path:

```python
if cfg.use_qk_norm:
    layer.children.append(q_norm_stats(...))
    layer.children.append(k_norm_stats(...))
```

---

## F. MLA layernorms are missing (`q_a_layernorm`, `kv_a_layernorm`)

### Problem

`mla.py` currently counts:

- `q_a_proj`
- `q_b_proj`
- `kv_a_proj`
- absorbed `kv_b` weights
- `q_absorb`
- `v_expand`
- `o_proj`

but it misses the real MLA layernorms:

- `q_a_layernorm`
- `kv_a_layernorm`

### Correct parameter math

For DeepSeek/GLM-style MLA:

- `q_a_layernorm` params = `q_lora_rank`
- `kv_a_layernorm` params = `kv_lora_rank`

For GLM-5:

- `2048 + 512 = 2560 extra params/layer`

### Correct FLOPs (approximate)

Per token, per layer:

- `q_a_layernorm` ≈ `4 * q_lora_rank`
- `kv_a_layernorm` ≈ `4 * kv_lora_rank`

### Fix

Add these as explicit children in `mla_proj_stats(...)`.

---

## G. DSA indexer math is wrong: it is not a softmax

### Problem

`dsa.py` models the indexer scoring step as:

- dense QK dot products
- then `softmax`
- then top-k

That does **not** match the reference GLM-5 indexer.

### Better description of the real indexer

The reference logic is approximately:

1. `q = wq_b(q_resid)` -> reshape to `[B, S_q, H_idx, D_idx]`
2. `k = k_norm(wk(hidden_states))` -> `[B, T_kv, D_idx]`
3. `weights = weights_proj(hidden_states)` -> `[B, S_q, H_idx]`
4. `scores = einsum(q, k)` -> `[B, S_q, H_idx, T_kv]`
5. `scores = relu(scores)`
6. `index_scores = weighted_sum_over_heads(scores, weights)` -> `[B, S_q, T_kv]`
7. `topk(index_scores)`

So the indexer uses:

- **ReLU**
- **weighted sum**
- **top-k**

not softmax.

### Correct FLOPs (backend-independent algebraic version)

Let:

- `Q = q_len`
- `T = kv_len`
- `H = index_n_heads`
- `D = index_head_dim`

Then:

- dot products:  
  `2 * B * Q * T * H * D`
- ReLU:  
  `1 * B * Q * T * H`
- head-weighted reduction:  
  about `2 * B * Q * T * H`
- top-k selection:  
  backend/algorithm dependent (not a softmax term)

### Fix

Replace this:

```python
flops_softmax = 5 * B * s * k_sel
```

with:

```python
flops_relu = B * q_len * kv_len * index_n_heads
flops_weighted_sum = 2 * B * q_len * kv_len * index_n_heads
flops_topk = ...  # optional approximate term
```

and **remove softmax from the README and code path**.

---

## H. DSA top-k should depend on KV length, not query length

### Problem

Current code uses:

```python
k_sel = min(index_topk, seq_len)
```

This is wrong for generation.

If:

- `q_len = 1`
- `kv_len = 200000`

then the real sparse attention still selects up to:

- `min(index_topk, kv_len)`

not 1.

### Fix

All attention calculators need two lengths:

- `q_len`
- `kv_len`

For DSA:

```python
k_sel = min(index_topk, kv_len)
```

not `min(index_topk, q_len)`.

---

## I. One `seq_len` is not enough: split prefill and decode

### Problem

The whole calculator currently assumes:

- query length = key/value length = `seq_len`

That is only true for a no-cache prefill pass.

For cached generation:

- `q_len = number of new tokens`
- `kv_len = past_len + q_len`

This changes:

- attention FLOPs
- DSA FLOPs
- HBM reads from cache
- top-k selection size
- per-step memory traffic

### Correct design

Every attention-like function should take:

- `q_len`
- `kv_len`

and every cache-like function should take:

- `cache_len`

not just `seq_len`.

### Suggested API shape

```python
attention_stats(
    q_len: int,
    kv_len: int,
    ...
)

mla_attention_stats(
    q_len: int,
    kv_len: int,
    ...
)

dsa_indexer_stats(
    q_len: int,
    kv_len: int,
    ...
)

kv_cache_stats(
    cache_len: int,
    ...
)
```

---

## J. MLA / DSA cache layout is runtime-dependent and must be explicit

### Problem

Your current calculator hardcodes the **compressed-cache absorbed MLA** view:

- MLA cache per token = `kv_lora_rank + qk_rope_head_dim`
- DSA+MLA adds `index_head_dim`

This is a valid **specialized runtime model**, but it is **not the only real runtime**.

### There are at least two distinct runtime modes

#### Mode 1: compressed / absorbed cache (your current assumption)

Cache per token, per layer:

- MLA:  
  `kv_lora_rank + qk_rope_head_dim`
- DSA+MLA:  
  `kv_lora_rank + qk_rope_head_dim + index_head_dim`

This is the small-cache theoretical path.

#### Mode 2: expanded K/V cache (current HF DeepSeek-V3 / GLM-4.7-Flash / GLM-5 path)

Main attention cache per token, per layer:

- keys: `num_attention_heads * qk_head_dim`
- values: `num_attention_heads * v_head_dim`

So:

- expanded MLA cache =  
  `num_attention_heads * qk_head_dim + num_attention_heads * v_head_dim`

and for GLM-5 add the indexer key cache:

- expanded DSA+MLA =  
  `num_attention_heads * qk_head_dim + num_attention_heads * v_head_dim + index_head_dim`

### Why this matters numerically

#### GLM-5

- compressed assumption:  
  `512 + 64 + 128 = 704 elements/token/layer`
- expanded-cache HF-style:  
  `64 * (256 + 256) + 128 = 32896 elements/token/layer`

That is a difference of about:

- **46.7x**

#### GLM-4.7-Flash

- compressed assumption:  
  `512 + 64 = 576`
- expanded-cache HF-style:  
  `20 * (256 + 256) = 10240`

difference:

- **17.8x**

### Fix

Make cache layout a **first-class runtime option**:

```python
cache_layout = "standard" | "mla_compressed" | "mla_expanded"
```

Then compute KV cache and HBM with the selected layout.

Without this, “memory correct” is not well-defined for GLM-5 / GLM-4.7-Flash.

---

## K. DSA HBM / activation memory also needs a runtime mode

### Problem

Your README and `dsa.py` use a streaming / blockwise-topk assumption:

- compute: dense-ish score evaluation
- storage: only top-k

This is fine for a fused blockwise selector kernel.

But the eager PyTorch/HF fallback path materializes a dense intermediate score tensor of shape roughly:

- `[B, Q, H_idx, T]`

before reducing across heads and taking top-k.

### Fix

Add two modes for DSA indexer memory:

- `dsa_indexer_mode = "fused_topk"`
- `dsa_indexer_mode = "eager_dense_scores"`

Then:

- fused mode:
  - storage ~ `O(B * Q * k_sel)`
- eager mode:
  - storage ~ `O(B * Q * H_idx * T)`

This is a major memory difference.

---

## L. `output_proj` bias rule is wrong for GLM-4.7 base

### Problem

In base GLM-4.7:

- `q_proj`, `k_proj`, `v_proj` use `bias=config.attention_bias`
- `o_proj` uses `bias=False`

Your current standard-attention path passes the same `cfg.attention_bias` flag into `output_proj_stats(...)`, so base GLM-4.7 would incorrectly count an `o_proj` bias.

### Fix

Bias should be architecture-specific:

#### Base GLM-4.7 standard attention
- Q/K/V bias = `attention_bias`
- O bias = `False`

#### MLA / DeepSeek-style attention
- `q_a_proj`, `kv_a_proj`, `o_proj` bias = `attention_bias`
- `q_b_proj`, `kv_b_proj` bias = `False`
- direct `q_proj` (when no q_lora) bias = `False`

---

## M. `num_nextn_predict_layers` / MTP is ignored

### Problem

The released configs include:

- `num_nextn_predict_layers = 1`

but your calculator does not model any MTP / next-n prediction block.

### Why this matters

If you compare against public model-card headline parameter counts, this can cause a mismatch.

This is probably one reason why:

- GLM-4.7 base no-MTP calculator math is around **352.8B**
- public headline is around **358B**

while the active-parameter math is already correct.

### Fix

Two acceptable choices:

#### Option 1 — explicit exclusion
Add:

```python
include_mtp = False
```

and state clearly:

- “Totals exclude next-n prediction / MTP blocks.”

#### Option 2 — model it
If you want to match headline totals more closely, add an optional MTP component when `num_nextn_predict_layers > 0`.

At minimum, the omission should be documented; otherwise users will think the tool is wrong when comparing against model cards.

---

## N. Activation memory (`act_bytes`) is not an exact peak-memory model

### Problem

The current code usually aggregates activations with:

- `max(child.act_bytes)`

This is easy to implement, but it is not an exact liveness-based peak-memory estimate.

Examples:

- SwiGLU needs the gate branch and up branch alive together before multiplication.
- Attention often needs Q/K/V and score/output buffers in overlapping lifetimes.
- MoE dispatch has extra routing / gather / scatter buffers.

### Fix

Two possible fixes:

#### Practical fix
Rename this metric to something like:

- `working_set_bytes_estimate`

and document that it is not exact peak memory.

#### More exact fix
Implement a simple liveness scheduler for each compound module.

For example, FFN peak should consider:

- input
- gate output
- up output
- output buffer

not only the max child tensor.

---

## O. Small completeness issues

These are not the main blockers for GLM-5 / GLM-4.7, but they are still worth fixing.

### O1. INT4 byte rounding

`dtype_bytes(INT4) = 0.5` and many call sites use `int(n * 0.5)`.

That rounds down for odd element counts.

Use:

- packed bytes = `ceil(n / 2)`

instead.

### O2. Embedding index bytes

Embedding index reads are currently multiplied by compute dtype bytes.

Token IDs are normally integer types, not bf16/fp16.

This is a small error, but the correct model would use separate `index_dtype_bytes`.

### O3. FFN elementwise FLOPs are approximate

Current SwiGLU / GELU elementwise FLOPs are very rough.

That is acceptable if you document them as approximations.

---

## Recommended redesign

If your goal is “real config in -> correct GLM FLOPs/memory out”, I recommend this redesign.

## 1. Add architecture fields to `ModelConfig`

You need these fields available in the calculator layer:

- `first_k_dense_replace`
- `mlp_layer_types`
- `n_routed_experts`
- `n_shared_experts`
- `use_qk_norm`
- `num_nextn_predict_layers`
- `qk_head_dim`
- `qk_nope_head_dim`
- `qk_rope_head_dim`
- `v_head_dim`
- `q_lora_rank`
- `kv_lora_rank`
- `head_dim`
- `attention_runtime`
- `cache_layout`

## 2. Separate standard attention shapes from hidden size

Use explicit widths:

- `q_width`
- `kv_width`
- `attn_out_width`
- `proj_out_width`

## 3. Separate prefill length from decode cache length

Use:

- `q_len`
- `kv_len`
- `cache_len`

## 4. Add runtime modes for MLA / DSA

Suggested enum values:

- `standard_gqa`
- `mla_absorbed_compressed_cache`
- `mla_expanded_cache`
- `dsa_fused_sparse`
- `dsa_hf_dense_mask`

## 5. Model dense and sparse transformer layers separately

At minimum:

- `dense_layer_stats`
- `sparse_layer_stats`

and aggregate by layer index.

---

## Suggested code patch order

If you want to fix this with code agents later, this is the order I would use.

### Patch 1 — make the config expressive enough
Add fields for:

- dense/sparse layer pattern
- shared experts
- qk norm
- q_len / kv_len split
- runtime/cache mode
- MTP inclusion flag

### Patch 2 — fix GLM-4.7 base attention shapes
Change:

- `output_proj_stats`
- `attention_stats`

to use explicit Q/K/V/O widths.

### Patch 3 — fix MoE
Add:

- shared experts
- dense prefix layers
- active shared-expert FLOPs

### Patch 4 — fix MLA projection math
Add:

- `q_a_layernorm`
- `kv_a_layernorm`
- runtime choice: absorbed vs expanded-cache

### Patch 5 — fix DSA
Replace:

- softmax term

with:

- ReLU
- head-weighted reduction
- top-k selection

and use `kv_len`, not `seq_len`, for the candidate length.

### Patch 6 — fix cache math
Add:

- `cache_layout`
- compressed vs expanded MLA cache formulas

### Patch 7 — add optional MTP
Either:
- support it explicitly
or:
- label totals as “excluding MTP”.

---

## Minimal corrected formulas to implement first

If you want the smallest set of changes that gets GLM-5 and GLM-4.7 mostly right, implement these first.

### Standard GLM-4.7 attention

Let:

- `h = hidden_size`
- `a = num_attention_heads`
- `k = num_key_value_heads`
- `d = head_dim`
- `Q = q_len`
- `T = kv_len`

Then:

- `q_proj params = h * (a*d) + (a*d if bias else 0)`
- `k_proj params = h * (k*d) + (k*d if bias else 0)`
- `v_proj params = h * (k*d) + (k*d if bias else 0)`
- `o_proj params = (a*d) * h`  **(bias false for base GLM-4.7)**

Attention FLOPs:

- `QK^T = 2 * B * a * Q * T * d`
- `softmax ≈ 5 * B * a * Q * T`
- `AV = 2 * B * a * Q * T * d`

Attention HBM (standard, no flash):

- read `Q`: `B * Q * (a*d)`
- read `K`: `B * T * (k*d)`
- read `V`: `B * T * (k*d)`
- write/read attention matrix: `B * a * Q * T`
- write attention output: `B * Q * (a*d)`

### GLM-5 / GLM-4.7-Flash MLA projection additions

Add:

- `q_a_layernorm params = q_lora_rank`
- `kv_a_layernorm params = kv_lora_rank`

### GLM-5 DSA indexer

Let:

- `Q = q_len`
- `T = kv_len`
- `H = index_n_heads`
- `D = index_head_dim`

Then:

- score dot-products:  
  `2 * B * Q * T * H * D`
- ReLU:  
  `B * Q * T * H`
- head-weighted reduction:  
  `2 * B * Q * T * H`
- top-k selection: algorithm-dependent

No softmax term.

### Sparse-layer MLP pattern

For layer `i`:

```python
is_sparse = (i >= first_k_dense_replace)
```

unless `mlp_layer_types` is explicitly supplied.

### Sparse MoE params

For one sparse layer:

```python
router = h * n_routed_experts
routed = n_routed_experts * swiglu_params(h, moe_intermediate_size)
shared = swiglu_params(h, moe_intermediate_size * n_shared_experts)
total_sparse_moe = router + routed + shared
active_sparse_flops_per_token = (
    num_experts_per_tok * swiglu_flops_per_token(h, moe_intermediate_size)
    + swiglu_flops_per_token(h, moe_intermediate_size * n_shared_experts)
)
```

---

## Final assessment

### For GLM-5
Your current program is **not correct** yet for total params, FLOPs, or memory because it misses:

- first 3 dense layers
- shared experts
- MLA layernorms
- correct DSA indexer math
- decode-time `q_len != kv_len`
- explicit cache/runtime mode

### For GLM-4.7 base
Your current program is **not correct** yet because it misses:

- first 3 dense layers
- shared experts
- correct `o_proj` size
- correct Q/O tensor widths in attention HBM
- `q_norm` / `k_norm`
- optional MTP accounting

### The most dangerous trap
For GLM-4.7, the total parameter count can look close to the public number **for the wrong reason**.
So validate against:

- per-layer params
- active params
- attention projection shapes
- cache size
- not just the one final total.

