#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import fletcher.shapes: circle, ellipse, hexagon, house, trapezium
#import "@preview/thmbox:0.3.0" as thmbox-lib
#import "lib.typ": slide-footer, slide-header
#import "@preview/cetz:0.4.2": canvas, draw
#import "@preview/lilaq:0.5.0" as lq
#set page(
  paper: "presentation-16-9",
  footer: slide-footer(docname: "Final Year Project GEMM Slides"),
  header: slide-header,
  margin: 2cm,
)
#set text(size: 18pt, font: "Calibri")
#show: thmbox-lib.thmbox-init()
#let claim = thmbox-lib.claim.with(
  title-fonts: "Calibri",
  sans-fonts: "Calibri",
)
#let note = thmbox-lib.note.with(
  title-fonts: "Calibri",
  sans-fonts: "Calibri",
)
#let remark = thmbox-lib.remark.with(
  title-fonts: "Calibri",
  sans-fonts: "Calibri",
)
#let definition = thmbox-lib.definition.with(
  title-fonts: "Calibri",
  sans-fonts: "Calibri",
)
#set math.mat(delim: "[")

#align(horizon)[
  #text(size: 32pt, weight: "bold")[
    GEMM Kernels Used in GPTQ and AWQ Quantized Models
  ]

  #text(style: "italic")[January 2025]
]

#pagebreak()

= Quantization Configuration

- We sampled around 1300 models from the Hugging Face model hub.

== Claimed Quantization Methods

#let raw_claimed_quant_methods = json(
  bytes(
    "
  {
  \"gptq\": 3872986,
  \"unknown\": 4233749,
  \"bitsandbytes\": 2810790,
  \"compressed-tensors\": 3302734,
  \"awq\": 9011475,
  \"auto-round\": 19650,
  \"fp8\": 3718771,
  \"torchao\": 9089,
  \"modelopt\": 9528,
  \"gptq_marlin\": 1846,
  \"aqlm\": 373,
  \"smooth_quant\": 57,
  \"int8\": 11586,
  \"awq_marlin\": 4613,
  \"nvfp4\": 27109
}
",
  ),
)

#let thresholded_claimed_quant_methods = {
  let other_count = 0
  let filtered_methods = (:)
  for (method, count) in raw_claimed_quant_methods {
    if count < 50000 {
      other_count += count
    } else {
      filtered_methods.insert(method, count)
    }
  }
  filtered_methods.insert("other", other_count)
  filtered_methods.pairs().sorted(key: v => v.at(1))
}

#figure(caption: "Distribution of Quantization Methods, weighted by download counts")[
  #lq.diagram(
    width: 20cm,
    height: 3cm,
    yaxis: (
      ticks: thresholded_claimed_quant_methods.map(method => method.at(0)).map(align.with(right)).enumerate(),
      subticks: none,
    ),
    lq.hbar(
      thresholded_claimed_quant_methods.map(method => method.at(1)),
      range(0, thresholded_claimed_quant_methods.len()),
    ),
  )
]
- `compressed-tensors` indicates that the model is quantized by LLM Compressor, which integrates a variety of quantization algorithms, including RTN, SmoothQuant, AWQ, GPTQ, QuIP, and others.
- `unknown` indicates that the quantization method cannot be reliably inferred from model metadata (e.g., MLX-LM quantized models or models without explicit quantization tags).
- Other quantization methods mean literally what they say on the tin.

#pagebreak()

== Weights and Activation Bitwidth

- We can conclude that the most popular weight and activation bit-widths are 4-bit and 8-bit, 16-bit and 8-bit, respectively, approximately three orders of magnitude higher download counts.

#figure(caption: "Distribution Bit-Widths, weighted by download counts")[
  #columns(2)[
    #lq.diagram(
      title: "Weight Bit-Widths",
      width: 10cm,
      height: 3cm,
      yaxis: (
        ticks: (2, 3, 4, 6, 8).map(str).map(align.with(right)).enumerate(),
        subticks: none,
      ),
      lq.hbar(
        (6626, 7571, 18778372, 271, 8241516),
        range(0, 5),
      ),
    )
    #colbreak()
    #lq.diagram(
      title: "Activation Bit-Widths",
      width: 10cm,
      height: 2.0cm,
      yaxis: (
        ticks: (4, 8, 16).map(str).map(align.with(right)).enumerate(),
        subticks: none,
      ),
      lq.hbar(
        (3349, 8707148, 18323859),
        range(0, 3),
      ),
    )
  ]
  #lq.diagram(
    title: "Popular Weight and Activation Bit-Width Combinations",
    width: 20cm,
    height: 2.0cm,
    yaxis: (
      ticks: ("W4A8", "W4A16", "W8A8", "W8A16").map(align.with(right)).enumerate(),
      subticks: none,
    ),
    lq.hbar(
      (2805820, 15964443, 5873953, 2363234),
      range(0, 4),
    ),
  )
]

#pagebreak()

== GPTQ and AWQ Configurations

#figure(caption: "Distribution of GPTQ and AWQ -Specific Configurations")[
  #lq.diagram(
    title: [GPTQ #footnote[`g` means group size, and `ao` means input activation ordering (Act Order).]],
    width: 20cm,
    height: 3cm,
    yaxis: (
      ticks: ("other", "4-bit", "4-bit g128", "4-bit g128 ao", "8-bit g128").map(align.with(right)).enumerate(),
      subticks: none,
    ),
    lq.hbar(
      (14899, 466682, 2912458, 406584, 72363),
      range(0, 5),
    ),
  )
  #lq.diagram(
    title: [AWQ #footnote[`g` means group size, and `zp` means zero-point enabled (Asymmetric Quantization).]],
    width: 20cm,
    height: 2.0cm,
    yaxis: (
      ticks: ("other", "4-bit g128 zp", "4-bit g64 zp").map(str).map(align.with(right)).enumerate(),
      subticks: none,
    ),
    lq.hbar(
      (1464, 100886, 8909125),
      range(0, 3),
    ),
  )
]

= GPTQ Linear Layer

- Since GPTQ is compatible with arbitrary quantization grid, GPTQ kernels can support a wide range of quantization formats, including those used by AWQ, because they support arbitrary grouping, scales, and zero-points.
- These features must be supported and can be enabled/disabled individually:
  - Group-wise scale
  - Zero-point
  - Quantized scale
  - Quantized zero-point
  - Activation reordering
  - Float dequantization
  - Bias addition
  - Float32 accumulation
- Marlin is the most likely kernel to perform GEMM for GPTQ models in vLLM.

== Marlin Kernel
=== Path for W4A16 with Group-Wise Scale #footnote[Channel-wise scale can be reduced to group-wise scale with group size of one entire row.] and Activation Reordering

==== Act Order Reduction
- If the full activation vector is available on a single device#footnote[No tensor parallelism along the K dimension], we can reorder the input activation channels rather than indexing `g_idx` at dequantization time.
  - If it is impossible, the input activation channel can also be reordered by `g_idx`, but it can not be completely eliminated.

==== Matrix Multiplication
- For each tile:
  - Dequantize the weight matrix $w_q = s "fp16"(w_"uint4" - 8)$
  - Perform matrix multiplication $X^"T" W^"T"$ , with $"fp16"$ scalar multiplication and $"fp32"$ accumulation.

#pagebreak()

==== Write to Output Activation
- Round $"fp32"$ accumulation to $"fp16"$ output activation. (RTN)
$
  y_"fp16" = round(y_"fp32")
$

=== Other Paths

==== Zero-Points
- If a zero-point exists and it is quantized, we dequantize it first $z_q = s "fp16"(z_"uint4")$, otherwise $z_q = z_"fp16"$.
- Then, we substract the zero-point during dequantization:
$
  w_q = s "fp16"(w_"uint4") - z_q
$
- In cases other than W4A16 such as W8A16, the zero-point may be substracted before scaling:
$
  w_q = s "fp16"(w_T - z_T)
$

==== Activation Quantization
- If activation quantization is enabled, we quantize the input activation to $T in {"int8", "fp8_e4m3"}$, typically per-token #footnote[Quantization to $"fp8_e4m3"$ may use per-channel scale] absmax.
$
  s = round(max(|x|) / (|max(T)|)), x_T = round(x / s)
$
where $a$ is the input activation, $s$ is the scale, and $a_T$ is the activation quantized to type $T$.
- Replace all $"fp16"$ with $T$ in the  weight dequantization and matrix multiplication steps and use the first-substract-then-scale dequantization formula if zero-point exists.

==== NVFP4 and MXFP4
- If weight is quantized to NVFP4 or MXFP4 format, we need to dequantize the scales using global scale first $s_q = s_"global" s_"fp8"$.
- In MXFP4, if activation quantization is enabled, round up the scale factor.

#pagebreak()

== CPU GPTQ Kernel
- There is no attempts to remove activation reordering on CPU.
- Most things are done in $"fp32"$ instead of $"fp16"$.
- Only W4A16 int4 is supported, and zero-point must be quantized.
- Int to float conversion is done by table lookup rather than arithmetic operations.
$
  w_q &= "fp16"(("fp32"(w_"uint4") - 8) "fp32"(s_"fp16")) &("Without Zeropoint")\
  
  w_q &= "fp16"(("fp32"(w_"uint4") - "fp32"(z_"uint4")) "fp32"(s_"fp16")) &("With Zeropoint")
$

#pagebreak()

= AWQ Linear Layer
- Most AWQ models use asymmetric quantization with zero-points and a default group size of $64$.
- The path divergence to *AWQ Dequantize + Matmul* and *AWQ GEMM* is determined by a heuristic based on the input sequence length.
- In vLLM, when number of tokens is at least $256$, we use AWQ Dequantize + Matmul path; otherwise, we use AWQ GEMM path.
- Another insight is that AWQ format weights can be reduced to GPTQ format, so vLLM also allow users to directly use GPTQ kernels for AWQ models.

== AWQ Dequantize + Matmul Path

$
  w_q = s "fp16"(w_"uint4") - s "fp16"(z_"uint4")
$

where $w_q$ is the dequantized weight matrix, $w_"uint4"$ is the quantized weight, $z_"uint4"$ is the zero-point, and $s$ is the scale, with respect to each group.

After dequantization, call `torch.matmul` to perform the matrix multiplication.


#pagebreak()

== AWQ GEMM Path
- For each tile:
  - Dequantize the weight matrix on-the-fly same as the AWQ Dequantize + Matmul path.
  - Perform matrix multiplication $X^"T" W^"T"$ , with $"fp16"$ scalar multiplication and $"fp32"$ accumulation.
  - Round to nearest $"fp16"$ output activation when writing back to output.

#note[
  About AWQ/GPTQ MoE modules: They are completely same in terms of quantization and dequantization. The dedicated MoE kernels are for MoE-specific arguments such as different weight shape (one extra dimension for experts).
]

== CPU AWQ Kernel
- No such dedicated kernel exists.
- Call GPTQ's.

#pagebreak()

= KVCache Quantization
- VLLM only supports per-tensor fp8 RTN for KVCache quantization.

#note(title: "Notation")[
  In KVCache quatization, we need to manipulate a lot of dynamic objects like the KVCache, so we use $<-$ for assignments in this section for convenience.

  $"softmax"$ and $"concat"$ operates column vectors.

  $[dot]_(T^(a times b))$ means reshape the matrix from $T^(...)$ to $T^(a times b)$.
]

#pagebreak()
== General Math of Attention Modules
=== Multi-Head Attention (MHA) in General (Not exactly how it is implemented in vLLM):
$
  (Q, K, V) <- X, quad S <- "mask"(K^"T" Q), quad Y <- V "softmax"(f S) quad ("for each head")
$
where:
- $X in RR^(k h times s) times RR^(k h times s) times RR^(k h times s)$ and $Y in RR^(v h times s)$ are the input and output activation matrices, respectively.
- $Q in RR^(k h times s)$, $K in RR^(k h times s)$, $V in RR^(v h times s)$, and $S in RR^(h times s times s)$ are the query, key, value, and attention score matrices, respectively.
- $"mask"(dot)$ is a masking function to mask out invalid attention scores to $-infinity$.
- $f$ is a pre-determined scaling factor, typically $(log_2 e) / sqrt(k)$ if $"softmax"(x)_i = 2^(x_i) / (sum_j 2^(x_j))$ for each column vector $x$.
- $s$ is the sequence length, $k, v$ is the query and value dimension, and $h$ is the number of heads.
- Cached variables are $K$ and $V$.

#pagebreak()

=== Multi-head Latent Attention (MLA) in General (Not exactly how it is implemented in vLLM):
==== An introduction of Multi-head Latent Attention
- MLA is a variant of MHA aiming at compressing the KVCache size by forcing the most of cache values into a latent space.
- A review of how MHA is used:
  - $W_O"MHA"("PE"(W_Q H), "PE"(W_K H), W_V H)$ where $"PE"(dot)$ is positional encoding. 
  - It expands to $sum_i^h W_O_i W_V_i H "softmax"(f "mask"("PE"(W_K_i H)^"T" "PE"(W_Q_i H))) $.
- An intuition of MLA:
  - If there is no positional encoding, the $S$ part reduces to $"softmax"(f "mask"((W_K_i H)^"T" (W_Q_i H))) = "softmax"(f "mask"(H^"T" (W_K_i)^"T" W_Q_i H))$.
  - We find that $(W_K_i)^"T" W_Q_i$ is a constant, so we replace it with a single learnable matrix $W_italic("KQ")_i$.
  - Therefore it becomes $sum_i^h W_O_i W_V_i H "softmax"(f "mask"(H^"T" W_italic("KQ")_i H))$.
  - It will be nice if we can just cache $H$ instead of $W_K_i H, W_V_i H$.
  - Anyway, we still want some sort of positional encoding, so we introduce positional encoding to a small amount of dimensions in queries and keys. Let's say we introduce additional key and value terms $(K_"PE")^"T" Q_i = "PE"(W_(K_"PE") H)^"T" "PE"(W_(Q_"PE")_i H)$ in the $S$ part.
  - So it becomes:
  $
    sum_i^h W_O_i W_V_i H "softmax"(f "mask"(H^"T" W_italic("KQ")_i H + "PE"(W_K_"PE" H)^"T" "PE"(W_(Q_"PE")_i H)))
  $
  - Actually, we don't want to cache $H$ entirely but only a low rank projection of it $C = W_C H$:
  $
    sum_i^h W_O_i W_V_i #rect(inset: 10pt, stroke: green)[$#text(fill: red)[$W_C H$] "softmax"(f "mask"((#text(fill: red)[$W_C H$])^"T" #text(fill: blue)[$W_italic("KQ")_i H$] + #text(fill: red)[$"PE"(W_K_"PE" H)$]^"T" #text(fill: blue)[$"PE"(W_(Q_"PE")_i H)$]))$]
  $
  - Red parts are variables to be cached, and blue parts are variables only related to new tokens.
  - In practice, there are more tricks on this equation, but we have already able to derive the MLA, and further tricks are pre- and post-processing on MLA's input and output.
#pagebreak()
Formal definition of MLA:
$
  (Q, Q_"PE", K_"PE", C) <- X, quad S <- "mask"(C^"T" Q + K_"PE"^"T" Q_"PE"), quad Y <- C "softmax"(f S) quad ("for each head")
$
where:
- $X in RR^(c times s) times RR^( k_"pe" h times s) times RR^(c h times s) times RR^(k_"pe" h times s)$ and $Y in RR^(r h times s)$ are the input and output activation matrices, respectively. *Remember that all heads share the same cached latent matrix $C$.*
- $C in RR^(c times s)$, $K_"PE" in RR^(k_"pe" h times s)$, $Q in RR^(c h times s)$, $Q_"PE" in RR^(k_"pe" h times s)$, and $S in RR^(h times s times s)$ are the latent cache, positional key cache, main query, positional query, and attention score matrices, respectively.
- $c$ is the latent dimension, $k_"pe"$ is the positional key/query dimension.
- Other notations are the same as MHA.
- Cached variables are $C$ and $K_"PE"$.
- So the expression on the previous page can be rewrite to:
$
  W_O W_V "MLA"(W_Q H,"PE"(W_(Q_"PE")_i H),"PE"(W_K_"PE" H), W_C H)
$

#pagebreak()

- A comparison between MHA and MLA:
  $
    &#text(fill: red)[$V_i$] "softmax"(&f &"mask"(&#text(fill: red)[$K_i$]^"T" &Q_i&)&) quad ("MHA") \
    &#text(fill: green)[$C$] "softmax"(&f &"mask"(#text(fill: green)[$C^"T" Q_i$ + ]&#text(fill: green)[$K_"PE"$]^"T" &(Q_"PE")_i&)&) quad ("MLA")
  $
  - $V$ vectors are replaced with one single $C$ vector shared by all heads.
  - $K$ vectors are replaced with one single $K_"PE"$ vector shared by all heads.
  - $C^T Q_i$ terms appear in the attention score calculation.
  - Therefore, we can reuse most of the MHA implementation for MLA with minor modifications.
  - Head dimensions can be merged to query demensions since only queries are different across heads in MLA(i.e. $n$ queries with $h$ heads can be merged to $n h$ queries with one head.), so that we don't need to worry about multi-head operations.
=== General Attention Module for Both MHA and MLA
$
  (Q_V, Q, K, V) <- X, quad S_i <- "mask"(K_i^"T" Q + V_i^"T" (Q_V)_i), quad Y_i <- V_i "softmax"(f S_i)
$

#pagebreak()

== Flash Attention 3 Implementation:
- Attention modules must have static scaling factor $s_q, s_k, s_v$ for quantizing $Q, K, V$ to $"fp8_e4m3"$, respectively. No per-token/per-channel scale or dynamic scale is allowed.
  - Extract $Q, K^"new", V^"new"$ from input activation $X$ and reshape them to matrices.
  $
    ((Q_V)_"fp16", Q_"fp16", K_"fp16"^"new", V_"fp16"^"new") <- [X_"fp16"]_("fp16"^(v times h times s) times "fp16"^(k times h times s) times "fp16"^(k times h times s) times "fp16"^(v times h times s))
  $
  - First quantize and cache $K$ and $V$:
  $
    K_"fp8"^"new" <- round(K_"fp16"^"new" / s_k), V_"fp8"^"new" <- round(V_"fp16"^"new" / s_v)
  $
  - Cache that into KVCache.
  $
    K_"fp8" <- "concat"(K_"fp8", K_"fp8"^"new"), V_"fp8" <- "concat"(V_"fp8", V_"fp8"^"new")
  $
  - Quantize $Q$:
  $
    (Q_V)_"fp8"^"new" <- round((Q_V)_"fp16"^"new" / s_q), Q_"fp8"^"new" <- round(Q_"fp16"^"new" / s_q) 
  $
  - Call Flash Attention 3 Steps (Not exactly in this way, explained later).
  $
  Y_"fp32" <- round(V_"fp8" dot "softmax"(f dot s_q dot s_k dot "mask"(K_"fp8"^"T" Q_"fp8"^"new" + II(Q_V != emptyset)V_"fp8"^"T" (Q_V)_"fp8"^"new")) dot s_v) ("for each head")
  $
  - Write back to output activation, rounding to $"fp16"$ and reshaping to $v h times s$
  $
    Y_"fp16" <- [round(Y_"fp32")_"fp16"]_("fp16"^(v h times s))
  $
  - All matrix multiplications have scalar multiplications in $"fp8_e4m3"$ and accumulations in $"fp32"$.
#pagebreak()
=== Explanation of how $Y_"fp32" <- "Attention"(Q_V, Q, K, V)$ is Computed
- In practice, KVCaches are made up of pages, with each page containing $16$ tokens.
- Since there is token's queries and heads work independently, there is no harm to assume that there is only one token's query vector $Q$ and one head, and you can generalize it easily.
- First, we assume that there is a function that process a page of KVCache $"step": (Q: "fp8"^k, Q_V: "fp8"^v union {emptyset}, K: "fp8"^(p times k), V: "fp8"^(p times v), f: "fp32", w: "fp32", n: "fp32"^v, a: "fp32"^v) -> (w^(*): "fp32", n^*: "fp32", a^*:"fp32"^v)$
- Additionally, if context parrellism is enabled, the log-sum-exp is also useful for cross-device reduction, so it must be returned by the MLA/MHA module. However, we omit it here for simplicity.
- *MLA models use MHA at prefilling stage and use MLA only at decoding stage.*
  $
    ("MLA") quad W_V "Attention"(W_K^T A, B, C, D) = "Attention"(emptyset, mat(A; B), mat(W_K D; C), W_V D) quad ("MHA")
  $
- Then, the overall computation can be described as:
#pagebreak()
  $
    &w <- 0 "// sum of attention weights"\
    &n <- -infinity_"fp32"^v "// maximum attention score of all previous pages"\
    &a <- 0_"fp32"^v "// accumulated weighted sum of values of all previous pages times 256"\
    &f^"scaled" <- f dot s_q dot s_k "// softmax scaling factor scaled by s_q and s_k"\
    &"for each page" (K^"page"_"fp8", V^"page"_"fp8") " in the sequence":\
    &quad (w, n, a) <- "step"(Q, Q_V, K^"page"_"fp8", V^"page"_"fp8", f^"scaled", w, n, a)\
    &y_"fp32" <- a dot (w^(-1) dot 2^(-8) dot s_v)
  $
- Here, $f^"scaled"$ is the softmax scaling factor scaled by $s_q$ and $s_k$, $w$ is the accumulated attention weights, $n$ is the maximum attention score of all previous pages, and $a$ is the accumulated weighted sum of values of all previous pages times $2^8$.
- Why $2^8$? Because the maximum exponent of $"fp8_e4m3"$ is $8$, and the attention score subtracted by the maximum attention score is at most $0$, so we use the entire space of exponents.
#pagebreak()

- $"step"(dot)$ definition:
$
  &"step"(Q: "fp8"^k, Q_V: "fp8"^v union {emptyset}, K: "fp8"^(k times p), V: "fp8"^(v times p), f: "fp32", w: "fp32", n: "fp32"^v, a: "fp32"^v):\
  &quad s_"fp32" <- K^"T" Q + II(Q_V != emptyset)V^"T" Q_V quad quad "// require " s_v = s_k " if " Q_V " is present." \
  &quad n^*_"fp32" <- max({n} union s_"fp32")\
  &quad w^*_"fp32" <- w dot 2^(f dot (n^*_"fp32" - n)) + sum_i 2^(f dot ([s_"fp32"]_i - n^*_"fp32"))\
  &quad a^*_"fp32"^v <- a dot 2^(f dot (s_"fp32" - n^*_"fp32")) + V dot "fp8"(2^(f dot (s_"fp32" - n^*_"fp32") + 8))\
  &quad "return" (w^*_"fp32", n^*_"fp32", a^*_"fp32"^v)
$
- Notice that the first call to $"step"(dot)$ use a special version so that it can zero out nessessary variables and avoid computing things like $max({n} union s)$ and $2^(-infinity)$ to save computation.
- The two GEMM operations $K^"T" Q + V^"T" Q_V$ and $V dot "fp8"(2^(f dot (s_"fp32" - n^*_"fp32") + 8))$ are call GEMM I and GEMM II, respectively.

#pagebreak()

== Cutlass Implementation
- Used if computational capability == 10, MLA and all dequantization scales are $1.0$.
- Use a more straightforward MLA-only implementation.
- Using same steps from Flash Attention 3 Implementation to quantize and cache $K$ and $V$, and quantize $Q$.
- Then, concatenate $V, K$ into $K^*$ when it moves them from global memory to shared memory
$
  K^*_"fp8" <- mat(V_"fp8"; K_"fp8"), quad Q^*_"fp8" <- mat((Q_V)_"fp8"; Q_"fp8")
$
- Do GEMM I:
$
  S_"fp32" <- (K^*_"fp8")^T Q^*
$

#pagebreak()

- Compute softmax and round to $"fp8"$
$
  P_"fp8" <- round("softmax"(f S_"fp32"))
$
- Do GEMM II:
$
  Y_"fp32" <- V_"fp8" P_"fp8"
$
- Write back to output activation, rounding to $"fp16"$ and reshaping to $v h times s$
$
  Y_"fp16" <- [round(Y_"fp32")_"fp16"]_("fp16"^(v h times s))
$

#pagebreak()

== FlashInfer Implementation
- Used if computational capability == 10.
- Call TRTLLM-Gen's attention module. #footnote[FlashInfer itself include an attention kernel called xqa, but not used by default in vLLM.]
  - TRTLLM-Gen is a half-open-source library developed by NVIDIA.
  - Only kernel selection part are open-sourced.
  - No available source code for the attention kernel itself.

== CPU Implementation
- No KVCache quantization is supported on CPU.

#pagebreak()

= References

#table(
  columns: (auto, auto, auto, auto),
  align: (left, left, left, left),
  [*Quantization Method*], [*Device*], [*Link*], [*Function*],
  [GPTQ], [CUDA], [#link("https://github.com/vllm-project/vllm/blob/main/csrc/quantization/gptq_marlin/marlin_template.h")[marlin_template.h]], [Marlin()],
  [GPTQ], [CUDA], [#link("https://github.com/vllm-project/vllm/blob/main/csrc/moe/marlin_moe_wna16/marlin_template.h")[marlin_template.h]], [Marlin()],
  [AWQ], [CUDA], [#link("https://github.com/vllm-project/vllm/blob/main/csrc/quantization/awq/gemm_kernels.cu")[gemm_kernels.cu]], [gemm_forward_4bit_cuda_m16nXk32()],
  [GPTQ], [CPU], [#link("https://github.com/vllm-project/vllm/blob/main/csrc/cpu/cpu_wna16.cpp")[cpu_wna16.cpp]], [cpu_gemm_wna16_impl()],
  [KVCache FP8], [CUDA], [
    #link("https://github.com/vllm-project/flash-attention/blob/main/hopper/mainloop_fwd_sm80.hpp")[mainloop_fwd_sm80.hpp]\
    #link("https://github.com/vllm-project/flash-attention/blob/main/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp")[...sm90_tma_gmma_ws.hpp]
  ], [mma()],
  [KVCache FP8], [CUDA], [#link("https://github.com/vllm-project/vllm/blob/main/csrc/attention/mla/cutlass_sm100_mla/kernel/sm100_fmha_mla_tma_warpspecialized.hpp")[...tma_warpspecialized.hpp]], [operator()()]
)