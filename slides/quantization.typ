// #import "@preview/touying:0.6.1": *
// #import "@preview/polylux:0.4.0": *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import fletcher.shapes: circle, ellipse, hexagon, house, rect, trapezium
#import "@preview/thmbox:0.3.0" as thmbox-lib
#import "lib.typ": slide-footer, slide-header
#import "@preview/cetz:0.4.2": canvas, draw
#set page(
  paper: "presentation-16-9",
  footer: slide-footer(docname: "Final Year Project Quantization Slides"),
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
#let dot-matrix((x, y), rows: 10, cols: 10, space: 8pt, prefix: none) = {
  import draw: *
  for i in range(0, rows) {
    for j in range(0, cols) {
      circle((x + j * space, y - i * space), radius: 1pt, fill: black, name: if prefix != none {
        prefix + "_(" + str(i) + "," + str(j) + ")"
      } else {
        none
      })
    }
  }
}
#let elements-in-dot-matrix(prefix, x: 0, y: 0, rows: 10, cols: 10) = (
  prefix + "_(" + str(x) + "," + str(y) + ")",
  prefix + "_(" + str(x + rows - 1) + "," + str(y + cols - 1) + ")",
)

#align(horizon)[
  #text(size: 32pt, weight: "bold")[
    Quantized Inference for Large Language Models
  ]

  #text(style: "italic")[November 2025]
]

#pagebreak()

#claim("Linear Layer Notation")[
  Ignore batching for now, we use the following notations for a linear layer:
  $ Y = W X $
  where $W in RR^(C_"out" times C_"in")$ is the weight matrix, $X in RR^(C_"in" times S)$ and $Y in RR^(C_"out" times S)$ are the input and output activations. Here, $C_"in"$ and $C_"out"$ are the channel dimensions respectively, and $S$ is the sequence length.

  Left multiplication is the most case this equation appear in papers. When we refer to a row, we mean the vector that selects a specific output channel.
]
#remark("No Bias in LLMs")[
  Unlike traditional neural networks, large language models do not use a bias term in their linear layers.
]
#note("Transposed Computation in Modern Libraries")[
  Many modern LLM inference libraries @exllamav2 run their computation with transposed matrices for better memory access patterns in batched inference. In such cases, the computation is performed as:
  $ Y^T = X^T W^T $
  Linear module in Pytorch @Ansel_PyTorch_2_Faster_2024 store the original weight matrix $W$ but *quantization implementations usually store $W^T$*.
]
#claim("Quantization Notation")[
  We denote the quantized weight matrix in type $"type"$ as $W_"type" in "type"^(C_"out" times C_"in")$, and the dequantized weight matrix as $W_"q" in RR^(C_"out" times C_"in")$.
]

#pagebreak()

#align(horizon)[
  = Formats
  Here we are introducing some format of compressed weights used in quantized inference.

  #line(length: 100%)

  #columns(2)[
    - Primitive Types
    #colbreak()
    - Tensor Compression Formats
  ]
]

#pagebreak()

== Primitive Types
- standard integers:
  - int16: 16-bit signed integer
  - int8: 8-bit signed integer
  - int4: 4-bit signed integer
- standard floating-point:
  - fp32: full precision @8766229
  - fp16: half precision @8766229
  - bf16: bfloat16 format @googlecloud2019bfloat16
  - fp8_e4m3: 8-bit float with 4 exponent bits and 3 mantissa bits
  - fp8_e5m2: 8-bit float with 5 exponent bits and 2 mantissa bits
- lookup table based:
  - normal-float: 4-bit normal float @dettmers2023qlora
  - llama.cpp IQ-series: llama.cpp's own lookup table based formats @llamacpp
  - k-means: quantization levels learned via k-means clustering  @han2016deepcompressioncompressingdeep

#pagebreak()

=== Standard Types
- $"int"N$ and $"uint"N$ are typical $N$-bit signed and unsigned integers respectively, stored in modulo $2^N$ format, and there isn't any variant used in LLM quantization.
- Standard floating-point consists of $1$ sign bits, $E$ exponent bits, and $M$ mantissa bits, denoted as $"fp"(E + M + 1)\_"e"E"m"M$. But there are some tricks apart from the standard IEEE-754 format:
  - Flush-to-zero: To speed up computation, some operations treat subnormal numbers as zeros, and all results that are subnormal are flushed to zero.
  - No-Infinity: In some formats like $"fp"8\_"e"5"m"2\_"fn"$, NaNs are represented as all ones in the exponent and mantissa bits, and all other patterns are normal or subnormal numbers, without place for infinity.
  - Unsigned-zero: Some formats like $"fp"8\_"e"4"m"3\_"fnuz"$, NaNs are represented as negative zero in standard floating-point, so there is only one zero representation. The place originally for NaNs and infinity are used for normal numbers.
  - Unsigned: Some formats like $"fp"8\_"e"8"m"0$ do not have sign bits, and only represent non-negative numbers.

#pagebreak()

=== Normal Float
- This is a $k$-bit floating-point format proposed in QLoRA @dettmers2023qlora. Typically, $k=4$.
  - Assume the distribution of weights is roughly normal.
  - Use $1$ value for zero, $2^(k - 1) - 1$ values for negative numbers, and $2^(k - 1)$ values for positive ones.
  - The non-zero values are evenly distributed on the quantiles of a normal distribution with variance around $0.5$ in postive or negative domain respectively.
  #align(center)[
    #figure(
      table(
        stroke: none,
        inset: 16pt,
        columns: (auto, auto, auto, auto),
        table.hline(),
        table.vline(),
        $-1.00$, $-0.70$, $-0.53$, $-0.39$,
        table.vline(),
        $-0.28$, $-0.18$, $-0.09$, $0.00$,
        $0.08$, $0.16$, $0.25$, $0.34$,
        $0.44$, $0.56$, $0.70$, $1.00$,
        table.hline(),
      ),
      caption: "Quantization levels of 4-bit normal float",
    )
  ]
#pagebreak()

=== Llama.cpp IQ-series
- A series of lookup table used in llama.cpp @llamacpp is used for 8-dimension vector quantization.
  - IQ1_S/IQ1_M: a lookup table $f_"iq1": "int11" -> "int8"^8$. @llamacpppullrequest5999
    - One 11-bit integer is unpacked to 8 consecutive 8-bit weights in a row.
    - The lookup is the first step of dequantization. Refer to later sections for complete procedure.
    - The concrete method of constructing or training the lookup table is documented.
  - IQ2_XSS: a lookup table $f_"iq2xss": "int8" -> "int8"^8$ for 2-bit.
  - IQ2_XS: two lookup tables $f_"iq2xs": "int8" -> "int8"^8$ and $f_"ksign_iq2xs": "int7" -> {0, 1}^8$ for 2.3125-bit.
  - IQ2_S: a lookup table $f_"iq2s": "int10" -> "int8"^8$ for 2.5625-bit.
  - IQ3_XXS: a lookup table $f_"iq3xs": "int6" -> "int8"^8$ for 3.0625-bit.
  - IQ3_S: a lookup table $f_"iq3s": "int8" -> "int8"^8$ for 3.3125-bit.
- Another lookup table used in llama.cpp for dequantization:
  - IQ4_S/IQ4_NL: a lookup table $f_"iq4s": "int4" -> "int8"$.

#pagebreak()

=== K-means Quantization
- This is a series of trained quantization levels proposed in Deep Compression @han2016deepcompressioncompressingdeep.
  - Currently, this type is heavily related to Hessian-based non-uniform quantization.
  - The error of a quantization is approximated by the Taylor expansion of the loss function. $ Delta L = g( w ) Delta w + 1/2 Delta w^T H( w ) Delta w + o( || Delta w ||^2 ) $
  - Neglecting the higher-order terms, the first-order term (since the training has converged), and non-diagonal terms of the Hessian (small by experience), we have $ Delta L approx 1/2 sum_i H_(i,i) ( w_i - w_i^q )^2 $
  - This is a weighted k-means clustering problem with weights $H_(i,i)$.
  - At dequantization time, we need to store the cluster centroids as well as the cluster assignments for each weight.
#pagebreak()

== Scaling
- Scaling Methods:
  - Absmax: scale by the absolute maximum value. @dettmers2022llmint8
  - Zeropoint: scale by a learned zeropoint and scale factor.
- Granularity of Scaling:
  - Vector-wise: each row has its own scale factor. @dettmers2022llmint8
  - Group-wise: each group of rows share a scale factor. @frantar2023gptqaccurateposttrainingquantization
- Hardware-Accelerated Low-Bit Float Formats:
  - MXFP4: 4-bit float with shared power-of-two scale factor per group of 32 weights. @opencompute2023ocpmicroscalingformatsmxv10specification
  - NVFP4: 4-bit float with shared fp8 scale factor per group of 16 weights. @nvidia2023introducingnvfp4forefficientandaccuratelowprecisioninference
- Additional Tricks:
  - Nibble Ordering @lin2024awqactivationawareweightquantization

#pagebreak()

=== Absmax and Zeropoint Scaling
- Absmax scaling:
  - The algorithm store the absolute maximum value $s = max_i {|bold(w)_i|} )$
  $
    bold(w)_q = bold(w)_"int8" s / (2^(8 - 1) - 1)
  $
- Zeropoint scaling:
  - The algorithm store a zeropoint $z = (min_i {bold(w)_i} + max_i {bold(w)_i}) / 2$ and a scale factor $s = max_i {bold(w)_i} - min_i {bold(w)_i}$
  $
    bold(w)_q = bold(w)_"int8" s / (2^(8) - 2) + z
  $
#remark(title: "Symmetric Distribution")[
  Weights in linear layers of LLMs are usually symmetrically distributed around zero.
]

#pagebreak()

=== Vector-wise and Group-wise Scaling
- Vector-wise scaling:
  - Each row has its own scale factor (and zeropoint if applicable).
  - Introduced in LLM.int8() @dettmers2022llmint8.
- Group-wise scaling:
  - Each group of a fixed number of elements share a scale factor (and zeropoint if applicable).
  - Introduced in GPTQ @frantar2023gptqaccurateposttrainingquantization, and the paper selected a group size of 128 for 4 and 3 bits quantization, but also mentioned that smaller group sizes allows even lower bit-widths.

#remark(title: "Non-sibling Groups")[
  In many work, groups are *not constrained* to be siblings in memory or matrix. For example, in GPTQ @frantar2023gptqaccurateposttrainingquantization, the group can involve arbitrary rows in a weight matrix.
]

=== Hardware-Accelerated Low-Bit Float Formats

- MXFP4 and NVFP4 are two low-bit floating-point formats supported by nvidia Blackwell architecture GPUs @nvidia2023introducingnvfp4forefficientandaccuratelowprecisioninference and directly supported through PyTorch @Ansel_PyTorch_2_Faster_2024 and TorchAO @torchao_maintainers_and_contributors_torchao_PyTorch_native_2024.
- Basically they are 4-bit floating-point group-wize scaled and then globally scaled.
  $ bold(w)_q = bold(w)_"fp4_e2m1" dot s_"group" dot s_"global" $
- The type of global scale factors for both formats is $"fp32"$, and the comparison is as follows:
  #align(center)[
    #figure(
      table(
        stroke: none,
        inset: 16pt,
        columns: (auto, auto, auto, auto),
        table.hline(),
        [Type], [Element Type], [Group Size], [Group Scale Type],
        table.hline(),
        [MXFP4], [fp4_e2m1], [32], [unsigned fp4_e8m0],
        [NVFP4], [fp4_e2m1], [16], [fp4_e5m2],
        table.hline(),
      ),
      caption: "Comparison of of MXFP4 and NVFP4 formats",
    )
  ]

#pagebreak()
== Dense and Sparse Decomposition
- LLM.int8() @dettmers2022llmint8 introduced a method that quantizes the most activation channels in 8-bit integers, and stores a small amount of outlier channels in full precision.
  - Activations are represented as a tuple of three tensors: quantized dense activations, outlier channel indices, and full-precision outlier activations.
- SqueezeLLM @kim2024squeezellmdenseandsparsequantization decomposes the weight matrix into a dense quantized matrix for elements within a threshold, and a sparse full-precision matrix for outlier elements.
  - Weights are represented as a dense quantized matrix and a compressed row sparse (CSR) full-precision matrix.
- Multiplication is done by performing both dense and sparse multiplications and summing the results. (e.g. $Y=W_"q" X=W_"dense"X + W_"sparse"X$)
  - CSR format is a format that pytorch already have efficient kernels for. @Ansel_PyTorch_2_Faster_2024
    - A CSR matrix is represented by three tensors: values, column indices, and row pointers.
    - Each element in the row pointers points to the start of its corresponding row in the values and column indices tensors.

#pagebreak()

== Additional Tricks
=== Nibble Ordering
- Nibble ordering is a method proposed in AWQ @lin2024awqactivationawareweightquantization to rearrange the low and high nibbles of 4-bit quantized weights
#align(center)[
  #figure(
    caption: "An example of nibble ordering, 8 4-bit weights are unpacked from a 32-bit integer",
  )[
    #diagram(
      node((0, 0), table(
        columns: (2em, 2em, 2em, 2em, 2em, 2em, 2em, 2em),
        inset: 8pt,
        stroke: none,
        table.cell(colspan: 8)[`w`],
        table.hline(),
        $w_0$, $w_4$, $w_1$, $w_5$, $w_2$, $w_6$, $w_3$, $w_7$,
        table.hline(),
      )),
      edge((0, 0), (0, 1), "-->", [Unpack]),
    )
    #columns(2)[
      #table(
        columns: (2em, 2em, 2em, 2em, 2em, 2em, 2em, 2em),
        inset: 8pt,
        stroke: none,
        table.cell(colspan: 8)[`(w & 0xf0f0f0f0) >> 4`],
        table.hline(),
        $0$, $w_0$, $0$, $w_1$, $0$, $w_2$, $0$, $w_3$,
        table.hline(),
      )
      #colbreak()
      #table(
        columns: (2em, 2em, 2em, 2em, 2em, 2em, 2em, 2em),
        inset: 8pt,
        stroke: none,
        table.cell(colspan: 8)[`w & 0x0f0f0f0f`],
        table.hline(),
        $0$, $w_4$, $0$, $w_5$, $0$, $w_6$, $0$, $w_7$,
        table.hline(),
      )
    ]
  ]
]

#pagebreak()

#align(horizon)[
  = Algorithms
  Here we are investigating some algorithms supported by industrial LLM inference libraries #footnote[
    In descending order of Github stars as of Dec. 2025.
  ].
]

#line(length: 100%)

#columns(3, gutter: 0pt)[
  - llama.cpp @llamacpp
  - vLLM @kwon2023efficient
  - DeepSpeed-FastGen @deepspeedfastgen
  #colbreak()
  - MLC LLM @mlc-llm
  - SGLang @sglang
  - TensorRT-LLM @tensorrtllm
  #colbreak()
  - Text Generation Inference @huggingfacetextgenerationinference
  - bitsandbytes @dettmers2022llmint8 @dettmers2023qlora @dettmers2022optimizers
  - Core ML Tools @applecoremltools
]

#pagebreak()

== Classification
- From the prospetive of when quantization is applied:
  - Post-Training Quantization: The model is quantized after full-precision training.
  - Quantization-Aware Training: The model is trained with quantization simulated during training.
  - Fine-Tuning with Quantization: The model is first quantized, then fine-tuned with quantization.
- From the prospetive of quantization target:
  - Weight Quantization: Quantization applied to model weights only.
  - Activation Quantization: Both the weights and activations are quantized.
  - KV Cache Quantization: Quantization applied to the key-value cache during inference.
- Currently, only post-training quantization is well-studied for LLMs.
- Weight quantization is the most commonly used method in LLM inference libraries.


#definition(title: "Round To Nearest")[
  A trival method of quantization is rounding each weight to its nearest quantization level. This is called "Round To Nearest" (RTN) method, and we are not going to talk about it much.
]

#pagebreak()
- The following methods have Tier-0 level support in industrial LLM inference libraries:
#align(center)[
  #figure(
    caption: "Mature Quantization Methods for LLMs",
  )[
    #table(
      columns: (auto, auto, auto, auto),
      inset: 12pt,
      align: horizon,
      stroke: none,
      [],
      table.vline(),
      [Post-Training Quantization],
      [Fine-Tuning with Quantization],
      [
        Quantization-Aware Training
        #footnote[
          Currently, there isn't serious implementation of Quantization-Aware Training for LLMs.
        ]
      ],
      table.hline(),
      [Weight],
      [
        RTN\
        GPTQ @frantar2023gptqaccurateposttrainingquantization\
        AWQ @lin2024awqactivationawareweightquantization\
      ],
      [
        QLoRA @dettmers2023qlora
      ],
      [],
      [Activation],
      [
        RTN\
        LLM.int8() @dettmers2022llmint8\
        SmoothQuant @xiao2023smoothquant\
      ],
      [],
      [],
      [KV Cache],
      [
        RTN
      ],
      [],
      [],
    )
  ]
]
#pagebreak()
- We have also confirm some methods that are popular in academia but not in industry:
  - Have more than 300 citations on Google Scholar.
  - Only one or no mainstream LLM inference library supports them.
#align(center)[
  #figure(
    kind: "list",
    supplement: [List],
    caption: "Methods Popular in Academia but Poorly Supported in Industry",
  )[
    #align(start)[
      #block(stroke: black, inset: 16pt)[
        #columns(4)[
          #text(weight: "bold")[
            Weight Quantization
          ]
          - SqueezeLLM @kim2024squeezellmdenseandsparsequantization
          - SpQR @dettmers2023spqr
          - QuIP @NEURIPS2023_0df38cd1

          #colbreak()
          #text(weight: "bold")[
            Activation Quantization
          ]
          - ZeroQuant #footnote[
              Early quantization method, the fundamental of many modern methods, but not supported in industrial libraries.
            ] @NEURIPS2022_adf7fa39
          - QuaRot @ashkboos2024quarot
          - OmniQuant @shao2024omniquantomnidirectionallycalibratedquantization

          #colbreak()
          #text(weight: "bold")[
            KV Cache Quantization
          ]
          - KVQuant @hooper2024kvquant
          - KIVI @liu2024kivi

          #colbreak()
          #text(weight: "bold")[
            Quantization-Aware Training
          ]
          - LLM-QAT @liu2023llmqatdatafreequantizationaware
        ]
      ]
    ]
  ]
]

- We are not going to discuss them in detail.

#pagebreak()

== GPTQ
- GPTQ @frantar2023gptqaccurateposttrainingquantization is a weight quantization method following the spirit of Optimal Brain Quantization.

=== Optimal Brain Quantization @frantar2023optimalbraincompressionframework
- Take layer-wise reconstruction error in the calibration as the objective.
$
  E = || W X - W_"q" X ||^2
$
- Approximate the objective with second-order Taylor expansion.
$
  E approx 1/2 (W - W_"q") H (W - W_"q")^T
$
- Derive the Hessian matrix.
$
  H approx 2 X X^T
$
#pagebreak()
- The errors in each row is independent, so the quantization of each row can be done independently. We denote the error and weights in a row as $e$ and $w$ respectively.
- The minimum error $Delta e$ cause by quantizing a single weight $w_i$ to $[w_i]_"q"$ while compensating the rest weights with $delta$ is:
$
  Delta e = ([w_i]_"q" - w_i)^2 / [H^(-1)]_(i, i), delta = - (w_i - [w_i]_"q") / [H^(-1)]_(i, i) dot [H^(-1)]_(:, i)
$
- Hessian Inverse can be maintained to avoid invering H multiple times, where $square_(-i)$ means removing the $i$-th row and column.
$
  H^(-1)_(-i) = (H^(-1) - [H^(-1)]_(:, i) [H^(-1)]_(i, :) / [H^(-1)]_(i, i))_(-i)
$
- Iteratively greedily quantize the weight with the smallest $Delta e$ gives the quantized weights.

#pagebreak()

=== Arbitrary Order Insight
- Experiment shows that the order of quantizing weights in a row does not significantly affect the final accuracy.
- If we quantize weights in a fixed order in all rows, we can avoid maintaining Hessian inverse for each row since the Hessian only depends $X$ but not $W$.
- If we quantize weights from left to right, the number of Hessian inverse updates can be reduced to $C_"in"$ from $C_"in" times C_"out"$.
#remark(title: "Actorder Input Channel Reordering")[
  #text(style: "italic")[Not mentioned in the paper but in the official implementation.]

  Reorder the input channels by the descending order of the second order derivative of the reconstruction error with respect to each column to ensure that important columns are quantized first so that later weights can compensate for the quantization error of important columns. The order is permuted back after quantization. See weight format in @gptq-dequantization-kernel.
]

#pagebreak()

=== Lazy-Batch Update
- Hessian inverse update can be unfriendly to memory cache.
- Using a block size $B$, we can quantize $B$ weights in a row before updating the Hessian inverse.
- When an element is update, only update the Hessian inverse within the current block.
- This method does not affect the rounding decision in the following weights within the current block.
- When a block is finished, update the full Hessian inverse and weights for the consequent blocks.
- Suppose $I$ is the block, we can update the Hessian inverse and weight compensation at the end of the block as:
$
  delta = - (w_I - [w_I]_"q") ([H^(-1)]_(I, I))^(-1) [H^(-1)]_(:, I)
$
$
  H^(-1)_(-I) = (H^(-1) - [H^(-1)]_(:, I) ([H^(-1)]_(I, I))^(-1) [H^(-1)]_(I, :))_(-I)
$

#pagebreak()

=== Cholesky Reformulation
- The quantization result can be arbitrarily bad if the Hessian inverse becomes non-positive definite due to numerical errors caused by iterative updates.
- We only use each row of the Hessian inverse from the diagonal elements to the end.
- Our updates are equivalent to performing a Cholesky decomposition of the Hessian matrix.
- We can precompute the Cholesky decomposition of the Hessian inverse using dedicated Cholesky kernels to improve numerical stability.

=== Group-Wise Quantization
- Using different zeros and scales for each groups of $G$ consecutive elements.
- The common group size is $G=128$ for 4-bit and 3-bit quantization.
- Extreme compression such as 2-bit quantization may use a group size of $G=8$.

=== Inference Time
- Linear module can be directly replaced GPTQ's linear module, using a fused GEMM kernel that directly consumes quantized weights.

#pagebreak()

== AWQ
=== Protecting Important Weights
- Prior work GPTQ heavily relies on the Hessian matrix to compute the compensation for quantization error, but the Hessian matrix is not stable across corpus from different domains.
- AWQ @lin2024awqactivationawareweightquantization find that, protecting specific around 1% of the weights will significantly improve the accuracy of quantized models.
- Further study shows that some input channels are more important than others, and they can be identified by activation magnitudes during calibration, without the need of Hessian matrix.
- Experiment shows that selecting top 0.1% channels and keeping them in full precision already improves the accuracy a lot.
- The relative order of the input activation magnitudes much more stable compared to the Hessian matrix across different corpus.

#pagebreak()

=== Scaling Channels
- Instead of keeping some channels in full precision, AWQ scale the important channels up with factor $s$ before quantization so that the error can be reduced due to more quantization levels in the effective range.
$
  W X = (W "diag"(s)) ("diag"(s)^(-1) X)
$
- Scaling the entire weight matrix is unfeasible, since it changes the zeros and scales (i.e. $max_(i, j) {W_(i, j)} - min_(i, j) {W_(i, j)}$), and the scaling factor $s$ is cancelled by the increased of scales.
- But this works in grouped quantization. A *group* sharing same zeros and scales is made up of $128$ consecutive elements *in a row*, and the weights multiplied to an *input channel* are located *in a column*. See the figure in @awq-dequantization-kernel.
- Thus, scaling each input channel only changes at most $1$ element in each group, which will highly unlikely change the zeros and scales of each group. This method end up looking like each weight being scaled by both group-wise scales and input-channel-wise scales.
- A grid search to find $s$ for selected 0.1% channels is fine, and use $s_i=1$ for other channels.

#pagebreak()

=== Inference Time
- Since the weight is scaled up in columns, we need to scale activations down in respective rows during inference.
- If there is an RMSNorm layer before the linear layer, we can change the weight of that RMSNorm layer to absorb the scaling of activations. An RMSNorm layer is defined token-wise as: $ y = x / sqrt(1 / C sum_(i=1)^C x_i^2 + epsilon) dot.o w $ and we can update the weight $w$ to $w dot.o s^(-1)$ to absorb the scaling of activations.
- If there is no RMSNorm layer before the linear layer, we have to insert a scaling layer $y = x dot.o s^(-1)$ before the regular linear layer is replaced by AWQ's linear layer.
- The AWQ's linear layer can be implemented as a fused GEMM kernel that directly consumes quantized weights. In addtion, AWQ will write dequantized weights to memory if the sequence length is long enough to amortize the dequantization cost.

#pagebreak()

== LLM.int8()
- LLM.int8() @dettmers2022llmint8 is an activation quantization method that keeps outlier channels in full precision.
- A hyperparameter $t$ is selected as the threshold for outlier channels, and every row in the input activations with magnitude larger than $t$ is marked as an outlier channel.
- For outlier channels, we dequantize the weight to fp16 to perform fp16 GEMM. For regular channels, we quantize the activations to int8 and perform int8 GEMM fused with dequantization.

#figure(
  caption: "LLM.int8() at Inference Time",
)[
  #diagram(
    spacing: 16pt,
    edge-stroke: 1pt,
    edge-corner-radius: 5pt,
    mark-scale: 60%,
    node((0, 0), $X$),
    edge("->", stroke: yellow),
    node((1, 0), [Quantize], corner-radius: 5pt, inset: 10pt, shape: rect, fill: green.lighten(50%)),
    edge((1, 0), "u,r", "->", [Regular Channels (Per-Token Scale)], label-pos: 1em, label-side: left, stroke: blue),
    edge((1, 0), "d,r,ru", "-->", [Channel Indices], label-pos: 60%, label-side: right),
    edge((1, 0), "dd,rr", "->", [Outlier Channels], label-pos: 1em, label-side: right, stroke: yellow),
    node((2, 0), $W_"int8"$),
    edge("->", stroke: blue),
    node((2, -1), [Int8 GEMM], corner-radius: 5pt, inset: 10pt, shape: rect, fill: blue.lighten(50%)),
    edge("->", stroke: blue),
    node((3, -1), [Dequantize], corner-radius: 5pt, inset: 10pt, shape: rect, fill: purple.lighten(50%)),
    edge((2, 0), (3, 0), "->", stroke: blue),
    node((3, 0), [Dequantize], corner-radius: 5pt, inset: 10pt, shape: rect, fill: red.lighten(50%)),
    edge("->", stroke: yellow),
    node((3, 2), [Fp16 GEMM], corner-radius: 5pt, inset: 10pt, shape: rect, fill: orange.lighten(50%)),
    node((4, 0), [Add], corner-radius: 5pt, inset: 10pt, shape: rect, fill: gray.lighten(50%)),
    edge((3, -1), "r,d", "->", stroke: yellow),
    edge((3, 2), "r,uu", "->", stroke: yellow),
    edge("->", stroke: yellow),
    node((5, 0), $Y$),

    node((7, -1), [Int8]),
    edge("r", "->", stroke: blue),
    node((7, 0), [Fp16]),
    edge("r", "->", stroke: yellow),
    node((7, 1), [Indices]),
    edge("r", "-->"),
  )
]

#pagebreak()

#remark(title: "Row-wise Weight Quantization and Column-wise Activation Quantization")[
  To make use of int8 GEMM kernels, the input activations must be quantized column-wise, while the weights must be quantized row-wise. This is the reason why LLM.int8() identify outlier rows but quantize column-wise.

  #figure(
    caption: "Caption: GEMM-Dequantize example of group-wise absmax quantization",
  )[
    #align(horizon)[
      #box(height: 5em, width: 8em, canvas({
        import draw: *
        dot-matrix((0pt, 0pt), rows: 12, cols: 17, space: 8pt, prefix: "w")
        rect-around(
          "w_(1,2)",
          stroke: black,
          padding: 3pt,
          radius: 2pt,
        )
      }))
      #box(height: 5em, $=$)
      #box(height: 5em, width: 8em, canvas({
        import draw: *
        dot-matrix((0pt, 0pt), rows: 12, cols: 8, space: 8pt, prefix: "w")
        rect-around(
          ..elements-in-dot-matrix("w", x: 1, y: 0, rows: 1, cols: 4),
          stroke: red,
          padding: 3pt,
          radius: 2pt,
        )
        rect-around(
          ..elements-in-dot-matrix("w", x: 1, y: 4, rows: 1, cols: 4),
          stroke: green,
          padding: 3pt,
          radius: 2pt,
        )
      }))
      #box(height: 5em, "  ")
      #box(height: 5em, width: 8em, canvas({
        import draw: *
        dot-matrix((0pt, 0pt), rows: 8, cols: 17, space: 8pt, prefix: "w")
        rect-around(
          ..elements-in-dot-matrix("w", x: 0, y: 2, rows: 4, cols: 1),
          stroke: purple,
          padding: 3pt,
          radius: 2pt,
        )
        rect-around(
          ..elements-in-dot-matrix("w", x: 4, y: 2, rows: 4, cols: 1),
          stroke: blue,
          padding: 3pt,
          radius: 2pt,
        )
      }))

      #box(height: 1em, width: 8em, $Y$)
      #box(height: 1em, $=$)
      #box(height: 1em, width: 8em, $W$)
      #box(height: 1em, "  ")
      #box(height: 1em, width: 8em, $X$)
      $
        Y_(square.filled) = s_(text(square.filled, fill: #red)) s_(text(square.filled, fill: #purple)) "fp16"(W_("int4", text(square.filled, fill: #red)) X_("int8", text(square.filled, fill: #purple))) + s_(text(square.filled, fill: #green)) s_(text(square.filled, fill: #blue)) "fp16"(W_("int4", text(square.filled, fill: #green)) X_("int8", text(square.filled, fill: #blue)))
      $
    ]
  ]
]

#pagebreak()

== SmoothQuant
- SmoothQuant @xiao2023smoothquant is an activation quantization method that migrate the outliers from activations to weights by scaling up the weights and scaling down the activations.
- There are much more outliers in activations than weights since there is no regularization on activations during training, making activation quantization much harder.
- SmoothQuant scale down the activations by a factor $s$ and scale up the weights by the same factor $s$ to keep the output unchanged.
- Given a hyperparameter $alpha in [0, 1]$ and calibration data $X$, the scaling factor for each input channel $i$ is computed as:
$
  s_i = (max_j {X_(i, j)})^alpha (max_j {W_(j, i)})^(1 - alpha)
$
- Then the computation is reformulated as:
$
  hat(Y) = (W "diag"(s))_"q" ("diag"(s^(-1)) X)_"q"
$

#pagebreak()

== QLoRA

- QLoRA @dettmers2023qlora is a fine-tuning with quantization method that uses Low-Rank Adaptation (LoRA) @hu2021loralowrankadaptationlarge to fine-tune quantized LLMs.
- Fine-tuning LLMs in full precision requires huge memory consumption, unfeasible for most users.
- Its basic idea is to store the pre-trained weights in quantized format, using group-wise absmax in fp4 or nf4, while keeping the LoRA weights in full precision.
- To further reduce memory consumption, QLoRA uses double quantization, which quantizes the scales of the weights again using the same method as the first quantization.
- During fine-tuning, only the LoRA weights are updated, while the pre-trained weights remain unchanged.
- Weights are fully dequantized during forward and backward passes.
- This is *not an inference-friendly method*, since there is a lot of table lookups and floating-point operations slowing down the inference speed, but it preserves the accuracy better for fine-tuning tasks.

#pagebreak()

== Other Interesting Methods
- llama.cpp @llamacpp treat every $8$ consecutive weights as a vector and quantize the vectors instead of each weight individually.
- BitNet @wang2025bitnetv2native4bit use Fast Hadamard Transform (FHT) to smooth the weight distribution before quantization.
- OneBit @xu2024onebitextremelylowbitlarge keeps only the signs in full-rank and decomposes the magnitude matrix into a product of two low-rank matrices.
- OmniQuant @shao2024omniquantomnidirectionallycalibratedquantization show that clipping parameters can be learned through differentiable objective functions.

#pagebreak()

== Quantization Scheme and Recipe
=== Quantization Scheme
- A quantization scheme defines the data type of the quantized weights, activations.
- Quantization schemes greatly affect the choice of GEMM kernels used during inference.
- KV Cache quantization is a separate part since they are not directly involved in GEMM operations, thus not showing in quantization schemes.
- Common quantization schemes include:
  - W8A8: Weights and activations are quantized to int8.
  - W4A4_NVFP4: Weights and activations are quantized to nvidia nvfp4.
  - W4A8_FP8: Weights and activations are quantized to fp8.
  - W4A8: Weights are quantized to int4, activations are in int8.
  - W4A16: Weights are quantized to int4, activations are in fp16.
- To be exact, quantization schemes also contain other information such as group size, scaling method, etc.

#pagebreak()

=== Quantization Recipe
- A quantization recipe defines the specific quantization algorithms used to quantize weights, activations, and KV cache.
- These three parts are orthogonal to each other, and mathematically can be combined arbitrarily with a reasonable quantization scheme.
- In fact, only a few combinations are supported in industrial LLM inference libraries.
- Here are some popular quantization recipes supported in vLLM. @nvidiatensorrtllmtutorialquantization
#figure(
  caption: "Popular Quantization Recipes for LLMs",
)[
  #table(
    columns: (auto, auto, auto, auto),
    inset: 6pt,
    align: horizon,
    stroke: none,
    [Name], [Weight], [Activation], [KV Cache],
    table.hline(),
    [W4A16 AWQ], [AWQ int4], [fp16], [fp16],
    [W4A16 GPTQ], [GPTQ int4], [fp16], [fp16],
    [W8A8 GPTQ], [GPTQ int8], [SmoothQuant int8], [int8],
    [FP8], [RTN fp8], [RTN fp8], [fp8],
    [FP4], [RTN nvfp4], [RTN nvfp4], [nvfp4],
  )
]

#pagebreak()

=== Additional Tips
- Skipping Quantization for Specific Layers
  - Typically, the embedding layer and the lm_head layer are not quantized to preserve accuracy.
  - Some models skips quite a large amount of layers. For example, this one @huggingfacedeepseek-r1-qwen3-8b-unsloth-nf4-condig-json.

=== Compitiblity

- Many quantization methods do not have their own format at inference time, but completely use existing layers during inference.
  - For example, OmniQuant @shao2024omniquantomnidirectionallycalibratedquantization use AWQ @lin2024awqactivationawareweightquantization's linear layer during inference. Thus all framework supporting AWQ can support OmniQuant.
  - We will have a table of features supported by different quantization schemes in vLLM.

#pagebreak()

#figure(
  caption: "Features supported in the schemes in vLLM",
)[
  #table(
    columns: (auto, auto, auto, auto, auto),
    inset: 8pt,
    align: horizon,
    stroke: none,
    [Scheme], [Group-wise], [Act-Order], [Asymmetric Weight], [Asymmetric Activation],
    table.hline(),
    [W8A16 FP8], [No], [No], [No], [No],
    [W8A8 INT8], [No], [No], [No], [Yes],
    [W8A8 FP8], [No], [No], [No], [No],
    [GPTQ], [Yes], [Yes], [Yes], [No],
    [AWQ], [Yes], [No], [Yes], [No],
    [W4A16 NVFP4], [Yes], [No], [Yes], [No],
    [W4A8 INT], [Yes], [No], [No], [No],
    [W4A8 FP8], [Yes], [Yes], [Yes], [No],
    [W4A4 NVFP4], [Yes], [No], [No], [No],
  )
]

#pagebreak()

#align(horizon)[
  = Kernels
  We are here to dive into some implementation details of quantized inference for LLMs.
  #line(length: 100%)
  #columns(2)[
    - Weight Dequantization Kernels
    - Activation Quantization Kernels
    #colbreak()
    - Low-Precision GEMM Kernels
    - Low-Precision Nonlinear Kernels
  ]
]

#pagebreak()

== Weight Dequantization Kernels
- This part is mainly used to clarify the concrete weight format used in real implementations.
- Weights does not nessarily need to be dequantized during inference.
  - Weight-only quantization methods can use a fused low-precision GEMM kernel that directly consumes quantized weights. @frantar2023gptqaccurateposttrainingquantization
  - Activation quantization methods can quantize the activations to work with quantized weights directly. @dettmers2022llmint8
  - AWQ @lin2024awqactivationawareweightquantization does dequantize weights if the sequence length is long enough.
#note(title: "Column-Major vs Row-Major")[
  In this section, when we refer to row-major format, we mean that $W_(i, 0), W_(i, 1), ..., W_(i, C_"in" - 1)$ is stored contiguously in memory for each row $i$ (i.e. $W$ is contiguous), and when we refer to column-major format, we mean that $W_(0, j), W_(1, j), ..., W_(C_"out" - 1, j)$ is stored contiguously in memory for each column $j$ (i.e. $W^T$ is contiguous).
]

#pagebreak()

=== GPTQ Dequantization Kernel
<gptq-dequantization-kernel>
- `g_idx` is a mapping from columns to group indices, shaped as $(C_"in")$, typed $"int32"$.
- The weight tensor format used in GPTQ @frantar2023gptqaccurateposttrainingquantization is shown below:
#figure(
  caption: "GPTQ Weight Tensor Format, Column-Major View, Showing First 11 Rows and 37 Columns, Assuming Bit-Width of 4",
)[
  #canvas({
    import draw: *
    dot-matrix((0pt, 0pt), rows: 11, cols: 37, space: 8pt, prefix: "w")
    rect-around(
      ..elements-in-dot-matrix("w", x: 0, y: 0, rows: 8, cols: 1),
      stroke: green,
      padding: 3pt,
      radius: 2pt,
      name: "uint32",
    )
    content((-1em, -(10 * 8pt) / 2 - 1em), anchor: "mid-east")[$C_"out"$]
    content(((37 * 8pt) / 2, 1em))[$C_"in"$]
    content((-2em, -(7 * 8pt) / 2), anchor: "mid-east")[8 weights packed into one uint32]
    set-style(mark: (end: ">"))
    line((-2em, -(7 * 8pt) / 2), "uint32", stroke: green)
    content(((36 * 8pt) + 1em, -(10 * 8pt)), anchor: "mid-west")[$W_"int4"$]
    rect-around(
      ..elements-in-dot-matrix("w", x: 0, y: 3, rows: 11, cols: 1),
      stroke: gray,
      padding: 3pt,
      radius: 2pt,
      name: "g_idx_1",
    )
    rect-around(
      ..elements-in-dot-matrix("w", x: 0, y: 9, rows: 11, cols: 1),
      stroke: gray,
      padding: 3pt,
      radius: 2pt,
      name: "g_idx_2",
    )
    rect-around(
      ..elements-in-dot-matrix("w", x: 0, y: 24, rows: 11, cols: 1),
      stroke: gray,
      padding: 3pt,
      radius: 2pt,
      name: "g_idx_3",
    )
    content(
      ((36 * 8pt) / 2, -(10 * 8pt) - 2em),
      anchor: "mid",
    )[Columns with same `g_idx` share same group scales and zeros]
    set-style(mark: (end: ">"))
    line(((36 * 8pt) / 2, -(10 * 8pt) - 2em + 0.5em), (3 * 8pt, -10 * 8pt - 3pt - 1pt), stroke: gray)
    set-style(mark: (end: ">"))
    line(((36 * 8pt) / 2, -(10 * 8pt) - 2em + 0.5em), (9 * 8pt, -10 * 8pt - 3pt - 1pt), stroke: gray)
    set-style(mark: (end: ">"))
    line(((36 * 8pt) / 2, -(10 * 8pt) - 2em + 0.5em), (24 * 8pt, -10 * 8pt - 3pt - 1pt), stroke: gray)
    rect-around(
      ..elements-in-dot-matrix("w", x: 0, y: 3, rows: 1, cols: 1),
      stroke: red,
      padding: 1pt,
      radius: 1pt,
    )
    rect-around(
      ..elements-in-dot-matrix("w", x: 0, y: 9, rows: 1, cols: 1),
      stroke: red,
      padding: 1pt,
      radius: 1pt,
    )
    rect-around(
      ..elements-in-dot-matrix("w", x: 0, y: 24, rows: 1, cols: 1),
      stroke: red,
      padding: 1pt,
      radius: 1pt,
    )
    rect-around(
      ..elements-in-dot-matrix("w", x: 1, y: 3, rows: 1, cols: 1),
      stroke: blue,
      padding: 1pt,
      radius: 1pt,
    )
    rect-around(
      ..elements-in-dot-matrix("w", x: 1, y: 9, rows: 1, cols: 1),
      stroke: blue,
      padding: 1pt,
      radius: 1pt,
    )
    rect-around(
      ..elements-in-dot-matrix("w", x: 1, y: 24, rows: 1, cols: 1),
      stroke: blue,
      padding: 1pt,
      radius: 1pt,
    )
    rect-around(
      ..elements-in-dot-matrix("w", x: 2, y: 3, rows: 1, cols: 1),
      stroke: purple,
      padding: 1pt,
      radius: 1pt,
    )
    rect-around(
      ..elements-in-dot-matrix("w", x: 2, y: 9, rows: 1, cols: 1),
      stroke: purple,
      padding: 1pt,
      radius: 1pt,
    )
    rect-around(
      ..elements-in-dot-matrix("w", x: 2, y: 24, rows: 1, cols: 1),
      stroke: purple,
      padding: 1pt,
      radius: 1pt,
    )
    rect-around(
      ..elements-in-dot-matrix("w", x: 3, y: 3, rows: 1, cols: 1),
      stroke: orange,
      padding: 1pt,
      radius: 1pt,
    )
    rect-around(
      ..elements-in-dot-matrix("w", x: 3, y: 9, rows: 1, cols: 1),
      stroke: orange,
      padding: 1pt,
      radius: 1pt,
    )
    rect-around(
      ..elements-in-dot-matrix("w", x: 3, y: 24, rows: 1, cols: 1),
      stroke: orange,
      padding: 1pt,
      radius: 1pt,
    )
    rect-around(
      ..elements-in-dot-matrix("w", x: 4, y: 3, rows: 1, cols: 1),
      stroke: yellow,
      padding: 1pt,
      radius: 1pt,
    )
    rect-around(
      ..elements-in-dot-matrix("w", x: 4, y: 9, rows: 1, cols: 1),
      stroke: yellow,
      padding: 1pt,
      radius: 1pt,
    )
    rect-around(
      ..elements-in-dot-matrix("w", x: 4, y: 24, rows: 1, cols: 1),
      stroke: yellow,
      padding: 1pt,
      radius: 1pt,
    )
    content(((36 * 8pt) / 2, 3em), anchor: "mid")[
      Zero and scale is determined by the `g_idx` value of the column and the row the weight is in.
    ]
    set-style(mark: (end: ">"))
    line(((36 * 8pt) / 2, 3em - 0.5em), (3 * 8pt, 1pt + 1pt), stroke: red)
    set-style(mark: (end: ">"))
    line(((36 * 8pt) / 2, 3em - 0.5em), (9 * 8pt, 1pt + 1pt), stroke: red)
    set-style(mark: (end: ">"))
    line(((36 * 8pt) / 2, 3em - 0.5em), (24 * 8pt, 1pt + 1pt), stroke: red)
    content((-2em, -(10 * 8pt)), anchor: "mid-east")[Similar uint32 packing format for zeros]
    content((-2em, -(10 * 8pt) - 1em), anchor: "mid-east")[But zeros $Z$ are stored as $Z - 1$]
    content(((36 * 8pt) + 2em, 0pt), anchor: "mid-west")[Fp16 for scales]
  })
]

#pagebreak()
- Here we present the pseudocode of GPTQ dequantization kernel:
#figure(
  caption: "GPTQ Dequantization Kernel Pseudocode",
)[
  #text(size: 8pt)[
    ```cpp
    /**
     * Dequantize a GPTQ quantized weight matrix, in transposed format.
     *
     * @param g_idx: shape (C_in,), group indices for each row
     * @param qweights: shape (C_in, C_out / (32 / bits)), each uint32_t contains packed quantized values
     * @param qzeros: shape (num_groups, C_out / (32 / bits)), each uint32_t contains packed zero points
     * @param scales: shape (num_groups, C_out), fp16 scaling factors
     * @param max_q: maximum quantized value (2^bits - 1)
     * @param bits: bit-width of quantization (e.g., 2, 3, 4)
     * @param rows: C_in
     * @param cols: C_out
     * @param dequantized: shape (C_in, C_out), output dequantized weights
     */
    inline void gptq_dequantize_kernel(const uint32_t g_idx[],
                                       const uint32_t qweights[],
                                       const uint32_t qzeros[],
                                       const fp16_t scales[], int max_q,
                                       int bits, int rows, int cols,
                                       fp16_t dequantized[]) {
      int packed_count = 32 / bits;
      for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
          uint32_t g = g_idx[y];
          uint32_t iweights = qweights[y * cols + x];
          int wshift = (x % packed_count) * bits;
          uint32_t weight = (iweights >> wshift) & max_q;
          fp32_t scale = static_cast<fp32_t>(scales[g * cols + x]);
          uint32_t izeros = qzeros[g * cols / packed_count + (x / packed_count)];
          uint32_t zero = ((izeros >> wshift) & max_q) + 1; // Notice that gptq zeros are stored as (zero - 1)
          fp16_t dequantized_value = fp16_t(fp32_t(weight - zero) * scale);
          dequantized[y * cols + x] = dequantized_value;
        }
      }
    }
    ```
  ]
]

#pagebreak()

=== AWQ Dequantization Kernel
<awq-dequantization-kernel>
- The weight tensor format of AWQ @lin2024awqactivationawareweightquantization is shown below:
#figure(
  caption: "AWQ Weight Tensor Format, Column-Major View, Showing First 11 Rows and 37 Columns, Assuming Group Size of 32 and Bit-Width of 4",
)[
  #canvas({
    import draw: *
    dot-matrix((0pt, 0pt), rows: 11, cols: 37, space: 8pt, prefix: "w")
    rect-around(
      ..elements-in-dot-matrix("w", x: 0, y: 0, rows: 8, cols: 1),
      stroke: green,
      padding: 3pt,
      radius: 2pt,
      name: "uint32",
    )
    content((-1em, -(10 * 8pt) / 2 - 1em), anchor: "mid-east")[$C_"out"$]
    content(((37 * 8pt) / 2, 1em))[$C_"in"$]
    content((-2em, -(7 * 8pt) / 2), anchor: "mid-east")[8 weights packed into one uint32]
    set-style(mark: (end: ">"))
    line((-2em, -(7 * 8pt) / 2), "uint32", stroke: green)
    rect-around(
      ..elements-in-dot-matrix("w", x: 10, y: 0, rows: 1, cols: 32),
      stroke: red,
      padding: 3pt,
      radius: 2pt,
      name: "group",
    )
    content((-2em, -(10 * 8pt)), anchor: "mid-east")[32 weights sharing same scale and zero]
    set-style(mark: (end: ">"))
    line((-2em, -(10 * 8pt)), "group", stroke: red)
    rect-around(
      ..elements-in-dot-matrix("w", x: 0, y: 36, rows: 11, cols: 1),
      stroke: blue,
      padding: 3pt,
      radius: 2pt,
      name: "channel",
    )
    content(((36 * 8pt) + 2em, -(10 * 8pt) / 2), anchor: "mid-west")[One scaled channel]
    set-style(mark: (end: ">"))
    line(((36 * 8pt) + 2em, -(10 * 8pt) / 2), "channel", stroke: blue)
    content(((36 * 8pt) + 1em, -(10 * 8pt)), anchor: "mid-west")[$W_"int4"$]
    content((-2em, 0pt), anchor: "mid-east")[Similar uint32 packing format for zeros]
    content(((36 * 8pt) + 2em, 0pt), anchor: "mid-west")[Fp16 for scales]
  })
]
- The weights are reordered in AWQ when packed into uint32 values, which is shown below:
#figure(
  caption: "AWQ Weight Ordering in a uint32 Value, 4-bit Weights",
)[
  #table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    inset: 12pt,
    stroke: none,
    table.hline(start: 1, end: 9),
    [Lower],
    $w_0$, $w_2$, $w_4$, $w_6$, $w_1$, $w_3$, $w_5$, $w_7$,
    [Higher],
    table.hline(start: 1, end: 9),
  )
]

#pagebreak()
- Here we present the pseudocode of AWQ dequantization kernel:
#figure(
  caption: "AWQ Dequantization Kernel Pseudocode",
)[
  #text(size: 8pt)[
    ```cpp
    /**
     * Dequantize a AWQ quantized weight matrix, in transposed format.
     *
     * @param qweights: shape (C_in, C_out / 8), each uint32_t contains 8 4-bit quantized values
     * @param scales: shape (C_in / group_size, C_out * 8), fp16 scaling factors
     * @param zeros: shape (C_in / group_size, C_out), uint32_t zeros packed 8 4-bit zero points
     * @param cols: C_out / 8
     * @param rows: C_in
     * @param group_size: number of rows per group for scales and zeros
     * @param dequantized: shape (C_in, C_out), output dequantized weights
     */
    inline void awq_dequantize(const uint32_t== qweights[],
                              const fp16_t scales[],
                              const uint32_t zeros[],
                              int cols, int rows, int group_size,
                              fp16_t dequantized[]) {
      for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) { // Iterate each packed uint32_t
          uint32_t izeros = zeros[y / group_size * cols + x];
          uint32_t iweights = qweights[y * cols + x];
          for (int i = 0; i < 8; i++) { // Iterate each 4-bit weight in the uint32_t
            int shift = reverse_awq_order()[i] * 4;
            uint32_t iweight = (iweights >> shift) & 0xF;
            uint32_t izero = (izeros >> shift) & 0xF;
            fp16_t scale = scales[y / group_size * cols * 8 + x * 8 + i];
            fp16_t dequantized_value = fp16_t(int(iweight) - int(izero)) * scale;
            dequantized[y * cols * 8 + x * 8 + i] = dequantized_value;
          }
        }
      }
    }
    ```
  ]
]

#pagebreak()

=== Absmax Dequantization Kernel
- The weight tensor format of absmax is simply one $"fp32"$ absolute maximum value per row, followed by the quantized weight matrix in row-major format.
  - It is the simplest weight format, typically used in papers about activation quantization @dettmers2022llmint8 @xiao2023smoothquant.
  - *In deployment, these activation quantization methods works with different weight quantization methods, and the weight format is not necessarily absmax*.
- Here we present the pseudocode of absmax dequantization kernel:
#figure(
  caption: "Absmax Weight Dequantization Kernel Pseudocode",
)[
  #text(size: 8pt)[
    ```cpp
    /**
     * Dequantize an absmax quantized weight matrix. *Non-transposed format.*
     *
     * @param qweight: shape (rows, cols), each int8_t contains quantized values
     * @param absmax: shape (rows,), fp32 absolute maximum values for each row
     * @param rows: number of rows in the weight matrix
     * @param cols: number of columns in the weight matrix
     * @param dequantized: shape (rows, cols), output dequantized weights
     */
    inline void absmax_weight_dequantize(const int8_t qweight[], const fp32_t absmax[], int rows, int cols, fp16_t dequantized[]) {
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          dequantized[i * cols + j] = fp16_t(fp32_t(qweight[i * cols + j]) * absmax[i] / 127.0f);
        }
      }
    }
    ```
  ]
]

#pagebreak()

=== QLoRA Dequantization Kernel
<qlora-dequantization-kernel>
- The weight tensor format of QLoRA @dettmers2023qlora is simply group-wise 4-bit quantized absmax format.
- A group can go across multiple rows, they only need to be siblings in memory.
- Elements in the weight tensor are typed $"nf4"$ or $"fp4_e2m1"$.
- Here we present the pseudocode of QLoRA dequantization kernel:
#figure(
  caption: "QLoRA Dequantization Kernel Pseudocode",
)[
  #text(size: 8pt)[
    ```cpp
    /**
     * Dequantize a QLoRA quantized weight matrix. *Non-transposed format.*
     *
     * @param qweight: shape (num / 2,), each uint8_t contains two 4-bit quantized values
     * @param absmax: shape (num / group_size,), fp16 absolute maximum values per
     * @param group_size: number of values per absmax
     * @param num: total number of values
     * @param dequantized: shape (num,), output dequantized weights
     * @param nf4: whether the quantized values are in NormalFloat4 format
     */
    inline void qlora_dequantize_kernel(const uint8_t qweight[], const fp16_t absmax[], int group_size, int num, bool nf4, fp16_t dequantized[]) {
      for (int i = 0; i < num; i++) {
        uint8_t iv = (qweight[i / 2] >> ((i & 1) * 4)) & 0x0F;
        fp64_t qv = nf4 ? nf4::lookup(iv) : fp4_e2m1::lookup(iv);
        dequantized[i] = fp16_t(qv * fp64_t(absmax[i / group_size]));
      }
    }
    ```
  ]
]

#pagebreak()

=== NVFP4 Dequantization Kernel

<nvfp4-dequantization-kernel>
- The NVFP4 weight format used here is a *linear block-scale* format (no swizzle / no permutation).
- It consists of two separate buffers:
  - `weight_packed`: packed 4-bit weight codes in row-major order.
  - `weight_scale_fp8`: per-row scale atoms, stored as FP8-E4M3 bytes.
  - A global scale scalar `global_scale_weight` is also used during quantization.
- Nibble order is lower $4$ bits for first $8$ weights, higher $4$ bits for next $8$ weights in a byte. Every $8$ weights is packed into one $"uint32"$.
- Each scale atom corresponds to $16$ weights.
- If the number of columns is not a multiple of $16$, $0$ is used to pad the remaining weights in the last group.

#pagebreak()

- Here we present the pseudocode of NVFP4 dequantization kernel.
#figure(
  caption: "NVFP4 Dequantization Kernel Pseudocode",
)[
  #columns(2)[
    #text(size: 8pt)[
      ```cpp
      /**
       * Dequantize NVFP4 weights (linear block-scale, not swizzled).
       *
       * @param weight_packed: shape (m, k/2), uint8, two 4-bit codes per byte
       * @param weight_scale_fp8: shape (rounded_m, rounded_k), uint8, FP8-E4M3 scale atoms
       * @param global_scale_weight: fp32 scalar used during quantization
       * @param m, k: original matrix shape (k must be even)
       * @param out: shape (m, k), fp32 output, row-major
       */
      inline void nvfp4_dequantize(const uint8_t* weight_packed,
                                  const uint8_t* weight_scale_fp8,
                                  fp32_t global_scale_weight,
                                  int m, int k, float* out) {
        int packed_k = k / 2;
        int sf_per_row = (k + 16 - 1) / 16;          // ceil(k/16)
        int rounded_k = ((sf_per_row + 4 - 1) / 4) * 4;

        static const fp32_t lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
      ```
    ]
    #colbreak()
    #text(size: 8pt)[
      ```cpp
        for (int i = 0; i < m; ++i) {
          const uint8_t* row_packed = weight_packed + size_t(i) * packed_k;
          const uint8_t* row_scales = weight_scale_fp8 + size_t(i) * rounded_k;

          for (int j = 0; j < packed_k; ++j) {
            uint8_t b = row_packed[j];
            uint8_t n0 = b & 0x0F;          // for t = 2*j
            uint8_t n1 = (b >> 4) & 0x0F;   // for t = 2*j + 1

            uint8_t ns[2] = {n0, n1};
            for (int s = 0; s < 2; ++s) {
              int t = 2 * j + s;
              if (t >= k) continue;

              int g = t / 16;
              if (g >= rounded_k) { out[size_t(i) * k + t] = 0.0f; continue; }

              bool sign = (ns[s] & 0x8) != 0;
              int idx = ns[s] & 0x7;
              fp32_t v = sign ? -lut[idx] : lut[idx];

              // FP8(E4M3) -> fp32 decode (implementation-specific)
              fp32_t sf = fp8_e4m3_to_fp32(row_scales[g]);
              out[size_t(i) * k + t] = v * (sf / global_scale_weight);
            }
          }
        }
      }
      ```
    ]
  ]
]

#pagebreak()

=== GGML llama.cpp Series Dequantization Kernels
- In the llama.cpp @llamacpp, a quantized model store their weights in different formats.
  - Each tensor has its own `type` field, indicating the compressed format.
- These format can be classified into several categories:
  - `I`, `F` and `BF` series: Just unquantized intgers or floats. e.g. `F16` is fp16.
  - `Q` series: Group-wise linearly quantized weights, with a scale and with or without a zero point. e.g. `Q4_0`, `Q5_1`, `Q8_0`.
  - `Q_K` series: Q Series variants with scales quantized to a higer bit-width than quants. e.g. `Q4_K`, `Q5_K`.
  - `TQ` series: Tenery or quaternary quantized weights. e.g. `TQ1` (Tenery), `TQ2` (Quaternary).
  - `IQ` series: Lookup-table vector-quantized weights, use suffix to indicate the bit-width when the integer part of the bit-width is same. (`xxs` < `xs` < `s` < `m`). e.g. `IQ1_xs`, `IQ2_s`.
- Different from vLLM, llama.cpp and ollama is optimized for CPU inference, thus they use a memory layout that is unfriendly for GPU kernels.

#pagebreak()

==== GGML Memory layout

- Blocks: The weight matrix is divided into blocks, each block can be dequantized independently with its own scale and zero point inside the block instead of a separate scale/zero tensor.
  - In some formats, headers appear at the end of each block instead of the beginning, but we still refer to them as block headers for simplicity.
- Group: A block has multiple elements, and this is the group concept used in other quantization methods, which means one group has one scale/zero point.
- The memory layout is shown below, and the matrix is stored in row-major format:

#figure(
  caption: "GGML Memory Layout for Quantized Weights",
)[
  #table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    stroke: none,
    inset: 6pt,
    table.vline(),
    [],
    $[w_0]_"q"$,
    table.vline(end: 1),
    $[w_1]_"q"$,
    table.vline(end: 1),
    $...$,
    $[w_g]_"q"$,
    table.vline(end: 1),
    $...$,
    $...$,
    [],
    $[w_(b g)]_"q"$,
    table.vline(end: 1),
    $...$,
    $...$,
    $...$,
    [Header],
    table.vline(end: 2),
    table.cell(colspan: 3)[Group 0],
    table.vline(),
    table.cell(colspan: 2)[Group 1],
    table.vline(),
    $...$,
    [Header],
    table.vline(end: 2),
    table.cell(colspan: 2)[Group 0],
    table.vline(end: 2),
    $...$,
    $...$,
    table.cell(colspan: 7)[Block 0], table.vline(end: 3), table.cell(colspan: 4)[Block 1], table.vline(end: 3), $...$,
    table.cell(colspan: 12)[Tensor], table.vline(),
  )
]

#pagebreak()

==== `Q` Series Dequantization Kernel
- Each group has 32 elements, and each block has only 1 group.
- Suffix `_0` means no zero point $m eq.delta 0$, `_1` means with zero point $m$.
- At the header of each block, there is one scale $d_"fp16"$ and optionally one zero point $m_"fp16"$.
- GGML decompose the bit-width to $l$ and $s$, where $"bitwidth" = l + s$ and $l = 2^n, h = 2^m, n > m$ for some integers $n, m$ if the bit-width is not a power of two.
- GGML first store the $l$-bit lower bits of all quantized values, then store the $s$-bit higher bits of all quantized values.
- GGML use same trick of reordering elements as AWQ when packing quantized values into bytes.
#figure(
  caption: "Example of block layout",
)[
  #columns(2)[
    ```cpp
    struct BlockQ4_1 {
      fp16_t d;
      fp16_t m;
      uint8_t  ql[32 / 2];
    };
    ```
    #colbreak()
    ```cpp
    struct BlockQ5_0 {
      fp16_t d;
      uint8_t  ql[32 / 2];
      uint8_t  qh[32 / 8];
    };
    ```
  ]
]
#pagebreak()

- Here we present the pseudocode of GGML Q Series dequantization kernel:
#figure(
  caption: "GGML Q Series Dequantization Kernel Pseudocode",
)[
  #text(size: 8pt)[
    ```cpp
    /**
     * Dequantize a row in GGML Q Series quantized weight matrix. *Non-transposed format.*
     * @param x: pointer to the quantized weight matrix
     * @param k: number of columns in the weight matrix
     * @param dequantized: shape (rows, k), output dequantized weights
     * @param bits: bit-width of quantization (e.g., 4, 5, 8)
     * @param has_zero_point: whether the quantization has zero point
     */
    inline void ggml_dequantize_row_q(const void *x, int64_t k, fp32_t *dequantized, int bits, bool has_zero_point) {
      constexpr int lowbits = bits & -bits;
      constexpr int highbits = bits - lowbits;
      int blocks = k / 32;
      int block_size = (1 + has_zero_point) * sizeof(fp16_t) + 32 * bits / 8;
      for (int i = 0; i < blocks; i++) {
        const uint8_t *block_ptr = (const uint8_t *)(x) + i * block_size;
        const fp16_t *header_ptr = (const fp16_t *)(block_ptr);
        fp32_t d = fp32_t(header_ptr[0]);
        fp32_t m = has_zero_point ? fp32_t(header_ptr[1]) : 0.0;
        const uint8_t *highbits_ptr = block_ptr + (1 + has_zero_point) * sizeof(fp16_t);
        const uint8_t *lowbits_ptr = highbits_ptr + 32 * highbits / 8;
        for (int j = 0; j < 32; j++) {
          int quant = int(lowbits_ptr[j % (32 / lowbits)]) >> (j / (32 / lowbits) * lowbits) &
              ((1 << lowbits) - 1);
          if (highbits > 0) {
            quant |= (int(highbits_ptr[j % (32 / highbits)]) >> (j / (32 / highbits) * highbits) &
                      ((1 << highbits) - 1)) << lowbits;
          }
          dequantized[i * 32 + j] = fp32_t(d * (quant - (1 << (Bits - 1))) + m);
        }
      }
    }
    ```
  ]
]

#pagebreak()

==== GGML `Q_K` Series Dequantization Kernel
- 16 or 8 groups in a block, each group has 16 or 32 elements respectively.
- The header may appear at the beginning or the end of each block.
- The quantization may or may not have zero point.
- We are going to show the definition of all $5$ `Q_K` formats used in GGML in next page. Remember:
  - The format use zero point if and only if there is a member variable named `m` in the block struct.
  - We denote bit from the lower $i$-th to the higher $j$-th (exclusive) of $dot$ as $[dot]_(i:j)$. (e.g. $11_(1:3) = 1_"int2"$)
  - The layout of `scales` of `Q2_K` (with minimum value $m$) is:
$
  ([m_0]_"uint4", [d_0]_"uint4"), ([m_1]_"uint4", [d_1]_"uint4"), ([m_2]_"uint4", [d_2]_"uint4"), ([m_3]_"uint4", [d_3]_"uint4"), \
  ([m_4]_"uint4", [d_4]_"uint4"), ([m_5]_"uint4", [d_5]_"uint4"), ([m_6]_"uint4", [d_6]_"uint4"), ([m_7]_"uint4", [d_7]_"uint4"), \
  ([m_8]_"uint4", [d_8]_"uint4"), ([m_9]_"uint4", [d_9]_"uint4"), ([m_10]_"uint4", [d_10]_"uint4"), ([m_11]_"uint4", [d_11]_"uint4"), \
  ([m_12]_"uint4", [d_12]_"uint4"), ([m_13]_"uint4", [d_13]_"uint4"), ([m_14]_"uint4", [d_14]_"uint4"), ([m_15]_"uint4", [d_15]_"uint4")
$

- The layout of `scales` of `Q3_K` (without minimum value $m$) is:
$
  ([[d_8]_"uint6"]_(0:4), [[d_0]_"uint6"]_(0:4)), ([[d_9]_"uint6"]_(0:4), [[d_1]_"uint6"]_(0:4)), ([[d_10]_"uint6"]_(0:4), [[d_2]_"uint6"]_(0:4)), \
  ([[d_11]_"uint6"]_(0:4), [[d_3]_"uint6"]_(0:4)), ([[d_12]_"uint6"]_(0:4), [[d_4]_"uint6"]_(0:4)), ([[d_13]_"uint6"]_(0:4), [[d_5]_"uint6"]_(0:4)),\
  ([[d_14]_"uint6"]_(0:4), [[d_6]_"uint6"]_(0:4)), ([[d_15]_"uint6"]_(0:4), [[d_7]_"uint6"]_(0:4)), \
  ([[d_12]_"uint6"]_(4:6), [[d_8]_"uint6"]_(4:6), [[d_4]_"uint6"]_(4:6), [[d_0]_"uint6"]_(4:6)), \
  ([[d_13]_"uint6"]_(4:6), [[d_9]_"uint6"]_(4:6), [[d_5]_"uint6"]_(4:6), [[d_1]_"uint6"]_(4:6)), \
  ([[d_14]_"uint6"]_(4:6), [[d_10]_"uint6"]_(4:6), [[d_6]_"uint6"]_(4:6), [[d_2]_"uint6"]_(4:6)), \
  ([[d_15]_"uint6"]_(4:6), [[d_11]_"uint6"]_(4:6), [[d_7]_"uint6"]_(4:6), [[d_3]_"uint6"]_(4:6))
$

- The layout of `scales` of `Q4_K` and `Q5_K` is:
$
  ([[d_4]_"uint6"]_(4:6), [d_0]_"uint6"), ([[d_5]_"uint6"]_(4:6), [d_1]_"uint6"), ([[d_6]_"uint6"]_(4:6), [d_2]_"uint6"), ([[d_7]_"uint6"]_(4:6), [d_3]_"uint6"), \
  ([[m_4]_"uint6"]_(4:6), [m_0]_"uint6"), ([[m_5]_"uint6"]_(4:6), [m_1]_"uint6"), ([[m_6]_"uint6"]_(4:6), [m_2]_"uint6"), ([[m_7]_"uint6"]_(4:6), [m_3]_"uint6"), \
  ([[m_4]_"uint6"]_(0:4), [[d_4]_"uint6"]_(0:4)), ([[m_5]_"uint6"]_(0:4), [[d_5]_"uint6"]_(0:4)),\
  ([[m_6]_"uint6"]_(0:4), [[d_6]_"uint6"]_(0:4)), ([[m_7]_"uint6"]_(0:4), [[d_7]_"uint6"]_(0:4))
$
- The layout of `scales` of `Q6_K` (without minimum value $m$) is:#footnote[Elements stored contiguously.]
$
  d_"int8"
$

#pagebreak()

#figure(
  caption: "Definition of BlockQ2_K, BlockQ3_K and BlockQ4_K",
)[
  #columns(3)[
    ```cpp
    struct BlockQ2_K {
      uint8_t scales[16];
      uint8_t ql[64];
      fp16_t d_block;
      fp16_t m_block;
    };

    struct BlockQ3_K {
      uint8_t qh[32];
      uint8_t ql[64];
      uint8_t scales[12];
      fp16_t d_block;
    };
    ```
    #colbreak()
    ```cpp
    struct BlockQ4_K {
      fp16_t d_block;
      fp16_t m_block;
      uint8_t scales[12];
      uint8_t ql[128];
    };

    struct BlockQ5_K {
      fp16_t d_block;
      fp16_t m_block;
      uint8_t scales[12];
      uint8_t qh[32];
      uint8_t ql[128];
    };
    ```
    #colbreak()
    ```cpp
    struct BlockQ6_K {
      uint8_t ql[64];
      uint8_t qh[32];
      uint8_t scales[16];
      fp16_t d_block;
    };
    ```
  ]
]

#pagebreak()

- The formula of $w_q$ for all `Q_K` formats with a zero point (minimum value) in a block is:
$
  [w_i]_"q" = d_"block" [d_(floor(i / G))]_"uint" [w_i]_"uint" - m_"block" [m_(floor(i / G))]_"uint"
$
- The formula of $w_q$ for `Q3_K` in a block without a zero point is:#footnote[Notice that some integers are interpreted as signed integers while others are unsigned here.]
$
  [w_i]_"q" = d_"block" ([d_(floor(i / 16))]_"uint6" - 32) [w_i]_"int3"
$
- The formula of $w_q$ for `Q6_K` in a block without a zero point is:#footnote[Notice that some integers are interpreted as signed integers while others are unsigned here.]
$
  [w_i]_"q" = d_"block" [d_(floor(i / 8))]_"int8" ([w_i]_"uint6" - 32)
$

#pagebreak()

- For `Q2_K`, `Q3_K`, every $128$ `ql` or `qh` elements are stored in a contiguous memory region. For example, in `Q2_K`'s `ql` array:
$
  ([w_96]_"uint2", [w_64]_"uint2", [w_32]_"uint2", [w_0]_"uint2"), & ..., & ([w_127]_"uint2", [w_95]_"uint2", [w_63]_"uint2", [w_31]_"uint2"), \
  ([w_224]_"uint2", [w_192]_"uint2", [w_160]_"uint2", [w_128]_"uint2"), & ..., & ([w_255]_"uint2", [w_223]_"uint2", [w_191]_"uint2", [w_159]_"uint2")
$
- For `Q4_K`, `Q5_K`, `Q6_K`, every $32$ `ql` or `qh` elements are stored in a contiguous memory region. For example, in `Q5_K`'s `ql` array:
$
  ([w_16]_"uint4", [w_0]_"uint4"), & ..., & ([w_31]_"uint4", [w_15]_"uint4"), & ([w_48]_"uint4", [w_32]_"uint4"), & ..., & ([w_63]_"uint4", [w_47]_"uint4"), \
  ([w_80]_"uint4", [w_64]_"uint4"), & ..., & ([w_95]_"uint4", [w_79]_"uint4"), & ([w_112]_"uint4", [w_96]_"uint4"), & ..., & ([w_127]_"uint4", [w_111]_"uint4"), \
  ([w_144]_"uint4", [w_128]_"uint4"), & ..., & ([w_159]_"uint4", [w_143]_"uint4"), & ([w_176]_"uint4", [w_160]_"uint4"), & ..., & ([w_191]_"uint4", [w_175]_"uint4"), \
  ([w_208]_"uint4", [w_192]_"uint4"), & ..., & ([w_223]_"uint4", [w_207]_"uint4"), & ([w_240]_"uint4", [w_224]_"uint4"), & ..., & ([w_255]_"uint4", [w_239]_"uint4")
$

#pagebreak()

- Here we present the pseudocode of GGML Q_K Series dequantization kernel:
#figure(
  caption: "GGML Q_K Series Dequantization Kernel Pseudocode (To be continued)",
)[
  #columns(2)[
    #text(size: 8pt)[
      ```cpp
      /**
       * Dequantize a row in GGML Q_K quantized weight matrix. *Non-transposed
       * format.*
       * @param x: pointer to the quantized weight matrix
       * @param k: number of columns in the weight matrix
       * @param dequantized: shape (rows, k), output dequantized weights
       * @param bits: bit-width of quantization (e.g., 2, 3, 4, 5, 6)
       */
      inline int ggml_dequantize_row_q_k(const void *x, uint64_t k,
                                        fp32_t *dequantized, int bits) {
        uint64_t blocks = k / 256;
        size_t block_size =
            (size_t[]){0,
                      0,
                      sizeof(uint8_t) * (16 + 64) + sizeof(fp16_t) * 2,
                      sizeof(uint8_t) * (32 + 64 + 12) + sizeof(fp16_t),
                      sizeof(fp16_t) * 2 + sizeof(uint8_t) * (12 + 128),
                      sizeof(fp16_t) * 2 + sizeof(uint8_t) * (12 + 32 + 128),
                      sizeof(uint8_t) * (64 + 32 + 16) + sizeof(fp16_t)}[bits];
        for (uint64_t i = 0; i < blocks; i++) {
          const uint8_t *block_ptr = (const uint8_t *)(x) + i * block_size;
          const uint8_t *elements_ptr;
          const uint8_t *scales_ptr;
          fp32_t d_block;
          fp32_t m_block;
          int group_size;
          fp32_t d_q[16];
          fp32_t m_q[16];
      ```
    ]
    #text(size: 8pt)[
      ```cpp
        switch (bits) {
          case 2:
            elements_ptr = block_ptr + sizeof(uint8_t) * 16;
            scales_ptr = block_ptr;
            d_block =
                fp32_t((const fp16_t *)(block_ptr + sizeof(uint8_t) * (16 + 64))[0]);
            m_block =
                fp32_t((const fp16_t *)(block_ptr + sizeof(uint8_t) * (16 + 64))[1]);
            group_size = 16;
            for (int j = 0; j < 16; j++) {
              uint8_t scale_byte = scales_ptr[j];
              d_q[j] = fp32_t(scale_byte & 0x0F) * d_block;
              m_q[j] = fp32_t(scale_byte >> 4) * m_block;
            }
          case 3:
            elements_ptr = block_ptr;
            scales_ptr = block_ptr + sizeof(uint8_t) * (32 + 64);
            d_block = fp32_t(
                (const fp16_t *)(block_ptr + sizeof(uint8_t) * (32 + 64 + 12))[0]);
            m_block = 0.0f;
            group_size = 16;
            for (int j = 0; j < 16; j++) {
              d_q[j] = fp32_t((((scales_ptr[j % 8] >> (j / 2)) & 0x0F) |
                              ((scales_ptr[8 + j % 4] >> (j / 4)) << 4)) -
                              32) *
                      d_block;
              m_q[j] = 0.0f;
            }
      ```
    ]
  ]
]
#pagebreak()
#figure(
  caption: "GGML Q_K Series Dequantization Kernel Pseudocode (Continued)",
)[
  #columns(2)[
    #text(size: 8pt)[
      ```cpp
          case 4:
          case 5:
            elements_ptr = block_ptr + sizeof(fp16_t) * 2 + sizeof(uint8_t) * 12;
            scales_ptr = block_ptr + sizeof(fp16_t) * 2;
            d_block = fp32_t((const fp16_t *)(block_ptr)[0]);
            m_block = fp32_t((const fp16_t *)(block_ptr)[1]);
            group_size = 32;
            for (int j = 0; j < 4; j++) {
              d_q[j] = fp32_t(scales_ptr[j] & 0x3F) * d_block;
              m_q[j] = fp32_t(scales_ptr[4 + j] & 0x3F) * m_block;
            }
            for (int j = 0; j < 4; j++) {
              d_q[4 + j] =
                  fp32_t((scales_ptr[8 + j] & 0x0F) | (scales_ptr[j] >> 6) << 4) *
                  d_block;
              m_q[4 + j] =
                  fp32_t((scales_ptr[8 + j] >> 4) | (scales_ptr[4 + j] >> 6) << 4) *
                  m_block;
            }
          case 6:
            elements_ptr = block_ptr;
            scales_ptr = block_ptr + sizeof(uint8_t) * (64 + 32);
            d_block =
                fp32_t((const fp16_t *)(block_ptr + sizeof(uint8_t) * (64 + 32))[0]);
            m_block = 0.0f;
            group_size = 16;
            for (int j = 0; j < 16; j++) {
              d_q[j] =
                  fp32_t(reinterpret_cast<const int8_t *>(scales_ptr)[j]) * d_block;
              m_q[j] = -32 * d_q[j];
            }
          default:
            terminate();
          }
          int groups = 256 / group_size;
      ```
    ]
    #colbreak()
    #text(size: 8pt)[
      ```cpp
          for (int j = 0; j < groups; j++) {
            for (int l = 0; l < group_size; l++) {
              const uint8_t *highbits_arr = elements_ptr;
              int idx = j * group_size + l;
              int length = groups * group_size;
              int highbits = bits & -bits;
              int lowbits = bits - highbits;
              size_t high_total_byte_count = length * highbits / 8;
              const uint8_t *lowbits_ptr = highbits_arr + high_total_byte_count;
              int idx_mod = bits <= 3 ? 128 : 32;
              int idx_rem = idx % idx_mod;
              highbits_arr += idx / idx_mod * highbits / 8;
              lowbits_ptr += idx / idx_mod * lowbits / 8;
              size_t high_byte_count = idx_mod * highbits / 8;
              int highbit_value = int(highbits_arr[idx_mod % high_byte_count]) >>
                                      (idx_rem / high_byte_count * highbits) &
                                  ((1 << highbits) - 1);
              size_t low_byte_count = idx_mod * lowbits / 8;
              int lowbit_value = int(lowbits_ptr[idx_mod % low_byte_count]) >>
                                    (idx_rem / low_byte_count * lowbits) &
                                ((1 << lowbits) - 1);
              int quant = lowbit_value | (highbit_value << lowbits);
              if (bits == 3 && quant >= 4) {
                quant -= 8;
              }
              dequantized[i * 256 + j * group_size + l] =
                  d_q[j] * fp32_t(quant) - m_q[j];
            }
          }
        }
      }
      ```
    ]
  ]
]

#pagebreak()

==== GGML `TQ` Series Dequantization Kernel
- Each block has only one group with 256 elements, and scale $d$ is stored after the elements.
- `TQ1` is ternary quantization with values in $V = {-1, 0, 1}$. This is actually BitNet @wang2025bitnetv2native4bit.
  - In general, it's possible to store $5$ ternary values in one byte, we will show how to do it later. For simplicity, we denote the bytes storing some ternary values as $t(v_0, ..., v_4)$.
  - The layout of a group is: ($160$ in $32$ bytes, $80$ in $16$ bytes, $16$ in $4$ bytes with $4$ padded $0$)
$
  t([w_128]_V, [w_96]_V, [w_64]_V, [w_32]_V, [w_0]_V), & ..., & t([w_159]_V, [w_127]_V, [w_95]_V, [w_63]_V, [w_31]_V), \
  t([w_224]_V, [w_208]_V, [w_192]_V, [w_176]_V, [w_160]_V), & ..., & t([w_239]_V, [w_223]_V, [w_207]_V, [w_191]_V, [w_175]_V), \
  t(0, [w_252]_V, [w_248]_V, [w_244]_V, [w_240]_V), & ... & t(0, [w_255]_V, [w_251]_V, [w_247]_V, [w_243]_V)
$
- `TQ2` is quaternary quantization with values in $U = {-1, 0, 1, 2}$.
  - The layout of a group is same as the `QK_6`, `QK_5`, `QK_4` formats shown before, that is, every $32$ elements are stored in a contiguous memory region.

#pagebreak()

- Here we show that storing $5$ ternary values in one byte is possible:
$
  t(v_0, v_1, v_2, v_3, v_4) = ceil((3^(-1)(v_4 + 1) + 3^(-2)(v_3 + 1) + 3^(-3)(v_2 + 1) + 3^(-4)(v_1 + 1) + 3^(-5)(v_0 + 1)) times 256) \
$
- And the inverse function is:
$
  t^(-1)(u)_i = floor((3((3^(4 - i) u) mod 256)) / 256) - 1 "for" i = 0, 1, 2, 3, 4
$
- This basically first interpreted $0.a_4 a_3 a_2 a_1 a_0$ as a base-$3$ number, then store it as an fixed point number in base-$2$.
- $v -> 3^0 v_0 + 3^1 v_1 + 3^2 v_2 + 3^3 v_3 + 3^4 v_4$ is not used because it will require to modulo $3$ in its inverse function, much slower than modulo $256$.

#pagebreak()

- Here we present the pseudocode of GGML TQ Series dequantization kernel:
#figure(
  caption: "GGML TQ Series Dequantization Kernel Pseudocode",
)[
  #columns(2)[
    #text(size: 8pt)[
      ```cpp
      /**
       * Dequantize a row in GGML TQ1 quantized weight matrix. *Non-transposed
       * format.*
       * @param x: pointer to the quantized weight matrix
       * @param k: number of columns in the weight matrix
       * @param dequantized: shape (rows, k), output dequantized weights
       */
      inline void ggml_dequantize_row_tq1(const void *x, int64_t k,
                                          fp32_t *dequantized) {
        constexpr int p3[6] = {1, 3, 9, 27, 81, 243};
        uint64_t blocks = k / 256;
        size_t block_size = sizeof(uint8_t) * (32 + 16 + 4) + sizeof(fp16_t);
        for (uint64_t i = 0; i < blocks; i++) {
          const uint8_t *ptr = (const uint8_t *)(x) + i * block_size;
          fp32_t d = fp32_t(((const fp16_t *)(ptr + 32 + 16 + 4))[0]);
          for (int j = 0; j < 160; j++) {
            int q = ((3 * (int(ptr[j % 32]) * p3[j / 32] & 0xff)) >> 8) - 1;
            dequantized[i * 256 + j] = d * fp32_t(q);
          }
          ptr += 32;
          for (int j = 0; j < 80; j++) {
            int q = ((3 * (int(ptr[j % 16]) * p3[j / 16] & 0xff)) >> 8) - 1;
            dequantized[i * 256 + 160 + j] = d * fp32_t(q);
          }
          ptr += 16;
          for (int j = 0; j < 16; j++) {
            int q = ((3 * (int(ptr[j % 4]) * p3[j / 4] & 0xff)) >> 8) - 1;
            dequantized[i * 256 + 160 + 80 + j] = d * fp32_t(q);
          }
        }
      }
      ```
    ]
    #colbreak()
    #text(size: 8pt)[
      ```cpp
      /**
       * Dequantize a row in GGML TQ2 quantized weight matrix. *Non-transposed
       * format.*
       * @param x: pointer to the quantized weight matrix
       * @param k: number of columns in the weight matrix
       * @param dequantized: shape (rows, k), output dequantized weights
       */
      inline void ggml_dequantize_row_tq2(const void *x, int64_t k,
                                          fp32_t *dequantized) {
        int64_t blocks = k / 256;
        size_t block_size = sizeof(uint8_t) * 64 + sizeof(fp16_t);
        for (int64_t i = 0; i < blocks; i++) {
          const uint8_t *ptr = (const uint8_t *)(x) + i * block_size;
          fp32_t d = fp32_t(((const fp16_t *)(ptr + sizeof(uint8_t) * 64))[0]);
          for (int j = 0; j < 8; j++) {
            for (int l = 0; l < 32; l++) {
              int q = ((int(ptr[j * 8 + l % 8]) >> (l / 8 * 2)) & 0x03) - 1;
              dequantized[i * 256 + j * 32 + l] = d * fp32_t(q);
            }
          }
        }
      }
      ```
    ]
  ]
]

#pagebreak()

==== GGML `IQ` Series Dequantization Kernel
- As described before, `IQ` series are heavily rely on lookup-tables.
- Since there are $9$ different variant in this series, we only show the most aggressive `IQ1_S` here.
===== `IQ1_S` Dequantization Kernel
- $1.5625$-bit quantization based on vector quantization based on codebook $f_"iq1s": "int11" -> "int8"^8$.
- Each block has $8$ groups, $256$ elements in total.
- Each group has $4$ vectors, $32$ elements in total.
- Each vector has $8$ elements, containing contiguous elements.
- Dequantization formula:
$
  [w_(8i:8i + 8)]_"q" = [d_"block"]_"fp16" (2 [h_"group"]_"uint3" + 1) (f_"iq1s" ([v_i]_"int11") + [delta_"group"]_{-1/8, 1/8}) \
  "where" [delta_"group"]_{-1/8, 1/8} = cases(-1/8 & "if" [delta_"group"]_"int1" = -1, 1/8 & "if" [delta_"group"]_"int1" = 0) \
$

#pagebreak()

#figure(
  caption: [Explanation of bit allocation `IQ1_S`],
)[
  #canvas({
    import draw: *
    dot-matrix((0pt, 0pt), cols: 43, rows: 1, space: 12pt, prefix: "w")
    rect-around(
      ..elements-in-dot-matrix("w", rows: 1, cols: 32, x: 0, y: 0),
      stroke: gray,
      padding: 5pt,
      radius: 2pt,
      name: "group",
    )
    rect-around(
      ..elements-in-dot-matrix("w", rows: 1, cols: 8, x: 0, y: 0),
      stroke: green,
      padding: 3pt,
      radius: 2pt,
      name: "vector-g",
    )
    rect-around(
      ..elements-in-dot-matrix("w", rows: 1, cols: 8, x: 0, y: 8),
      stroke: purple,
      padding: 3pt,
      radius: 2pt,
      name: "vector-p",
    )
    rect-around(
      ..elements-in-dot-matrix("w", rows: 1, cols: 8, x: 0, y: 16),
      stroke: blue,
      padding: 3pt,
      radius: 2pt,
      name: "vector-b",
    )
    rect-around(
      ..elements-in-dot-matrix("w", rows: 1, cols: 8, x: 0, y: 24),
      stroke: red,
      padding: 3pt,
      radius: 2pt,
      name: "vector-r",
    )
    content(
      (0em - 2em, 0em),
      anchor: "mid-east",
    )[
      One Group
    ]
    content(
      (0em, -2em),
      anchor: "mid-east",
      name: "group-level",
    )[
      $[delta_"group"]_"int1"$ $[h_"group"]_"uint3"$
    ]
    set-style(mark: (end: ">"))
    line((0em - 6pt, 0em), "group-level", stroke: gray)
    content(
      (0em + 12pt * 8, -2em),
      anchor: "mid-east",
      name: "vector-g-level",
    )[
      $[v_0]_"int11"$
    ]
    set-style(mark: (end: ">"))
    line("vector-g", "vector-g-level", stroke: green)
    content(
      (0em + 12pt * 16, -2em),
      anchor: "mid-east",
      name: "vector-p-level",
    )[
      $[v_1]_"int11"$
    ]
    set-style(mark: (end: ">"))
    line("vector-p", "vector-p-level", stroke: purple)
    content(
      (0em + 12pt * 24, -2em),
      anchor: "mid-east",
      name: "vector-b-level",
    )[
      $[v_2]_"int11"$
    ]
    set-style(mark: (end: ">"))
    line("vector-b", "vector-b-level", stroke: blue)
    content(
      (0em + 12pt * 32, -2em),
      anchor: "mid-east",
      name: "vector-r-level",
    )[
      $[v_3]_"int11"$
    ]
    set-style(mark: (end: ">"))
    line("vector-r", "vector-r-level", stroke: red)
    set-style(mark: (end: ">", start: ">"))
    line(
      (rel: (0, -2em), to: "group-level.west"),
      (rel: (0, -2em), to: "vector-r-level.east"),
      stroke: black,
      name: "group-total",
    )
    content(
      (rel: (0, -1em), to: "group-total"),
      anchor: "mid",
    )[
      $48$ bits total in a group
    ]

    for i in range(0, 8) {
      rect(
        (3em * i, -7em),
        (3em * (i + 1) - 5pt, -8em),
        name: "group-" + str(i),
      )
    }
    content(
      (rel: (-2em, 0em), to: "group-0.south-west"),
      anchor: "south-east",
      name: "block-level",
    )[
      $[d_"block"]_"fp16"$
    ]
    line(
      (rel: (0, -2em), to: "block-level.west"),
      (rel: (0, -2em), to: "group-7.east"),
      stroke: black,
      name: "block-total",
    )
    content(
      (rel: (0, -1em), to: "block-total"),
      anchor: "north",
      name: "block-bits",
    )[
      $16 + 48 times 8 = 400$ bits total in a block, $400 / 256 = 1.5675$ bits on average
    ]
    content(
      (rel: (0, -1em), to: "block-bits"),
      anchor: "north",
    )[
      Block Memory Layout:
      - `fp16 d`: $[d_"block"]_"fp16"$ for the block
      - `uint8[32] qs`: $[[v_i]_"int11"]_(0:8)$ for each vector
      - `uint16[8] qh`: Packed $[delta_"group"]_"int1", [h_"group"]_"uint3", [[v_3]_"int11"]_(8:11), ..., [[v_0]_"int11"]_(8:11)$ for each group
    ]
    set-style(mark: (start: none, end: none))
    line(
      "group-0.north-west",
      "group.south-west",
      stroke: (
        paint: gray,
        dash: (4pt, 2pt),
        cap: "round",
      ),
    )
    line(
      "group-0.north-west",
      "group.south-west",
      stroke: (
        paint: gray,
        dash: (4pt, 2pt),
        cap: "round",
      ),
    )
    line(
      "group-0.north-east",
      "group.south-east",
      stroke: (
        paint: gray,
        dash: (4pt, 2pt),
        cap: "round",
      ),
    )
  })
]

#pagebreak()

- Here we present the pseudocode of GGML IQ1_S dequantization kernel:

#figure(
  caption: "GGML IQ1_S Dequantization Kernel Pseudocode",
)[
  #text(size: 8pt)[
    ```cpp
    /**
      * Dequantize a row in GGML IQ1 quantized weight matrix. *Non-transposed
      * format.*
      * @param x: pointer to the quantized weight matrix
      * @param k: number of columns in the weight matrix
      * @param dequantized: shape (rows, k), output dequantized weights
      * @param table: lookup table for dequantization
    */
    inline void ggml_dequantize_row_iq1_s(const void *x, int64_t k, fp32_t *dequantized,
                                          const int8_t table[1 << 11][8]) {
      int64_t blocks = k / 256;
      size_t block_size = sizeof(fp16_t) + sizeof(uint8_t) * 32 + sizeof(uint16_t) * 8;
      for (int64_t i = 0; i < blocks; i++) {
        const uint8_t *ptr = (const uint8_t *)(x) + i * block_size;
        fp32_t d = fp32_t(((const fp16_t *)(ptr))[0]);
        const uint8_t *qs = ptr + sizeof(fp16_t);
        const uint16_t *qh = (const uint16_t *)(ptr + sizeof(fp16_t) + sizeof(uint8_t) * 32);
        for (int j = 0; j < 8; j++) {
          fp32_t delta = qh[j] & 0x8000 ? -1.0 / 8 : 1.0 / 8;
          fp32_t dl = d * (2 * ((qh[j] >> 12) & 0x7) + 1);
          for (int l = 0; l < 4; l++) {
            int vec = ((qh[j] >> (l * 3)) & 0x7) << 8 | qs[j * 4 + l];
            for (int m = 0; m < 8; m++) {
              dequantized[i * 256 + j * 32 + l * 8 + m] = dl * (fp32_t(table[vec][m]) + delta);
            }
          }
        }
      }
    }
    ```
  ]
]

#pagebreak()

= References
#bibliography("references.bib", style: "ieee", title: none)
