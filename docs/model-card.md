# Model card

## Model and purpose

acclass is a classifier for shell commands and structured action descriptions, derived
from ModernBERT. It predicts nine effects. See [semantics](semantics.md) for each label.

It predicts effects, not permission. It is a fast pre-filter: if it finds no risky effect,
a caller can allow the action quickly and send everything else to a slower review. It is
not a sandbox, an authorization system or a safety guarantee.

- **Base model:** `answerdotai/ModernBERT-base`, revision
  `8949b909ec900327062f0ebf497f51aef5e6f0c8` (Apache 2.0).
- **Outputs:** nine independent sigmoid outputs on mean-pooled tokens.
- **Data:** 26,852 rows of original synthetic data. The data is not published.
- **License:** original model changes use [Apache 2.0](../LICENSE). Third-party terms are in
  the [notices](../legal/ATTRIBUTIONS.md). This repository has no model weights.

| Download | Size | Use |
| --- | ---: | --- |
| Q8_0 GGUF | 151.7 MiB | Default |
| Q5_0 GGUF | 98.2 MiB | Smaller, same speed on CPU |
| FP32 safetensors | 570.7 MiB | Full weights. Serve with `serve --quant f32`. |

Q8 and Q5 are versions of the model with 8-bit and 5-bit weights. Norms and the classifier
stay in FP32.

## Input and output

The engine accepts UTF-8 text. A shell input is the exact command string. An action input is
a JSON string in the [action input format](engine-api.md#action-input-format).

The limit is 2,048 tokens, including special tokens. The engine does not truncate or pad
inputs. It rejects inputs above the limit and inputs with the tokenizer's unknown token.

Each label has a probability and a boolean. The boolean is true when
`probability >= threshold`. The release model uses threshold 0.5 for every label. The engine
reads the thresholds from `thresholds.json`, so check them if you use another bundle.

Calibration makes a probability closer to the true rate. The model uses one global
temperature, T = 1.76, built into the classifier weights. Expected calibration error (10 bins)
changes from 0.011 to 0.005 on the calibration rows. On the evaluation rows it is 0.004 for
shell and 0.006 for action.

## Evaluation

All numbers use 3,793 held-out evaluation rows at threshold 0.5, unless noted. The 95%
intervals resample whole groups of related rows (1,000 draws), because rows in a group are
not independent. Macro F1 is the average F1 of the nine labels.

The data has two parts:

- **Base rows** are original synthetic commands and invented action descriptions, written as
  contrasting cases.
- **LLM-authored rows** are inputs that a state-of-the-art open-weights LLM generated.

Base rows are the better estimate. See [limits](#limits).

| Evaluation subset | Rows | Macro F1 (95% CI) | Micro F1 | Exact nine-label match |
| --- | ---: | --- | ---: | ---: |
| Shell, all | 2,250 | **0.940** (0.925–0.952) | 0.952 | 0.83 |
| Shell, base rows | 533 | 0.925 (0.885–0.959) | 0.932 | 0.79 |
| Shell, LLM-authored | 1,717 | 0.945 (0.933–0.956) | 0.958 | 0.85 |
| Action, all | 1,543 | **0.961** (0.948–0.971) | 0.969 | 0.86 |
| Action, base rows | 701 | 0.956 (0.941–0.987) | 0.979 | 0.90 |
| Action, LLM-authored | 842 | 0.953 (0.934–0.964) | 0.960 | 0.83 |

F1 for each label (positive rows in brackets):

| Label | Shell | Action |
| --- | --- | --- |
| `local_state_change` | 0.96 (1,092) | 0.98 (599) |
| `outbound` | 0.99 (987) | 0.99 (898) |
| `egress` | 0.92 (245) | 0.84 (240) |
| `remote_state_change` | 0.96 (374) | 0.96 (517) |
| `delegated_execution` | 0.99 (498) | 0.99 (605) |
| `exposes_secrets` | 0.90 (278) | 0.96 (106) |
| `destructive` | 0.94 (880) | 0.94 (510) |
| `host_control_change` | 0.95 (317) | 0.98 (220) |
| `fetched_or_generated_execution` | 0.87 (269) | 0.99 (217) |

Weakest results:

- Shell `fetched_or_generated_execution` on base rows: F1 0.72 (59 positives, precision 0.68:
  it marks too many rows).
- Action `exposes_secrets` on base rows: F1 0.82 (16 positives).
- Action `egress`: F1 0.84.

## Quantization

A flip is a boolean that differs from FP32 at threshold 0.5. The comparison uses all
evaluation rows, which is 34,137 decisions.

| Format | Size | Macro F1 shell / action | Flips |
| --- | ---: | --- | ---: |
| FP32 | 571 MiB | 0.940 / 0.961 | reference |
| Q8_0 (default) | 151.7 MiB | 0.941 / 0.960 | 19 |
| Q5_0 | 98.2 MiB | 0.939 / 0.960 | 85 |

## Latency

These values cover one model pass, one request at a time, without tokenizer or socket time.
They are rough values from shared machines. Your values can differ. Values are median / 95th
percentile, in ms. Apple values use this model. The others use an earlier model with the same
architecture.

| Hardware | FP32 | Q8 | Q5 |
| --- | --- | --- | --- |
| Apple M4 Pro, Metal | 8.5 / 53 | 6.5 / 50 | 6.9 / 53 |
| Apple M4 Pro, CPU, 8 threads | 43 / 228 | 25 / 131 | 24 / 130 |
| Arm Cortex-X925, CPU, 8 threads | 76 / 438 | 21 / 326 | 21 / 222 |
| AMD Ryzen 5 6650U, CPU, 6 threads | 66 / 760 | 50 / 717 | 51 / 544 |
| NVIDIA GB10, CUDA, up to 16 tokens (median) | 4.8 | 4.2 | not measured |
| NVIDIA GB10, CUDA, about 1,800 tokens (median) | 102 | 77 | not measured |

CUDA needs a build from source. After 2 seconds idle, a short command on M4 Pro takes about
25 ms on Metal and 42 ms on the CPU with Q8. Other Apple chips are not measured.

## Limits

Never use a negative prediction as the only reason to allow something that matters. Never use
a prediction as an irreversible block.

- **It does not guarantee safety.** False negatives occur. A false negative can cause a caller
  to allow a risky action without review. Keep OS isolation and review for anything that
  matters.
- **An LLM wrote the labels,** in two runs of one model family. They are not independent human
  truth.
- **Most of the data is LLM-generated.** LLM-authored evaluation rows share style with
  LLM-authored training rows, so their scores are optimistic. Base rows are the better estimate.
- **Accuracy on your own commands is not measured.**
- **We looked at the evaluation rows once before this model was final.** They are not perfectly
  untouched.
- **Action mode was trained on invented tool descriptions.** Each description states all effects
  of its tool. Real MCP or custom-tool descriptions can be wrong or incomplete. Treat action
  predictions as hints that need review.
- **The model sees only the text.** It does not see the working directory, files, environment or
  user intent. Predictions describe effects inferred from text, not observed execution.
- **The scores say nothing about resistance to adversarial inputs.**
- **Path scope is a separate heuristic.** It cannot prove that a command stays inside a folder,
  and it cannot identify every effect.

A low probability does not prove that an effect is absent. A high probability does not mean that
the action is forbidden. Keep an explicit policy for uncertain inputs and engine errors. See
[semantics](semantics.md) and [privacy](privacy.md).
