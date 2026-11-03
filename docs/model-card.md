# Model card

## Model and purpose

acclass is a ModernBERT-derived multi-label classifier for shell commands and
structured action descriptions. It predicts nine effects. See
[semantics](semantics.md) for the meaning of each label.

It predicts effects, not permission. It is a fast pre-filter: when it is
confident that nothing risky happens, a caller can allow the action quickly.
Everything else goes to a slower review. It is not a sandbox, an authorization
system or a safety guarantee.

- **Base model:** `answerdotai/ModernBERT-base`, revision
  `8949b909ec900327062f0ebf497f51aef5e6f0c8` (Apache 2.0).
- **Outputs:** nine independent sigmoid outputs on mean-pooled tokens.
- **Data:** 26,852 rows of original synthesized data. The data is not published.
- **License:** original classifier modifications use
  [Apache 2.0](../LICENSE). Third-party terms are in the
  [notices](../legal/ATTRIBUTIONS.md). Model weights are not in this source tree.

| Download | Size | Use |
| --- | ---: | --- |
| Q8_0 GGUF | 151.7 MiB | Default |
| Q5_0 GGUF | 98.2 MiB | Smaller, same speed on CPU |
| FP32 safetensors | 570.7 MiB | Exact FP32 serving; use `serve --quant f32` |

Quantized files keep norms and the classifier in FP32.

## Input and output

The engine accepts UTF-8 text. A shell input is the exact command string. An
action input is a JSON string in the
[action input format](engine-api.md#action-input-format).

The input limit is 2,048 tokens, including special tokens. The engine does not
truncate or pad inputs. It rejects inputs above the limit and inputs that
contain the tokenizer's unknown token.

Each label returns a probability and a boolean. The boolean is true when
`probability >= threshold`. The release model uses threshold 0.5 for every
label. The engine reads the thresholds from `thresholds.json`, so check them if
you use a different bundle.

Scores are calibrated with one global temperature, T = 1.76. Expected
calibration error (10 bins) changes from 0.011 to 0.005 on the calibration rows.
On the evaluation rows it is 0.004 for shell and 0.006 for action.

## Evaluation

All numbers are on 3,793 held-out evaluation rows at threshold 0.5, unless
noted. The 95% intervals resample whole groups of related rows (1,000 draws),
because rows in a group are not independent. "Base rows" and "LLM-authored"
rows are two separately produced parts of the dataset. Base rows are the better
estimate; see [limits](#limits).

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

- Shell `fetched_or_generated_execution` on base rows: F1 0.72 (59 positives,
  precision 0.68: it over-flags).
- Action `exposes_secrets` on base rows: F1 0.82 (16 positives).
- Action `egress`: F1 0.84.

## Quantization

A flip is a boolean that differs from FP32 at threshold 0.5. The comparison uses
all evaluation rows, which is 34,137 decisions.

| Format | Size | Macro F1 shell / action | Flips |
| --- | ---: | --- | ---: |
| FP32 | 571 MiB | 0.940 / 0.961 | reference |
| Q8_0 (default) | 151.7 MiB | 0.941 / 0.960 | 19 |
| Q5_0 | 98.2 MiB | 0.939 / 0.960 | 85 |

## Latency

Forward pass only, one request at a time. These are rough values from shared
machines, measured on an earlier model with the same architecture. Your values
can be different.

| Hardware class | FP32 | Q8 | Q5 |
| --- | --- | --- | --- |
| ARM CPU, 8 threads, p50 / p95 | 92 / 509 ms | 34 / 388 ms | 37 / 327 ms |
| x86 CPU, 6 threads, p50 / p95 | 66 / 760 ms | 50 / 717 ms | 51 / 544 ms |
| CUDA GPU, p50, up to 16 / about 1,800 tokens | 4.8 / 102 ms | 4.2 / 77 ms | not measured |

On the ARM CPU, Q8 takes about 16 ms for short commands (up to 32 tokens).

## Limits

Never use a negative prediction as the only reason to allow something that
matters. Never use a prediction as an irreversible block.

- **Not a safety guarantee.** False negatives occur and can fast-allow risky
  actions. Keep OS isolation and review for anything that matters.
- **Labels are LLM judgments,** from two runs of one model family. They are not
  independent human truth.
- **Much of the data is LLM-authored.** LLM-authored evaluation rows share style
  with LLM-authored training rows, so scores on them are optimistic. Base rows
  are the better estimate.
- **Real-world accuracy on your commands is unmeasured.**
- **The evaluation rows were opened once before this model was final.** They are
  not perfectly untouched.
- **Action mode was trained on stipulated, fictional action descriptions.** Real
  MCP or custom-tool metadata does not attest what a backend really does. Treat
  action predictions as hints that need review.
- **The model sees only the text.** It does not see the working directory,
  files, environment or user intent. Predictions describe effects inferred from
  text, not observed execution.
- **Evaluation scores do not establish resistance to adversarial inputs.**
- **Path scope is a separate heuristic.** It cannot prove that a command stays
  inside a directory or identify every effect.

A low probability is not proof that an effect is absent. A high probability does
not establish that the action is forbidden. Keep an explicit policy for uncertain
inputs and engine errors. See [semantics](semantics.md) and [privacy](privacy.md).
