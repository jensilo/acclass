# Architecture

## Request path

1. The `classify` client sends a JSON line over a Unix socket.
2. The resident `serve` process tokenizes the complete input.
3. The ModernBERT-derived encoder produces nine sigmoid probabilities.
4. The engine compares each probability with its bundle threshold.
5. For shell inputs, a separate Bash parser estimates path scope.
6. The daemon returns a JSON line with predictions, scope and artifact identity.

The daemon loads one model and warms it up before it listens. Model results
and path scope are separate signals. Neither is a sandbox or a permission
system. See [semantics](semantics.md) and [API](engine-api.md).

## Model contract

The encoder has 22 layers and hidden size 768. It uses rotary position
embeddings, global and sliding attention, mean pooling and nine output heads.
The engine validates the fixed architecture, ordered heads, finite thresholds
and full-input contract before serving.

The token limit is 2,048, including special tokens. Tokenizer truncation and
padding are disabled. Unknown tokens and invalid lengths cause errors rather
than partial-input predictions.

A model bundle contains:

- `config.json`
- `tokenizer.json`
- `thresholds.json`
- `heads.json`
- `model_contract.json`
- `model.gguf` or `model.safetensors`

The daemon hashes the selected weights, metadata and serving binary. It also
checks file metadata before and after loading to detect changes during load.
Clients can compare the returned identity with a trusted expected identity.

## Weights and devices

By default, `serve` loads `model.gguf` if present. Otherwise, it loads
`model.safetensors` with runtime Q8 quantization. An explicit `--quant q8`,
`--quant q5` or `--quant f32` selects safetensors. `--gguf` conflicts with
`--quant`.

`convert` reads safetensors on CPU and creates a new Q8 or Q5 GGUF bundle.
It does not overwrite an existing output directory. It copies model metadata
and records the converted weight hash in the output contract.

CPU inference uses Candle plus custom quantized matrix and attention kernels.
On macOS, Metal support is compiled in. CUDA is an optional build feature.
`--device auto` tries CUDA in a CUDA build, otherwise Metal on macOS,
otherwise CPU. Only automatic selection can fall back to CPU, and only when
device initialization fails. Model-loading and inference errors are not
fallback conditions. Explicit `cpu`, `cuda` and `metal` selections do not
silently switch devices.

## Source layout

| Path | Role |
| --- | --- |
| `engine/src/main.rs` | CLI, daemon, protocol, model validation and conversion |
| `engine/src/model.rs` | Encoder, classification head and model loading |
| `engine/src/qgemm.rs` | Quantized CPU matrix kernels |
| `engine/src/attention.rs` | CPU attention kernels |
| `engine/src/metal.rs`, `engine/src/metal.metal` | Metal integration and kernels |
| `engine/src/scope.rs` | Bash path-scope analysis |
| `engine/tests/client.rs` | Thin-client fixtures without model weights |
| `.cargo/config.toml` | CPU target flags |
| `install/install.sh` | Release download and installation |
| `legal/` | Model and dependency notices |

See [builds](releasing.md), [privacy](privacy.md) and the
[model card](model-card.md).
