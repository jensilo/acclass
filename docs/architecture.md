# Architecture

## Request path

1. The `classify` client sends a JSON line over a Unix socket.
2. The daemon (`acclass serve`) tokenizes the complete input.
3. The encoder produces one probability for each of the nine labels.
4. The engine compares each probability with the threshold of that label.
5. For shell inputs, a separate Bash parser estimates the path scope.
6. The daemon returns a JSON line with predictions, scope and identity.

Model results and path scope are separate signals. Neither is a sandbox or a
permission system. See [semantics](semantics.md) and the [API](engine-api.md).

## Model

The encoder is derived from ModernBERT. It has 22 layers and hidden size 768. It uses
rotary position embeddings, global and sliding attention, mean pooling and nine
output heads.

The token limit is 2,048, including special tokens. Tokenizer truncation and padding
are off. Unknown tokens and invalid lengths cause errors. The model never predicts
from part of an input.

A model bundle contains:

- `config.json`
- `tokenizer.json`
- `thresholds.json`
- `heads.json`
- `model_contract.json`
- `model.gguf` or `model.safetensors`

## Daemon start

1. The daemon selects the device.
2. It hashes the selected weights, the metadata files and its own binary. Clients can
   compare these hashes with values that they trust.
3. It validates the architecture, labels, thresholds and input limit of the bundle.
4. It loads the model. It stops if file metadata changed during the load.
5. It runs one test inference.
6. It starts to listen on the socket.

## Weights

| `serve` option | Weights |
| --- | --- |
| None | `model.gguf` if present. Otherwise `model.safetensors`, quantized to Q8 at start. |
| `--gguf` | `model.gguf`. Conflicts with `--quant`. |
| `--quant q8`, `--quant q5` | `model.safetensors`, quantized at start |
| `--quant f32` | `model.safetensors`, unquantized |

`convert` reads safetensors on the CPU and creates a new Q8 or Q5 GGUF bundle. It does
not overwrite an existing folder. It copies the metadata and records the new weight
hash in `model_contract.json`.

## Devices

CPU inference uses Candle with custom quantized matrix and attention kernels. macOS
builds include Metal. CUDA is an optional build feature. The release has a CUDA engine for
Linux aarch64.

| `--device` | Behaviour |
| --- | --- |
| `auto` (default) | CUDA in a CUDA build, otherwise Metal on macOS, otherwise CPU. Falls back to CPU only if device initialization fails. |
| `cpu`, `cuda`, `metal` | Uses this device, or fails. |

Model loading errors and inference errors never cause a fallback.

## Source layout

| Path | Role |
| --- | --- |
| `engine/src/main.rs` | CLI, daemon, protocol, model validation and conversion |
| `engine/src/model.rs` | Encoder, classification head and model loading |
| `engine/src/qgemm.rs` | Quantized CPU matrix kernels |
| `engine/src/attention.rs` | CPU attention kernels |
| `engine/src/metal.rs`, `engine/src/metal.metal` | Metal integration and kernels |
| `engine/src/scope.rs` | Bash path-scope analysis |
| `engine/tests/client.rs` | Client tests without model weights |
| `.cargo/config.toml` | CPU target flags |
| `install/install.sh` | Release download and installation |
| `legal/` | Model and dependency notices |

See [builds](releasing.md), [privacy](privacy.md) and the
[model card](model-card.md).
