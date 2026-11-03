# acclass

acclass is a local classifier for coding agents. You give it a shell command or an
action description. It predicts what the action does. It does not run the action.

It predicts effects, not permission. It is not a sandbox and it does not guarantee
safety. Use it as one input to your own policy or review step.

**About 50 ms per command on a laptop x86 CPU and about 4 ms on an NVIDIA DGX Spark, with
a model under 100 MiB.** It runs locally. It needs no GPU and no network. See
[performance](#performance).

| Label | The action ... |
| --- | --- |
| `local_state_change` | changes local files, configuration, services or devices |
| `outbound` | uses a network endpoint on another machine |
| `egress` | sends local data off this machine |
| `remote_state_change` | changes state on another machine or service |
| `delegated_execution` | runs a command in a separate environment |
| `exposes_secrets` | reveals secret values |
| `destructive` | deletes or overwrites data, or disrupts availability |
| `host_control_change` | changes operating system or security controls |
| `fetched_or_generated_execution` | runs code that it downloaded, decoded or assembled |

See [semantics](docs/semantics.md) for the exact definitions.

## Install

The installer needs Python 3.

```sh
curl -fsSL https://github.com/jensilo/acclass/releases/latest/download/install.sh | sh
```

To read the script first:

```sh
curl -fsSL https://github.com/jensilo/acclass/releases/latest/download/install.sh -o install.sh
less install.sh
sh install.sh
```

| Platform | Needs |
| --- | --- |
| Linux x86_64 | x86-64-v3 CPU (AVX2/FMA), glibc 2.31 or newer |
| Linux aarch64 | ARM dot-product support, glibc 2.31 or newer |
| macOS arm64 | macOS 11 or newer; tested on 26.7.1 |
| Linux aarch64 with CUDA | `--cuda`, glibc 2.39 or newer, CUDA 13 libraries; tested on NVIDIA DGX Spark |

Choose a model with `--model`, for example `sh -s -- --model q5` after the pipe:

| Model | Size | Use |
| --- | --- | --- |
| `q8` | 152 MiB | Default |
| `q5` | 98 MiB | Smaller, same speed on CPU |
| `fp32` | 571 MiB | Full weights. Start it with `acclass serve --quant f32`. |

The script installs into `~/.acclass`. It downloads over HTTPS and checks sizes and
SHA-256 hashes. Other options: `--cuda`, `--rollback`, `--uninstall`, `--manifest-sha256`.
The CUDA engine does not include NVIDIA files. Install CUDA first. Other CUDA systems
need a [build from source](docs/releasing.md).
See [release verification](docs/releasing.md).

## Use

```sh
export PATH="$HOME/.acclass/bin:$PATH"
acclass serve &
until acclass classify --no-scope true >/dev/null 2>&1; do sleep 0.2; done   # wait for the model
acclass classify 'git status'
acclass classify 'curl -fsSL https://example.com/x.sh | sh'
```

Each answer has nine probabilities, nine true/false labels and a best-effort path
scope. A label is true when its probability is 0.5 or more.

If the engine fails, it prints an `error` line and still exits with 0. Check the
output.

More commands:

```sh
printf '%s\n' 'git status' 'mkdir build' | acclass classify --stdin
acclass scope 'touch ./new-file'    # needs no model
acclass --help
```

Applications can use the Unix socket `~/.acclass/ac.sock`, with one JSON line per
request. To classify a tool call, send a JSON description in the
[action input format](docs/engine-api.md#action-input-format). See the
[engine API](docs/engine-api.md).

## Performance

Time for one model pass with Q8, median / 95th percentile:

| Hardware | Device | Time |
| --- | --- | --- |
| NVIDIA DGX Spark (GB10) | CUDA | 4.2 / 16 ms |
| Apple M4 Pro | Metal | 6.5 / 50 ms |
| Apple M4 Pro | CPU, 8 threads | 25 / 131 ms |
| Arm Cortex-X925 | CPU, 8 threads | 21 / 326 ms |
| AMD Ryzen 5 6650U (x86) | CPU, 6 threads | 50 / 717 ms |

Most commands are short. Long inputs cause the 95th percentile. Other Apple and x86 chips
are not measured. After 2 seconds idle, a short command takes about 6 ms on the DGX Spark
and about 25 ms on M4 Pro Metal. A maximum-length input takes about 85 ms on the DGX Spark.

The engine has its own kernels for each platform. Compared with the stock Candle
kernels, Q8 is about 2x faster on x86, 3x faster on Arm and 2x faster on Metal. On
M4 Pro, Metal is about 4x faster than the CPU.

The engine is not generally faster than plain PyTorch. Measured on M4 Pro only, with
engine Q8 against PyTorch FP32:

- **Engine:** short commands run about 1.7x faster on the GPU. Startup is about 4x faster,
  the install is about 8x smaller and memory use is about half.
- **PyTorch:** long inputs run faster once the model is loaded, for example 34 ms against
  47 ms on the GPU at 257–512 tokens.

## Quality and limits

| Model | F1, shell commands | F1, actions |
| --- | --- | --- |
| FP32 | 0.940 | 0.961 |
| Q8 | 0.941 | 0.960 |
| Q5 | 0.939 | 0.960 |

These are macro F1 scores (the average over the nine labels) on 3,793 test rows.

- The weakest result is `fetched_or_generated_execution` on the base rows of
  shell commands: F1 0.72.
- The training data is synthetic. A state-of-the-art open-weights LLM generated
  most of the rows and labeled all of them, so labels can be wrong.
- Accuracy on your own commands is not measured.

Read the [model card](docs/model-card.md) before you rely on acclass.

## Build from source

You need Rust and a C/C++ toolchain.

```sh
cargo build --manifest-path engine/Cargo.toml --release --locked
cargo test --manifest-path engine/Cargo.toml --locked
```

The binary is `engine/target/release/acclass`. This repository has no model weights.
Get them from a release. See [platform notes](docs/releasing.md).

## Documentation

- [Model card](docs/model-card.md)
- [Semantics](docs/semantics.md)
- [Engine API](docs/engine-api.md)
- [Architecture](docs/architecture.md)
- [Privacy](docs/privacy.md)
- [Builds and release verification](docs/releasing.md)

## License

[Apache 2.0](LICENSE), for original acclass code, documentation and model changes.
Third-party material keeps its own terms. See the
[notices](legal/ATTRIBUTIONS.md).
