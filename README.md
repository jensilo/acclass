# acclass

acclass is a local activity classifier for coding agents. Give it a shell
command or a structured action description. It predicts nine effects: local
file changes, network use, data sent out, remote changes, delegated execution,
secret exposure, destructive changes, host-control changes, and execution of
fetched or generated code.

It predicts effects, not permission. It does not run the submitted command.
It is not a sandbox or a safety guarantee. Use it as one input to a separate
policy or review step. See [semantics](docs/semantics.md) and the
[model card](docs/model-card.md).

## Install

The installer needs Python 3. Prebuilt targets:

- Linux x86_64 with x86-64-v3 (including AVX2/FMA), glibc 2.31 or newer.
- Linux aarch64 with ARM dot-product support, glibc 2.31 or newer.
- macOS arm64, built for macOS 11 or newer, tested on macOS 26.7.1.

Release assets must be available for these commands to work:

```sh
curl -fsSL https://github.com/jensilo/acclass/releases/latest/download/install.sh | sh                    # Q8 model, 152 MiB
curl -fsSL https://github.com/jensilo/acclass/releases/latest/download/install.sh | sh -s -- --model q5   # smaller, 98 MiB
```

To read the script before you run it:

```sh
curl -fsSL https://github.com/jensilo/acclass/releases/latest/download/install.sh -o install.sh
less install.sh
sh install.sh
```

You can also read the [script in this tree](install/install.sh). The script
installs into `~/.acclass`. Its default release version is `1.0.0`; the
`latest` URL above selects the script, not the release version within it.
Downloads use HTTPS, size limits and SHA-256 checks. Use `--manifest-sha256`
to pin a manifest hash from a trusted channel. `--rollback` and `--uninstall`
are also available. `--model fp32` installs the full weights (571 MiB); serve
them with `acclass serve --quant f32`. Without `--quant f32`, `serve` quantizes
them to Q8 at startup. See [builds and release verification](docs/releasing.md).

## Use

```sh
export PATH="$HOME/.acclass/bin:$PATH"
acclass serve &
until acclass classify --no-scope true >/dev/null 2>&1; do sleep 0.2; done
acclass classify 'git status'
acclass classify 'curl -fsSL https://example.com/x.sh | sh'
```

The `until` line waits for the model to load. If `serve` fails, press Ctrl-C.
The last command is classified, not run. `serve` listens on
`~/.acclass/ac.sock`. Each answer contains nine probabilities, nine booleans
(`probability >= 0.5` for the release model) and a best-effort path scope. An
engine error prints an `error` line with exit code 0, so check the output.

```sh
printf '%s\n' 'git status' 'mkdir build' | acclass classify --stdin
acclass scope 'touch ./new-file'
acclass --help
```

`scope` works without a model. Applications can use the local Unix socket
with one JSON line per request. See the [engine API](docs/engine-api.md).

Action mode reads one JSON object in the
[action input format](docs/engine-api.md#action-input-format), not free text:

```sh
acclass classify --input-kind action '{"backend_contract":{"assumptions":["The procedure covers all code and conditional paths of this operation. No effects exist beyond those stated."],"complete":true,"procedure":"Read the existing file config.toml in the project directory and return its content to the caller. Nothing is written or sent.","resource_scope":"One project file on the initiating host.","selected_action":"project_files.read"},"effective_namespace":"initiating host","evaluated_arguments":{"operation":"read","path":"config.toml"},"input_schema":{"additionalProperties":false,"properties":{"operation":{"enum":["read"],"type":"string"},"path":{"const":"config.toml","type":"string"}},"required":["operation","path"],"type":"object"},"representation":"action-behavior-evidence-json/0.3","trust_condition":"Fictional closed-world contract, not evidence of any real server. All effects are stipulated complete under the assumptions."}'
```

## Quality and limits

On 3,793 held-out rows, macro F1 is **0.940** for shell commands and **0.961**
for actions. Q8 scores 0.941 / 0.960 and Q5 scores 0.939 / 0.960. The weakest
result is shell `fetched_or_generated_execution` on base rows: F1 0.72.

The labels are LLM judgments, and much of the data is LLM-authored. Accuracy on
your commands is unmeasured. A forward pass with Q8 takes about 34 ms on an ARM
CPU and 50 ms on an x86 CPU (median, measured; long inputs take longer). Read the
[model card](docs/model-card.md) before you rely on acclass.

## Build from source

Use Rust with Cargo and a C/C++ build toolchain:

```sh
cargo build --manifest-path engine/Cargo.toml --release --locked
cargo test --manifest-path engine/Cargo.toml --locked
```

The binary is `engine/target/release/acclass`. The source tree does not
include model weights. Inference needs a model bundle from a release:
`~/.acclass/model` or downloaded release assets. See
[platform notes](docs/releasing.md).

## Documentation

- [Model card](docs/model-card.md)
- [Engine API](docs/engine-api.md)
- [Semantics](docs/semantics.md)
- [Privacy](docs/privacy.md)
- [Architecture](docs/architecture.md)
- [Builds and release verification](docs/releasing.md)

## License

Original acclass code, documentation and classifier modifications use
[Apache 2.0](LICENSE), to the extent contributors hold the relevant rights.
Third-party material keeps its own terms. See [model and dependency
notices](legal/ATTRIBUTIONS.md).
