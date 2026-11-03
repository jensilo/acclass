# Builds and release verification

## Build from source

You need Rust and a C/C++ toolchain. From the repository root:

```sh
cargo build --manifest-path engine/Cargo.toml --release --locked
cargo test --manifest-path engine/Cargo.toml --locked
```

The binary is `engine/target/release/acclass`. The tests need no model weights.

Inference needs a model bundle from a release. See [architecture](architecture.md) for
its files. Use one of these sources:

- An existing installation: `--model ~/.acclass/model`.
- Release assets. Put the files of one model in a folder and remove the prefix from
  their names:

```sh
mkdir model
for f in q8-*; do cp "$f" "model/${f#q8-}"; done
engine/target/release/acclass serve --model model
```

## Platforms

| Platform | Prebuilt | Needs |
| --- | --- | --- |
| Linux x86_64 | Yes | x86-64-v3 CPU |
| Linux aarch64 | Yes | ARM dot-product support |
| macOS arm64 | Yes | Apple build tools and SDK, if you build from source |
| macOS Intel | No | |
| Windows | No | The engine uses Unix sockets. It is not a native Windows program. |

`.cargo/config.toml` sets the same CPU requirements for source builds. To build for an
older CPU, change these flags. Quantized CPU inference is then slower. There are no
prebuilt binaries for older CPUs.

macOS builds enable Metal. CUDA is optional. To use it, add `--features cuda` and use a
compatible CUDA toolchain. See [device selection](architecture.md#devices).

## Verify a release

A release has these files, with no folders:

- `release.json` (the manifest), `install.sh` and `SHA256SUMS`
- `acclass-VERSION-PLATFORM.tar.gz`, where `PLATFORM` is `linux-x86_64`,
  `linux-aarch64` or `macos-arm64`
- Model files with the prefix `q8-`, `q5-` or `fp32-`

To check downloaded files:

1. Download the assets and `SHA256SUMS` into one folder.
2. Check `SHA256SUMS` against a source other than this download.
3. Run this command in the folder:

```sh
sha256sum -c SHA256SUMS
# macOS alternative:
shasum -a 256 -c SHA256SUMS
```

To install with a pinned manifest:

1. Read the installer.
2. Get the SHA-256 hash of `release.json` from a source other than this download.
3. Replace both placeholders and run this command. The URL must use HTTPS.

```sh
sh install/install.sh --manifest HTTPS_RELEASE_JSON_URL --manifest-sha256 TRUSTED_SHA256
```

The installer compares the manifest with the hash. It then checks the size and hash of
each asset against the manifest. A checksum from the same download does not prove who
published the file.

See [installation](../README.md#install) and the [notices](../legal/ATTRIBUTIONS.md).
