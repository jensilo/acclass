# Builds and release verification

## Build from source

Use Rust with Cargo and a C/C++ build toolchain. From the repository root:

```sh
cargo build --manifest-path engine/Cargo.toml --release --locked
cargo test --manifest-path engine/Cargo.toml --locked
```

The binary is `engine/target/release/acclass`. Tests use fixtures and do not
need model weights.

Running inference needs a model bundle from a release; see
[architecture](architecture.md) for its files. Use one of these sources:

- An existing installation: `--model ~/.acclass/model`.
- Downloaded release assets. Put one model's files in a directory and remove the
  prefix from their names:

```sh
mkdir model
for f in q8-*; do cp "$f" "model/${f#q8-}"; done
engine/target/release/acclass serve --model model
```

The installer targets Linux x86_64 with x86-64-v3, Linux aarch64 with ARM
dot-product support, and macOS arm64. The checked-in Cargo flags require
x86-64-v3 on x86_64 and enable dot-product instructions on aarch64. Building
from source with these flags does not remove those CPU requirements. Review
and change target flags for older CPUs; that configuration is not established
by the prebuilt target list. No Intel macOS binary is provided. The engine
uses Unix sockets and is not a native Windows program.

macOS builds enable Metal and need the Apple build tools and SDK. CUDA is
optional: add `--features cuda` to build with a compatible CUDA toolchain.
The standard build does not enable CUDA. See [device selection](architecture.md).

## Verify a release

Release assets are flat GitHub release files:

- `release.json`, `install.sh` and `SHA256SUMS`.
- Engine archives: `acclass-VERSION-PLATFORM.tar.gz`, where `PLATFORM` is
  `linux-x86_64`, `linux-aarch64` or `macos-arm64`.
- Model files with the prefix `q8-`, `q5-` or `fp32-`.

Download the assets and `SHA256SUMS` from the chosen release into one directory.
Check the checksum file through a trusted channel before you rely on it. From
that directory:

```sh
sha256sum -c SHA256SUMS
# macOS alternative:
shasum -a 256 -c SHA256SUMS
```

Read the installer before running it. Pin the SHA-256 of `release.json`
from an independently trusted channel:

```sh
sh install/install.sh --manifest HTTPS_RELEASE_JSON_URL --manifest-sha256 TRUSTED_SHA256
```

Replace both placeholders. The URL must use HTTPS. The installer verifies
the manifest pin and then checks each downloaded asset's declared size and
SHA-256. A checksum from the same untrusted download does not establish
publisher authenticity. See [installation](../README.md#install) and
[notices](../legal/ATTRIBUTIONS.md).
