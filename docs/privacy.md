# Privacy

## Inference

The engine loads model files from local storage and serves requests over a local Unix
socket. Its inference path has no network client and no telemetry. It does not run the
command that it classifies.

| Data | Content |
| --- | --- |
| Requests | Command text. Can include working-directory and home paths. |
| Replies | Probabilities, path information and token IDs. Token IDs can reveal the input. |

Treat requests and replies as sensitive. Do not submit secrets unless the input, output
and logging rules of the calling application permit it.

- The CLI prints replies to standard output. The application decides whether to store
  or forward them.
- The engine writes no command history.
- The daemon prints startup status, including model and socket paths.
- Clients and error paths can print diagnostics. Logs or process capture outside the
  engine can keep this output.
- Scope analysis can check whether a local path exists. It does not read file contents
  or run the command. Scope is not a privacy or containment guarantee. See
  [semantics](semantics.md).

## Socket access

- The daemon sets socket permissions to `0600`.
- It creates missing parent folders with mode `0700`. It does not change an existing
  parent folder. Use a folder that you control.
- Socket permissions do not protect against other processes of the same user or
  against an administrator.
- The engine hashes the model files and its binary at start, and includes the hashes in
  replies. These hashes identify the loaded files. They do not encrypt requests and do
  not show that the files are trustworthy.

## Installation

The installer contacts the HTTPS URLs of `release.json` (the manifest) and the release
assets. These servers can see the download requests. Inference does not use them.

- The installer checks the size and hash of each asset against the manifest.
- To pin the manifest, use `--manifest-sha256` with a hash from another source. See
  [release verification](releasing.md).
- The default installation folder is `~/.acclass`. `--home` or `ACCLASS_HOME` changes it.
- The installer keeps the current version and one previous version for `--rollback`.
- `--uninstall` removes the files that the installer manages. It leaves unrelated files.
- `--uninstall` does not stop a running daemon. Stop it yourself.
