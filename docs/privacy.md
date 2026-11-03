# Privacy

## Inference

The engine loads model files from local storage and serves requests over a
local Unix socket. Its inference path has no network client or telemetry
sender. It does not execute the command being classified.

Requests contain command text and can include working-directory and home
paths. Replies contain probabilities, path information and token IDs.
Token IDs can reveal input text. Treat requests and replies as sensitive.
The CLI prints replies to standard output; applications decide whether to
store or forward them.

The engine does not write a command-history log. Its daemon prints startup
status, including model and socket paths, to standard output. Its clients
and error paths can print diagnostics. Logs or process capture outside the
engine can retain these outputs.

Scope analysis can check local path existence. It does not read file
contents or execute the submitted command. Scope is not a privacy or
containment guarantee. See [semantics](semantics.md).

## Socket access

The daemon sets socket permissions to `0600`. It creates missing parent
directories with mode `0700`; it does not change permissions on an existing
parent directory. Use a directory you control. Socket permissions do not
protect against other processes with the same user identity or an
administrator.

The engine hashes model artifacts and its binary at startup and includes
those hashes in replies. This identifies loaded files; it does not encrypt
requests or establish that those files are trusted.

## Installation

The installer contacts the HTTPS release manifest and asset URLs. The
servers can observe download requests. Installation downloads are separate
from local inference. The installer checks declared asset sizes and hashes;
use `--manifest-sha256` with an independently trusted hash when you need to
pin the manifest. See [release verification](releasing.md).

The default installation root is `~/.acclass`. The installer keeps the
current installation and can retain a previous version for rollback.
`--uninstall` removes installer-managed files; it can leave unrelated files.
Stop a running daemon separately. Do not submit secrets unless the calling
application's input, output and logging policy permits it.
