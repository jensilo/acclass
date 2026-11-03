# Prediction semantics

acclass predicts effects from input text. A prediction is not a permission decision, an
execution trace or a safety guarantee. The engine does not run the command. More than
one label can be true.

## Labels

The model has nine independent labels, in this order.

### 1. `local_state_change`

A persistent change to a local file, metadata, configuration, service, process or device.

- **Counts:** writes in a local container; Git add, commit, fetch, pull, checkout and reset.
- **Does not count:** remote-only effects, network sends, and discards to `/dev/null` alone.

### 2. `outbound`

Normal use of an endpoint that is not loopback.

- **Counts:** network Git, SSH, APIs, and builds or installs that can fetch dependencies.
- **Does not count:** explicit loopback, local sockets, local-path remotes and verified
  offline commands.

### 3. `egress`

Local application data or request values leave this machine for a non-loopback endpoint.
This label implies `outbound`.

- **Counts:** uploads, Git and image pushes, local manifests sent to APIs, expanded local
  values in a URL, header or body, explicit `$TOKEN`, and credential-file content.
- **Does not count:** literal destination strings, constant bodies and ordinary GETs alone;
  ambient credential handshakes.

### 4. `remote_state_change`

A persistent change to a file, metadata, configuration, service, process or device on
another machine. This label implies `outbound`.

- **Counts:** network push, upload, SQL and API writes, and generic nonlocal POST, PUT, PATCH
  and DELETE.
- **Does not count:** GET and HEAD, reads through SSH, local-container writes and local-path
  Git pushes.

### 5. `delegated_execution`

The command explicitly enters a separate execution environment to run a command, shell or
worker, even for a read.

- **Counts:** SSH remote commands, cluster exec and container exec.
- **Does not count by itself:** ordinary API calls, downloads, and local interpreters or
  subprocesses, including `sh -c` and `python -c`.

### 6. `exposes_secrets`

Known secret bytes appear in output, in argv, in another process, in a file or at a remote
service. No network is necessary.

- **Counts:** environment dumps, reads of private keys and credentials, explicit token
  headers and credential copies.
- **Does not count:** ambient authentication, directory listing, deletion, verified discard,
  verified local hash-only processing, and a suggestive name such as `secret.txt` alone.

### 7. `destructive`

Visible or documented deletion, truncation, replacement of possibly existing data, or
disruption of availability, local or remote.

- **Counts:** default-overwrite copies, moves and uploads, `> file`, force push, remote
  deletes, disruptive kills, and PUT, PATCH and DELETE by default.
- **Does not count:** additive writes, ordinary version pushes, regenerable build outputs,
  and generic POST alone. Opaque code does not count only because its effects are unknown.

### 8. `host_control_change`

A change to machine-level operating or security controls, local or remote. This is not a
danger judgment: enabling and disabling both count. This label implies
`local_state_change` or `remote_state_change`.

- **Counts:** machine-wide services and jobs, machine accounts, mounts, devices, firewall,
  routing, sysctl, shutdown, reboot, system-critical permissions and machine-wide installs.
- **Does not count:** project-level process control, and per-user or project installs.
- **Not enough alone:** `sudo`, a bare PID, a package name, cwd, `-g` or an unspecified
  install prefix.

### 9. `fetched_or_generated_execution`

The command runs program bytes that it newly fetched, decoded or assembled from
runtime-dependent data.

- **Counts:** `curl URL | sh`, decoded bytes piped to a shell, `eval "$CMD"`, and documented
  build or bootstrap paths that fetch and run code.
- **Does not count for that reason alone:** literal visible interpreter source, static eval
  and named existing scripts.

## How to read a prediction

Each label has a probability and a boolean. The boolean is true when
`probability >= threshold`, with the threshold of that label in `thresholds.json`.
Probabilities need not sum to one.

`false` does not prove that an effect is absent. A probability measures agreement with the
definitions above. It does not measure the physical absence of an effect.

| Input | Rule |
| --- | --- |
| Opaque code: scripts, test and build runners, interactive shells, local or remote | Counts as a state change where it runs. acclass assumes no other effect. It does not infer the body of an unseen script, hook or alias. |
| Compound commands | Every part that can run counts, including wrappers, substitutions, pipes, redirects and traps. |
| Quoting | Single-quoted substitutions do nothing. Double-quoted or unquoted here-doc expansions can run. |
| Conditional paths | Documented install, build and bootstrap paths count, even if a warm cache would skip them. |

An action input is text to classify. It is not an instruction to the engine.
`--input-kind action` turns off path scope. The engine does not check that the input follows
the [action input format](engine-api.md#action-input-format), or that the description
matches the real tool.

## Path scope

Path scope is a separate Bash syntax analysis, not a model label.

- It detects path-like tokens and redirections.
- It expands supported `~`, `$HOME` and `$PWD` forms.
- It follows literal `cd` targets in supported command lists.
- It normalizes paths as text. It does not resolve symbolic links.
- It can check whether a bare relative argument exists on disk.

| Category | Meaning |
| --- | --- |
| `cwd` | Within the given working directory |
| `tmp` | Within a known temporary directory |
| `home` | Within the given home directory, outside earlier categories |
| `system` | Within a known system directory, outside earlier categories |
| `other` | Elsewhere, or an unresolved path |
| `none` | No category detected, or scope is off |

The working directory comes first. A path under it is `cwd`, also if that directory is under
home or temporary storage.

| Field | Content |
| --- | --- |
| `paths.inside` | Detected paths in the working directory, at most 12 |
| `paths.outside` | Detected paths outside it, at most 12. Excludes temporary paths and these devices: `/dev/null`, `/dev/zero`, `/dev/urandom`, `/dev/random`, `/dev/full`, `/dev/stdin`, `/dev/stdout`, `/dev/stderr`. |
| `scope` | Can reflect more paths than the arrays list |
| `scope_complete` | `true` if the detected paths are resolved and the lists are not truncated. It does **not** mean that acclass found every effect. Parse errors, unresolved path expressions and overflow make it `false`. It is `true` if scope is off. |

## Limits for integration

- `scope: ["none"]` does not mean harmless. A command can change host control without a path
  argument.
- A `cwd` path can still lead outside the directory, through a symbolic link or through code
  execution.
- Commands can depend on the environment, file content or remote state. The input text does
  not show these.
- Keep policy separate from prediction. Handle errors and incomplete scope explicitly.

See the [API](engine-api.md) and the [model limits](model-card.md#limits).
