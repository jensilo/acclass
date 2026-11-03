# Prediction semantics

acclass predicts effects from input text. Effects can overlap. A prediction
is not a permission decision, an execution trace or a safety guarantee.
The engine does not run the submitted command.

## Labels

The output heads are nine independent labels, in this order.

1. **`local_state_change`:** a persistent change to a local file, metadata,
   configuration, service, process or device. Writes in a local container count.
   Git add, commit, fetch, pull, checkout and reset count. Remote-only effects,
   network sends and discards to `/dev/null` alone do not count.
2. **`outbound`:** normal use of a non-loopback endpoint. This includes network
   Git, SSH, APIs, and builds or installs that can fetch dependencies. Explicit
   loopback, local sockets, local-path remotes and verified offline commands do
   not count.
3. **`egress`:** locally derived application content or request values cross a
   non-loopback boundary. This includes uploads, Git and image pushes, local
   manifests sent to APIs, and expanded local values in a URL, header or body.
   Literal destination strings, constant bodies and ordinary GETs alone do not
   count. Ambient credential handshakes do not count; explicit `$TOKEN` or
   credential-file content does. Egress implies `outbound`.
4. **`remote_state_change`:** a persistent change to a nonlocal file, metadata,
   configuration, service, process or device. This includes network push, upload,
   SQL and API writes. Generic nonlocal POST, PUT, PATCH and DELETE count; GET and
   HEAD do not. Reads through SSH, local-container writes and local-path Git
   pushes do not count. This label implies `outbound`.
5. **`delegated_execution`:** the command explicitly enters a distinct execution
   environment to run a command, shell or separate worker, even for a read. SSH
   remote commands, cluster exec and container exec count. Ordinary API calls,
   downloads and local interpreters or subprocesses, including `sh -c` and
   `python -c`, do not count by themselves.
6. **`exposes_secrets`:** known secret bytes are revealed through output, argv,
   another process, a file or a remote service. Environment dumps, private-key
   and credential reads, explicit token headers and credential copies count. No
   network is necessary. Ambient authentication, directory listing, deletion,
   verified discard and verified local hash-only processing do not count. A
   suggestive name such as `secret.txt` alone does not count.
7. **`destructive`:** visible or documented deletion, truncation, replacement of
   possibly existing data, or availability disruption, local or remote. This
   includes default-overwrite copies, moves and uploads, `> file`, force push,
   remote deletes and disruptive kills. Additive writes, ordinary version pushes
   and regenerable build outputs do not count. Opaque code is not destructive only
   because its effects are unknown. Generic POST alone does not count; PUT, PATCH
   and DELETE do by default.
8. **`host_control_change`:** a change to machine-level operating or security
   controls at a local or remote target. This includes machine-wide services and
   jobs, machine accounts, mounts, devices, firewall, routing, sysctl, shutdown,
   reboot, system-critical permissions and machine-wide installs. Enabling and
   disabling both count; this is not a danger judgment. Project-level process
   control and per-user or project installs do not count. `sudo`, a bare PID, a
   package name, cwd, `-g` or an unspecified install prefix alone is not
   sufficient. This label implies `local_state_change` or `remote_state_change`.
9. **`fetched_or_generated_execution`:** executed program bytes are newly
   fetched, decoded or assembled from runtime-dependent data. `curl URL | sh`,
   decoded bytes piped to a shell and `eval "$CMD"` count. Documented build or
   bootstrap paths that fetch and run code count. Literal visible interpreter
   source, static eval and named existing scripts do not count for that reason
   alone.

## How to read a prediction

Each label has a probability and a boolean. The engine sets the boolean when
`probability >= threshold`, using that label's value in `thresholds.json`.
Probabilities need not sum to one. More than one boolean can be true.

`false` is not proof of absent effects. Labels follow the written conventions
above, so probabilities measure agreement with these conventions, not the
physical absence of an effect.

- **Opaque execution** counts as a state change at its execution target. Local
  scripts, test and build runners and interactive shells count like remote ones.
  Opacity does not establish any other effect. The model does not infer the body
  of an unseen script, hook or alias.
- **Compound commands:** every part that can run counts, including wrappers,
  substitutions, pipes, redirects and traps.
- **Quoting matters:** single-quoted substitutions are inert. Active
  double-quoted or unquoted here-doc expansions can execute.
- **Conditional paths:** documented install, build and bootstrap paths count,
  even if a warm cache would skip them.

An action input is input text, not an instruction to the engine.
`--input-kind action` disables shell path analysis. The engine does not verify
that the input follows the [action input format](engine-api.md#action-input-format)
or that the description fully represents an external tool's behavior.

## Path scope

Path scope is a separate Bash syntax analysis, not a model label. It detects
path-like tokens and redirections, expands supported `~`, `$HOME` and `$PWD`
forms, and tracks literal `cd` targets in supported command lists.
It uses lexical path normalization. It does not resolve symbolic links.
It can check whether a bare relative argument exists on disk.

| Category | Meaning |
| --- | --- |
| `cwd` | Within the supplied working directory |
| `tmp` | Within a recognized temporary directory |
| `home` | Within the supplied home directory, outside earlier categories |
| `system` | Within a recognized system directory, outside earlier categories |
| `other` | Elsewhere, or an unresolved path |
| `none` | No category detected, or scope analysis was disabled |

The working directory has priority over the other categories. A path under
it is `cwd`, even if that directory is also under home or temporary storage.

`paths.inside` lists detected working-directory paths. `paths.outside` lists
selected paths outside it. Temporary paths and certain standard device
paths are not included in `outside`. Each array holds at most 12 unique
paths; scope categories can still reflect additional paths.

`scope_complete` means that the detected path inventory is resolved and not
truncated. It does **not** mean that every command effect was found. Parse
errors, unresolved path expressions and inventory overflow can make it
false. When analysis is disabled, it is true with empty path arrays.

## Integration limits

Do not infer that `scope: ["none"]` means harmless. A command can change host
control without a path argument. A path classified as `cwd` can still lead
outside the directory through a symbolic link or through code execution.
Commands can also depend on environment, file content or remote state that
is not in the input text.

Keep policy separate from prediction. Handle errors and incomplete scope
explicitly. See the [API](engine-api.md) and [model limits](model-card.md).
