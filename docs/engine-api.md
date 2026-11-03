# Engine API

## Command line

```sh
acclass serve [--model DIR] [--socket PATH] [--gguf | --quant q8|q5|f32] \
  [--device auto|cpu|cuda|metal]
acclass classify [--socket PATH] [--cwd DIR] [--home DIR] \
  [--input-kind shell|action] [--no-scope] [--stdin | CMD...]
acclass scope [--cwd DIR] [--home DIR] [--stdin | CMD]
acclass convert [--model DIR] [--out DIR] [--quant q8|q5]
```

`--help` and `--version` do not load a model or connect to the daemon.
The default model directory is `~/.acclass/model`. The default socket is
`~/.acclass/ac.sock`. `classify` sends text; it does not execute that text.
Quote a command so that your own shell does not expand it first.

`classify --stdin` reads one command per nonblank line. Each response is
printed as one JSON line. Without `--stdin`, each command argument is a
separate request. `scope --stdin` reads all standard input as one command.
`scope` needs neither model weights nor a daemon.

`--` makes all remaining arguments commands: `acclass classify -- --version`
classifies the text `--version`. Unknown options are also treated as commands,
so check their spelling.

`serve` starts to accept requests only after it loads and warms up the model.
To wait for it from a shell:

```sh
acclass serve &
until acclass classify --no-scope true >/dev/null 2>&1; do sleep 0.2; done
```

If `serve` fails, this loop does not stop. Press Ctrl-C and read the `serve` error.

### Exit codes of `classify`

| Result | Exit code |
| --- | --- |
| Reply received, including an `error` reply | 0 |
| Invalid option, or no connection to the socket | 2 |
| Lost connection, invalid or oversized reply, or oversized request | 1 |

The daemon's `error` reply prints to standard output with exit code 0. Always
check each output line for `error`.

### Threads

On macOS, the default is the performance cores, at most 8. On other systems, set
`RAYON_NUM_THREADS` to the number of physical or performance cores. Set
`CANDLE_NUM_THREADS` to the same number. Otherwise, a lower `RAYON_NUM_THREADS`,
`taskset` or a shared machine can oversubscribe the cores.

## Socket protocol

The transport is newline-delimited UTF-8 JSON over a Unix socket. A connection
can carry multiple requests. End each request with a newline:

```json
{"cmd":"git status","cwd":"/project","home":"/user","input_kind":"shell","analyze_scope":true}
```

| Field | Meaning |
| --- | --- |
| `cmd` | Required string to classify |
| `cwd` | Scope working directory; defaults to the daemon's working directory |
| `home` | Scope home directory; defaults to the daemon's `$HOME` |
| `input_kind` | Use `action` to disable shell scope; otherwise shell scope is used |
| `analyze_scope` | Set to `false` to disable scope analysis |

Use absolute `cwd` and `home` paths. The CLI sends its own working directory
and home directory. Direct socket clients must send them if daemon defaults
are not suitable. `input_kind` does not select a different model or rewrite
the submitted text. The CLI accepts only `shell` or `action`. Only the exact
`cmd` string is a model feature; `cwd` and `home` affect scope only.

## Action input format

For `input_kind: "action"`, `cmd` is one JSON object as a string. The model was
trained on stipulated contracts in this format. Other text can produce a result,
but the evaluation does not apply to it. The engine does not validate the format.

| Field | Meaning |
| --- | --- |
| `representation` | Always `action-behavior-evidence-json/0.3` |
| `trust_condition` | Statement that the contract is stipulated, not a backend attestation |
| `effective_namespace` | Where effects occur, for example `initiating host`, `local container` or `nonlocal passive service` |
| `input_schema` | JSON Schema of the action's arguments |
| `evaluated_arguments` | The complete argument values of this call |
| `backend_contract.selected_action` | Name of the selected action |
| `backend_contract.complete` | `true`: the procedure states all effects |
| `backend_contract.resource_scope` | The resources that the action uses |
| `backend_contract.assumptions` | List of assumptions that the procedure depends on |
| `backend_contract.procedure` | What the action does, step by step |

Encode the object as compact UTF-8 JSON with sorted keys, as this Python call does:
`json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`.
Do not append labels or context. Example, with line breaks for this page only:

```json
{"backend_contract":{"assumptions":["The procedure covers all code and conditional paths of this operation. No effects exist beyond those stated."],
"complete":true,"procedure":"Read the existing file config.toml in the project directory and return its content to the caller. Nothing is written or sent.",
"resource_scope":"One project file on the initiating host.","selected_action":"project_files.read"},
"effective_namespace":"initiating host","evaluated_arguments":{"operation":"read","path":"config.toml"},
"input_schema":{"additionalProperties":false,"properties":{"operation":{"enum":["read"],"type":"string"},"path":{"const":"config.toml","type":"string"}},"required":["operation","path"],"type":"object"},
"representation":"action-behavior-evidence-json/0.3",
"trust_condition":"Fictional closed-world contract, not evidence of any real server. All effects are stipulated complete under the assumptions."}
```

Real tool metadata does not attest what a backend does. Treat predictions for
real tool calls as hints that need review. See the [limits](model-card.md#limits).

## Classification response

| Field | Meaning |
| --- | --- |
| `probs` | Nine head names mapped to probabilities |
| `preds` | The same head names mapped to booleans, using bundle thresholds |
| `ms` | Model forward-pass time in milliseconds; not end-to-end latency |
| `token_count` | Token count including special tokens |
| `token_ids` | Tokenized input; treat it as sensitive input data |
| `input_complete` | `true` after successful full-input classification |
| `scope` | Detected categories: `cwd`, `tmp`, `home`, `system`, `other`, or `none` |
| `paths` | `inside` and `outside` path arrays |
| `scope_complete` | Whether the detected path inventory is resolved and not truncated |
| `identity` | Loaded artifact and binary identity |

See [semantics](semantics.md) for the head names and scope limits.
When scope is disabled, the engine returns `scope: ["none"]`, empty path
arrays, and `scope_complete: true`. These values do not mean the input has
no file effects.

`identity` contains protocol `acclass-daemon/1`, `loading_mode`, an
`artifacts` map of SHA-256 hashes for the loaded weights and model metadata,
and `binary_sha256`. Loading modes are `gguf`, `q8-runtime`, `q5-runtime`
and `f32`. Compare identity with a trusted expected value when required;
identity alone does not establish trust.

## Readiness and errors

```json
{"ping":true}
```

The reply contains `ready: true`, the ordered `heads`,
`context_limit_inclusive: 2048`, and `identity`. Readiness is returned by a
daemon that has loaded and warmed up its model.

Errors use an `error` string and include `identity` in daemon replies:

```json
{"error":"missing field 'cmd'"}
```

The example shows the error field only. Do not treat an error or a lost
connection as a negative prediction. The CLI reports a lost connection, invalid
response or oversized response as a failure. It prints an `error` reply with
[exit code 0](#exit-codes-of-classify).

The daemon limits a request line to 256 KiB and allows at most 64 active
connections. It sets read and write timeouts to 30 seconds. Oversized lines
are rejected; if the remaining line cannot be drained within the bound,
the connection is closed. Invalid UTF-8, invalid JSON, incompatible model
metadata, unknown tokens and inputs above 2,048 tokens are not silently
clipped into valid classifications.

## Python client

```python
import json, os, socket

with socket.socket(socket.AF_UNIX) as client:
    client.connect(os.path.expanduser("~/.acclass/ac.sock"))
    client.sendall(json.dumps({"cmd": "git status", "analyze_scope": False}).encode() + b"\n")
    with client.makefile("rb") as response:
        result = json.loads(response.readline())
        if "error" in result:
            raise RuntimeError(result["error"])
        print(result["probs"])
```

Production clients must set time and byte limits, require all nine valid labels
and the full-input fields, and check the model identity. Decisions, permissions,
policy, context, privacy and OS isolation belong to the caller, not the classifier.

See [privacy](privacy.md) for socket permissions and data handling, and
[architecture](architecture.md) for model loading and device selection.
