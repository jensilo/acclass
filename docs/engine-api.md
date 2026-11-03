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

| Default | Value |
| --- | --- |
| Model folder | `~/.acclass/model` |
| Socket | `~/.acclass/ac.sock` |

- `--help` and `--version` do not load a model or connect to the daemon. The daemon is
  the `acclass serve` process.
- `classify` sends text. It does not run the text. Quote a command so that your shell
  does not expand it first.
- `classify --stdin` reads one command from each nonblank line. It prints one JSON
  line for each command.
- Without `--stdin`, `classify` sends each argument as a separate request.
- `scope --stdin` reads all input as one command. `scope` needs no model and no daemon.
- After `--`, all arguments are commands: `acclass classify -- --version` classifies
  the text `--version`.
- `classify` also treats an unknown option as a command. Check the spelling.

`serve` accepts requests only after it loads and tests the model. To wait for it:

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

The CLI prints an `error` reply to standard output and exits with 0. Check each output
line for `error`.

### Threads

On macOS, the default is the performance cores, at most 8. On other systems, set
`RAYON_NUM_THREADS` to the number of physical or performance cores, and set
`CANDLE_NUM_THREADS` to the same number. Without this, a lower `RAYON_NUM_THREADS`,
`taskset` or a shared machine can oversubscribe the cores.

## Socket protocol

The protocol uses UTF-8 JSON lines over a Unix socket. One connection can carry many
requests. End each request with a newline:

```json
{"cmd":"git status","cwd":"/project","home":"/user","input_kind":"shell","analyze_scope":true}
```

| Field | Meaning |
| --- | --- |
| `cmd` | Required. The string to classify. |
| `cwd` | Working directory for scope. Default: the daemon's working directory. |
| `home` | Home directory for scope. Default: the daemon's `$HOME`. |
| `input_kind` | `action` turns off scope. All other values use shell scope. |
| `analyze_scope` | `false` turns off scope. |

- Use absolute `cwd` and `home` paths. The CLI sends its own values. Socket clients
  must send them if the daemon's defaults are wrong for the request.
- The model reads only the exact `cmd` string. `cwd` and `home` affect scope only.
- `input_kind` does not select another model or change the text.
- The CLI accepts only `shell` or `action`.

## Action input format

For `input_kind: "action"`, `cmd` is one JSON object as a string. We trained the model
on invented tool descriptions in this format. Each description states all effects of
its tool. Other text can produce a result, but the evaluation does not apply to it. The
engine does not validate the format.

| Field | Meaning |
| --- | --- |
| `representation` | Always `action-behavior-evidence-json/0.3` |
| `trust_condition` | States that the description is an assumption, not a proof of what a backend does |
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
Do not add labels or context. This example has line breaks for this page only:

```json
{"backend_contract":{"assumptions":["The procedure covers all code and conditional paths of this operation. No effects exist beyond those stated."],
"complete":true,"procedure":"Read the existing file config.toml in the project directory and return its content to the caller. Nothing is written or sent.",
"resource_scope":"One project file on the initiating host.","selected_action":"project_files.read"},
"effective_namespace":"initiating host","evaluated_arguments":{"operation":"read","path":"config.toml"},
"input_schema":{"additionalProperties":false,"properties":{"operation":{"enum":["read"],"type":"string"},"path":{"const":"config.toml","type":"string"}},"required":["operation","path"],"type":"object"},
"representation":"action-behavior-evidence-json/0.3",
"trust_condition":"Fictional closed-world contract, not evidence of any real server. All effects are stipulated complete under the assumptions."}
```

Real tool descriptions can be wrong or incomplete. Treat predictions for real tool
calls as hints that need review. See the [limits](model-card.md#limits).

## Classification response

| Field | Meaning |
| --- | --- |
| `probs` | Nine label names mapped to probabilities |
| `preds` | The same names mapped to booleans, using the bundle's thresholds |
| `ms` | Model time in milliseconds, without tokenizer, scope and socket time |
| `token_count` | Number of tokens, including special tokens |
| `token_ids` | Tokenized input. Treat it as sensitive input data. |
| `input_complete` | `true` after the model read the full input |
| `scope` | Detected categories: `cwd`, `tmp`, `home`, `system`, `other` or `none` |
| `paths` | `inside` and `outside` path arrays |
| `scope_complete` | `true` if the detected paths are resolved and the list is not truncated |
| `identity` | Hashes of the loaded files and the binary |

See [semantics](semantics.md) for the labels and the scope limits.

If scope is off, the response has `scope: ["none"]`, empty path arrays and
`scope_complete: true`. These values do not mean that the input has no file effects.

`identity` contains:

| Field | Meaning |
| --- | --- |
| `protocol` | `acclass-daemon/1` |
| `loading_mode` | `gguf`, `q8-runtime`, `q5-runtime` or `f32` |
| `artifacts` | SHA-256 hashes of the loaded weights and model metadata |
| `binary_sha256` | SHA-256 hash of the daemon binary |

Compare `identity` with a value that you trust. The hashes alone do not show that the
files are trustworthy.

## Readiness

```json
{"ping":true}
```

The reply contains `ready: true`, the ordered `heads`,
`context_limit_inclusive: 2048` and `identity`. The daemon listens only after it loads
and tests its model.

## Errors

An error reply has an `error` string:

```json
{"error":"missing field 'cmd'"}
```

Most error replies also contain `identity`. Replies to an oversized line or to invalid
UTF-8 do not.

Do not treat an error or a lost connection as a negative prediction.

| Condition | Daemon behaviour |
| --- | --- |
| Line above 256 KiB | Replies with an error. Closes the connection if it cannot read to the end of the line within its limit. |
| Invalid UTF-8, invalid JSON | Replies with an error. |
| Unknown token, above 2,048 tokens | Replies with an error. It never cuts the input. |
| More than 64 connections | Closes the new connection. |
| No data or no write progress for 30 seconds | Read or write timeout. |
| Incompatible model metadata | Refuses to start. |

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

A production client must:

- set time and byte limits
- require all nine valid labels and the full-input fields
- check the model identity

The caller owns decisions, permissions, policy, context, privacy and OS isolation.

See [privacy](privacy.md) for socket permissions and
[architecture](architecture.md) for model loading and devices.
