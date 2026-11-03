// acclass: full-input ModernBERT nine-head inference engine (CUDA, Metal or CPU).
//
//   acclass serve    [--model DIR] [--socket PATH]   resident daemon on a unix socket
//   acclass classify [--socket PATH] CMD|<stdin>   thin client, one JSON response per line
//
// Protocol (NDJSON over unix socket):
//   request : {"cmd": "...", "cwd": "/abs/dir", "home": "/abs/home"}
//   response: {"probs": {...}, "preds": {...}, "ms": 12.3,
//              "scope": ["cwd"], "paths": {"inside": [...], "outside": [...]}}
//   error   : {"error": "..."}
// cwd/home are optional and default to the daemon's process cwd/$HOME.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod attention;
#[cfg(target_os = "macos")]
mod metal;
mod model;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod qgemm;
mod scope;

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::MetadataExt;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const HIDDEN: usize = 768;
const INTERMEDIATE: usize = 1152;
const LAYERS: usize = 22;

const MAX_REQUEST: usize = 256 * 1024;
const MAX_CONNS: usize = 64;
const PROTOCOL: &str = "acclass-daemon/1";
const MODEL_METADATA: [&str; 5] = ["config.json", "tokenizer.json", "thresholds.json", "heads.json", "model_contract.json"];

// Streaming SHA-256 keeps startup memory bounded even for full-precision weights.
fn sha256_file(path: &Path) -> std::io::Result<String> {
    sha256_reader(std::fs::File::open(path)?)
}

fn sha256_reader(mut reader: impl Read) -> std::io::Result<String> {
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = match reader.read(&mut buffer) {
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            other => other?,
        };
        if n == 0 { break; }
        hasher.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn identity_paths(dir: &Path, use_gguf: bool) -> std::io::Result<Vec<PathBuf>> {
    let mut paths: Vec<_> = MODEL_METADATA.iter().map(|name| dir.join(name)).collect();
    paths.push(dir.join(if use_gguf { "model.gguf" } else { "model.safetensors" }));
    paths.push(std::env::current_exe()?);
    Ok(paths)
}

fn artifact_fingerprints(paths: &[PathBuf]) -> std::io::Result<Vec<(u64, u64, u64, i64, i64, i64, i64)>> {
    paths.iter().map(|path| {
        let m = std::fs::metadata(path)?;
        Ok((m.dev(), m.ino(), m.len(), m.mtime(), m.mtime_nsec(), m.ctime(), m.ctime_nsec()))
    }).collect()
}

fn daemon_identity(dir: &Path, quant: Option<GgmlDType>, use_gguf: bool) -> std::io::Result<Value> {
    let paths = identity_paths(dir, use_gguf)?;
    let mut artifacts = serde_json::Map::new();
    for path in &paths[..paths.len()-1] {
        artifacts.insert(path.file_name().unwrap().to_string_lossy().into_owned(), json!(sha256_file(path)?));
    }
    let mode = match (use_gguf, quant) {
        (true, _) => "gguf",
        (false, Some(GgmlDType::Q5_0)) => "q5-runtime",
        (false, Some(_)) => "q8-runtime",
        (false, None) => "f32",
    };
    Ok(json!({"protocol": PROTOCOL, "loading_mode": mode,
        "artifacts": artifacts, "binary_sha256": sha256_file(paths.last().unwrap())?}))
}

static ACTIVE_CONNS: AtomicUsize = AtomicUsize::new(0);

struct ConnGuard;
impl Drop for ConnGuard {
    fn drop(&mut self) {
        ACTIVE_CONNS.fetch_sub(1, Ordering::SeqCst);
    }
}

const HEADS: [&str; 9] = [
    "local_state_change", "outbound", "egress", "remote_state_change",
    "delegated_execution", "exposes_secrets", "destructive",
    "host_control_change", "fetched_or_generated_execution",
];

fn stdin_utf8(bytes: Vec<u8>) -> Result<String, String> {
    String::from_utf8(bytes).map_err(|e| format!("invalid utf-8 in stdin: {e}"))
}

fn home_dir() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/"))
}

fn default_socket() -> PathBuf {
    home_dir().join(".acclass/ac.sock")
}

const HELP: &str = "acclass – standalone activity classifier (any harness)

Usage:
  acclass serve [--model DIR] [--socket PATH] [--gguf | --quant q8|q5|f32]
                [--device auto|cpu|cuda|metal]
  acclass classify [--socket PATH] [--cwd DIR] [--home DIR]
                   [--input-kind shell|action] [--no-scope] [--stdin | CMD...]
  acclass scope [--cwd DIR] [--home DIR] [--stdin | CMD]
  acclass convert [--model DIR] [--out DIR] [--quant q8|q5]

Options:
  -h, --help       Show this help without loading a model
  -V, --version    Show version without loading a model
  --model DIR      Model bundle (default: ~/.acclass/model)
  --socket PATH    Unix socket for any harness (default: ~/.acclass/ac.sock)
  --device NAME    Serve only: auto (default), cpu, cuda, metal; CUDA needs a cuda build
  --gguf           Load model.gguf explicitly; conflicts with --quant
  --quant q8       Load model.safetensors with runtime Q8 quantization
  --quant q5       Opt-in small: runtime Q5_0 (serve) or a Q5_0 GGUF (convert)
  --quant f32      Load model.safetensors as FP32, never GGUF (no FP16 alias)
  --out DIR        New converted GGUF bundle (default: out/model_ninehead_q8)
  --cwd DIR        Scope working directory (default: current directory)
  --home DIR       Scope home directory (default: $HOME)
  --stdin          Read classify commands per line; scope reads all stdin
  --input-kind     shell (default) or action
  --no-scope       Disable scope analysis for classify
  --               Treat remaining arguments as commands

Serving defaults to model.gguf when present, including when both weight files
exist. Otherwise it loads model.safetensors with runtime Q8 quantization.
An explicit --quant always selects safetensors. convert reads safetensors on CPU.
Auto tries CUDA in cuda builds, otherwise Metal on macOS, otherwise CPU.
Only auto may fall back to CPU, and only if device initialization fails.
";

fn cli_error(message: &str) -> ! {
    eprintln!("acclass: {message}\nTry 'acclass --help' for usage.");
    std::process::exit(2);
}

fn option_value(args: &mut impl Iterator<Item = String>, option: &str) -> String {
    args.next().unwrap_or_else(|| cli_error(&format!("{option} needs a value")))
}

// `quant`: None = not given; Some(None) = f32; Some(Some(dtype)) = runtime quantization.
fn loading_mode(dir: &Path, gguf: bool, quant: Option<Option<GgmlDType>>) -> Result<(Option<GgmlDType>, bool), &'static str> {
    if gguf && quant.is_some() {
        return Err("--gguf conflicts with --quant; choose GGUF or safetensors");
    }
    Ok((quant.unwrap_or(Some(GgmlDType::Q8_0)), gguf || (quant.is_none() && dir.join("model.gguf").is_file())))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeviceChoice {
    Auto,
    Cpu,
    Cuda,
    Metal,
}

impl DeviceChoice {
    fn parse(value: &str) -> Result<Self, &'static str> {
        match value {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "metal" => Ok(Self::Metal),
            _ => Err("--device must be auto, cpu, cuda or metal"),
        }
    }
}

// The initializer is separate so failure policy can be tested without GPU hardware.
// This fallback applies only to initialization, never model loading or inference.
fn select_device_with(
    requested: DeviceChoice,
    auto_backend: DeviceChoice,
    initialize: impl FnOnce(DeviceChoice) -> std::io::Result<Device>,
) -> std::io::Result<Device> {
    let target = if requested == DeviceChoice::Auto { auto_backend } else { requested };
    if target == DeviceChoice::Cpu { return Ok(Device::Cpu); }
    match initialize(target) {
        Err(e) if requested == DeviceChoice::Auto => {
            eprintln!("# acclass: {target:?} initialization failed: {e}; falling back to CPU");
            Ok(Device::Cpu)
        }
        result => result,
    }
}

fn select_device(requested: DeviceChoice) -> std::io::Result<Device> {
    let auto_backend = if cfg!(feature = "cuda") { DeviceChoice::Cuda }
        else if cfg!(target_os = "macos") { DeviceChoice::Metal }
        else { DeviceChoice::Cpu };
    select_device_with(requested, auto_backend, |target| {
        let result = match target {
            DeviceChoice::Cuda if !cfg!(feature = "cuda") => return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported, "CUDA is not enabled in this build; rebuild with --features cuda")),
            DeviceChoice::Cuda => Device::new_cuda(0),
            DeviceChoice::Metal if !cfg!(target_os = "macos") => return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported, "Metal is only supported on macOS")),
            DeviceChoice::Metal => Device::new_metal(0),
            _ => unreachable!("only accelerator initialization is delegated"),
        };
        result.map_err(|e| std::io::Error::other(format!("cannot initialize {target:?} device 0: {e}")))
    })
}

fn load_thresholds(model_dir: &Path) -> std::io::Result<Vec<f32>> {
    let raw = std::fs::read_to_string(model_dir.join("thresholds.json"))?;
    let v: Value = serde_json::from_str(&raw)?;
    let invalid = || std::io::Error::new(std::io::ErrorKind::InvalidData, "expected exactly nine finite thresholds in [0,1]");
    let object = v.as_object().ok_or_else(invalid)?;
    if object.len() != HEADS.len() { return Err(invalid()); }
    HEADS.iter().map(|h| {
        let value = object.get(*h).and_then(Value::as_f64).ok_or_else(invalid)?;
        if !value.is_finite() || !(0.0..=1.0).contains(&value) { return Err(invalid()); }
        Ok(value as f32)
    }).collect()
}

struct Engine {
    model: model::ModernBert9Head,
    tokenizer: tokenizers::Tokenizer,
    thresholds: Vec<f32>,
    identity: Value,
}
impl Engine {
    fn run(&self, cmd: &str) -> Result<Value, String> {
        let enc = self
            .tokenizer
            .encode(cmd, true)
            .map_err(|e| format!("tokenizer: {e}"))?;
        let ids = enc.get_ids().to_vec();
        validate_token_length(ids.len())?;
        if self.tokenizer.token_to_id("[UNK]").is_some_and(|unk| ids.contains(&unk)) {
            return Err("unknown input token; model input is incomplete".into());
        }
        let t0 = Instant::now();
        let probs = self
            .model
            .forward(&ids)
            .map_err(|e| format!("forward: {e}"))?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if probs.len() != HEADS.len() || probs.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) {
            return Err("invalid nine-head model probabilities".into());
        }
        let probs_map: serde_json::Map<String, Value> = HEADS
            .iter()
            .zip(&probs)
            .map(|(h, p)| (h.to_string(), json!(p)))
            .collect();
        let preds_map: serde_json::Map<String, Value> = HEADS
            .iter()
            .zip(&probs)
            .zip(&self.thresholds)
            .map(|((h, p), t)| (h.to_string(), json!(p >= t)))
            .collect();
        Ok(json!({"probs": probs_map, "preds": preds_map, "ms": ms,
                  "token_count": ids.len(), "input_complete": true, "token_ids": ids}))
    }
}

fn validate_token_length(length: usize) -> Result<(), String> {
    if length == 0 || length > model::MAX_LEN {
        return Err(format!("input must contain 1..={} inclusive tokens; clipping is forbidden", model::MAX_LEN));
    }
    Ok(())
}

fn validate_model_contract(dir: &Path) -> std::io::Result<()> {
    let invalid = || std::io::Error::new(std::io::ErrorKind::InvalidData, "incompatible model; expected nine heads and full 2048-token contract");
    let heads: Vec<String> = serde_json::from_str(&std::fs::read_to_string(dir.join("heads.json"))?)?;
    let contract: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("model_contract.json"))?)?;
    if heads.iter().map(String::as_str).collect::<Vec<_>>() != HEADS ||
        contract["schema"] != "acclass-vnext-model/1" || contract["heads"] != json!(HEADS) ||
        contract["num_labels"] != 9 || contract["context_limit_inclusive"] != model::MAX_LEN ||
        contract["add_special_tokens"] != true || contract.get("truncation") != Some(&Value::Null) ||
        contract.get("padding") != Some(&Value::Null) || contract["threshold_comparison"] != ">=" ||
        contract["pooling"] != "attention-mask-aware-mean" {
        return Err(invalid());
    }
    load_thresholds(dir)?;
    let config: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
    validate_model_config(&config)?;
    Ok(())
}

// The forward pass is fixed to this architecture. Reject metadata that would
// silently give different pooling, attention, bias, or position semantics.
fn validate_model_config(config: &Value) -> std::io::Result<()> {
    let expected = json!({
        "model_type": "modernbert", "hidden_size": HIDDEN, "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS, "num_attention_heads": 12, "vocab_size": 50368,
        "classifier_pooling": "mean", "classifier_activation": "gelu", "hidden_activation": "gelu",
        "local_attention": 128, "global_attn_every_n_layers": 3, "norm_bias": false,
        "mlp_bias": false, "attention_bias": false, "classifier_bias": false,
        "norm_eps": 1e-5, "pad_token_id": 50283,
    });
    let invalid = || std::io::Error::new(std::io::ErrorKind::InvalidData, "unsupported ModernBERT architecture configuration");
    for (key, value) in expected.as_object().unwrap() {
        if config.get(key) != Some(value) { return Err(invalid()); }
    }
    let layer_types: Vec<&str> = (0..LAYERS).map(|i| if i % 3 == 0 { "full_attention" } else { "sliding_attention" }).collect();
    if config["layer_types"] != json!(layer_types) ||
        config["max_position_embeddings"].as_u64().is_none_or(|limit| limit < model::MAX_LEN as u64) {
        return Err(invalid());
    }
    for (kind, theta) in [("full_attention", 160000.0), ("sliding_attention", 10000.0)] {
        let rope = &config["rope_parameters"][kind];
        if rope["rope_type"] != "default" || rope["rope_theta"].as_f64() != Some(theta) {
            return Err(invalid());
        }
    }
    if config["id2label"].as_object().is_none_or(|labels| labels.len() != HEADS.len()) ||
        HEADS.iter().enumerate().any(|(i, h)| config["id2label"][i.to_string()] != *h) {
        return Err(invalid());
    }
    Ok(())
}

fn write_quant_contract(model_dir: &Path, out_dir: &Path, dtype: GgmlDType) -> Result<(), Box<dyn std::error::Error>> {
    let name = format!("{dtype:?}");
    let mut contract: Value = serde_json::from_str(&std::fs::read_to_string(model_dir.join("model_contract.json"))?)?;
    let source_hash = contract["model_safetensors_sha256"].as_str()
        .ok_or("source contract lacks model_safetensors_sha256")?.to_owned();
    contract["source_dtype"] = json!("float32");
    contract["dtype"] = json!(format!("mixed-{name}/F32"));
    contract["quantization"] = json!(name);
    contract["model_gguf_sha256"] = json!(sha256_file(&out_dir.join("model.gguf"))?);
    contract["source_model_safetensors_sha256"] = json!(source_hash);
    let mut serialized = serde_json::to_vec_pretty(&contract)?;
    serialized.push(b'\n');
    std::fs::write(out_dir.join("model_contract.json"), serialized)?;
    Ok(())
}

/// Write a pre-quantized GGUF: the large matrices and embedding as `q`
/// (Q8_0, or the opt-in small Q5_0), norms/classifier as F32. Copy the
/// tokenizer/thresholds/config alongside so the output dir is a complete model bundle.
fn convert(model_dir: PathBuf, out_dir: PathBuf, q: GgmlDType) -> Result<(), Box<dyn std::error::Error>> {
    validate_model_contract(&model_dir)?;
    // Never overwrite an existing model artifact directory.
    std::fs::create_dir(&out_dir)?;
    let dev = Device::Cpu;
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &[model_dir.join("model.safetensors")],
            DType::F32,
            &dev,
        )?
    };
    let mut t: Vec<(String, QTensor)> = Vec::new();

    fn push<S: Into<candle_core::Shape>>(
        t: &mut Vec<(String, QTensor)>,
        vb: &candle_nn::VarBuilder<'_>,
        name: String,
        dims: S,
        dtype: GgmlDType,
    ) -> candle_core::Result<()> {
        let w = vb.get(dims, &name)?;
        t.push((name, QTensor::quantize(&w, dtype)?));
        Ok(())
    }

    for i in 0..LAYERS {
        let p = format!("model.layers.{i}");
        push(&mut t, &vb, format!("{p}.attn.Wqkv.weight"), (3 * HIDDEN, HIDDEN), q)?;
        push(&mut t, &vb, format!("{p}.attn.Wo.weight"), (HIDDEN, HIDDEN), q)?;
        push(&mut t, &vb, format!("{p}.mlp.Wi.weight"), (2 * INTERMEDIATE, HIDDEN), q)?;
        push(&mut t, &vb, format!("{p}.mlp.Wo.weight"), (HIDDEN, INTERMEDIATE), q)?;
        if i != 0 {
            push(&mut t, &vb, format!("{p}.attn_norm.weight"), (HIDDEN,), GgmlDType::F32)?;
        }
        push(&mut t, &vb, format!("{p}.mlp_norm.weight"), (HIDDEN,), GgmlDType::F32)?;
    }
    push(&mut t, &vb, "head.dense.weight".into(), (HIDDEN, HIDDEN), q)?;
    push(&mut t, &vb, "head.norm.weight".into(), (HIDDEN,), GgmlDType::F32)?;
    push(&mut t, &vb, "model.embeddings.tok_embeddings.weight".into(), (50368, HIDDEN), q)?;
    push(&mut t, &vb, "model.embeddings.norm.weight".into(), (HIDDEN,), GgmlDType::F32)?;
    push(&mut t, &vb, "model.final_norm.weight".into(), (HIDDEN,), GgmlDType::F32)?;
    push(&mut t, &vb, "classifier.weight".into(), (HEADS.len(), HIDDEN), GgmlDType::F32)?;
    push(&mut t, &vb, "classifier.bias".into(), (HEADS.len(),), GgmlDType::F32)?;

    let refs: Vec<(&str, &QTensor)> = t.iter().map(|(n, q)| (n.as_str(), q)).collect();
    let gguf_path = out_dir.join("model.gguf");
    let mut out = std::fs::File::create(&gguf_path)?;
    gguf_file::write(&mut out, &[], &refs)?;
    drop(out);
    println!("# acclass: wrote {} ({} tensors)", gguf_path.display(), t.len());

    for f in ["tokenizer.json", "config.json", "heads.json", "thresholds.json"] {
        std::fs::copy(model_dir.join(f), out_dir.join(f))?;
    }
    write_quant_contract(&model_dir, &out_dir, q)?;
    Ok(())
}

fn serve(model_dir: PathBuf, socket: PathBuf, quant: Option<GgmlDType>, use_gguf: bool, requested_device: DeviceChoice) -> std::io::Result<()> {
    // Fail unsupported explicit choices promptly, before hashing large artifacts.
    let device = select_device(requested_device)?;
    println!("# acclass: inference backend: {:?}", device.location());
    println!(
        "# acclass: loading model from {} (quant={quant:?}, gguf={use_gguf}) ...",
        model_dir.display()
    );
    let identity_start = Instant::now();
    let identity_files = identity_paths(&model_dir, use_gguf)?;
    let fingerprints = artifact_fingerprints(&identity_files)?;
    let identity = daemon_identity(&model_dir, quant, use_gguf)?;
    println!("# acclass: artifact identity hashed in {:.1}ms", identity_start.elapsed().as_secs_f64() * 1e3);
    validate_model_contract(&model_dir)?;
    let mut tokenizer = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    tokenizer.with_truncation(None).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    tokenizer.with_padding(None);
    let thresholds = load_thresholds(&model_dir)?;
    let model = if use_gguf {
        model::ModernBert9Head::load_gguf(&model_dir.join("model.gguf"), device)
    } else {
        model::ModernBert9Head::load(&model_dir, device, quant)
    }.map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    if artifact_fingerprints(&identity_files)? != fingerprints {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "classifier artifacts changed while loading"));
    }
    let engine = Arc::new(Engine { model, tokenizer, thresholds, identity });

    // Warmup: one throwaway inference absorbs the Metal pipeline JIT cost so
    // the first real command is not 100x slower than the rest.
    engine.run("git status").map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    if socket.exists() {
        if probe_listening(&socket) {
            eprintln!("# acclass: daemon already running on {}", socket.display());
            std::process::exit(1);
        }
        let _ = std::fs::remove_file(&socket);
    }
    prepare_socket_parent(&socket)?;
    let listener = UnixListener::bind(&socket)?;
    std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600))?;
    println!("# acclass: listening on {}", socket.display());

    loop {
        let stream = match listener.accept() {
            Ok((s, _)) => s,
            Err(_) => {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
        };
        if ACTIVE_CONNS.fetch_add(1, Ordering::SeqCst) >= MAX_CONNS {
            ACTIVE_CONNS.fetch_sub(1, Ordering::SeqCst);
            drop(stream);
            continue;
        }
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || {
            let _guard = ConnGuard;
            handle_conn(stream, engine);
        });
    }
}

fn prepare_socket_parent(socket: &Path) -> std::io::Result<()> {
    if let Some(parent) = socket.parent().filter(|p| !p.as_os_str().is_empty()) {
        // Never chmod an existing shared directory such as /tmp or the project.
        if !parent.exists() {
            use std::os::unix::fs::DirBuilderExt;
            std::fs::DirBuilder::new().recursive(true).mode(0o700).create(parent)?;
        }
    }
    Ok(())
}

/// Return true if something is already accepting connections at `socket`.
fn probe_listening(socket: &Path) -> bool {
    UnixStream::connect(socket).is_ok()
}

fn handle_conn(stream: UnixStream, engine: Arc<Engine>) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    });
    let mut stream = stream;
    let default_cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let default_home = home_dir();
    loop {
        let line = match read_line_bounded(&mut reader, MAX_REQUEST) {
            Line::Eof => return,
            Line::TooLargeUnterminated => {
                let _ = writeln!(stream, "{}", json!({"error": "command too large; connection closed"}));
                return;
            }
            Line::TooLarge => {
                if writeln!(stream, "{}", json!({"error": "command too large"})).is_err() {
                    return;
                }
                continue;
            }
            Line::Invalid => {
                if writeln!(stream, "{}", json!({"error": "invalid utf-8"})).is_err() {
                    return;
                }
                continue;
            }
            Line::Line(line) => line,
        };
        let mut resp = match serde_json::from_str::<Value>(line.trim()) {
            // Readiness attests the exact loaded model and serving binary.
            Ok(req) if req.get("ping").and_then(Value::as_bool) == Some(true) =>
                json!({"ready": true, "heads": HEADS, "context_limit_inclusive": model::MAX_LEN}),
            Ok(req) => match req.get("cmd").and_then(|c| c.as_str()) {
                Some(cmd) => {
                    let cwd = req
                        .get("cwd")
                        .and_then(|v| v.as_str())
                        .map(PathBuf::from)
                        .unwrap_or_else(|| default_cwd.clone());
                    let home = req
                        .get("home")
                        .and_then(|v| v.as_str())
                        .map(PathBuf::from)
                        .unwrap_or_else(|| default_home.clone());
                    match engine.run(cmd) {
                        Ok(mut v) => {
                            if req.get("input_kind").and_then(Value::as_str) == Some("action") || req.get("analyze_scope").and_then(Value::as_bool) == Some(false) {
                                if let Value::Object(map) = &mut v {
                                    map.insert("scope".into(), json!(["none"]));
                                    map.insert("paths".into(), json!({"inside": [], "outside": []}));
                                    map.insert("scope_complete".into(), json!(true));
                                }
                            } else {
                                let report = scope::analyze(cmd, &cwd, &home);
                                if let Value::Object(map) = &mut v {
                                    map.insert("scope".into(), json!(report.scope));
                                    map.insert("paths".into(), json!(report.paths));
                                    map.insert("scope_complete".into(), json!(report.scope_complete));
                                }
                            }
                            v
                        }
                        Err(e) => json!({"error": e}),
                    }
                }
                None => json!({"error": "missing field 'cmd'"}),
            },
            Err(e) => json!({"error": format!("bad json: {e}")}),
        };
        resp.as_object_mut().unwrap().insert("identity".into(), engine.identity.clone());
        if writeln!(stream, "{resp}").is_err() {
            return;
        }
    }
}

enum Line {
    Eof,
    Line(String),
    TooLarge,
    TooLargeUnterminated,
    Invalid,
}

/// Read one newline-terminated request, up to `max` bytes. An oversized line
/// is drained (bounded) so the next request can still be read.
fn read_line_bounded<R: BufRead>(reader: &mut R, max: usize) -> Line {
    let mut buf: Vec<u8> = Vec::new();
    loop {
        let mut byte = [0u8; 1];
        match reader.read(&mut byte) {
            Ok(0) => {
                return if buf.is_empty() {
                    Line::Eof
                } else {
                    decode(buf)
                };
            }
            Ok(_) => {
                if byte[0] == b'\n' {
                    return decode(buf);
                }
                buf.push(byte[0]);
                if buf.len() > max {
                    return drain(reader);
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Line::Eof,
        }
    }
}

fn decode(buf: Vec<u8>) -> Line {
    match String::from_utf8(buf) {
        Ok(s) => Line::Line(s),
        Err(_) => Line::Invalid,
    }
}

/// Discard the rest of an oversized line, bounded to 8 MiB.
fn drain<R: BufRead>(reader: &mut R) -> Line {
    let limit = 8 * 1024 * 1024;
    let mut seen = 0usize;
    loop {
        let mut byte = [0u8; 1];
        match reader.read(&mut byte) {
            Ok(0) => return Line::TooLargeUnterminated,
            Ok(_) => {
                seen += 1;
                if byte[0] == b'\n' { return Line::TooLarge; }
                if seen > limit { return Line::TooLargeUnterminated; }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return Line::TooLargeUnterminated,
        }
    }
}

fn classify(
    socket: PathBuf,
    cmds: Vec<String>,
    use_stdin: bool,
    cwd: PathBuf,
    home: PathBuf,
    input_kind: &str,
    analyze_scope: bool,
) -> std::io::Result<()> {
    let stream = UnixStream::connect(&socket).unwrap_or_else(|e| {
        eprintln!(
            "# acclass: cannot connect to {}: {e}\n# start the daemon with: acclass serve --model <dir>",
            socket.display()
        );
        std::process::exit(2);
    });
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    let dup = stream.try_clone()?;
    let mut writer = std::io::BufWriter::new(dup);
    let mut reader = BufReader::new(stream);

    let cwd = cwd.to_string_lossy().to_string();
    let home = home.to_string_lossy().to_string();
    let emit = |cmd: &str, writer: &mut dyn Write, reader: &mut BufReader<UnixStream>| -> std::io::Result<()> {
        let request = json!({"cmd": cmd, "cwd": cwd, "home": home, "input_kind": input_kind, "analyze_scope": analyze_scope}).to_string();
        if request.len() > MAX_REQUEST {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "command too large; clipping is forbidden"));
        }
        writeln!(writer, "{request}")?;
        writer.flush()?;
        match read_line_bounded(reader, MAX_REQUEST) {
            Line::Line(line) => writeln!(std::io::stdout().lock(), "{line}"),
            Line::Eof => Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "classifier closed before a response")),
            _ => Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid or oversized classifier response")),
        }
    };

    if use_stdin {
        for line in std::io::stdin().lock().lines() {
            let line = line?; // Invalid UTF-8/read errors must not silently drop an input.
            if !line.trim().is_empty() {
                emit(&line, &mut writer, &mut reader)?;
            }
        }
    } else {
        for cmd in &cmds {
            emit(cmd, &mut writer, &mut reader)?;
        }
    }
    Ok(())
}

/// Apple M: rayon would use every core, and E-cores slow each parallel section. Use
/// the P-cores, at most 8 (more gave nothing on M4 Pro). Environment settings win.
#[cfg(target_os = "macos")]
fn default_cpu_threads() {
    if std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return;
    }
    let threads = candle_core::utils::get_num_threads().min(8).to_string();
    // Safe: main has not started any other thread yet.
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", &threads);
        if std::env::var_os("CANDLE_NUM_THREADS").is_none() {
            std::env::set_var("CANDLE_NUM_THREADS", &threads);
        }
    }
}

fn main() {
    #[cfg(target_os = "macos")]
    default_cpu_threads();
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| {
        eprint!("{HELP}");
        std::process::exit(1);
    });
    match mode.as_str() {
        "--help" | "-h" => { print!("{HELP}"); return; }
        "--version" | "-V" => { println!("acclass {}", env!("CARGO_PKG_VERSION")); return; }
        "serve" | "classify" | "scope" | "convert" => {}
        other => cli_error(&format!("unknown mode: {other}")),
    }
    let mut model_dir = home_dir().join(".acclass/model");
    let mut out_dir = PathBuf::from("out/model_ninehead_q8");
    let mut socket = default_socket();
    let mut quant = None;
    let mut use_gguf = false;
    let mut requested_device = DeviceChoice::Auto;
    let mut cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut home = home_dir();
    let mut rest: Vec<String> = Vec::new();
    let mut use_stdin = false;
    let mut input_kind = String::from("shell");
    let mut analyze_scope = true;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--help" | "-h" => { print!("{HELP}"); return; }
            "--version" | "-V" => { println!("acclass {}", env!("CARGO_PKG_VERSION")); return; }
            "--" => { rest.extend(args); break; }
            "--model" => model_dir = PathBuf::from(option_value(&mut args, "--model")),
            "--out" => out_dir = PathBuf::from(option_value(&mut args, "--out")),
            "--socket" => socket = PathBuf::from(option_value(&mut args, "--socket")),
            "--cwd" => cwd = PathBuf::from(option_value(&mut args, "--cwd")),
            "--home" => home = PathBuf::from(option_value(&mut args, "--home")),
            "--device" => {
                if mode != "serve" { cli_error("--device is only supported by serve"); }
                requested_device = DeviceChoice::parse(&option_value(&mut args, "--device"))
                    .unwrap_or_else(|e| cli_error(e));
            }
            "--gguf" => use_gguf = true,
            "--quant" => match args.next().as_deref() {
                Some("q8") => quant = Some(Some(GgmlDType::Q8_0)),
                Some("q5") => quant = Some(Some(GgmlDType::Q5_0)),
                Some("f32") => quant = Some(None),
                _ => cli_error("--quant must be q8, q5 or f32; FP16 is not supported"),
            },
            "--stdin" => use_stdin = true,
            "--no-scope" => analyze_scope = false,
            "--input-kind" => {
                input_kind = option_value(&mut args, "--input-kind");
                if !matches!(input_kind.as_str(), "shell" | "action") { cli_error("--input-kind must be shell or action"); }
            }
            _ => rest.push(a),
        }
    }

    let res: Result<(), Box<dyn std::error::Error>> = match mode.as_str() {
        "serve" => {
            if !rest.is_empty() { cli_error("unexpected serve argument"); }
            let (quant, use_gguf) = loading_mode(&model_dir, use_gguf, quant)
                .unwrap_or_else(|e| cli_error(e));
            serve(model_dir, socket, quant, use_gguf, requested_device).map_err(Into::into)
        }
        "classify" => classify(socket, rest, use_stdin, cwd, home, &input_kind, analyze_scope).map_err(Into::into),
        "scope" => {
            let cmd = if use_stdin {
                let mut buf = Vec::new();
                std::io::Read::read_to_end(&mut std::io::stdin(), &mut buf).expect("read stdin");
                match stdin_utf8(buf) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("# acclass: {e}");
                        std::process::exit(1);
                    }
                }
            } else {
                rest.join(" ")
            };
            let report = scope::analyze(&cmd, &cwd, &home);
            println!(
                "{}",
                serde_json::to_string(&report).expect("scope report serializes")
            );
            Ok(())
        }
        "convert" => {
            let q = match quant {
                None => GgmlDType::Q8_0,
                Some(Some(q)) => q,
                Some(None) => cli_error("convert writes a quantized GGUF; use --quant q8 or q5"),
            };
            convert(model_dir, out_dir, q)
        }
        other => {
            eprintln!("unknown mode: {other}");
            std::process::exit(1);
        }
    };
    if let Err(e) = res {
        eprintln!("# acclass: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn device_choices_are_exact() {
        for (value, expected) in [
            ("auto", DeviceChoice::Auto), ("cpu", DeviceChoice::Cpu),
            ("cuda", DeviceChoice::Cuda), ("metal", DeviceChoice::Metal),
        ] {
            assert_eq!(DeviceChoice::parse(value).unwrap(), expected);
        }
        for invalid in ["", "gpu", "CUDA", "cuda:0"] {
            assert!(DeviceChoice::parse(invalid).is_err());
        }
    }

    #[test]
    fn explicit_cpu_never_initializes_an_accelerator() {
        let device = select_device_with(DeviceChoice::Cpu, DeviceChoice::Cuda, |_| panic!("CPU must not initialize a GPU")).unwrap();
        assert!(device.is_cpu());
        assert!(select_device(DeviceChoice::Cpu).unwrap().is_cpu());
    }

    #[test]
    fn automatic_device_falls_back_only_on_initialization_failure() {
        for backend in [DeviceChoice::Cuda, DeviceChoice::Metal] {
            let mut initialized = false;
            let device = select_device_with(DeviceChoice::Auto, backend, |target| {
                assert_eq!(target, backend);
                initialized = true;
                Err(std::io::Error::other("synthetic initialization failure"))
            }).unwrap();
            assert!(initialized);
            assert!(device.is_cpu());
            // A successful initializer is also used directly without a retry.
            assert!(select_device_with(DeviceChoice::Auto, backend, |_| Ok(Device::Cpu)).unwrap().is_cpu());
        }
        assert!(select_device_with(DeviceChoice::Auto, DeviceChoice::Cpu,
            |_| panic!("CPU auto must not initialize a GPU")).unwrap().is_cpu());
    }

    #[test]
    fn explicit_accelerator_initialization_errors_are_not_fallbacks() {
        for requested in [DeviceChoice::Cuda, DeviceChoice::Metal] {
            let error = select_device_with(requested, DeviceChoice::Cpu, |target| {
                assert_eq!(target, requested);
                Err(std::io::Error::other("synthetic initialization failure"))
            }).unwrap_err();
            assert_eq!(error.to_string(), "synthetic initialization failure");
        }
    }

    #[test]
    fn serving_weight_selection_respects_explicit_precision() {
        let dir = std::env::temp_dir().join(format!("acclass-cli-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(loading_mode(&dir, false, None).unwrap(), (Some(GgmlDType::Q8_0), false));
        std::fs::write(dir.join("model.gguf"), b"fixture").unwrap();
        assert_eq!(loading_mode(&dir, false, None).unwrap(), (Some(GgmlDType::Q8_0), true));
        std::fs::write(dir.join("model.safetensors"), b"fixture").unwrap();
        assert_eq!(loading_mode(&dir, false, None).unwrap(), (Some(GgmlDType::Q8_0), true));
        assert_eq!(loading_mode(&dir, true, None).unwrap(), (Some(GgmlDType::Q8_0), true));
        assert_eq!(loading_mode(&dir, false, Some(None)).unwrap(), (None, false));
        assert_eq!(loading_mode(&dir, false, Some(Some(GgmlDType::Q8_0))).unwrap(), (Some(GgmlDType::Q8_0), false));
        let q5 = Some(GgmlDType::Q5_0);
        assert_eq!(loading_mode(&dir, false, Some(q5)).unwrap(), (q5, false));
        assert!(loading_mode(&dir, true, Some(None)).is_err());
        assert!(loading_mode(&dir, true, Some(Some(GgmlDType::Q8_0))).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn streaming_sha256_matches_standard_vectors_and_partial_reads() {
        for (input, digest) in [
            ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            ("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            ("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
        ] {
            assert_eq!(sha256_reader(input.as_bytes()).unwrap(), digest);
        }
        let million = vec![b'a'; 1_000_000];
        assert_eq!(sha256_reader(Cursor::new(&million)).unwrap(), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
        struct Partial(Cursor<Vec<u8>>);
        impl Read for Partial {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                let n = out.len().min(7);
                self.0.read(&mut out[..n])
            }
        }
        assert_eq!(sha256_reader(Partial(Cursor::new(million))).unwrap(), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    }

    #[test]
    fn q8_contract_pins_output_and_preserves_source_provenance() {
        let dir = std::env::temp_dir().join(format!("acclass-q8-contract-{}", std::process::id()));
        let source = dir.join("source");
        let output = dir.join("output");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::create_dir_all(&output).unwrap();
        let original = json!({"schema": "acclass-vnext-model/1", "dtype": "float32",
            "quantization": "unquantized; Q8_0-compatible matrices",
            "model_safetensors_sha256": "c".repeat(64), "heads": HEADS,
            "thresholds_sha256": "t".repeat(64), "policy": {"preserved": true}});
        let source_bytes = serde_json::to_vec(&original).unwrap();
        std::fs::write(source.join("model_contract.json"), &source_bytes).unwrap();
        std::fs::write(output.join("model.gguf"), b"abc").unwrap();
        write_quant_contract(&source, &output, GgmlDType::Q8_0).unwrap();
        let converted: Value = serde_json::from_slice(&std::fs::read(output.join("model_contract.json")).unwrap()).unwrap();
        assert_eq!(converted["source_dtype"], "float32");
        assert_eq!(converted["dtype"], "mixed-Q8_0/F32");
        assert_eq!(converted["quantization"], "Q8_0");
        assert_eq!(converted["model_gguf_sha256"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(converted["source_model_safetensors_sha256"], original["model_safetensors_sha256"]);
        for (key, value) in original.as_object().unwrap() {
            if key != "dtype" && key != "quantization" {
                assert_eq!(&converted[key], value, "changed {key}");
            }
        }
        assert_eq!(std::fs::read(source.join("model_contract.json")).unwrap(), source_bytes);
        assert_eq!(std::fs::read(output.join("model.gguf")).unwrap(), b"abc");
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn model_config() -> Value {
        let layers: Vec<&str> = (0..LAYERS).map(|i| if i % 3 == 0 { "full_attention" } else { "sliding_attention" }).collect();
        let labels: serde_json::Map<String, Value> = HEADS.iter().enumerate()
            .map(|(i, h)| (i.to_string(), json!(h))).collect();
        json!({
            "model_type": "modernbert", "hidden_size": 768, "intermediate_size": 1152,
            "num_hidden_layers": 22, "num_attention_heads": 12, "vocab_size": 50368,
            "classifier_pooling": "mean", "classifier_activation": "gelu", "hidden_activation": "gelu",
            "local_attention": 128, "global_attn_every_n_layers": 3, "norm_bias": false,
            "mlp_bias": false, "attention_bias": false, "classifier_bias": false,
            "norm_eps": 1e-5, "pad_token_id": 50283, "max_position_embeddings": 8192,
            "layer_types": layers, "id2label": labels,
            "rope_parameters": {
                "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}
            }
        })
    }

    #[test]
    fn architecture_metadata_cannot_change_fixed_forward_semantics() {
        let valid = model_config();
        assert!(validate_model_config(&valid).is_ok());
        for (key, value) in [
            ("classifier_pooling", json!("cls")), ("hidden_size", json!(1024)),
            ("norm_eps", json!(1e-6)), ("attention_bias", json!(true)),
            ("max_position_embeddings", json!(2047)), ("layer_types", json!([])),
            ("rope_parameters", json!({})), ("id2label", json!({})),
        ] {
            let mut invalid = valid.clone();
            invalid[key] = value;
            assert!(validate_model_config(&invalid).is_err(), "accepted {key}");
        }
        for key in valid.as_object().unwrap().keys() {
            let mut invalid = valid.clone();
            invalid.as_object_mut().unwrap().remove(key);
            assert!(validate_model_config(&invalid).is_err(), "missing {key}");
        }
    }

    #[test]
    fn model_bundle_requires_exact_heads_thresholds_and_full_input_contract() {
        let dir = std::env::temp_dir().join(format!("acclass-contract-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, value: &Value| {
            std::fs::write(dir.join(name), serde_json::to_vec(value).unwrap()).unwrap();
        };
        let thresholds: serde_json::Map<String, Value> = HEADS.iter().map(|h| (h.to_string(), json!(0.5))).collect();
        let thresholds = Value::Object(thresholds);
        let contract = json!({"schema": "acclass-vnext-model/1", "heads": HEADS, "num_labels": 9,
            "context_limit_inclusive": 2048, "add_special_tokens": true, "truncation": null,
            "padding": null, "pooling": "attention-mask-aware-mean", "threshold_comparison": ">="});
        write("heads.json", &json!(HEADS));
        write("thresholds.json", &thresholds);
        write("config.json", &model_config());
        write("model_contract.json", &contract);
        assert!(validate_model_contract(&dir).is_ok());
        for key in contract.as_object().unwrap().keys() {
            let mut invalid = contract.clone();
            invalid.as_object_mut().unwrap().remove(key);
            write("model_contract.json", &invalid);
            assert!(validate_model_contract(&dir).is_err(), "missing contract {key}");
        }
        write("model_contract.json", &contract);
        let mut reversed = HEADS.to_vec();
        reversed.reverse();
        write("heads.json", &json!(reversed));
        assert!(validate_model_contract(&dir).is_err());
        write("heads.json", &json!(HEADS));
        for value in [json!(-0.01), json!(1.01), json!("0.5"), Value::Null] {
            let mut invalid = thresholds.clone();
            invalid[HEADS[0]] = value;
            write("thresholds.json", &invalid);
            assert!(load_thresholds(&dir).is_err());
        }
        let mut extra = thresholds.clone();
        extra["legacy_head"] = json!(0.5);
        write("thresholds.json", &extra);
        assert!(load_thresholds(&dir).is_err());
        write("thresholds.json", &thresholds);
        assert!(validate_model_contract(&dir).is_ok());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn socket_setup_preserves_existing_parent_permissions() {
        let dir = std::env::temp_dir().join(format!("acclass-permissions-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        prepare_socket_parent(&dir.join("ac.sock")).unwrap();
        assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o755);
        prepare_socket_parent(&dir.join("new/nested/ac.sock")).unwrap();
        assert_eq!(std::fs::metadata(dir.join("new/nested")).unwrap().permissions().mode() & 0o777, 0o700);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn full_input_lengths_are_never_clipped() {
        assert!(validate_token_length(1).is_ok());
        assert!(validate_token_length(1453).is_ok());
        assert!(validate_token_length(2048).is_ok());
        assert!(validate_token_length(2049).is_err());
        assert!(validate_token_length(0).is_err());
    }

    #[test]
    fn bounded_reader_returns_lines() {
        let mut r = Cursor::new(b"{\"cmd\":\"ls\"}\nsecond\n".to_vec());
        match read_line_bounded(&mut r, 1024) {
            Line::Line(s) => assert_eq!(s, "{\"cmd\":\"ls\"}"),
            _ => panic!("expected a line"),
        }
        match read_line_bounded(&mut r, 1024) {
            Line::Line(s) => assert_eq!(s, "second"),
            _ => panic!("expected a line"),
        }
        assert!(matches!(read_line_bounded(&mut r, 1024), Line::Eof));
    }

    #[test]
    fn bounded_reader_rejects_oversized_and_resyncs() {
        let mut data = vec![b'x'; 2048];
        data.push(b'\n');
        data.extend_from_slice(b"ok\n");
        let mut r = Cursor::new(data);
        assert!(matches!(read_line_bounded(&mut r, 16), Line::TooLarge));
        match read_line_bounded(&mut r, 16) {
            Line::Line(s) => assert_eq!(s, "ok"),
            _ => panic!("expected resync after oversized line"),
        }
    }

    #[test]
    fn bounded_reader_never_treats_an_undrained_tail_as_a_new_request() {
        let mut data = vec![b'x'; 8 * 1024 * 1024 + 64];
        data.extend_from_slice(b"{\"cmd\":\"true\"}\n");
        let mut r = Cursor::new(data);
        assert!(matches!(read_line_bounded(&mut r, 16), Line::TooLargeUnterminated));
        let mut r = Cursor::new(vec![b'x'; 64]);
        assert!(matches!(read_line_bounded(&mut r, 16), Line::TooLargeUnterminated));
    }

    #[test]
    fn bounded_reader_accepts_exact_byte_limit_and_full_unicode() {
        let mut r = Cursor::new("☃☃\n".as_bytes());
        assert!(matches!(read_line_bounded(&mut r, 6), Line::Line(s) if s == "☃☃"));
        let mut r = Cursor::new("☃☃\n".as_bytes());
        assert!(matches!(read_line_bounded(&mut r, 5), Line::TooLarge));
    }

    #[test]
    fn bounded_reader_rejects_invalid_utf8() {
        let mut r = Cursor::new(b"\xff\xfe\n".to_vec());
        assert!(matches!(read_line_bounded(&mut r, 16), Line::Invalid));
    }

    #[test]
    fn stdin_decoder_rejects_invalid_utf8() {
        assert!(stdin_utf8(vec![0xff, 0xfe]).is_err());
        assert_eq!(stdin_utf8(b"cat x".to_vec()).unwrap(), "cat x");
    }

    #[test]
    fn probe_detects_live_socket() {
        let dir = std::env::temp_dir().join(format!("acclass-probe-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let sock = dir.join("probe.sock");
        let _ = std::fs::remove_file(&sock);
        assert!(!probe_listening(&sock));
        let listener = UnixListener::bind(&sock).expect("bind probe socket");
        assert!(probe_listening(&sock));
        drop(listener);
        let _ = std::fs::remove_file(&sock);
        let _ = std::fs::remove_dir(&dir);
    }
}
