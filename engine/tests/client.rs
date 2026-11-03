// Thin-client fixtures only. The listener never evaluates a command or loads weights.
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::DirBuilderExt;
use std::os::unix::net::UnixListener;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn client_fixture(stdin: Option<&[u8]>, expect_request: bool) -> std::process::Output {
    let dir = std::env::temp_dir().join(format!(
        "acclass-client-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::DirBuilder::new().mode(0o700).create(&dir).unwrap();
    let socket = dir.join("stub.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let server = std::thread::spawn(move || {
        let (connection, _) = listener.accept().unwrap();
        let mut request = String::new();
        BufReader::new(connection).read_line(&mut request).unwrap();
        assert_eq!(!request.is_empty(), expect_request);
        // Close without a reply: the client must report failure, not success
        // with empty stdout. For invalid input, no command reaches this socket.
    });
    let mut command = Command::new(env!("CARGO_BIN_EXE_acclass"));
    command.args(["classify", "--socket"]).arg(&socket)
        .stdout(Stdio::piped()).stderr(Stdio::piped());
    if stdin.is_some() { command.arg("--stdin").stdin(Stdio::piped()); }
    else { command.arg("true"); }
    let mut child = command.spawn().unwrap();
    if let Some(input) = stdin {
        child.stdin.take().unwrap().write_all(input).unwrap();
    }
    let output = child.wait_with_output().unwrap();
    server.join().unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    output
}

#[test]
fn help_and_version_do_not_load_models_or_connect() {
    for args in [
        vec!["--help"], vec!["-h"], vec!["serve", "--help"],
        vec!["classify", "-h"], vec!["scope", "--help"], vec!["convert", "-h"],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_acclass"))
            .args(args).env("HOME", "/nonexistent/acclass-help-home").output().unwrap();
        assert!(result.status.success());
        assert!(result.stderr.is_empty());
        let help = String::from_utf8(result.stdout).unwrap();
        assert!(help.contains("Usage:"));
        assert!(help.contains("~/.acclass/model"));
        assert!(help.contains("~/.acclass/ac.sock"));
        assert!(help.contains("both weight files"));
        assert!(help.contains("--device auto|cpu|cuda|metal"));
    }
    for args in [vec!["--version"], vec!["-V"], vec!["serve", "--version"], vec!["serve", "-V"]] {
        let result = Command::new(env!("CARGO_BIN_EXE_acclass"))
            .args(args).env("HOME", "/nonexistent/acclass-version-home").output().unwrap();
        assert!(result.status.success());
        assert!(result.stderr.is_empty());
        assert_eq!(String::from_utf8(result.stdout).unwrap(), format!("acclass {}\n", env!("CARGO_PKG_VERSION")));
    }
}

#[test]
fn invalid_precision_and_conflicts_fail_before_model_loading() {
    for args in [
        vec!["serve", "--quant", "f16"],
        vec!["serve", "--gguf", "--quant", "f32"],
        vec!["serve", "--quant", "f32", "--gguf"],
        vec!["serve", "--model"],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_acclass")).args(args).output().unwrap();
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty());
        let error = String::from_utf8(result.stderr).unwrap();
        assert!(error.contains("--help"));
        assert!(!error.contains("panicked"));
    }
}

#[test]
fn invalid_device_options_fail_before_model_loading() {
    for args in [
        vec!["serve", "--device"],
        vec!["serve", "--device", "gpu"],
        vec!["serve", "--device", "CUDA"],
        vec!["classify", "--device", "cpu"],
        vec!["scope", "--device", "auto"],
        vec!["convert", "--device", "cpu"],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_acclass")).args(args).output().unwrap();
        assert_eq!(result.status.code(), Some(2));
        assert!(result.stdout.is_empty());
        let error = String::from_utf8(result.stderr).unwrap();
        assert!(error.contains("--device"));
        assert!(error.contains("--help"));
        assert!(!error.contains("panicked"));
    }
}

#[test]
fn unsupported_devices_fail_before_artifact_hashing() {
    let mut unsupported = Vec::new();
    if !cfg!(feature = "cuda") { unsupported.push(("cuda", "--features cuda")); }
    if !cfg!(target_os = "macos") { unsupported.push(("metal", "only supported on macOS")); }
    for (device, message) in unsupported {
        let result = Command::new(env!("CARGO_BIN_EXE_acclass"))
            .args(["serve", "--device", device, "--model", "/nonexistent/acclass-device-model"])
            .output().unwrap();
        assert_eq!(result.status.code(), Some(1));
        assert!(result.stdout.is_empty());
        let error = String::from_utf8(result.stderr).unwrap();
        assert!(error.contains(message), "{error}");
        assert!(!error.contains("No such file"), "device must fail before model I/O: {error}");
    }
}

#[test]
fn cpu_device_is_logged_before_missing_model_error() {
    let result = Command::new(env!("CARGO_BIN_EXE_acclass"))
        .args(["serve", "--device", "cpu", "--model", "/nonexistent/acclass-device-model"])
        .output().unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert!(String::from_utf8(result.stdout).unwrap().contains("inference backend: Cpu"));
    assert!(String::from_utf8(result.stderr).unwrap().contains("No such file"));
}

#[cfg(all(not(feature = "cuda"), not(target_os = "macos")))]
#[test]
fn default_and_auto_device_preserve_cpu_serving() {
    for extra in [vec![], vec!["--device", "auto"]] {
        let result = Command::new(env!("CARGO_BIN_EXE_acclass"))
            .args(["serve", "--model", "/nonexistent/acclass-device-model"])
            .args(extra).output().unwrap();
        assert_eq!(result.status.code(), Some(1));
        assert!(String::from_utf8(result.stdout).unwrap().contains("inference backend: Cpu"));
        assert!(!String::from_utf8(result.stderr).unwrap().contains("falling back"));
    }
}

#[test]
fn default_socket_is_under_standalone_home_not_xdg_cache() {
    let result = Command::new(env!("CARGO_BIN_EXE_acclass"))
        .args(["classify", "true"])
        .env("HOME", "/nonexistent/acclass-client-home")
        .env("XDG_CACHE_HOME", "/nonexistent/xdg-cache")
        .output().unwrap();
    assert!(!result.status.success());
    let error = String::from_utf8(result.stderr).unwrap();
    assert!(error.contains("/nonexistent/acclass-client-home/.acclass/ac.sock"));
    assert!(!error.contains("xdg-cache"));
}

#[test]
fn closed_socket_is_not_a_successful_empty_classification() {
    let result = client_fixture(None, true);
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8(result.stderr).unwrap().contains("closed before a response"));
}

#[test]
fn invalid_utf8_stdin_is_not_silently_skipped() {
    let result = client_fixture(Some(b"\xff\n"), false);
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
}

#[test]
fn oversized_stdin_is_neither_sent_nor_clipped() {
    let mut input = vec![b'x'; 256 * 1024];
    input.push(b'\n');
    let result = client_fixture(Some(&input), false);
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8(result.stderr).unwrap().contains("clipping is forbidden"));
}
