// Bash command scope analysis.
//
// Parses a command with tree-sitter-bash, walks it for path-like argument and
// redirection tokens, expands `~`/`$HOME`/`$PWD` against the tool's cwd/home,
// tracks literal `cd` targets across `&&`/`;`/newline lists, and classifies
// every resolved path as one of cwd/tmp/home/system/other. The extension's
// per-primitive policy consumes the result (`scope` + `paths` + `scope_complete`).
// `scope_complete` covers the detected path inventory, not all command effects.
//
// Scope is a *set of categories*; a command is sensitive if any category is
// sensitive (home/system/other). See docs/architecture.md.
//
// Bare relative arguments (`foo` in `rm -rf foo`) are only counted when the
// token exists on disk, or when the command is known to operate on file paths
// (see CREATE_COMMANDS: `touch new` counts even though `new` is absent). This
// keeps `rm foo` scoped while `systemctl restart nginx` stays scope=none.
// The factual host-control head handles operations without path arguments.

use serde::Serialize;
use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};
use tree_sitter::Node;

const MAX_PATHS: usize = 12;

/// Commands whose bare arguments are file paths even when the target does not
/// exist yet (creation/rename). Prevents false blocks on `touch new`, `mkdir build`.
const CREATE_COMMANDS: [&str; 12] = [
    "touch", "mkdir", "rmdir", "ln", "install", "truncate", "mkfifo", "mknod", "cp", "mv",
    "rsync", "dd",
];

#[derive(Serialize, Default)]
pub struct Report {
    /// Detected path inventory is resolved and was not truncated.
    /// This does not claim the analyzer identified every command effect.
    pub scope_complete: bool,
    pub scope: Vec<String>,
    pub paths: Paths,
}

#[derive(Serialize, Default)]
pub struct Paths {
    pub inside: Vec<String>,
    pub outside: Vec<String>,
}

#[derive(Clone)]
struct State {
    /// Current shell directory, mutated by literal `cd`.
    cwd: PathBuf,
    home: PathBuf,
    /// Set when a `cd` target could not be resolved; subsequent relative paths
    /// are then treated as unknown (scope `other`) rather than assumed cwd.
    unknown_cwd: bool,
}

struct Acc {
    scope_complete: bool,
    scope: BTreeSet<&'static str>,
    inside: Vec<String>,
    outside: Vec<String>,
    seen: BTreeSet<String>,
    /// The tool's original cwd; fixed for the whole command.
    cwd: PathBuf,
    home: PathBuf,
}

impl Acc {
    fn new(cwd: PathBuf, home: PathBuf) -> Self {
        Self {
            scope_complete: true,
            scope: BTreeSet::new(),
            inside: Vec::new(),
            outside: Vec::new(),
            seen: BTreeSet::new(),
            cwd,
            home,
        }
    }

    fn add(&mut self, p: PathBuf) {
        let s = p.to_string_lossy().to_string();
        if !self.seen.insert(s.clone()) {
            return;
        }
        if within(&p, &self.cwd) {
            self.scope.insert("cwd");
            if self.inside.len() < MAX_PATHS {
                self.inside.push(s);
            } else {
                self.scope_complete = false;
            }
        } else if is_tmp(&p) {
            self.scope.insert("tmp");
        } else if within(&p, &self.home) {
            self.scope.insert("home");
            self.add_outside(s);
        } else if is_system(&p) {
            self.scope.insert("system");
            if !is_benign_dev(&p) {
                self.add_outside(s);
            }
        } else {
            self.scope.insert("other");
            self.add_outside(s);
        }
    }

    fn add_outside(&mut self, path: String) {
        if self.outside.len() < MAX_PATHS {
            self.outside.push(path);
        } else {
            self.scope_complete = false;
        }
    }

    fn add_unresolved(&mut self) {
        self.scope_complete = false;
        self.scope.insert("other");
    }

    fn finish(self) -> Report {
        let mut scope: Vec<String> = self.scope.into_iter().map(str::to_string).collect();
        if scope.is_empty() {
            scope.push("none".to_string());
        }
        Report {
            scope_complete: self.scope_complete,
            scope,
            paths: Paths {
                inside: self.inside,
                outside: self.outside,
            },
        }
    }
}

/// Parse `cmd` and classify the paths it touches, relative to `cwd`/`home`.
pub fn analyze(cmd: &str, cwd: &Path, home: &Path) -> Report {
    let home = normalize_path(home);
    let cwd = normalize_path(cwd);
    let mut st = State {
        cwd: cwd.clone(),
        home: home.clone(),
        unknown_cwd: false,
    };
    let mut acc = Acc::new(cwd, home);
    let src = cmd.as_bytes();

    let mut parser = tree_sitter::Parser::new();
    let language: tree_sitter::Language = tree_sitter_bash::LANGUAGE.into();
    if parser.set_language(&language).is_err() {
        acc.add_unresolved();
        return acc.finish();
    }
    let Some(tree) = parser.parse(cmd, None) else {
        acc.add_unresolved();
        return acc.finish();
    };
    let root = tree.root_node();
    if root.has_error() {
        acc.add_unresolved();
    }
    let named = named_children(root);
    // A bare path (e.g. a file-op descriptor) parses as a single token.
    if named.len() == 1 && is_token_kind(named[0].kind()) {
        trace_node(named[0], src, &st, &mut acc, false);
    } else {
        walk(root, src, &mut st, &mut acc);
    }
    acc.finish()
}

fn walk(n: Node, src: &[u8], st: &mut State, acc: &mut Acc) {
    match n.kind() {
        // `;`-lists and redirected statements run in the same shell, so `cd`
        // propagates across their children in order.
        "program" | "list" | "compound_statement" | "redirected_statement" => {
            for ch in named_children(n) {
                walk(ch, src, st, acc);
            }
        }
        // Pipeline elements run in subshells: a `cd` in one does not leak.
        "pipeline" => {
            for ch in named_children(n) {
                let mut sub = st.clone();
                walk(ch, src, &mut sub, acc);
            }
        }
        // Subshells and command substitutions get an isolated cwd.
        "subshell" | "command_substitution" => {
            let mut sub = st.clone();
            for ch in named_children(n) {
                walk(ch, src, &mut sub, acc);
            }
        }
        "command" => process_command(n, src, st, acc),
        "file_redirect" => process_file_redirect(n, src, st, acc),
        "heredoc_redirect" | "herestring_redirect" => walk_redirect_body(n, src, st, acc),
        "variable_assignment" => {}
        k if is_token_kind(k) => {
            // Reached only for bare tokens outside a `command` node (e.g. the
            // whole command is one path).
            trace_node(n, src, st, acc, false);
        }
        _ => {
            for ch in named_children(n) {
                walk(ch, src, st, acc);
            }
        }
    }
}

fn process_command(n: Node, src: &[u8], st: &mut State, acc: &mut Acc) {
    let children = named_children(n);
    let name_node = children.iter().find(|c| c.kind() == "command_name").copied();
    let name = name_node.map(|c| text(c, src).trim().to_string());
    let is_create = name.as_deref().is_some_and(is_create_command);

    if let Some(node) = name_node {
        // `./deploy.sh` and `/usr/bin/x` are file accesses; a bare `rm` is not.
        for ch in named_children(node) {
            if name.as_deref().is_some_and(looks_like_path_token) {
                trace_node(ch, src, st, acc, false);
            } else {
                scan_substitutions(ch, src, st, acc);
            }
        }
    }

    if name.as_deref() == Some("cd") {
        let mut tokens = children
            .iter()
            .skip_while(|c| c.kind() != "command_name")
            .skip(1)
            .filter(|c| is_token_kind(c.kind()));
        let target = match tokens.next() {
            Some(t) if text(*t, src) == "--" => tokens.next(),
            first => first,
        };
        match target {
            None => {
                st.cwd = st.home.clone();
                st.unknown_cwd = false;
            }
            Some(node) => apply_cd(*node, src, st),
        }
        if st.unknown_cwd {
            acc.scope_complete = false;
        }
        return;
    }

    let mut after_name = false;
    for ch in &children {
        if ch.kind() == "command_name" {
            after_name = true;
            continue;
        }
        if !after_name {
            continue; // variable_assignment before the command name
        }
        handle_arg(*ch, src, st, acc, is_create);
    }
}

fn handle_arg(n: Node, src: &[u8], st: &State, acc: &mut Acc, bare_ok: bool) {
    match n.kind() {
        "file_redirect" => process_file_redirect(n, src, st, acc),
        "heredoc_redirect" | "herestring_redirect" => walk_redirect_body(n, src, st, acc),
        "command_substitution" => {
            // Its side effects can be scanned, but its argument output is unknown.
            acc.scope_complete = false;
            let mut sub = st.clone();
            walk(n, src, &mut sub, acc);
        }
        k if is_token_kind(k) => trace_node(n, src, st, acc, bare_ok),
        _ => {
            let mut sub = st.clone();
            walk(n, src, &mut sub, acc);
        }
    }
}

fn process_file_redirect(n: Node, src: &[u8], st: &State, acc: &mut Acc) {
    // `2>&1` / `<&0` duplicate a file descriptor; not a path.
    let mut cursor = n.walk();
    for ch in n.children(&mut cursor) {
        if !ch.is_named() && matches!(text(ch, src), ">&" | "<&") {
            return;
        }
    }
    let dest = named_children(n)
        .into_iter()
        .rev()
        .find(|c| is_token_kind(c.kind()));
    let Some(dest) = dest else { return };
    if dest.kind() == "number" {
        return;
    }
    scan_substitutions(dest, src, st, acc);
    match resolve_node(dest, src, st) {
        // Redirects always create/write their target, existing or not.
        Some(r) if !r.unresolved => trace_token(&r.text, r.tilde, st, acc, true),
        // A redirect to an unknown target is a write we cannot place: unsafe.
        _ => acc.add_unresolved(),
    }
}

fn apply_cd(n: Node, src: &[u8], st: &mut State) {
    let Some(r) = resolve_node(n, src, st) else {
        st.unknown_cwd = true;
        return;
    };
    if r.unresolved {
        st.unknown_cwd = true;
        return;
    }
    let tilde = r.tilde;
    let target = r.text.trim();
    if target.is_empty() || target == "-" {
        st.unknown_cwd = true;
        return;
    }
    let expanded_tilde = tilde && target.starts_with('~');
    let relative = !(target.starts_with('/') || expanded_tilde);
    if st.unknown_cwd && relative {
        return; // still unknown
    }
    match resolve_str(target, tilde, st) {
        Some(p) => {
            st.cwd = p;
            st.unknown_cwd = false;
        }
        None => st.unknown_cwd = true,
    }
}

/// Add one token if it is a path we can place.
///
/// `bare_ok` trusts a bare relative name even when it is absent (creation
/// commands). Untraceable path tokens mark the inventory incomplete.
fn trace_token(
    token: &str,
    tilde: bool,
    st: &State,
    acc: &mut Acc,
    bare_ok: bool,
) {
    let s = token.trim();
    if s.is_empty() || s.contains("://") {
        return;
    }
    let cand = split_assignment(s);
    if cand.is_empty() || cand.starts_with('-') {
        return;
    }
    let expanded_tilde = tilde && cand.starts_with('~');
    let relative = !(cand.starts_with('/') || expanded_tilde);
    if st.unknown_cwd && relative {
        acc.add_unresolved();
        return;
    }
    let explicit = cand.contains('/') || expanded_tilde || cand == "." || cand == "..";
    if !bare_ok && !explicit && !has_glob(cand) && !st.cwd.join(cand).exists() {
        return; // a bare name that is not on disk and not obviously a path
    }
    match resolve_str(cand, tilde, st) {
        Some(p) => acc.add(p),
        None => acc.add_unresolved(),
    }
}

/// Resolve a token and record its scope. A resolvable path prefix is kept
/// even when a later `$VAR`/`$(...)` is unknown; a fully unresolved token is
/// omitted from paths unless its raw text is path-like, in which case it is
/// `other`. Either way, unresolved output makes the inventory incomplete.
fn trace_node(n: Node, src: &[u8], st: &State, acc: &mut Acc, bare_ok: bool) {
    scan_substitutions(n, src, st, acc);
    match resolve_node(n, src, st) {
        Some(r) if !r.unresolved => trace_token(&r.text, r.tilde, st, acc, bare_ok),
        unresolved => {
            acc.scope_complete = false;
            if let Some(r) = unresolved.filter(|r| looks_like_path_token(&r.text)) {
                trace_token(&r.text, r.tilde, st, acc, bare_ok);
            } else if bare_ok || looks_like_path_token(text(n, src)) {
                acc.add_unresolved();
            }
        }
    }
}

/// Walk `command_substitution` nodes inside a token for side effects without
/// treating the surrounding literal text as a path.
fn scan_substitutions(n: Node, src: &[u8], st: &State, acc: &mut Acc) {
    if n.kind() == "command_substitution" {
        let mut sub = st.clone();
        walk(n, src, &mut sub, acc);
        return;
    }
    for ch in named_children(n) {
        scan_substitutions(ch, src, st, acc);
    }
}

/// Heredoc/herestring bodies are data, but an unquoted body can contain
/// `$(...)` commands that run. Walk those; ignore literal text.
fn walk_redirect_body(n: Node, src: &[u8], st: &State, acc: &mut Acc) {
    if n.kind() == "command_substitution" {
        let mut sub = st.clone();
        walk(n, src, &mut sub, acc);
        return;
    }
    for ch in named_children(n) {
        walk_redirect_body(ch, src, st, acc);
    }
}

fn is_create_command(name: &str) -> bool {
    let base = name.rsplit('/').next().unwrap_or(name);
    CREATE_COMMANDS.contains(&base)
}

/// `if=/dev/zero`, `of=/tmp/z`, `--out=/x` -> the value after `=`.
fn split_assignment(s: &str) -> &str {
    if let Some(i) = s.find('=') {
        if i > 0 && !s[..i].contains('/') {
            return &s[i + 1..];
        }
    }
    s
}

fn has_glob(s: &str) -> bool {
    s.bytes().any(|b| matches!(b, b'*' | b'?' | b'['))
}

/// Resolve a token to text plus whether a leading `~` should be expanded
/// (single/double quotes suppress tilde expansion).
/// Resolve a token to its literal prefix and whether a `$VAR`/`$(...)`
/// segment could not be resolved. `text` is the literal prefix before that
/// segment; unresolved tokens are fail-closed by the caller.
struct Resolved {
    text: String,
    tilde: bool,
    unresolved: bool,
}

fn resolve_node(n: Node, src: &[u8], st: &State) -> Option<Resolved> {
    match n.kind() {
        "word" | "number" | "string_content" => Some(Resolved {
            text: text(n, src).to_string(),
            tilde: true,
            unresolved: false,
        }),
        "raw_string" => Some(Resolved {
            text: text(n, src).trim_matches('\'').to_string(),
            tilde: false,
            unresolved: false,
        }),
        "string" | "translated_string" => {
            let mut out = String::new();
            let mut unresolved = false;
            for ch in named_children(n) {
                if unresolved {
                    break;
                }
                if ch.kind() == "string_content" {
                    out.push_str(text(ch, src));
                } else {
                    match resolve_node(ch, src, st) {
                        Some(r) => {
                            out.push_str(&r.text);
                            unresolved = r.unresolved;
                        }
                        None => unresolved = true,
                    }
                }
            }
            Some(Resolved {
                text: out,
                tilde: false,
                unresolved,
            })
        }
        "concatenation" | "concat" => {
            let mut out = String::new();
            let mut tilde = true;
            let mut unresolved = false;
            for (i, ch) in named_children(n).into_iter().enumerate() {
                if unresolved {
                    break;
                }
                match resolve_node(ch, src, st) {
                    Some(r) => {
                        if i == 0 {
                            tilde = r.tilde;
                        }
                        out.push_str(&r.text);
                        unresolved = r.unresolved;
                    }
                    None => unresolved = true,
                }
            }
            Some(Resolved {
                text: out,
                tilde,
                unresolved,
            })
        }
        "simple_expansion" | "expansion" => {
            let mut children = named_children(n);
            if children.len() != 1 || children[0].kind() != "variable_name" {
                return None; // `$1`, `${x:-y}`, `${#x}` ...
            }
            match text(children.remove(0), src) {
                "HOME" => Some(Resolved {
                    text: st.home.to_string_lossy().to_string(),
                    tilde: false,
                    unresolved: false,
                }),
                "PWD" => Some(Resolved {
                    text: st.cwd.to_string_lossy().to_string(),
                    tilde: false,
                    unresolved: false,
                }),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Expand `~` / `~/x` / `$HOME` / `$PWD` and resolve relatives against the
/// current shell cwd. Returns `None` for targets we cannot place.
fn resolve_str(s: &str, tilde: bool, st: &State) -> Option<PathBuf> {
    if tilde && s == "~" {
        return Some(st.home.clone());
    }
    if tilde && s.starts_with("~/") {
        return Some(normalize_path(&st.home.join(&s[2..])));
    }
    if s.starts_with('~') && tilde {
        return None; // ~user
    }
    if s.starts_with('/') {
        return Some(normalize_path(Path::new(s)));
    }
    if st.unknown_cwd {
        return None;
    }
    Some(normalize_path(&st.cwd.join(s)))
}

fn is_token_kind(k: &str) -> bool {
    matches!(
        k,
        "word"
            | "number"
            | "string"
            | "raw_string"
            | "translated_string"
            | "concatenation"
            | "simple_expansion"
            | "expansion"
    )
}

fn looks_like_path_token(s: &str) -> bool {
    s.contains('/') || s.starts_with('~') || s == "." || s == ".."
}

fn named_children(n: Node) -> Vec<Node> {
    let mut cursor = n.walk();
    n.children(&mut cursor).filter(|c| c.is_named()).collect()
}

fn text<'a>(n: Node, src: &'a [u8]) -> &'a str {
    n.utf8_text(src).unwrap_or("")
}

/// Strip `.`/`..` lexically. No filesystem access (paths may not exist).
fn normalize_path(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::RootDir => out.push("/"),
            Component::Normal(s) => out.push(s),
            Component::Prefix(_) => {}
        }
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    out
}

fn within(p: &Path, base: &Path) -> bool {
    p.starts_with(base)
}

fn starts_any(p: &Path, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|pre| within(p, Path::new(pre)))
}

fn is_tmp(p: &Path) -> bool {
    starts_any(
        p,
        &[
            "/tmp",
            "/private/tmp",
            "/var/tmp",
            "/var/folders",
            "/private/var/folders",
        ],
    )
}

const BENIGN_DEV: [&str; 8] = [
    "/dev/null",
    "/dev/zero",
    "/dev/urandom",
    "/dev/random",
    "/dev/full",
    "/dev/stdin",
    "/dev/stdout",
    "/dev/stderr",
];

fn is_benign_dev(p: &Path) -> bool {
    BENIGN_DEV.iter().any(|d| p == Path::new(d))
}

fn is_system(p: &Path) -> bool {
    if is_benign_dev(p) {
        return true;
    }
    starts_any(
        p,
        &[
            "/usr",
            "/bin",
            "/sbin",
            "/lib",
            "/libexec",
            "/opt",
            "/etc",
            "/private/etc",
            "/dev",
            "/sys",
            "/proc",
            "/boot",
            "/System",
            "/Library",
            "/Applications",
            "/var",
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scopes(cmd: &str) -> Vec<String> {
        analyze(cmd, Path::new(env!("CARGO_MANIFEST_DIR")), Path::new("/Users/u")).scope
    }

    fn outside(cmd: &str) -> Vec<String> {
        analyze(cmd, Path::new(env!("CARGO_MANIFEST_DIR")), Path::new("/Users/u")).paths.outside
    }

    #[test]
    fn bare_arguments_need_to_exist() {
        assert_eq!(scopes("rm -rf Cargo.toml"), vec!["cwd"]);
        assert_eq!(scopes("ls src"), vec!["cwd"]);
        assert_eq!(scopes("git status"), vec!["none"]);
        assert_eq!(scopes("rm missing_zzz"), vec!["none"]);
    }

    #[test]
    fn create_commands_trace_absent_targets() {
        assert_eq!(scopes("touch newfile_zzz"), vec!["cwd"]);
        assert_eq!(scopes("mkdir build_zzz"), vec!["cwd"]);
        assert_eq!(scopes("cp ./a ./b_zzz"), vec!["cwd"]);
    }

    #[test]
    fn explicit_paths_skip_existence() {
        assert_eq!(scopes("rm ./missing_zzz"), vec!["cwd"]);
        assert_eq!(scopes("rm -rf *.log"), vec!["cwd"]);
    }

    #[test]
    fn cd_changes_scope() {
        assert_eq!(scopes("cd /etc && rm ./x"), vec!["system"]);
        assert_eq!(scopes("cd /tmp && rm ./x"), vec!["tmp"]);
        assert_eq!(scopes("cd src && rm scope.rs"), vec!["cwd"]);
    }

    #[test]
    fn cd_with_unknown_var_is_other() {
        assert_eq!(scopes("cd $DIR && rm Cargo.toml"), vec!["other"]);
    }

    #[test]
    fn subshell_cd_does_not_leak() {
        assert_eq!(scopes("(cd /etc; rm ./x)"), vec!["system"]);
        assert_eq!(scopes("(cd /etc); rm ./x"), vec!["cwd"]);
    }

    #[test]
    fn home_and_tmp() {
        assert_eq!(scopes("cat ~/.pi/agent/auth.json"), vec!["home"]);
        assert_eq!(scopes("cat $HOME/.ssh/id_rsa"), vec!["home"]);
        assert_eq!(scopes("cat /tmp/x"), vec!["tmp"]);
        assert_eq!(scopes("echo hi > /dev/null"), vec!["system"]);
    }

    #[test]
    fn absolute_outside_is_other() {
        assert_eq!(scopes("rm -rf /"), vec!["other"]);
        assert_eq!(scopes("find /"), vec!["other"]);
    }

    #[test]
    fn system_reads() {
        assert_eq!(scopes("cat /etc/passwd"), vec!["system"]);
        assert_eq!(scopes("ls /usr/bin"), vec!["system"]);
    }

    #[test]
    fn option_values_and_dd() {
        assert_eq!(scopes("curl -o /tmp/f https://x.test"), vec!["tmp"]);
        assert_eq!(scopes("dd if=/dev/zero of=/tmp/z"), vec!["system", "tmp"]);
        assert_eq!(scopes("git -C /Users/u/other status"), vec!["home"]);
    }

    #[test]
    fn unknown_redirect_is_other() {
        assert_eq!(scopes("echo hi > $OUT"), vec!["other"]);
        assert_eq!(scopes("cat > /tmp/x"), vec!["tmp"]);
    }

    #[test]
    fn no_paths_is_none() {
        assert_eq!(scopes("pwd"), vec!["none"]);
        assert_eq!(scopes("iptables -F"), vec!["none"]);
        assert_eq!(scopes("echo $PATH"), vec!["none"]);
    }

    #[test]
    fn system_command_bare_args_are_not_cwd() {
        assert_eq!(scopes("systemctl restart nginx"), vec!["none"]);
        assert_eq!(scopes("npm install -g x"), vec!["none"]);
    }

    #[test]
    fn command_substitution_is_scanned() {
        assert_eq!(scopes("echo $(cat /etc/hosts)"), vec!["system"]);
        assert_eq!(scopes("rm $(find / -name x)"), vec!["other"]);
    }

    #[test]
    fn outside_paths_exclude_tmp_but_include_home() {
        assert_eq!(outside("cat ~/.pi/x /tmp/y"), vec!["/Users/u/.pi/x"]);
    }

    #[test]
    fn quoted_tilde_is_literal() {
        assert_eq!(scopes("cat '~/x'"), vec!["cwd"]);
        assert_eq!(scopes("cat \"~/x\""), vec!["cwd"]);
        assert_eq!(scopes("rm ~/foo"), vec!["home"]);
    }

    #[test]
    fn unresolved_tail_keeps_path_prefix() {
        assert_eq!(scopes("cat ~/Documents/$FILE"), vec!["home"]);
        assert_eq!(scopes("rm /tmp/$NAME"), vec!["tmp"]);
        assert_eq!(scopes("cat /etc/$(whoami).conf"), vec!["system"]);
        assert_eq!(scopes("cat $DIR/etc/passwd"), vec!["other"]);
    }

    #[test]
    fn bare_variable_arguments_stay_none() {
        assert_eq!(scopes("echo $PATH"), vec!["none"]);
        assert_eq!(scopes("echo $A$B"), vec!["none"]);
    }

    #[test]
    fn heredoc_body_is_scanned() {
        assert_eq!(
            scopes("cat <<EOF\n$(cat /etc/hosts)\nEOF\n"),
            vec!["system"]
        );
        assert_eq!(
            scopes("cat <<'EOF'\n$(cat /etc/hosts)\nEOF\n"),
            vec!["none"]
        );
        assert_eq!(scopes("cat <<<$(cat /etc/hosts)"), vec!["system"]);
    }

    #[test]
    fn resolved_inventory_is_complete_without_claiming_all_effects_known() {
        for cmd in ["pwd", "systemctl restart nginx", "cat /etc/hosts", "touch ./new", "cat /tmp/x"] {
            let report = analyze(cmd, Path::new("/project"), Path::new("/Users/u"));
            assert!(report.scope_complete, "{cmd}");
            assert_eq!(serde_json::to_value(&report).unwrap()["scope_complete"], true);
        }
    }

    #[test]
    fn outside_inventory_truncation_requires_review_even_with_same_category() {
        let allowed: Vec<String> = (0..MAX_PATHS)
            .map(|i| format!("/Users/u/allowed/{i}"))
            .collect();
        let base = format!("cat {}", allowed.join(" "));
        let complete = analyze(&base, Path::new("/project"), Path::new("/Users/u"));
        assert!(complete.scope_complete);
        assert_eq!(complete.paths.outside, allowed);
        for extra in ["/Users/u/private", "/private-data/file"] {
            let report = analyze(&format!("{base} {extra}"), Path::new("/project"), Path::new("/Users/u"));
            assert!(!report.scope_complete, "{extra}");
            assert_eq!(report.paths.outside, allowed);
            assert_eq!(report.paths.outside.len(), MAX_PATHS);
            assert_eq!(serde_json::to_value(&report).unwrap()["scope_complete"], false);
        }
        let duplicate = analyze(&format!("{base} {}", allowed[0]), Path::new("/project"), Path::new("/Users/u"));
        assert!(duplicate.scope_complete, "duplicate paths do not truncate the inventory");
    }

    #[test]
    fn inside_inventory_truncation_is_also_incomplete() {
        let cmd = format!("touch {}", (0..=MAX_PATHS).map(|i| format!("./{i}")).collect::<Vec<_>>().join(" "));
        let report = analyze(&cmd, Path::new("/project"), Path::new("/Users/u"));
        assert_eq!(report.scope, vec!["cwd"]);
        assert_eq!(report.paths.inside.len(), MAX_PATHS);
        assert!(!report.scope_complete);
    }

    #[test]
    fn mixed_resolved_and_unresolved_paths_are_incomplete() {
        for (cmd, expected_scope, expected_paths) in [
            ("cat ~/allowed/config ~/allowed/$FILE", vec!["home"], vec!["/Users/u/allowed/config", "/Users/u/allowed"]),
            ("cat ~/allowed/config $DIR/private", vec!["home", "other"], vec!["/Users/u/allowed/config"]),
            ("cat ~/allowed/config /etc/$FILE", vec!["home", "system"], vec!["/Users/u/allowed/config", "/etc"]),
            ("cat ~/allowed/config > $OUT", vec!["home", "other"], vec!["/Users/u/allowed/config"]),
        ] {
            let report = analyze(cmd, Path::new("/project"), Path::new("/Users/u"));
            assert!(!report.scope_complete, "{cmd}");
            assert_eq!(report.scope, expected_scope, "{cmd}");
            assert_eq!(report.paths.outside, expected_paths, "{cmd}");
        }
    }

    #[test]
    fn unresolved_targets_and_parser_errors_are_incomplete() {
        for cmd in ["cat ~unknown/file", "touch $FILE", "cat $FILE", "cd $DIR", "cd $DIR && touch ./x", "cat /tmp/$FILE", "echo > $OUT", "cat ~/allowed/config $(whoami)", "~/bin/$TOOL", "$BIN/tool", "cat '", "(cat /etc/hosts", "cat ./x >"] {
            assert!(!analyze(cmd, Path::new("/project"), Path::new("/Users/u")).scope_complete, "{cmd}");
        }
    }

    #[test]
    fn cd_skips_double_dash() {
        assert_eq!(scopes("cd -- /etc && rm ./x"), vec!["system"]);
        assert_eq!(scopes("cd -- /tmp && rm ./x"), vec!["tmp"]);
    }
}
