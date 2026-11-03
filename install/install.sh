#!/bin/sh
# acclass installer: prebuilt engine + model into ~/.acclass. Needs Python 3.
#   curl -fsSL https://github.com/jensilo/acclass/releases/latest/download/install.sh | sh                   # Q8 model (default)
#   curl -fsSL https://github.com/jensilo/acclass/releases/latest/download/install.sh | sh -s -- --model q5  # smaller model
# Run with --help for all options (other model, manifest pin, rollback, uninstall).
set -eu
umask 077
command -v python3 >/dev/null 2>&1 || { echo 'acclass install: Python 3 is required' >&2; exit 1; }
exec python3 - "$@" <<'PY'
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request

VERSION = '1.0.0'
# Explicit --manifest URLs override this default; relative asset URLs resolve
# against the manifest URL.
MANIFEST_URL = 'https://github.com/jensilo/acclass/releases/download/v{version}/release.json'
SCHEMA = 'acclass-release/2'
MODEL_FILES = {
    'q8': {'model.gguf', 'tokenizer.json', 'config.json', 'heads.json', 'thresholds.json', 'model_contract.json'},
    'q5': {'model.gguf', 'tokenizer.json', 'config.json', 'heads.json', 'thresholds.json', 'model_contract.json'},
    'fp32': {'model.safetensors', 'tokenizer.json', 'config.json', 'heads.json', 'thresholds.json', 'model_contract.json'},
}
X86_64_V3 = {'avx', 'avx2', 'bmi1', 'bmi2', 'f16c', 'fma', 'movbe', 'abm', 'xsave'}


class Fail(Exception):
    pass


def interrupted(signum, frame):
    raise KeyboardInterrupt()


signal.signal(signal.SIGTERM, interrupted)


def require(condition, message):
    if not condition:
        raise Fail(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cpu_flags(cpuinfo, key):
    for line in cpuinfo.splitlines():
        name, _, value = line.partition(':')
        if name.strip().lower() == key:
            return set(value.split())
    return set()


def detect_platform(system=None, machine=None, cpuinfo=None):
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if system == 'Darwin':
        if machine == 'x86_64':
            try:  # Rosetta reports x86_64; the native arm64 binary still runs.
                if subprocess.run(['sysctl', '-n', 'sysctl.proc_translated'], capture_output=True,
                                  text=True).stdout.strip() == '1':
                    machine = 'arm64'
            except OSError:
                pass
        require(machine == 'arm64', 'no prebuilt binary for Intel macOS; build from source (docs/releasing.md)')
        return 'macos-arm64'
    require(system == 'Linux', f'no prebuilt binary for {system}; build from source (docs/releasing.md)')
    if cpuinfo is None:
        try:
            cpuinfo = Path('/proc/cpuinfo').read_text()
        except OSError:
            cpuinfo = ''
    if machine in ('x86_64', 'amd64'):
        missing = X86_64_V3 - cpu_flags(cpuinfo, 'flags')
        require(not missing, 'this CPU lacks x86-64-v3 (AVX2/FMA, missing: ' + ' '.join(sorted(missing)) +
                '); build from source (docs/releasing.md)')
        return 'linux-x86_64'
    if machine in ('aarch64', 'arm64'):
        require('asimddp' in cpu_flags(cpuinfo, 'features'),
                'this CPU lacks ARMv8.2 dotprod (asimddp); build from source (docs/releasing.md)')
        return 'linux-aarch64'
    raise Fail(f'no prebuilt binary for Linux {machine}; build from source (docs/releasing.md)')


def require_https(url):
    parsed = urllib.parse.urlparse(url)
    require(parsed.scheme == 'https' and parsed.hostname and
            parsed.username is None and parsed.password is None,
            'refusing non-HTTPS/credential-bearing URL')


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url, destination, expected_sha=None, expected_size=None):
    """Download HTTPS only; cap the manifest at 2 MiB and assets at declared sizes."""
    require_https(url)
    h = hashlib.sha256()
    size = 0
    limit = expected_size if expected_size is not None else 2 * 1024 * 1024
    try:
        source = urllib.request.build_opener(HTTPSRedirect()).open(url, timeout=60)
        with source, open(destination, 'xb') as out:
            require_https(source.geturl())
            while True:
                chunk = source.read(min(1 << 20, limit - size + 1))
                if not chunk:
                    break
                size += len(chunk)
                require(size <= limit, f'download exceeds size limit: {url}')
                h.update(chunk)
                out.write(chunk)
    except OSError as error:
        raise Fail(f'download failed: {url}: {error}') from None
    require(expected_size is None or size == expected_size, f'size mismatch: {url}')
    require(expected_sha is None or h.hexdigest() == expected_sha, f'SHA-256 mismatch: {url}')
    return h.hexdigest()


def resolve(base, url):
    return urllib.parse.urljoin(base, url)


def asset(entry, name):
    require(isinstance(entry, dict) and isinstance(entry.get('url'), str) and
            re.fullmatch(r'[0-9a-f]{64}', str(entry.get('sha256'))) and
            type(entry.get('size')) is int and entry['size'] >= 0,
            f'invalid manifest entry: {name}')
    return entry


def load_manifest(location, pin, scratch):
    path = scratch / 'release.json'
    digest = fetch(location, path)
    require(pin is None or digest == pin, 'release manifest does not match --manifest-sha256')
    try:
        manifest = json.loads(path.read_text())
    except ValueError:
        raise Fail('release manifest is not valid JSON') from None
    require(isinstance(manifest, dict) and manifest.get('schema') == SCHEMA and
            isinstance(manifest.get('version'), str) and isinstance(manifest.get('binaries'), dict) and
            isinstance(manifest.get('models'), dict), f'unsupported release manifest (expected {SCHEMA})')
    return manifest, path


def safe_extract(archive, destination):
    """Regular files only, under bin/, LICENSE/, share/ or MODEL-CARD.md; no links or traversal."""
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        for m in members:
            parts = m.name.split('/')
            require((m.isfile() or m.isdir()) and not m.name.startswith('/') and '..' not in parts and
                    (parts[0] in ('bin', 'LICENSE', 'share') or m.name == 'MODEL-CARD.md'),
                    f'unexpected archive member: {m.name}')
        for m in members:
            target = destination / m.name
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as src, open(target, 'xb') as dst:
                shutil.copyfileobj(src, dst)
            target.chmod(0o755 if m.name == 'bin/acclass' or m.name.endswith('.sh') else 0o644)


def relink(path, target):
    """Atomically point symlink `path` at `target` (rename over the old link)."""
    temporary = path.with_name(f'.{path.name}.{os.getpid()}')
    if temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(target)
    os.replace(temporary, path)


def publish(stage, destination, retired):
    """Restore the retained version if replacement fails or is interrupted."""
    try:
        if destination.exists():
            destination.rename(retired)
        stage.rename(destination)
    except BaseException:
        if retired.exists() and not destination.exists():
            retired.rename(destination)
        raise


def managed_links(home):
    for name in ('bin', 'model'):
        link = home / name
        require(not link.exists() and not link.is_symlink() or
                (link.is_symlink() and os.readlink(link) == f'current/{name}'),
                f'{link} is not managed by this installer; move it away or run --uninstall')


class Lock:
    def __init__(self, home):
        self.path = home / '.lock'

    def __enter__(self):
        try:
            self.path.mkdir()
        except FileExistsError:
            raise Fail(f'another install is running (or remove a stale {self.path})') from None

    def __exit__(self, *exc):
        self.path.rmdir()


def install(args, home):
    target = detect_platform()
    home.mkdir(parents=True, exist_ok=True)
    managed_links(home)
    with Lock(home), tempfile.TemporaryDirectory(prefix='.staging-', dir=home) as scratch:
        scratch = Path(scratch)
        location = args.manifest or MANIFEST_URL.format(version=args.version)
        manifest, manifest_path = load_manifest(location, args.manifest_sha256, scratch)
        version = manifest['version']
        require(re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', version), 'invalid release version')
        require(target in manifest['binaries'], f'release {version} has no binary for {target}')
        files = manifest['models'].get(args.model)
        require(isinstance(files, dict) and set(files) == MODEL_FILES[args.model],
                f'release {version} has no complete {args.model} model')
        stage = scratch / 'stage'
        (stage / 'model').mkdir(parents=True)
        binary = asset(manifest['binaries'][target], target)
        print(f'acclass {version}: {target} engine, {args.model} model', flush=True)
        fetch(resolve(location, binary['url']), scratch / 'engine.tar.gz', binary['sha256'], binary['size'])
        safe_extract(scratch / 'engine.tar.gz', stage)
        require((stage / 'bin/acclass').is_file(), 'engine archive lacks bin/acclass')
        for name in sorted(files):
            entry = asset(files[name], name)
            print(f'  {name} ({entry["size"] / 2**20:.1f} MiB)', flush=True)
            fetch(resolve(location, entry['url']), stage / 'model' / name, entry['sha256'], entry['size'])
            (stage / 'model' / name).chmod(0o644)
        try:
            out = subprocess.run([str(stage / 'bin/acclass'), '--version'], capture_output=True, text=True,
                                 timeout=30).stdout.strip()
        except OSError as error:
            raise Fail(f'engine does not run on this machine: {error}') from None
        require(out == f'acclass {version}', f'engine does not run on this machine (got {out!r})')
        shutil.copyfile(manifest_path, stage / 'release.json')
        (stage / 'INSTALLATION.json').write_text(json.dumps({
            'schema': 'acclass-installation/2', 'version': version, 'model': args.model, 'platform': target,
            'manifest_sha256': sha256(manifest_path),
            'files_sha256': {p.relative_to(stage).as_posix(): sha256(p)
                             for p in sorted(stage.rglob('*')) if p.is_file()}}, indent=2, sort_keys=True) + '\n')
        versions = home / 'versions'
        versions.mkdir(exist_ok=True)
        name = f'{version}-{args.model}'
        current = home / 'current'
        old = os.readlink(current) if current.is_symlink() else None
        require(old is None or re.fullmatch(r'versions/[A-Za-z0-9._-]+', old),
                f'{current} is not managed by this installer; run --uninstall first')
        require(current.is_symlink() or not current.exists(), f'{current} is not managed by this installer')
        destination = versions / name
        publish(stage, destination, scratch / 'retired')
        relink(current, f'versions/{name}')
        for link in ('bin', 'model'):
            if not (home / link).is_symlink():
                (home / link).symlink_to(f'current/{link}')
        # Keep one previous version for --rollback; delete older ones.
        previous = home / 'previous'
        if old and old != f'versions/{name}' and (home / old).is_dir():
            relink(previous, old)
        keep = {name}
        if previous.is_symlink():
            kept = Path(os.readlink(previous)).name
            if kept != name and (versions / kept).is_dir():
                keep.add(kept)
            else:
                previous.unlink()
        for entry in versions.iterdir():
            if entry.name not in keep:
                shutil.rmtree(entry) if entry.is_dir() and not entry.is_symlink() else entry.unlink()
    print(f'Installed acclass {version} ({args.model}) in {home}.')
    if old and Path(old).name != name:
        print(f'Previous version {Path(old).name} kept; undo with: install.sh --rollback')
    paths = '' if home == Path('~/.acclass').expanduser() else f' --model {home / "model"} --socket {home / "ac.sock"}'
    print(f'Add {home / "bin"} to PATH, then run: acclass serve{paths}')
    print(f'and in another shell: acclass classify{paths and " --socket " + str(home / "ac.sock")} "git status"')
    print('A running daemon keeps the old model until you restart it.')


def rollback(home):
    current, previous = home / 'current', home / 'previous'
    require(current.is_symlink() and previous.is_symlink() and previous.resolve().is_dir(),
            'no previous version to roll back to')
    managed_links(home)
    with Lock(home):
        old, new = os.readlink(current), os.readlink(previous)
        relink(current, new)
        relink(previous, old)
    print(f'Switched to {Path(new).name} (previous: {Path(old).name}). Restart a running daemon.')


def uninstall(home):
    if not home.exists():
        print(f'Nothing installed at {home}.')
        return
    managed_links(home)
    with Lock(home):
        for name in ('bin', 'model', 'current', 'previous'):
            if (home / name).is_symlink():
                (home / name).unlink()
        if (home / 'versions').is_dir():
            shutil.rmtree(home / 'versions')
        if (home / 'ac.sock').is_socket():
            (home / 'ac.sock').unlink()
    leftovers = sorted(p.name for p in home.iterdir() if p.name != '.lock')
    if not leftovers:
        home.rmdir()
        print(f'Removed {home}.')
    else:
        print(f'Removed acclass from {home}; left unrelated files: {", ".join(leftovers)}')
    print('Stop a running acclass daemon yourself. Pi configuration, if any, still names the old paths.')


def main():
    parser = argparse.ArgumentParser(prog='install.sh', description='Install the acclass engine and model.')
    parser.add_argument('--model', choices=['q8', 'q5', 'fp32'], default='q8',
                        help='q8: default (152 MiB); q5: smaller (98 MiB); '
                             'fp32: full weights for further training (571 MiB; serve with --quant f32)')
    parser.add_argument('--version', default=VERSION, help=f'release version (default {VERSION})')
    parser.add_argument('--manifest', help='HTTPS release.json URL (overrides --version)')
    parser.add_argument('--manifest-sha256', help='independently trusted SHA-256 of release.json')
    parser.add_argument('--home', type=Path, default=Path(os.environ.get('ACCLASS_HOME', '~/.acclass')),
                        help='install root (default ~/.acclass or $ACCLASS_HOME)')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--rollback', action='store_true', help='switch back to the previous version')
    action.add_argument('--uninstall', action='store_true', help='remove the installation')
    args = parser.parse_args()
    home = args.home.expanduser().absolute()
    require(not home.is_symlink(), f'{home} must not be a symlink')
    require(args.manifest_sha256 is None or re.fullmatch(r'[0-9a-f]{64}', args.manifest_sha256),
            '--manifest-sha256 must be 64 lowercase hex characters')
    if args.uninstall:
        uninstall(home)
    elif args.rollback:
        rollback(home)
    else:
        install(args, home)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print(f'acclass install: {str(error) or "interrupted"}', file=sys.stderr)
        sys.exit(1)
PY
