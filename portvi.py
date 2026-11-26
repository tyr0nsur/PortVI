"""Build the pinned native runtime and exercise it with open test programs."""
import argparse
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "engine"
WORK = ROOT / ".work"
TARGET = ROOT / "target"
EXE = TARGET / "release" / "portvi.exe"
_CACHE_NAME = re.compile(r"portvi-([a-f0-9]{20})-[a-z0-9_]{8}")
_CACHE_OWNER = b"PortVI CLI managed Cargo cache v1\n"


def _linked(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def _claim(cache, state):
    # Atomic rename awards the idle cache to exactly one runner. A crash
    # leaves an active lease for inspection, even if a Cargo child survives.
    marker = cache / (".portvi-" + state)
    try:
        if _linked(cache) or not cache.is_dir() or _linked(marker) or not marker.is_file():
            return False
        if marker.stat().st_size != len(_CACHE_OWNER) or marker.read_bytes() != _CACHE_OWNER:
            return False
        marker.rename(cache / ".portvi-active")
    except FileNotFoundError:
        return False
    return True


@contextmanager
def build_cache(repo, revision):
    """Reuse this revision and retire completed, runner-owned jobs."""
    if not re.fullmatch(r"[a-f0-9]{20}", revision):
        raise ValueError("Invalid PortVI build revision")
    root = repo.resolve() / "target" / "builds"
    for parent in (root.parent, root):
        parent.mkdir(exist_ok=True)
        if _linked(parent) or parent.resolve(strict=True) != parent:
            raise RuntimeError(f"Unexpected build directory: {parent}")
    current = None
    try:
        for cache in root.iterdir():
            match = _CACHE_NAME.fullmatch(cache.name)
            if not match:
                continue
            if not _claim(cache, "failed"):
                if not _claim(cache, "ready"):
                    continue
                if match[1] == revision and current is None:
                    current = cache
                    continue
            # Never descend into links or junctions during cleanup. Python
            # rmtree on Windows removes nested junctions without walking them.
            if cache.resolve(strict=True).parent != root or _linked(cache):
                raise RuntimeError(f"Unexpected cache path: {cache}")
            try:
                shutil.rmtree(cache)
            except OSError:
                # Retry any remaining files before the next build.
                if cache.is_dir():
                    active = cache / ".portvi-active"
                    if active.exists():
                        active.rename(cache / ".portvi-failed")
                    else:
                        (cache / ".portvi-failed").write_bytes(_CACHE_OWNER)
                raise
        if current is None:
            current = Path(tempfile.mkdtemp(prefix=f"portvi-{revision}-", dir=root))
            (current / ".portvi-active").write_bytes(_CACHE_OWNER)
        yield current
    except Exception:
        if current is not None:
            (current / ".portvi-active").rename(current / ".portvi-failed")
        raise
    else:
        (current / ".portvi-active").rename(current / ".portvi-ready")


def git(directory, *args, capture=False):
    return subprocess.run(
        ["git", "-C", str(directory), *map(str, args)],
        check=True, text=True, stdout=subprocess.PIPE if capture else None,
    ).stdout


def native_tree():
    # The gitlink selects the engine. Moving an upstream branch cannot change a build.
    entry = git(ROOT, "ls-files", "--stage", "--", "engine", capture=True).split()
    if len(entry) != 4 or entry[0] != "160000" or entry[2] != "0":
        raise RuntimeError("The engine gitlink is missing or conflicted.")
    revision = entry[1]
    if not (ENGINE / ".git").exists():
        git(ROOT, "submodule", "update", "--init", "--", "engine")
    if git(ENGINE, "rev-parse", "HEAD", capture=True).strip() != revision:
        raise RuntimeError("The engine checkout differs from the gitlink. Check git submodule status.")
    patches = sorted((ROOT / "patches").glob("*.patch"))
    identity = hashlib.sha256(revision.encode("ascii"))
    for patch in patches:
        identity.update(patch.name.encode("utf-8") + b"\0")
        identity.update(hashlib.sha256(patch.read_bytes()).digest())
    WORK.mkdir(exist_ok=True)
    source = WORK / identity.hexdigest()[:20]
    if source.is_dir():
        return source
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=WORK) as staging:
        checkout = Path(staging) / "engine"
        git(ROOT, "clone", "--quiet", "--no-checkout", "-c", "core.autocrlf=false",
            "--", ENGINE, checkout)
        git(checkout, "checkout", "--quiet", "--detach", revision)
        for patch in patches:
            git(checkout, "apply", "--index", "--whitespace=error", patch)
        try:
            checkout.rename(source)
        except OSError:
            # Another build may have finished preparing the identical input.
            if not source.is_dir():
                raise
    return source


def toolchain(source):
    if sys.platform != "win32" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("Native execution requires Windows x64.")
    import tomllib
    with (source / "rust-toolchain.toml").open("rb") as manifest:
        version = tomllib.load(manifest)["toolchain"]["channel"]
    env = os.environ.copy()
    if not shutil.which("cargo"):
        private = Path(env.get("LOCALAPPDATA", "")) / "PortVI" / "toolchain"
        cargo_bin = private / "cargo" / "bin"
        if not (cargo_bin / "cargo.exe").is_file():
            raise RuntimeError("Install rustup with the x86_64-pc-windows-msvc toolchain.")
        env.update(CARGO_HOME=str(private / "cargo"), RUSTUP_HOME=str(private / "rustup"))
        env["PATH"] = str(cargo_bin) + os.pathsep + env.get("PATH", "")
    # Use explicit file names so subprocess lookup also sees the private toolchain.
    cargo = shutil.which("cargo", path=env["PATH"])
    rustc = shutil.which("rustc", path=env["PATH"])
    if not cargo or not rustc:
        raise RuntimeError("Both cargo and rustc must be available through rustup.")
    # Cargo's incremental output must belong to this engine and patch set.
    env["CARGO_TARGET_DIR"] = str(TARGET / "builds" / source.name)
    return cargo, rustc, "+" + version, env


def smoke(executable, rustc, version, env):
    with tempfile.TemporaryDirectory(prefix="probe-", dir=WORK) as staging:
        folder = Path(staging)
        probe = folder / "portvi_probe.exe"
        subprocess.run([
            rustc, version, "--edition=2024", "-C", "panic=abort", "-C", "opt-level=2",
            "-C", "link-arg=/ENTRY:probe_entry", "-C", "link-arg=/SUBSYSTEM:CONSOLE",
            str(ROOT / "probes" / "runtime.rs"), "-o", str(probe),
        ], cwd=folder, env=env, check=True)
        # A scratch directory keeps developer settings and saves out of this test.
        run_env = env.copy()
        for name in list(run_env):
            if name.upper().startswith("VICTORY_"):
                del run_env[name]
        run_env.update(VICTORY_CACHE_DIR=str(folder / "cache"), VICTORY_DATA_DIR=str(folder / "data"),
                       VICTORY_LOG="info")
        result = subprocess.run(
            [str(executable), "run", str(probe), "--execute", "--strict"], env=run_env,
            cwd=folder, capture_output=True, text=True, errors="replace", timeout=45,
        )
        print(result.stdout, end="")
        if result.returncode != 73 or "title exited with 73" not in result.stdout.splitlines():
            sys.stderr.write(result.stderr)
            raise RuntimeError(f"Native probe did not return its success marker (host exit {result.returncode}).")
        print("PASS: source-built x64 probe initialized the runtime, round-tripped a BSTR and returned 73.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "build", "test", "smoke", "gpu", "texture"))
    args = parser.parse_args()
    source = native_tree()
    if args.action == "prepare":
        print(source)
        return
    cargo, rustc, version, env = toolchain(source)
    with build_cache(ROOT, source.name) as cache:
        env["CARGO_TARGET_DIR"] = str(cache)
        run_action(args.action, source, cargo, rustc, version, env)


def run_action(action, source, cargo, rustc, version, env):
    if action in {"gpu", "texture"}:
        env.pop("VICTORY_WARP", None)
        env.pop("VICTORY_D3D12_WARP", None)
        if action == "gpu":
            case = "emit::valu_tests::v_med3_f32_on_physical_adapter"
            package = ["-p", "victory-shader", "--features", "host"]
            options = ["--ignored"]
            evidence = ["physical adapter: "]
        else:
            case = "modules::dstorage::tests::streamed_bc4_texels_survive_a_real_gpu_upload"
            package = ["-p", "victory-hle"]
            options = []
            env["PORTVI_BC4_REQUIRE_HARDWARE"] = "1"
            evidence = ["BC4 adapter: ", "software=false", "32/32 expected texels", "32/32 stale texels detected"]
        result = subprocess.run([
            cargo, version, "test", "--locked", *package, "--lib", case,
            "--", "--exact", *options, "--nocapture",
        ], cwd=source, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
           text=True, errors="replace")
        print(result.stdout, end="")
        result.check_returncode()
        for marker in [f"test {case} ... ok", *evidence]:
            if marker not in result.stdout:
                raise RuntimeError(f"Missing hardware test evidence: {marker}")
        return
    if action == "test":
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tools"],
                       cwd=ROOT, check=True)
        subprocess.run([cargo, version, "test", "--workspace", "--locked", "--", "--test-threads=1"],
                       cwd=source, env=env, check=True)
        return
    subprocess.run([cargo, version, "build", "--release", "--locked", "-p", "victory", "--bin", "portvi"],
                   cwd=source, env=env, check=True)
    executable = Path(env["CARGO_TARGET_DIR"]) / "release" / EXE.name
    EXE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="publish-", dir=EXE.parent) as staging:
        for artifact in (executable.with_suffix(".pdb"), executable):
            if artifact.is_file():
                staged = Path(staging) / artifact.name
                shutil.copy2(artifact, staged)
                staged.replace(EXE.parent / artifact.name)
    if action == "smoke":
        smoke(executable, rustc, version, env)
    else:
        print(EXE)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))
