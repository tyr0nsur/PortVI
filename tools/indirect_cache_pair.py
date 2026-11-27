"""Run an owned PortVI test program twice using cold and warm pipeline caches."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


def sha256(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def run_pair(command, output, env, timeout, expected_exit, metadata):
    output.mkdir()
    output = output.resolve(strict=True)
    manifest = {"command": command, **metadata, "sessions": []}
    with TemporaryDirectory(prefix=".scratch-", dir=output) as scratch_path:
        scratch = Path(scratch_path).resolve(strict=True)
        cache = scratch / "cache"
        cache.mkdir()
        for phase in ("cold", "warm"):
            data = scratch / f"data-{phase}"
            data.mkdir()
            pipeline_log = output / f"{phase}.pipelines.tsv"
            log = output / f"{phase}.log"
            run_env = {
                **env,
                "VICTORY_CACHE_DIR": str(cache),
                "VICTORY_DATA_DIR": str(data),
                "VICTORY_PIPELINE_LOG": str(pipeline_log),
            }
            record = {"phase": phase, "started_utc": datetime.now(timezone.utc).isoformat()}
            manifest["sessions"].append(record)
            try:
                with log.open("wb") as stream:
                    result = subprocess.run(
                        command, cwd=data, env=run_env, stdout=stream,
                        stderr=subprocess.STDOUT, timeout=timeout,
                    )
                record["exit_code"] = result.returncode
                if result.returncode != expected_exit:
                    raise RuntimeError(f"{phase} run returned {result.returncode}. See {log}")
                if not pipeline_log.is_file() or not pipeline_log.stat().st_size:
                    raise RuntimeError(f"{phase} run recorded no pipeline events. See {log}")
                with pipeline_log.open(encoding="utf-8") as events:
                    built = Counter(line.rstrip().rsplit("\t", 1)[-1] for line in events
                                    if line.startswith("built\t"))
                record["pipelines_made"] = built["made"]
                record["pipelines_loaded"] = built["loaded"]
                if phase == "cold" and not any(path.is_file() for path in cache.rglob("*")):
                    raise RuntimeError("The cold run left an empty cache.")
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                record["error"] = str(exc)
                raise
            finally:
                (output / "pair.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path, help="diagnostics-enabled PortVI runtime or Rust test binary")
    parser.add_argument("program", nargs="?", type=Path, help="finite source-built fixture you have permission to run")
    parser.add_argument("output", type=Path, help="new directory for logs and the comparison manifest")
    parser.add_argument("--libtest", metavar="TEST_NAME", help="run one exact, ignored test in a Rust test binary")
    parser.add_argument("--console", choices=("series-s", "series-x"), default="series-x")
    parser.add_argument("--remap", action="store_true", help="enable view remapping on both passes")
    parser.add_argument("--timeout", type=int, default=120, help="seconds allowed for each fixture")
    parser.add_argument("--expected-exit", type=int, default=0, help="required exit code for each pass")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if (args.program is None) == (args.libtest is None):
        parser.error("supply either a program for PortVI or --libtest for a Rust test binary")

    try:
        runtime = args.runtime.resolve(strict=True)
        env = {key: value for key, value in os.environ.items()
               if not key.upper().startswith(("VICTORY_", "PORTVI_", "XCL_"))}
        settings = {"VICTORY_LOG": "info", "VICTORY_VIEW_REMAP": str(int(args.remap))}
        metadata = {
            "runtime_sha256": sha256(runtime),
            "settings": settings,
            "expected_exit": args.expected_exit,
            "timeout_seconds": args.timeout,
        }
        if args.libtest:
            command = [str(runtime), args.libtest, "--ignored", "--exact", "--nocapture"]
        else:
            program = args.program.resolve(strict=True)
            command = [str(runtime), "run", str(program), "--execute", "--console", args.console]
            metadata["program_sha256"] = sha256(program)
        run_pair(command, args.output, {**env, **settings}, args.timeout, args.expected_exit, metadata)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
