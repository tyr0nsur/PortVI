# Test and build storage

Before starting another local build or hardware test, close the previous run's processes and release its disposable files. Save its revision, settings, logs, measurements and any evidence needed to reproduce a failure outside the scratch directory.

Use a separate scratch profile for each hardware run and remove it when the run finishes. Keep the source profile, fixtures and one reusable build cache for the current revision. The PortVI CLI reclaims its marked failed or obsolete Cargo caches before the next build or test.

After an interruption, check whether the runner, compiler, runtime or capture tool still uses the directory before reclaiming it. Confirm the cache belongs to the run and its resolved path is inside the expected scratch root. Preserve active runs, shared toolchains, user saves and unrelated worktrees.
