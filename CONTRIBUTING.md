# Contribute to PortVI

Our reference card is the desktop GeForce RTX 4060 with 8 GB. If you work on shader translation, compare GPU results or test on another GeForce, start with the [1080p lighting milestone](https://github.com/tyr0nsur/PortVI/issues/26). It includes the current rendering failure, the upstream implementation that translates the missing shaders on CPU, and ways to help using original fixtures. Questions and results are welcome in English or Spanish.

| Report | Where to share it |
| --- | --- |
| Ray-traced lighting on the RTX 4060 | [Rendering milestone](https://github.com/tyr0nsur/PortVI/issues/26) |
| Median shader or another GPU adapter | [GPU validation](https://github.com/tyr0nsur/PortVI/issues/1) |
| BC4 streaming and skipped-upload control | [Texture validation](https://github.com/tyr0nsur/PortVI/issues/2) |
| Cold and warm scene frame times | [Frame timing](https://github.com/tyr0nsur/PortVI/issues/3) |
| Fence wakeups | [Synchronization](https://github.com/tyr0nsur/PortVI/issues/4) |
| A new reproducible regression or hardware result | [Open a report](https://github.com/tyr0nsur/PortVI/issues/new/choose) |

## Run a hardware check

Follow the [Windows setup](README.md#build-and-run). Use `python portvi.py gpu` for the median shader or `python portvi.py texture` for the BC4 fixture. Run `git rev-parse HEAD` and note the exact adapter, driver, Windows build and whether the command used real hardware or WARP.

Include the command, observed and expected result, and a compact, sanitized excerpt. For a frame capture, include the scene, settings, duration, cold or warm cache state, displayed FPS and frame-time percentiles. Both successful and failing reproductions are helpful. [Frame-time protocol](docs/engineering-review.md#measuring-a-30-fps-target)

## Send a patch

Follow the [engine patch instructions](README.md#contribute). Include the smallest reproduction available, expected and observed results before and after the change, and any upstream commit or pull request you built on. Use authored or redistributable inputs. Run the affected tests and `python portvi.py test` before asking for review. If you changed a dashboard result, update its evidence and run `python tools/render_progress.py`.

## Community rules

Treat people with respect. Report results in the relevant thread and keep criticism focused on the work. Posts must exclude credentials, personal data, unauthorized game files, game executables, keys, leaks and links to obtain them. Share sanitized logs and redistribution-safe fixtures. Maintainers may remove violating content and restrict participation after repeated or serious violations. Flag a GitHub post to the maintainers, and use the platform reporting tool for abuse.
