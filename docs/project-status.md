# PortVI project status

Updated 27 November 2026. [Component baseline at `134b1bf`](https://github.com/tyr0nsur/PortVI/tree/134b1bf9b324c19c43958b62a549cfffd1451cd6)

PortVI targets the desktop GeForce RTX 4060 with 8 GB through VIctory’s native Windows x64 runtime. The first milestone is a correctly lit scene with a measured 1920 × 1080 internal render target and output. The longer-term target is 30 rendered FPS at that quality. [Work on this milestone](https://github.com/tyr0nsur/PortVI/issues/26)

The latest experimental scene run stayed up for 110 seconds. The three world captures showed subtitles on black. Its ray-tracing builder had 44 of 53 pipelines ready. A source translator from the DexNom project translated the nine rejected programs in a CPU comparison. Validating their GPU output on the RTX 4060 is the next step. [Scene experiment](https://github.com/tyr0nsur/PortVI/pull/24) · [Candidate translator](https://github.com/DexNom/VIctory/pull/64)

| Check | Observed result |
| --- | --- |
| Native loader | The source-built x64 fixture bound four imports, round-tripped a Windows Automation string with an embedded zero and exited with the expected code 73. |
| DXIL validation | At revision `134b1bf`, 25,613 of 26,084 programs passed Microsoft’s validator. This added one success to the previous 25,612. All earlier successes remained valid. |
| D3D12 pipeline creation | The RTX 4060 created all 36 previously blocked descriptor-read pipelines. |
| Buffer view readback | The RTX 4060 and WARP each matched the expected address, element count and format in all 24 rewritten buffer views. |
| Placed texture writes | The RTX 4060 and WARP matched both texels after dispatching through a placed UAV. [Fixture and results](engineering-review.md#native-engine-refresh) |
| Null texture reads | Both the RTX 4060 and WARP returned all 4,096 expected texels in a compute dispatch. [Fixture and device checks](engineering-review.md#native-graphics-refresh-of-26-november-2026) |
| Command-list lifetime | Both adapters passed the pipeline and root-signature retention and release fixture. |
| Wave64 on the RTX 4060 | The split-wave first-lane fixture returned the expected thread ID in groups of 64 and 128 threads. Native and split-wave row shuffles matched 143,360 lane results. [Device tests](engineering-review.md#wave64-on-a-32-lane-gpu) |
| Introductory sequence | A preliminary RTX 4060 capture showed partial geometry. Two 120 s runs recorded 96 and 121 successful DXGI presents. [Revision, output modes and frame times](engineering-review.md#preliminary-startup-frame-timings) |

[Engineering evidence and commands](engineering-review.md#component-tests) · [Windows CI](https://github.com/tyr0nsur/PortVI/actions/workflows/windows.yml?query=branch%3Amain) · [Contribute hardware results](../CONTRIBUTING.md)

When recording a scene milestone, link evidence for the exact PortVI revision, environment and test. Attribute upstream result reports to the project that ran them. The [frame-time protocol](engineering-review.md#measuring-a-30-fps-target) specifies paired cold and warm runs.
