# PortVI project status

Evidence snapshot on 25 November 2026. [PortVI revision `1c3810b`](https://github.com/tyr0nsur/PortVI/tree/1c3810bd51a987898ea62abed3ffa237fbf05e1b)

PortVI is an experimental Windows x64 compatibility runtime built on Marvisak’s VIctory. Its public work covers native platform bindings, translation to DXIL and repeatable Direct3D 12 hardware checks.

| Check | Observed result |
| --- | --- |
| Native loader | The source-built x64 fixture bound two platform imports and exited with the expected code 73. |
| DXIL validation | 25,612 of 26,084 sampled programs passed Microsoft’s validator. The pass added 42 valid programs across the same sample, with zero previous successes lost. |
| D3D12 pipeline creation | The RTX 4060 created all 36 previously blocked descriptor-read pipelines. |
| Buffer view readback | The RTX 4060 and WARP each matched the expected address, element count and format in all 24 rewritten buffer views. |
| GTA VI scene capture | PortVI has yet to publish a measured scene capture. Scene FPS and stability are unmeasured. |

[Engineering evidence and commands](engineering-review.md#descriptor-words-as-shader-data) · [Windows CI](https://github.com/tyr0nsur/PortVI/actions/runs/45680340859) · [Contribute hardware results](../CONTRIBUTING.md)

When recording a scene milestone, link evidence for the exact PortVI revision, environment and test. Attribute upstream result reports to the project that ran them. The [frame-time protocol](engineering-review.md#measuring-a-30-fps-target) specifies paired cold and warm runs.
