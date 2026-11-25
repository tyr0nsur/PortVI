# Native runtime engineering review

25 November 2026. PortVI pins the original `Marvisak/VIctory` main revision `940d32f` and applies fifteen patches. The latest handles compute masks carried across branches and loops. [Current engine](https://github.com/Marvisak/VIctory/commit/940d32f69b22265690a0b5a15d6ded572ba33baf) · [Previous PortVI revision](https://github.com/tyr0nsur/PortVI/tree/27aa4bc5f32346fac9da16aad236a5beaf583f6b)

## Reproduced on this revision

On Windows x64, 762 Rust workspace tests passed and 11 were marked ignored. One of the passing tests also ran in an isolated child process, bringing the total to 763 executions. Both Python frame-time tests passed. The physical-adapter median and texture probes ran separately. The open x64 program returned 73. Clippy passed for all targets and features with warnings treated as errors.

| Area | Reproduction and result |
| --- | --- |
| Native loader and CLI | The source-built x64 probe bound both platform imports. The CLI printed the title’s exit status and returned 73 to Windows. The smoke test checks both values. [Probe source](../probes/runtime.rs) |
| Shader cache | The upstream source and encoding fingerprints and effective loop limit identify the translator. PortVI also keys the emitter’s write-mark option, the device’s heap size and the slots that mirror title descriptors. Each compilation uses one snapshot of both limits. Obstructing an entry path during publication exercises the in-memory fallback, recovery on the next request and a subsequent warm disk hit. Upstream’s cache of translation refusals remains covered. [Cache and tests](../patches/0002-shader-cache-publication.patch) |
| Shader execution | The hardware-required median test identifies the adapter by the D3D12 device LUID and checks thirteen defined inputs. The gradient suite covers the independent widths of coordinates and derivatives, including minimum LOD, packed registers and unsupported forms. [Gradients and adapter tests](../patches/0003-rdna2-gradients-and-adapter.patch) |
| Depth bounds and console ABI | Test draws at depth 0.5 are culled outside 0.6–1.0, visible inside 0.4–0.6 and culled again outside 0.0–0.4. The engine checks host depth-bounds support before using it. The device-removal regression compares the console slot with D3D12. The queue and list tests dispatch through both generations of console tables. [D3D12 and ABI patch](../patches/0004-depth-bounds-and-console-abi.patch) |
| Staging bounds and descriptor identity | On the two-mip BC4 fixture, the final mip starts at byte 512 of a 520-byte allocation. A padded row pitch of 256 covers eight allocated bytes there. The diagnostic and detiler now use the same bounded length. SRV and UAV records intern the defined fields of a placed resource, independent of padding. [Regression and fix](../patches/0005-placed-views-and-staging-spans.patch) |
| Empty memory operations | On upstream, an unaligned zero-byte write dirtied an aliased 64 KiB page. A zero-length remap could also discard its alias link. The regression takes a clean page, exercises both empty calls, then confirms that a real one-byte write still reaches the alias. [Tracker fix and regression](../patches/0007-empty-memory-ranges.patch) |

## D3D12 descriptor limits and DXIL validation

The desktop RTX 4060 accepted 1,000,000 shader-visible view descriptors. Windows WARP accepted 2,097,152. The engine probes the device and uses that limit when allocating heaps and compiling shaders. On the RTX 4060 it reserves 721,469 slots for direct title indices, 262,144 for dynamic descriptors, 16,384 for host copy pairs and three terminal slots. Tests exercise allocations at the boundary, release and reuse, and a direct index at the cutoff alongside a working dynamic table. These changes integrate work by antoinerz1, JurajSimicevic and wbkepler. [Device probe](https://github.com/Marvisak/VIctory/pull/1777) · [Dynamic allocation](https://github.com/Marvisak/VIctory/pull/1751) · [Title-slot bounds](https://github.com/Marvisak/VIctory/pull/1779) · [PortVI integration and device test](../patches/0011-capped-title-views.patch)

For structured shader-resource views, the translator emits byte-addressed loads. The runtime preserves the original byte range and records the console stride so element indexing still advances by the correct number of bytes. A snapshot is aligned for both D3D12 raw views and the console element size. On hardware and WARP a 12-byte-stride fixture read the expected elements and a missing buffer read back zeros. The gather fixture now binds its byte-addressed input using the matching raw descriptor. [View conversion and regressions](../patches/0010-structured-buffer-reads.patch) · [Gather fixture](../patches/0003-rdna2-gradients-and-adapter.patch)

The emitter sets the DXIL module and container feature flags when a vertex or mesh stage declares an unordered-access view. Compute shaders that use derivatives select validator 1.8 and emit the matching pipeline-state metadata. The tests also check a vertex shader without the declared view, rejection under the earlier derivative profile and a compute dot product that still validates under 1.6. [BastienRodz’s UAV change](https://github.com/Marvisak/VIctory/pull/1989) · [Halfect’s compute derivative change](https://github.com/Marvisak/VIctory/pull/1065) · [Imported code and tests](../patches/0008-dxil-stage-flags-and-validator.patch)

## Linked mesh and pixel shaders

Direct3D 12 allows 32 KiB for the combined vertex and primitive outputs of a mesh shader. Each stage can translate individually while its pipeline still exceeds that budget. When the mesh stage is over budget, PortVI reads the pixel stage’s channel usage and removes unused mesh outputs. It keeps the declared vertex and primitive counts, checks that each required arbitrary channel has a compatible producer, and measures the emitted pair again. The linked cache includes both original containers, translator options, loop limit and device heap bounds. This adapts Texbirds’s work. [Upstream proposal](https://github.com/Marvisak/VIctory/pull/2398) · [DirectX mesh shader specification](https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html) · [Runtime, translator and tests](../patches/0013-linked-mesh-outputs.patch)

The public fixture declares 128 vertices and 128 primitives. Its standalone mesh output occupies 34,816 bytes. Paired with a pixel shader that reads four arbitrary channels, it uses 4,096 bytes. In a depth-only pipeline it uses 2,048 bytes. The runtime’s normal pipeline creation path dispatched the linked draw and read back (1, 0.5, 0.25, 1) at all 256 pixels on the RTX 4060 and WARP. The suite also exercises a consumer that still needs too much output, a declared channel the producer never writes, swapped stages and a second pipeline creation that reuses the linked pair.

## Floating-point conversion

PortVI translates `v_cvt_f32_f64` for round-to-nearest-even. The GPU suite checks fourteen register pairs against a CPU conversion, including signed zero, ties in both directions, the smallest representable FP32 subnormal, infinities and a NaN. A separate test covers inline operands and a literal encoding of −2.25. Directed rounding and a dynamically selected mode produce an explicit translation refusal. Calvix2k contributed the conversion and default-cybe contributed binary64 inline constants. [Original conversion](https://github.com/Marvisak/VIctory/pull/2348) · [Inline constants](https://github.com/Marvisak/VIctory/pull/1559) · [Our hardware reproduction](https://github.com/Marvisak/VIctory/pull/2348#issuecomment-7164049522) · [PortVI tests and fixes](../patches/0012-fp64-operands-and-rounding.patch)

## Compute lane masks

A compute shader can save its active lanes, use the resulting scalar condition to choose a value, then restore the original lanes. PortVI tracks that condition and lane masks used as ordinary numbers across control-flow joins. The emitted DXIL declares the wave size that these calculations require. Leaving that declaration out produced different values across subgroups on WARP during testing.

On the RTX 4060 the tests checked 32-lane outputs against a CPU reference, including an empty mask, a mask with only lane 31 and a branch joining a mask with a constant. The 64-lane programs passed the DXIL validator. The RTX 4060 reported 32 lanes and WARP reported four, so hardware readback of the 64-lane cases is open. An AMD hardware report can run the two `wave64` tests in `emit::synthetic_tests` and include the reported lane range. This work builds on Xvania7’s data-flow changes. [Upstream proposal](https://github.com/Marvisak/VIctory/pull/2289) · [Translator and readback tests](../patches/0014-compute-mask-values.patch)

### Loop-carried masks and scalar joins

A comparison produces one bit per lane. A loop can accumulate those bits into a mask, and a later branch can choose between that mask and an ordinary 32-bit word. Building the scalar word can uncover another mask dependency. PortVI fills both kinds of join until every live input has a value. In a 64-lane wave the two 32-bit halves identify the mask together. A high half written after a join is read where it exists. The implementation builds on nathanbzrr’s predecessor tracking in VIctory. Claude Code helped develop the loop readbacks. [Upstream change](https://github.com/Marvisak/VIctory/pull/385) · [Implementation and source-authored tests](../patches/0015-compute-mask-joins.patch)

The same 46 compute programs went through the Windows SDK DXIL validator at each step. All 46 originally stopped at a lane-mask join. With upstream’s tracking on the PortVI base, 33 emitted programs contained a boolean phi with no incoming value. A small public test reproduces that specific validator error and checks the returned word and lane bit on the GPU.

| Same 46 programs | DXIL emitted | Microsoft validator passed |
| --- | ---: | ---: |
| PortVI before this patch | 0 | 0 |
| VIctory PR 385 on that base | 46 | 13 |
| PortVI with the completed joins | 46 | 46 |

The desktop RTX 4060 (driver 32.0.16.2114) created 45 of the 46 D3D12 compute pipelines. The remaining program declares a fixed wave size of 64 and this adapter reports 32. Five authored Wave32 cases also checked two output words per lane over 20 GPU dispatches, matching all 1,280 reference words. The Wave64 loop readbacks still need a native 64-lane device. Across the full sample, 25,570 programs passed standalone DXIL validation, up from 25,524. These are translation and device measurements. Scene frame times have their own capture protocol below.

## Latest checks against Leonida and XCL

Leonida also merged a draw test for dual-source blending. XCL added parallel compilation for pieces of large direct compute dispatches and identified active pipelines in its device-removal reports. The XCL dispatch change targets pipeline creation time. It has a separate fallback for indirect dispatches and large loops. [Leonida draw test](https://github.com/ubiquitarianism/leonida/pull/1703) · [XCL compute dispatches](https://github.com/xcl-dev/xcl/pull/2080) · [XCL device-removal reports](https://github.com/xcl-dev/xcl/pull/1592)

| Project | Recent change and relevance to PortVI |
| --- | --- |
| Leonida | Its Vulkan runtime merged fixes for sizing a ReBAR arena, staging texture uploads and preserving offsets when importing host buffers. The parked-mask change is a separate open PR. With its parent’s translator our 12 authored dispatches matched 98 of 384 lane values. The PR matched all 384 on the same RTX 4060. [Merged runtime changes](https://github.com/ubiquitarianism/leonida/compare/71a769d7d85208f86c6e2e50263aca7df1becce9...3d4f661238f72fbc2133cf80649be1b258c2db91) · [Lane-slot PR](https://github.com/ubiquitarianism/leonida/pull/1803) · [Runnable Vulkan probe](../probes/leonida_lane_slots.py) |
| XCL | Its permlane tests passed on the RTX 4060. A further 42 native-wave32 and split-wave64 dispatches checked inactive rows and halves, all four instruction flag combinations and in-place writes. All 18,144 lane/register values matched XCL’s independent expected-value function. [Original tests and hardware report](https://github.com/xcl-dev/xcl/pull/478) |
| Leonida | Its Vulkan translator now unpacks compressed FP16 mesh exports and exports layer and viewport indices per primitive. Its maintainers report reaching the world phase after story loading and are still working towards a first confirmed world frame. These are useful cases for testing per-primitive outputs on Direct3D 12. [Compressed exports](https://github.com/ubiquitarianism/leonida/commit/d6efad1bbcf44d12ae2946adeb5081e27753be6f) · [Layer and viewport](https://github.com/ubiquitarianism/leonida/commit/c48cc2353abddb82a81e5cbac342ca81bac6fadf) · [Maintainer status](https://github.com/ubiquitarianism/leonida/blob/384abe5b06d08e349f4b4e4394c68f036da47dc7/README.md) |
| XCL | Its D3D12 backend corrected mesh triangle indices to nine bits and the index guard for a million-descriptor heap. Its NVIDIA integration branch reports about 5 FPS in a partially drawn world on an RTX 5090. The 14 FPS average also includes menus and loading. [Mesh index fix](https://github.com/xcl-dev/xcl/commit/3eab40a4bb48c15d4b79ed3f756deba52eb89bdb) · [Heap guard](https://github.com/xcl-dev/xcl/commit/19bef421e6f974fec2aa8289b228c78ab92f3013) · [Measurement and hardware](https://github.com/xcl-dev/xcl/blob/f0e1b2f149cc61231d04b838482f2fd5ec05b391/README.md) |
| Leonida | Its Vulkan backend changed the order of mesh-shader interface declarations after one AMD Windows driver failed on the old order. PortVI emits DXIL for D3D12. The cross-project check worth repeating is creating a real mesh pipeline on each driver after bytecode validation. [Merged change and hardware scope](https://github.com/ubiquitarianism/leonida/pull/802) |
| XCL | When a driver refuses a large compute pipeline, XCL now records the driver error, compile time, bytecode size and cache identity. These are useful fields when correlating PortVI’s pipeline failures with its offline shader results. Our driver checks here use small source-authored programs. [Merged D3D12 diagnostics](https://github.com/xcl-dev/xcl/pull/1918) |

## How the previous patches map to this engine

| Previous patches | Current implementation |
| --- | --- |
| 0001, 0004, 0011 · shader cache | The engine uses the shader’s own wave width. Patches 0002, 0009 and 0011 key the emitter options and both descriptor limits used for the compilation. The translated result survives a disk publication failure. |
| 0002, 0003, 0005 · CLI and adapter selection | Patch 0001 exposes `portvi.exe` and exercises it in the upstream CLI tests. Patch 0003 verifies the selected physical adapter and runs the hardware-required median fixture. |
| 0006, 0007, 0010 · depth and device methods | Patch 0004 restores the depth-stencil pipeline stream and dynamic bounds on supported adapters. The upstream device-removal, timestamp-frequency and command-list reset methods are exercised through their console slots. |
| 0008, 0009, 0014 · streaming | Upstream checks live indexed textures at submission. It tracks guarded pages and their aliases, hashes other committed memory, and marks writes again at completion. Patch 0006 exercises that path through an actual storage queue and GPU readback. [Upstream texture implementation](https://github.com/Marvisak/VIctory/blob/940d32f69b22265690a0b5a15d6ded572ba33baf/crates/victory-gpu/src/texture.rs) · [Upstream write tracking](https://github.com/Marvisak/VIctory/blob/940d32f69b22265690a0b5a15d6ded572ba33baf/crates/victory-gpu/src/watch.rs) |
| 0012, 0013 · resource identity and footprint size | Upstream already bounds detiling and normalizes the placed texture resource key. Patch 0005 covers the remaining diagnostic span and descriptor interning, with device-backed tests. |
| 0015, 0016 · G16 sampling | Patch 0003 adapts the decoder, DXIL lowering and original GPU fixtures to this revision’s translator. |

## BC4 storage and GPU readback

Run `python portvi.py texture` on Windows. The command identifies the adapter used by the rendering device and requires a physical GPU. The normal regression suite also exercises the fixture on software-capable CI hosts. [Fixture and sampling shader](../patches/0006-streamed-bc4-readback.patch) · [Tracking issue](https://github.com/tyr0nsur/PortVI/issues/2)

| Phase on an NVIDIA GeForce RTX 4060, driver 32.0.16.2114 | Measured result |
| --- | --- |
| Upload a uniform sentinel to an 8×4 BC4_UNORM texture using the runtime upload path | All 32 GPU samples matched the original value, 51 on a 0–255 scale. |
| Submit a file request through the storage queue, inflate zlib and decode two BCPack blocks into that same texture’s console memory | The completed request produced the 16 independently expected block bytes. |
| Sample the same host texture while deliberately withholding its next upload | All 32 samples still matched the sentinel. All 32 failed the reference for the new content. |
| Record the upload and sample again | All 32 texels matched the new reference. |

The fixture uses the runtime’s 4 KiB swizzle, detiling, staging and command submission. Its compute shader point-samples texel centers at mip zero. The oracle works directly from the authored endpoints and index progression. It allows 0.75 of one 8-bit intensity step for interpolants and 0.01 for endpoints. Forced WARP passed the same positive and negative controls locally. The hardware-only mode rejected that software run. The fixture measures an explicitly scheduled upload to a virtual destination. Aliased physical-page destinations and scene timing need separate measurements.

## G16 texture sampling

The console instruction `image_sample_d_g16` and its minimum-LOD form `image_sample_d_cl_g16` now lower to DXIL `sampleGrad`. Each gradient axis stores two 16-bit values per register and pads an odd last component independently. The coordinate-precision bit controls the coordinates and clamp. A decoder regression originally counted six address registers for a 2D case that requires four. The translator originally returned an unsupported-instruction error for both forms. [PortVI decoder, lowering and GPU tests](../patches/0003-rdna2-gradients-and-adapter.patch)

On the RTX 4060 and on WARP, the two new 2D forms passed 18 sampling cases per adapter. A seven-level texture encoded its own mip, x and y in its texels so a wrong register or level was visible in the readback from all 32 lanes. The suite also checks translation for 1D, 3D and array views. The cube, comparison and offset G16 forms are outside the implemented set.

A separate run checked the ordinary `image_sample_d` and `image_sample_d_cl` forms with 16-bit coordinates. RDNA2 still uses a full 32-bit register per gradient component for these opcodes. On a 2D sample the decoder originally read three address registers where five were required. The RTX 4060 returned mip 5, texel (0, 0) for a case that should read mip 2, texel (3, 1). With gradient width selected by the opcode and coordinate width selected by A16, all 18 ordinary-form cases passed per adapter as well. [Device regression and fix](../patches/0003-rdna2-gradients-and-adapter.patch) · [LLVM gradient and address packing](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/SIISelLowering.cpp#L10683-L10761) · [Leonida's independent f32-gradient fixture](https://github.com/ubiquitarianism/leonida/blob/ece08d3b1e2c51af15c133ae95480ad27cfb522b/runtime/test/shaders/ps_texture_grad_cl_a16.asm)

We also executed Leonida PR 1481's own SPIR-V translator and Vulkan renderer on this RTX 4060. Four 64×64 draws exercised both G16 instructions with and without A16. In the first two draws the gradient selected mip 1 on the right half. In the other two, a minimum LOD selected it. All 4,096 RGBA pixels of each draw matched the authored reference, and all seven of the PR's semantic tests passed with SPIR-V validation enabled.

To reproduce with Leonida revision `ece08d3b1`, build its `vk_texture_test` target, put `spirv-as` and `spirv-val` on PATH, then run `python probes/leonida_g16.py <leonida-checkout> <vk_texture_test.exe>`. [Four-case Vulkan fixture](../probes/leonida_g16.py) · [Leonida shader proposal](https://github.com/ubiquitarianism/leonida/pull/1481)

## Earlier upstream comparisons, 25 November 2026

These observations were recorded against the previous integration. Its measurements and patches are preserved at that revision. [Earlier engineering review](https://github.com/tyr0nsur/PortVI/blob/d0cb833051d5c100a790246c71d8363c7a5315c3/docs/engineering-review.md)

The comparison starts with the heads checked at 02:13 CET on 25 November. XCL’s corresponding main revision was `0cdc896`. [XCL comparison window through 02:25](https://github.com/xcl-dev/xcl/compare/0cdc896fa3b63ab781a5ba0dc0c278b240e02976...8f5fca5bb9ad6e695093d75294ad2cec023bba60)

| Project | Finding and PortVI decision |
| --- | --- |
| VIctory | The main branch’s hardware-test selection was already in our previous review. A separate integration branch reported a still-scene checkpoint on an RTX 3080 Ti with several settings changed together, including disabling ray tracing. PortVI keeps its existing defaults pending a comparable scene and timing capture. [VIctory main at the previous cutoff](https://github.com/Marvisak/VIctory/commit/ccfb02996eac831853e8324156f123eeff083e9d) · [Integration measurements](https://github.com/fxceless/victory-fork/pull/465) |
| fxceless | A new revision fixed duplicate placed textures when two resource descriptions differed only in ABI padding. PortVI now reads only the defined fields at the SRV and UAV entry points. This also gives descriptor interning and the host resource cache the same identity. A regression asks for the same address with two padding patterns, then changes its width and requires a distinct resource. [Upstream change](https://github.com/fxceless/victory-fork/commit/c96b4f2b8365bf43d54ab074940923ef8267f4c6) · [PortVI fix](https://github.com/tyr0nsur/PortVI/blob/d0cb833051d5c100a790246c71d8363c7a5315c3/patches/0013-placed-texture-identity.patch) |
| Leonida | Main corrected its documented count of 52 validated BVH-builder shaders and excluded a local indexing cache. Its open G16 sampling work and a corresponding decoder change in the VIctory fork identified separate packing of the two gradient axes. [Builder correction](https://github.com/ubiquitarianism/leonida/pull/1087) · [G16 sample proposal](https://github.com/ubiquitarianism/leonida/pull/1481) · [VIctory fork decoder](https://github.com/fxceless/victory-fork/pull/446) |
| XCL | The original ivall repository now resolves to xcl-dev/xcl. XCL reported reducing retained upload staging from 256 MiB to 144 MiB on a laptop RTX 4060 by changing how its buffer grows. It also corrected packed 3D mip tails. PortVI sizes each staging resource from the D3D12 footprint. Examining that allocation exposed a different local bug. [XCL staging measurements](https://github.com/xcl-dev/xcl/pull/2024) · [XCL volume texture fix](https://github.com/xcl-dev/xcl/pull/1990) |

At 02:41 CET a further check found two VIctory main commits limiting copy-back comparisons to pages marked by GPU writes. Leonida had merged the Windows and Proton story-loading work. Its authors reached the world-loading phase and had yet to verify a first world frame. XCL had reached `3da2abb`. An independent run on our desktop RTX 4060 confirmed its 132 MiB mixed RGBA/BC7 upload grew the staging buffer from 128 to 144 MiB and returned byte-exact pixels. [VIctory changes](https://github.com/Marvisak/VIctory/compare/ccfb02996eac831853e8324156f123eeff083e9d...940d32f69b22265690a0b5a15d6ded572ba33baf) · [Leonida integration](https://github.com/ubiquitarianism/leonida/pull/984) · [XCL hardware report](https://github.com/xcl-dev/xcl/pull/2024#issuecomment-7150085392) · [BC4 footprint report to VIctory](https://github.com/Marvisak/VIctory/pull/601#issuecomment-7149989674)

**Staging allocation bounds.** For the last mip of a two-mip 8×4 BC4 image, D3D12 reported a 256-byte row pitch and only eight allocated bytes from that mip’s offset to the end of the buffer. PortVI previously formed a 256-byte mutable slice there. Detiling now receives a slice bounded by the next footprint or the actual end of the allocation. The regression checks the device’s footprint and verifies that writing the final eight bytes leaves the preceding mip untouched. [PortVI fix and device regression](https://github.com/tyr0nsur/PortVI/blob/d0cb833051d5c100a790246c71d8363c7a5315c3/patches/0012-bound-texture-staging-footprints.patch)

## Reference projects and earlier experiments

**AnyXboxX has two execution paths.** Its isolated diagnostics run x86-64 code in Unicorn. Its separate `--natif` mode executes the machine code on the host CPU and supplies the platform runtime in C++. Its import report groups functions by provider. PortVI’s loader already prints a binding summary, so a useful follow-up is to check event ordering as well as successful binding in a small native callback fixture. [AnyXboxX execution modes and reports](https://github.com/LooperSalty/AnyXboxX/blob/b10149942d972ed2b1d7b721cce63347f086a0f8/README.md)

**Xenia and replayable frames.** Xenia can record and replay a GPU command stream. For a compact PortVI fixture, author a dispatch that writes an indirect argument, consume it on another queue and compare the readback to predetermined bytes. The same commands, waits and expected output can move between drivers. VIctory already persists driver pipeline objects, and the main project also has shader and pipeline prebuilding. The opportunity is reproducing one failing frame and checking its output across adapters. [Xenia GPU tracing](https://github.com/xenia-project/xenia/blob/master/docs/gpu.md) · [VIctory pipeline library](https://github.com/LooperSalty/victory/blob/fd021bf26a7003749d668e5d9bf6b234bfab2e47/crates/victory-gpu/src/pipeline/library.rs) · [Upstream prebuild](https://github.com/Marvisak/VIctory/commit/9c58a1c35a384552bde9b4c9b169731f282a15a1)

**xemu and hardware reference results.** The original Xbox graphics tests publish outputs produced by real hardware. The method transfers across GPU generations. For PortVI, use small programs with known outputs, keep the input and expected value public, and record GPU and driver with every comparison. The median test above is the first physical-adapter check here. The AMD descriptor-heap fix already under discussion upstream is another reason to repeat checks across vendors. [xemu graphics debugging](https://github.com/xemu-project/xemu-website/blob/master/docs/docs/dev/debug/graphics.md) · [Hardware reference results](https://github.com/abaire/nxdk_pgraph_tests_golden_results) · [Upstream AMD investigation](https://github.com/fxceless/victory-fork/pull/134)

**WinDurango and service behavior.** WinDurango implements Xbox One ERA and WinRT interfaces on Windows. Its D3D11.X backend targets an earlier graphics API. Resolving a function import leaves its timing contract to verify. A program can depend on when a callback fires, a fence signals or a queued read finishes. All 13 of the pinned VIctory storage tests passed here, including completion order, error propagation and staging limits. [WinDurango](https://github.com/WinDurango/WinDurango) · [Pinned storage tests](https://github.com/LooperSalty/victory/blob/fd021bf26a7003749d668e5d9bf6b234bfab2e47/crates/victory-hle/src/modules/dstorage.rs)

**Integrating the existing fence fix.** AndyLatimer's upstream PR 32 registers fences by address and tests waits satisfied by CPU signals, command-list writes and direct memory writes. Other contributors have already exercised those cases. At head `9c5328d` a dry run of its nine-file diff against the PortVI engine failed to apply in `fence.rs`, `host.rs`, `list.rs` and `queue.rs`. An integration should preserve the fork's deferred submission handling and repeat the PR's wakeup tests with it enabled. One public RTX 4090 comparison reported a device reset during loading both with and without the PR. [PR and test reports](https://github.com/Marvisak/VIctory/pull/32) · [Fork's comparison with newer upstream work](https://github.com/fxceless/victory-fork/issues/126)

**Xbox2Windows and XCL informed the texture fixture.** Xbox2Windows compares 16,384 bytes of a rendered frame and deliberately withholds an update to expose stale contents. An XCL contributor reported about 189 MB of BCPack requests with zero bytes filled on a 4060 Ti. PortVI now applies a skipped-upload control to a directly streamed BC4 resource and compares the GPU output against a reference derived from its authored pattern. [Xbox2Windows fixtures](https://github.com/R0Y00/Xbox2Windows) · [XCL texture measurements](https://github.com/xcl-dev/xcl/issues/878)

To include the default physical adapter's reported wave range with a GPU result, run these commands in an x64 Visual Studio developer shell from the project root after the first build. The probe calls Windows directly.

```powershell
cl /nologo /EHsc probes\adapter.cpp /Fotarget\adapter.obj /Fetarget\adapter.exe /link d3d12.lib dxgi.lib
.\target\adapter.exe
```

## Measuring a 30 FPS target

Run the same interactive scene for 120 seconds with the same build, driver, settings and resolution. Capture it once with a new cache directory and repeat with that directory populated. Record scene, drawn and missing work, observed visual faults and any device reset alongside the timings. A faster pass that skips geometry changes what the benchmark measures.

Start PresentMon while the scene is running. Keep frame generation disabled for a native-rendering benchmark. The summarizer reports all displayed intervals and includes the frame types when PresentMon provides them. [PresentMon console options and CSV fields](https://github.com/GameTechDev/PresentMon/blob/main/README-ConsoleApplication.md)

```powershell
New-Item -ItemType Directory -Force runs | Out-Null
PresentMon.exe --process_name portvi.exe --timed 120 --terminate_after_timed --output_file runs\warm.csv
python tools/frametimes.py runs\warm.csv
```

Read the game window's swap chain. The tool reports its displayed FPS, 95th and 99th percentile duration, worst duration and count of undisplayed rows. Each PID and swap chain has its own summary. At 30 FPS, one frame lasts about 33.33 ms. A 30 FPS average alongside a 90 ms 99th percentile means the slowest part of the run still contains visible stalls. The local tests above contain no scene-level performance measurement.
