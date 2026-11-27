//! Host pipelines for title pipelines (decision 0007, step 3): the
//! title's graphics state with the translated shaders, under a root
//! signature of our own that the translated shaders' bindings define
//! (`victory_shader::emit`): per stage one block of root constants for the
//! user data, root views for the title's root views, static samplers for
//! the immediate samplers, and directly indexed heaps.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use victory_shader::emit::{MEMORY_SPACE, ROOT_CBV_SPACE, SAMPLER_SPACE, USER_DATA_SPACE};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Interface, PCSTR};

use super::{Shader, Translated};
use crate::host::host;

/// A stream's graphics state, owned (the title's stream lives only for the
/// creation call). Subobjects the stream leaves out keep D3D12's defaults.
pub(crate) struct GraphicsState {
    blend: D3D12_BLEND_DESC,
    sample_mask: u32,
    rasterizer: D3D12_RASTERIZER_DESC,
    depth: D3D12_DEPTH_STENCIL_DESC,
    depth_bounds: bool,
    /// Input elements, their names pointing into `names`.
    inputs: Vec<D3D12_INPUT_ELEMENT_DESC>,
    names: Vec<CString>,
    strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
    pub(crate) topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    pub(crate) targets: D3D12_RT_FORMAT_ARRAY,
    depth_format: DXGI_FORMAT,
    samples: DXGI_SAMPLE_DESC,
}

impl Default for GraphicsState {
    fn default() -> Self {
        let target = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let keep = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        GraphicsState {
            blend: D3D12_BLEND_DESC {
                RenderTarget: [target; 8],
                ..Default::default()
            },
            sample_mask: u32::MAX,
            rasterizer: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_BACK,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            depth: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                StencilReadMask: 0xff,
                StencilWriteMask: 0xff,
                FrontFace: keep,
                BackFace: keep,
                ..Default::default()
            },
            depth_bounds: false,
            inputs: Vec::new(),
            names: Vec::new(),
            strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
            targets: D3D12_RT_FORMAT_ARRAY::default(),
            depth_format: DXGI_FORMAT_UNKNOWN,
            samples: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
        }
    }
}

/// Integer render target formats: D3D12 cannot blend into them.
fn integer_format(f: DXGI_FORMAT) -> bool {
    matches!(
        f,
        DXGI_FORMAT_R32G32B32A32_UINT
            | DXGI_FORMAT_R32G32B32A32_SINT
            | DXGI_FORMAT_R32G32B32_UINT
            | DXGI_FORMAT_R32G32B32_SINT
            | DXGI_FORMAT_R16G16B16A16_UINT
            | DXGI_FORMAT_R16G16B16A16_SINT
            | DXGI_FORMAT_R32G32_UINT
            | DXGI_FORMAT_R32G32_SINT
            | DXGI_FORMAT_R10G10B10A2_UINT
            | DXGI_FORMAT_R8G8B8A8_UINT
            | DXGI_FORMAT_R8G8B8A8_SINT
            | DXGI_FORMAT_R16G16_UINT
            | DXGI_FORMAT_R16G16_SINT
            | DXGI_FORMAT_R32_UINT
            | DXGI_FORMAT_R32_SINT
            | DXGI_FORMAT_R8G8_UINT
            | DXGI_FORMAT_R8G8_SINT
            | DXGI_FORMAT_R16_UINT
            | DXGI_FORMAT_R16_SINT
            | DXGI_FORMAT_R8_UINT
            | DXGI_FORMAT_R8_SINT
    )
}

/// The title's blend state as the host takes it: blending off on integer
/// render targets. The console ignores blending there, and titles leave it
/// on (the world sets it on an `R8_UINT` target), but D3D12 does not allow
/// it: vkd3d-proton refuses such a pipeline with `E_INVALIDARG` ("using
/// integer format is not supported"), and its draws are lost. With one
/// description for every target, each target gets its own copy so the
/// others keep theirs.
fn host_blend(blend: &D3D12_BLEND_DESC, targets: &D3D12_RT_FORMAT_ARRAY) -> D3D12_BLEND_DESC {
    let count = (targets.NumRenderTargets as usize).min(8);
    let per_target = blend.IndependentBlendEnable.as_bool();
    let blends = |i: usize| {
        blend.RenderTarget[if per_target { i } else { 0 }]
            .BlendEnable
            .as_bool()
    };
    if !(0..count).any(|i| integer_format(targets.RTFormats[i]) && blends(i)) {
        return *blend;
    }
    let mut host = *blend;
    if !per_target {
        host.IndependentBlendEnable = true.into();
        host.RenderTarget = [blend.RenderTarget[0]; 8];
    }
    for i in 0..count {
        if integer_format(targets.RTFormats[i]) {
            host.RenderTarget[i].BlendEnable = false.into();
        }
    }
    host
}

impl GraphicsState {
    /// Takes in one subobject of a stream: its type, then its data at +4
    /// (+8 for those holding pointers).
    ///
    /// # Safety
    /// `at` points at a whole subobject of type `kind`.
    pub(crate) unsafe fn read(&mut self, kind: u32, at: *const u8) {
        // SAFETY: the caller passes a whole subobject.
        unsafe {
            let data = at.add(4);
            match kind {
                8 => self.blend = data.cast::<D3D12_BLEND_DESC>().read_unaligned(),
                9 => self.sample_mask = data.cast::<u32>().read_unaligned(),
                10 => self.rasterizer = data.cast::<D3D12_RASTERIZER_DESC>().read_unaligned(),
                11 => {
                    self.depth = data.cast::<D3D12_DEPTH_STENCIL_DESC>().read_unaligned();
                    self.depth_bounds = false;
                }
                21 => {
                    // As before, read this console subobject with the PC layout.
                    let d = data.cast::<D3D12_DEPTH_STENCIL_DESC1>().read_unaligned();
                    self.depth_bounds = d.DepthBoundsTestEnable.as_bool();
                    self.depth = D3D12_DEPTH_STENCIL_DESC {
                        DepthEnable: d.DepthEnable,
                        DepthWriteMask: d.DepthWriteMask,
                        DepthFunc: d.DepthFunc,
                        StencilEnable: d.StencilEnable,
                        StencilReadMask: d.StencilReadMask,
                        StencilWriteMask: d.StencilWriteMask,
                        FrontFace: d.FrontFace,
                        BackFace: d.BackFace,
                    };
                }
                12 => {
                    let layout = at.add(8).cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned();
                    if layout.pInputElementDescs.is_null() {
                        return;
                    }
                    let elements = std::slice::from_raw_parts(
                        layout.pInputElementDescs,
                        layout.NumElements as usize,
                    );
                    self.names = elements
                        .iter()
                        .map(|e| CStr::from_ptr(e.SemanticName.0.cast()).to_owned())
                        .collect();
                    self.inputs = elements
                        .iter()
                        .zip(&self.names)
                        .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                            SemanticName: PCSTR(name.as_ptr().cast()),
                            Format: DXGI_FORMAT(
                                crate::resource::pc_format(e.Format.0 as u32) as i32
                            ),
                            ..*e
                        })
                        .collect();
                }
                13 => {
                    self.strip_cut =
                        D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(data.cast::<i32>().read_unaligned())
                }
                14 => {
                    self.topology =
                        D3D12_PRIMITIVE_TOPOLOGY_TYPE(data.cast::<i32>().read_unaligned())
                }
                15 => {
                    self.targets = data.cast::<D3D12_RT_FORMAT_ARRAY>().read_unaligned();
                    for f in &mut self.targets.RTFormats {
                        f.0 = crate::resource::pc_format(f.0 as u32) as i32;
                    }
                }
                16 => {
                    let f = data.cast::<u32>().read_unaligned();
                    self.depth_format = DXGI_FORMAT(crate::resource::pc_format(f) as i32);
                }
                17 => self.samples = data.cast::<DXGI_SAMPLE_DESC>().read_unaligned(),
                _ => {}
            }
        }
    }

    /// Everything the state holds, as text: part of a pipeline's name in
    /// the library and of the key a title pipeline is started early by.
    pub(crate) fn describe(&self) -> String {
        let inputs: Vec<String> = self
            .inputs
            .iter()
            .zip(&self.names)
            .map(|(e, name)| {
                format!(
                    "{name:?} {} {} {} {} {} {}",
                    e.SemanticIndex, e.Format.0, e.InputSlot, e.AlignedByteOffset, e.InputSlotClass.0, e.InstanceDataStepRate
                )
            })
            .collect();
        let depth_bounds = if self.depth_bounds { " depth bounds" } else { "" };
        format!(
            "{:?} {} {:?} {:?} {inputs:?} {} {} {:?} {} {:?}{depth_bounds}",
            self.blend,
            self.sample_mask,
            self.rasterizer,
            self.depth,
            self.strip_cut.0,
            self.topology.0,
            self.targets,
            self.depth_format.0,
            self.samples
        )
    }

    fn depth_with_bounds(&self, host: &crate::host::Host) -> Option<D3D12_DEPTH_STENCIL_DESC1> {
        if !self.depth_bounds {
            return None;
        }
        if !host.depth_bounds_supported {
            crate::host::warn_depth_bounds();
            return None;
        }
        let d = self.depth;
        Some(D3D12_DEPTH_STENCIL_DESC1 {
            DepthEnable: d.DepthEnable,
            DepthWriteMask: d.DepthWriteMask,
            DepthFunc: d.DepthFunc,
            StencilEnable: d.StencilEnable,
            StencilReadMask: d.StencilReadMask,
            StencilWriteMask: d.StencilWriteMask,
            FrontFace: d.FrontFace,
            BackFace: d.BackFace,
            DepthBoundsTestEnable: true.into(),
        })
    }

    /// The vertex buffer slots the input layout reads.
    fn slots(&self) -> u32 {
        self.inputs
            .iter()
            .fold(0, |m, e| m | 1 << e.InputSlot.min(31))
    }

    /// The input elements with their byte offsets worked out.
    fn elements(&self) -> Vec<Input> {
        let mut next = [0u32; 32];
        self.inputs
            .iter()
            .map(|e| {
                let slot = e.InputSlot.min(31) as usize;
                let format = e.Format.0 as u32;
                let bytes = crate::resource::element(format).map_or(0, |(b, _)| b as u32);
                // An appended element starts at the next multiple of
                // min(4, its size) after the previous one, where D3D12 (and
                // vkd3d) place it; the host pipeline gets the title's
                // descriptions, so the offsets read here must match.
                let offset = if e.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT {
                    next[slot]
                        .checked_next_multiple_of(bytes.clamp(1, 4))
                        .unwrap_or(u32::MAX)
                } else {
                    e.AlignedByteOffset
                };
                next[slot] = offset.saturating_add(bytes);
                Input {
                    slot: slot as u32,
                    offset,
                    format,
                    per_instance: e.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                }
            })
            .collect()
    }
}

/// One input element: vertex buffer slot, byte offset, format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Input {
    pub(crate) slot: u32,
    pub(crate) offset: u32,
    pub(crate) format: u32,
    pub(crate) per_instance: bool,
}

/// The console's own primitive topology types.
const RECTANGLES: i32 = 19;
const QUADS: i32 = 32;

/// One stage of a host pipeline and where its bindings go.
pub(crate) struct Stage {
    pub(crate) shader: Arc<Shader>,
    /// The root parameter of its user data, and its size in dwords.
    pub(crate) user_data: Option<(u32, u32)>,
    /// Root parameters of its root views: (title root parameter, ours).
    pub(crate) root_cbvs: Vec<(u8, u32)>,
    pub(crate) root_srvs: Vec<(u8, u32)>,
    pub(crate) root_uavs: Vec<(u8, u32)>,
    /// Our root parameters of the title-memory page table and of the
    /// written-pages bitmap (`crate::memory`), if it reads or writes there.
    pub(crate) memory: Option<u32>,
    pub(crate) memory_dirty: Option<u32>,
    /// Our root parameter of the heap's view addresses (`crate::memory`).
    pub(crate) view_addresses: Option<u32>,
}

impl Stage {
    pub(crate) fn translated(&self) -> &Translated {
        self.shader
            .translated()
            .expect("host pipelines are made from translated shaders")
    }
}

pub(crate) struct HostPipeline {
    /// A compute pipeline (else graphics).
    pub(crate) compute: bool,
    /// A mesh pipeline (graphics without vertex input).
    pub(crate) mesh: bool,
    pub(crate) pso: ID3D12PipelineState,
    pub(crate) root: ID3D12RootSignature,
    /// Vertex, then pixel stage, or the compute stage.
    pub(crate) stages: Vec<Stage>,
    /// A graphics pipeline of the console's rectangle lists whose geometry
    /// shader completes each rectangle (`victory_shader::emit::rectangle_gs`):
    /// it draws them as triangles, one per rectangle.
    pub(crate) rectangles: bool,
    /// The vertex buffer slots it reads (bit per slot).
    pub(crate) slots: u32,
    pub(crate) inputs: Vec<Input>,
    /// Its shaders, for reports of slow submissions.
    pub(crate) name: String,
    /// Our root parameters of the view-mark table and the buffer views'
    /// written-pages bitmap, if a stage marks its writes through views
    /// (`victory_shader::emit`, "Write marks").
    pub(crate) marks: Option<(u32, u32)>,
}

/// Logs a pipeline that gets no host pipeline, once per reason.
fn refuse(reason: &'static str, shaders: &[Arc<Shader>]) -> Option<HostPipeline> {
    static SEEN: std::sync::Mutex<Vec<&str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&reason) {
        seen.push(reason);
        tracing::warn!(?shaders, "no host pipeline: {reason} (logged once)");
    }
    None
}

impl HostPipeline {
    /// A host pipeline for vertex and pixel shaders with the graphics
    /// state `g`, or for one compute shader.
    pub(crate) fn new(shaders: &[Arc<Shader>], g: Option<&GraphicsState>) -> Option<HostPipeline> {
        let find = |stage| shaders.iter().find(|s| s.stage == stage);
        let compute = match shaders {
            [cs] if cs.stage == "compute" => Some(cs),
            _ => None,
        };
        if compute.is_none()
            && shaders
                .iter()
                .any(|s| !matches!(s.stage, "vertex" | "pixel" | "mesh"))
        {
            return refuse(
                "only vertex, mesh, pixel and compute stages are translated",
                shaders,
            );
        }
        let mesh = find("mesh");
        let stages = if let Some(cs) = compute {
            vec![(cs, D3D12_SHADER_VISIBILITY_ALL)]
        } else {
            let first = match (mesh, find("vertex")) {
                (Some(ms), _) => (ms, D3D12_SHADER_VISIBILITY_MESH),
                (None, Some(vs)) => (vs, D3D12_SHADER_VISIBILITY_VERTEX),
                (None, None) => return refuse("no vertex or mesh shader", shaders),
            };
            let mut stages = vec![first];
            stages.extend(find("pixel").map(|ps| (ps, D3D12_SHADER_VISIBILITY_PIXEL)));
            stages
        };
        if mesh.is_some() && !host()?.mesh_shaders() {
            return refuse("the host GPU has no mesh shaders (or no shader model 6.6)", shaders);
        }
        for (s, _) in &stages {
            s.translated()?;
        }
        // Mesh shaders whose standalone output footprint exceeds D3D12's 32 KiB
        // limit are re-translated together with their paired pixel shader,
        // pruning outputs the pixel shader never reads. Only a pair with
        // exactly the stages asked for replaces them; anything else keeps
        // the standalone translations (and the host's verdict on them).
        let pixel = find("pixel");
        let linked = mesh
            .and_then(|ms| {
                let bytes = victory_shader::emit::mesh_output_bytes(&ms.translated()?.host.dxil)?;
                if bytes <= victory_shader::emit::MESH_OUTPUT_LIMIT {
                    return None;
                }
                super::translate_linked(ms, pixel.map(Arc::as_ref))
            })
            .filter(|(_, ps)| ps.is_some() == pixel.is_some());
        let stages: Vec<(&Arc<Shader>, D3D12_SHADER_VISIBILITY)> = match &linked {
            Some((ms, ps)) => {
                let mut v = vec![(ms, D3D12_SHADER_VISIBILITY_MESH)];
                if let Some(ps) = ps {
                    v.push((ps, D3D12_SHADER_VISIBILITY_PIXEL));
                }
                v
            }
            None => stages,
        };
        let host = host()?;
        // `SFI0`: 64-bit atomics on typed resources and on heap resources.
        let atomics64 = stages.iter().any(|(s, _)| {
            let Some(t) = s.translated() else {
                return false;
            };
            victory_shader::Container::parse(&t.host.dxil)
                .ok()
                .and_then(|c| {
                    c.part(victory_shader::FourCC::SFI0)?
                        .data
                        .get(..4)?
                        .try_into()
                        .ok()
                })
                .is_some_and(|b: [u8; 4]| u32::from_le_bytes(b) & 0x1040_0000 != 0)
        });
        if atomics64 && !host.heap_atomics64() {
            return refuse(
                "the host GPU has no 64-bit atomics on typed heap resources",
                shaders,
            );
        }

        let aniso_point_mip_supported = crate::sampler::aniso_point_mip(&host.device);

        // The root signature the translated shaders' bindings define.
        let mut params: Vec<D3D12_ROOT_PARAMETER1> = Vec::new();
        let mut remap_param = None;
        let mut samplers = Vec::new();
        let mut bound = Vec::new();
        for &(shader, visibility) in &stages {
            let t = &shader.translated()?.host;
            let mut stage = Stage {
                shader: shader.clone(),
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                memory: None,
                memory_dirty: None,
                view_addresses: None,
            };
            let dwords = t.user_data_dwords();
            if dwords > 0 {
                stage.user_data = Some((params.len() as u32, dwords));
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: USER_DATA_SPACE,
                            Num32BitValues: dwords,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            for (slots, kind, out) in [
                (
                    &t.root_cbvs,
                    D3D12_ROOT_PARAMETER_TYPE_CBV,
                    &mut stage.root_cbvs,
                ),
                (
                    &t.root_srvs,
                    D3D12_ROOT_PARAMETER_TYPE_SRV,
                    &mut stage.root_srvs,
                ),
                (
                    &t.root_uavs,
                    D3D12_ROOT_PARAMETER_TYPE_UAV,
                    &mut stage.root_uavs,
                ),
            ] {
                for &slot in slots {
                    // All stages read the same map for the bound CBV/SRV/UAV heap.
                    // Each stage records its binding, but the signature pays for one root SRV.
                    let shared_remap = kind == D3D12_ROOT_PARAMETER_TYPE_SRV && slot == victory_shader::emit::VIEW_REMAP;
                    if shared_remap {
                        if let Some(index) = remap_param {
                            out.push((slot, index));
                            continue;
                        }
                        remap_param = Some(params.len() as u32);
                    }
                    out.push((slot, params.len() as u32));
                    params.push(D3D12_ROOT_PARAMETER1 {
                        ParameterType: kind,
                        Anonymous: D3D12_ROOT_PARAMETER1_0 {
                            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                                ShaderRegister: u32::from(slot),
                                RegisterSpace: ROOT_CBV_SPACE,
                                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                            },
                        },
                        ShaderVisibility: if shared_remap { D3D12_SHADER_VISIBILITY_ALL } else { visibility },
                    });
                }
            }
            for (used, kind, register, out) in [
                (t.memory, D3D12_ROOT_PARAMETER_TYPE_SRV, 0, &mut stage.memory),
                (t.memory_writes, D3D12_ROOT_PARAMETER_TYPE_UAV, 0, &mut stage.memory_dirty),
                (t.view_addresses, D3D12_ROOT_PARAMETER_TYPE_SRV, 1, &mut stage.view_addresses),
            ] {
                if used {
                    *out = Some(params.len() as u32);
                    params.push(D3D12_ROOT_PARAMETER1 {
                        ParameterType: kind,
                        Anonymous: D3D12_ROOT_PARAMETER1_0 {
                            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                                ShaderRegister: register,
                                RegisterSpace: MEMORY_SPACE,
                                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                            },
                        },
                        ShaderVisibility: visibility,
                    });
                }
            }
            for (n, s) in t.static_samplers.iter().enumerate() {
                samplers.push(D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER(s.filter_for_host(aniso_point_mip_supported) as i32),
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE(s.address[0] as i32),
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE(s.address[1] as i32),
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE(s.address[2] as i32),
                    MipLODBias: s.mip_lod_bias,
                    MaxAnisotropy: s.max_anisotropy,
                    ComparisonFunc: D3D12_COMPARISON_FUNC(s.comparison as i32),
                    BorderColor: D3D12_STATIC_BORDER_COLOR(s.border as i32),
                    MinLOD: s.min_lod,
                    MaxLOD: s.max_lod,
                    ShaderRegister: n as u32,
                    RegisterSpace: SAMPLER_SPACE,
                    ShaderVisibility: visibility,
                });
            }
            bound.push(stage);
        }
        // One pair for all stages: they share the heap and the bitmap.
        let marks = stages
            .iter()
            .any(|(s, _)| s.translated().is_some_and(|t| t.host.view_marks))
            .then(|| {
                let first = params.len() as u32;
                for (kind, register) in [
                    (D3D12_ROOT_PARAMETER_TYPE_SRV, victory_shader::emit::VIEW_MARKS),
                    (D3D12_ROOT_PARAMETER_TYPE_UAV, victory_shader::emit::VIEW_WRITES),
                ] {
                    params.push(D3D12_ROOT_PARAMETER1 {
                        ParameterType: kind,
                        Anonymous: D3D12_ROOT_PARAMETER1_0 {
                            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                                ShaderRegister: register,
                                RegisterSpace: MEMORY_SPACE,
                                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                            },
                        },
                        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                    });
                }
                (first, first + 1)
            });
        let (root, root_blob) = match root_signature(&host.device, &params, &samplers, compute.is_some()) {
            Ok(root) => root,
            Err(e) => {
                tracing::warn!(error = %e, ?shaders, "host root signature refused");
                return None;
            }
        };
        let code = |i: usize| {
            stages.get(i).and_then(|(s, _)| s.translated()).map_or(
                D3D12_SHADER_BYTECODE::default(),
                |t| D3D12_SHADER_BYTECODE {
                    pShaderBytecode: t.host.dxil.as_ptr().cast(),
                    BytecodeLength: t.host.dxil.len(),
                },
            )
        };
        if compute.is_some() {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // SAFETY: a borrowed pointer the description never releases.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                CS: code(0),
                ..Default::default()
            };
            let name = super::library::name("compute", &root_blob, &[&code_bytes(&code(0))], "");
            let made = super::library::pipeline(
                &name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateComputePipelineState(&desc) },
            );
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: true,
                mesh: false,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                rectangles: false,
                slots: 0,
                inputs: Vec::new(),
                marks,
            });
        }
        let Some(g) = g else {
            return refuse("graphics state not kept (plain description)", shaders);
        };
        let state = g.describe();
        let depth_with_bounds = g.depth_with_bounds(host);
        if mesh.is_some() {
            let name = super::library::name(
                "mesh",
                &root_blob,
                &[&code_bytes(&code(0)), &code_bytes(&code(1))],
                &state,
            );
            let ms = Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, code(0));
            let made = match depth_with_bounds {
                Some(depth) => stream_pipeline(
                    &host.device, &root, ms, code(1), g,
                    Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, depth),
                    &name,
                ),
                None => stream_pipeline(
                    &host.device, &root, ms, code(1), g,
                    Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, g.depth),
                    &name,
                ),
            };
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host mesh pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host mesh pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: false,
                mesh: true,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                rectangles: false,
                slots: 0,
                inputs: Vec::new(),
                marks,
            });
        }

        // Rectangle lists: a geometry shader made for the vertex shader adds
        // each rectangle's fourth corner (else rectangles are expanded on
        // the CPU from vertex inputs, see `draw`).
        let gs = match (g.topology.0, stages.first().and_then(|(s, _)| s.translated())) {
            (RECTANGLES, Some(vs)) => match victory_shader::emit::rectangle_gs(&vs.host.dxil) {
                Ok(gs) => gs,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "no rectangle geometry shader");
                    Vec::new()
                }
            },
            _ => Vec::new(),
        };
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: code(0),
            PS: code(1),
            GS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: if gs.is_empty() { std::ptr::null() } else { gs.as_ptr().cast() },
                BytecodeLength: gs.len(),
            },
            BlendState: host_blend(&g.blend, &g.targets),
            SampleMask: g.sample_mask,
            RasterizerState: g.rasterizer,
            DepthStencilState: g.depth,
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: g.inputs.as_ptr(),
                NumElements: g.inputs.len() as u32,
            },
            IBStripCutValue: g.strip_cut,
            PrimitiveTopologyType: host_topology(g),
            NumRenderTargets: g.targets.NumRenderTargets,
            RTVFormats: g.targets.RTFormats,
            DSVFormat: g.depth_format,
            SampleDesc: g.samples,
            ..Default::default()
        };
        let name = super::library::name(
            "graphics",
            &root_blob,
            &[&code_bytes(&code(0)), &code_bytes(&code(1)), &gs],
            &state,
        );
        let started = std::time::Instant::now();
        let made = if let Some(depth) = depth_with_bounds {
            let vertex = VertexStream {
                vs: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, desc.VS),
                gs: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, desc.GS),
                input: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, desc.InputLayout),
                strip_cut: Sub(
                    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE,
                    desc.IBStripCutValue,
                ),
            };
            stream_pipeline(
                &host.device, &root, vertex, desc.PS, g,
                Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, depth),
                &name,
            )
        } else {
            super::library::pipeline(
                &name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadGraphicsPipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateGraphicsPipelineState(&desc) },
            )
        };
        let pso = match made {
            Ok(pso) => pso,
            Err(e) => {
                tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                host.log_messages();
                return None;
            }
        };
        tracing::info!(?shaders, ms = started.elapsed().as_millis(), rectangles = !gs.is_empty(), "host pipeline");
        crate::host::name(&pso, || format!("{shaders:?}"));
        tracing::debug!(
            ?shaders,
            depth = ?g.depth,
            dsv = g.depth_format.0,
            targets = ?&g.targets.RTFormats[..g.targets.NumRenderTargets.min(8) as usize],
            blend = ?g.blend.RenderTarget[0],
            cull = g.rasterizer.CullMode.0,
            "host graphics state"
        );
        Some(HostPipeline {
            compute: false,
            mesh: false,
            pso,
            root,
            stages: bound,
            name: format!("{shaders:?}"),
            rectangles: !gs.is_empty(),
            slots: g.slots(),
            inputs: g.elements(),
            marks,
        })
    }
}

/// The host's topology type: the console's rectangle and quad types draw
/// as triangles on the host (see `draw`).
fn host_topology(g: &GraphicsState) -> D3D12_PRIMITIVE_TOPOLOGY_TYPE {
    match g.topology.0 {
        RECTANGLES | QUADS => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        _ => g.topology,
    }
}

/// A stream subobject: its type, then its description, pointer-aligned.
#[repr(C, align(8))]
struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);

/// What a vertex pipeline's stream has in place of the mesh shader (the
/// geometry shader is the rectangle lists' one, or empty).
#[repr(C)]
struct VertexStream {
    vs: Sub<D3D12_SHADER_BYTECODE>,
    gs: Sub<D3D12_SHADER_BYTECODE>,
    input: Sub<D3D12_INPUT_LAYOUT_DESC>,
    strip_cut: Sub<D3D12_INDEX_BUFFER_STRIP_CUT_VALUE>,
}

/// The stream form used by mesh pipelines and vertex pipelines with depth bounds.
fn stream_pipeline<S, D>(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    stages: S,
    ps: D3D12_SHADER_BYTECODE,
    g: &GraphicsState,
    depth: Sub<D>,
    name: &str,
) -> windows::core::Result<ID3D12PipelineState> {
    #[repr(C)]
    struct Stream<S, D> {
        root: Sub<*mut std::ffi::c_void>,
        stages: S,
        ps: Sub<D3D12_SHADER_BYTECODE>,
        blend: Sub<D3D12_BLEND_DESC>,
        mask: Sub<u32>,
        raster: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D>,
        targets: Sub<D3D12_RT_FORMAT_ARRAY>,
        dsv: Sub<DXGI_FORMAT>,
        samples: Sub<DXGI_SAMPLE_DESC>,
        topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
    }
    let mut stream = Stream {
        root: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root.as_raw()),
        stages,
        ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
        blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, host_blend(&g.blend, &g.targets)),
        mask: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, g.sample_mask),
        raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer),
        depth,
        targets: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, g.targets),
        dsv: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, g.depth_format),
        samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, g.samples),
        // A vertex pipeline needs its type. A mesh pipeline's, which the
        // host derives from the mesh shader anyway, is given too: a
        // pipeline library compares descriptions with it, and loads none
        // made from a stream without it.
        topology: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, host_topology(g)),
    };
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: size_of_val(&stream),
        pPipelineStateSubobjectStream: (&raw mut stream).cast(),
    };
    let device = device.cast::<ID3D12Device2>()?;
    super::library::pipeline(
        name,
        // SAFETY: the stream holds whole subobjects pointing at live data.
        |library, name| unsafe { library.LoadPipeline(name, &desc) },
        // SAFETY: as above.
        || unsafe { device.CreatePipelineState(&desc) },
    )
}

/// The bytes of shader code (none for an absent stage).
fn code_bytes(code: &D3D12_SHADER_BYTECODE) -> Vec<u8> {
    if code.pShaderBytecode.is_null() {
        return Vec::new();
    }
    // SAFETY: a translation's own bytes, alive while its shader is.
    unsafe { std::slice::from_raw_parts(code.pShaderBytecode.cast::<u8>(), code.BytecodeLength) }.to_vec()
}

fn root_signature(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER1],
    samplers: &[D3D12_STATIC_SAMPLER_DESC],
    compute: bool,
) -> windows::core::Result<(ID3D12RootSignature, Vec<u8>)> {
    let input = if compute {
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    } else {
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
    };
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: windows::Win32::Graphics::Direct3D12::D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: samplers.len() as u32,
                pStaticSamplers: samplers.as_ptr(),
                Flags: input
                    | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                    | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
            },
        },
    };
    // SAFETY: the description points at live arrays; the blob is read in
    // full before it is released.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        if let Err(e) = D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error)) {
            if let Some(error) = error {
                let text = std::slice::from_raw_parts(
                    error.GetBufferPointer().cast::<u8>(),
                    error.GetBufferSize(),
                );
                tracing::warn!("root signature: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        Ok((device.CreateRootSignature(0, bytes)?, bytes.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn targets(formats: &[DXGI_FORMAT]) -> D3D12_RT_FORMAT_ARRAY {
        let mut t = D3D12_RT_FORMAT_ARRAY {
            NumRenderTargets: formats.len() as u32,
            ..Default::default()
        };
        t.RTFormats[..formats.len()].copy_from_slice(formats);
        t
    }

    fn blending(independent: bool) -> D3D12_BLEND_DESC {
        let mut b = GraphicsState::default().blend;
        b.IndependentBlendEnable = independent.into();
        for t in &mut b.RenderTarget {
            t.BlendEnable = true.into();
            t.SrcBlend = D3D12_BLEND_SRC_ALPHA;
            t.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        }
        b
    }

    #[test]
    fn integer_targets_do_not_blend() {
        let t = targets(&[
            DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            DXGI_FORMAT_R8_UINT,
        ]);
        for independent in [false, true] {
            let host = host_blend(&blending(independent), &t);
            assert!(host.IndependentBlendEnable.as_bool());
            assert!(host.RenderTarget[0].BlendEnable.as_bool());
            assert!(host.RenderTarget[1].BlendEnable.as_bool());
            assert!(!host.RenderTarget[2].BlendEnable.as_bool());
            // The rest of the target's state is the title's.
            assert_eq!(host.RenderTarget[2].SrcBlend, D3D12_BLEND_SRC_ALPHA);
            assert_eq!(host.RenderTarget[0].DestBlend, D3D12_BLEND_INV_SRC_ALPHA);
        }
    }

    #[test]
    fn blend_states_without_integer_blending_are_kept() {
        // No integer target.
        let b = blending(false);
        let t = targets(&[DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R11G11B10_FLOAT]);
        assert_eq!(host_blend(&b, &t), b);
        // An integer target that does not blend.
        let mut b = blending(true);
        b.RenderTarget[1].BlendEnable = false.into();
        let t = targets(&[DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32_UINT]);
        assert_eq!(host_blend(&b, &t), b);
        // Integer formats past the targets in use.
        let mut t = targets(&[DXGI_FORMAT_R8G8B8A8_UNORM]);
        t.RTFormats[1] = DXGI_FORMAT_R8_UINT;
        assert_eq!(host_blend(&blending(false), &t), blending(false));
    }
    fn element(format: DXGI_FORMAT, offset: u32) -> D3D12_INPUT_ELEMENT_DESC {
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(c"TEXCOORD".as_ptr().cast()),
            SemanticIndex: 0,
            Format: format,
            InputSlot: 0,
            AlignedByteOffset: offset,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        }
    }

    fn offsets(inputs: &[D3D12_INPUT_ELEMENT_DESC]) -> Vec<u32> {
        let state = GraphicsState {
            inputs: inputs.to_vec(),
            ..GraphicsState::default()
        };
        state.elements().iter().map(|e| e.offset).collect()
    }

    /// An appended element after one of 1 or 2 bytes started right after
    /// it; D3D12 aligns it to min(4, its size), so the rectangle expansion
    /// read its vertices from the wrong bytes.
    #[test]
    fn appended_elements_are_aligned() {
        const APPEND: u32 = D3D12_APPEND_ALIGNED_ELEMENT;
        // RG8 (2 bytes) then RG32 float: 4, not 2.
        let a = [
            element(DXGI_FORMAT_R8G8_UNORM, 0),
            element(DXGI_FORMAT_R32G32_FLOAT, APPEND),
        ];
        assert_eq!(offsets(&a), [0, 4]);
        // R8 then R16 float: 2, not 1; then R8 packs right after it.
        let b = [
            element(DXGI_FORMAT_R8_UNORM, 0),
            element(DXGI_FORMAT_R16_FLOAT, APPEND),
            element(DXGI_FORMAT_R8_UNORM, APPEND),
        ];
        assert_eq!(offsets(&b), [0, 2, 4]);
        // Already aligned layouts are unchanged.
        let c = [
            element(DXGI_FORMAT_R8G8B8A8_UNORM, 0),
            element(DXGI_FORMAT_R32G32_FLOAT, APPEND),
            element(DXGI_FORMAT_R32G32B32_FLOAT, APPEND),
        ];
        assert_eq!(offsets(&c), [0, 4, 12]);
    }

    /// An explicit offset near 4 GiB overflowed the next appended offset
    /// (a panic in debug builds).
    #[test]
    fn a_huge_explicit_offset_does_not_overflow() {
        let inputs = [
            element(DXGI_FORMAT_R32G32B32A32_FLOAT, u32::MAX - 8),
            element(DXGI_FORMAT_R32_FLOAT, D3D12_APPEND_ALIGNED_ELEMENT),
        ];
        let _ = offsets(&inputs);
    }

    #[test]
    fn depth_bounds_follow_the_last_depth_subobject() {
        #[repr(C, align(8))]
        struct Sub21(u32, D3D12_DEPTH_STENCIL_DESC1);
        #[repr(C, align(8))]
        struct Sub11(u32, D3D12_DEPTH_STENCIL_DESC);
        let mut g = GraphicsState::default();
        let plain = g.describe();
        let sub = Sub21(
            21,
            D3D12_DEPTH_STENCIL_DESC1 {
                DepthBoundsTestEnable: true.into(),
                ..Default::default()
            },
        );
        let plain_sub = Sub11(11, D3D12_DEPTH_STENCIL_DESC::default());
        // SAFETY: whole subobjects of their kinds.
        unsafe { g.read(21, (&raw const sub).cast()) };
        assert!(g.depth_bounds && g.describe().ends_with(" depth bounds"));
        // A later plain depth subobject turns the bounds off again...
        // SAFETY: as above.
        unsafe { g.read(11, (&raw const plain_sub).cast()) };
        assert!(!g.depth_bounds);
        assert!(!g.describe().contains("depth bounds"));
        // ...and a later subobject 21 turns them back on.
        // SAFETY: as above.
        unsafe { g.read(21, (&raw const sub).cast()) };
        assert!(g.depth_bounds);
        // SAFETY: as above.
        unsafe { g.read(11, (&raw const plain_sub).cast()) };
        g.depth = GraphicsState::default().depth;
        assert_eq!(g.describe(), plain);
    }

    /// A subobject 21's depth and stencil fields reach the host's
    /// `DEPTH_STENCIL1` unchanged, with the bounds test on; a host without
    /// the bounds test gets none (the plain description is used instead).
    #[test]
    fn host_depth_stencil1_keeps_the_depth_and_stencil_fields() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        #[repr(C, align(8))]
        struct Sub21(u32, D3D12_DEPTH_STENCIL_DESC1);
        let front = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_ZERO,
            StencilDepthFailOp: D3D12_STENCIL_OP_INCR,
            StencilPassOp: D3D12_STENCIL_OP_REPLACE,
            StencilFunc: D3D12_COMPARISON_FUNC_EQUAL,
        };
        let back = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_INVERT,
            StencilDepthFailOp: D3D12_STENCIL_OP_DECR_SAT,
            StencilPassOp: D3D12_STENCIL_OP_INCR_SAT,
            StencilFunc: D3D12_COMPARISON_FUNC_GREATER_EQUAL,
        };
        let title = D3D12_DEPTH_STENCIL_DESC1 {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
            DepthFunc: D3D12_COMPARISON_FUNC_GREATER,
            StencilEnable: true.into(),
            StencilReadMask: 0x3c,
            StencilWriteMask: 0xc3,
            FrontFace: front,
            BackFace: back,
            DepthBoundsTestEnable: true.into(),
        };
        let mut g = GraphicsState::default();
        let sub = Sub21(21, title);
        // SAFETY: `sub` is a whole subobject 21.
        unsafe { g.read(21, (&raw const sub).cast()) };
        assert!(g.depth_bounds);

        let Some(d) = g.depth_with_bounds(host) else {
            assert!(!host.depth_bounds_supported, "a supporting host gets DEPTH_STENCIL1");
            eprintln!("no host depth bounds support; skipped");
            return;
        };
        assert_eq!(d, title);

        // Without the bounds test the plain description is used.
        g.depth_bounds = false;
        assert!(g.depth_with_bounds(host).is_none());
    }

    /// The PC `dxc.exe` (`VICTORY_PC_DXC`, else the newest Windows SDK's).
    fn dxc() -> Option<std::path::PathBuf> {
        let dxc = match std::env::var_os("VICTORY_PC_DXC") {
            Some(p) => std::path::PathBuf::from(p),
            None => victory_shader::emit::validator::Validator::default_path()?
                .with_file_name("dxc.exe"),
        };
        dxc.exists().then_some(dxc)
    }

    fn compile(dxc: &std::path::Path, dir: &std::path::Path, src: &str, profile: &str) -> Vec<u8> {
        let (hlsl, out) = (
            dir.join(format!("{profile}.hlsl")),
            dir.join(format!("{profile}.dxil")),
        );
        std::fs::write(&hlsl, src).unwrap();
        let status = std::process::Command::new(dxc)
            .args(["-nologo", "-T", profile, "-E", "main", "-Fo"])
            .arg(&out)
            .arg(&hlsl)
            .status()
            .unwrap();
        assert!(status.success(), "dxc failed for {profile}");
        std::fs::read(out).unwrap()
    }

    fn code(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
        D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        }
    }

    /// A full-screen triangle at depth 0.5.
    const VS: &str = "float4 main(uint i : SV_VertexID) : SV_Position {
        float2 p = float2((i << 1) & 2, i & 2);
        return float4(p * 2.0 - 1.0, 0.5, 1.0);
    }";
    const PS: &str = "float4 main() : SV_Target { return 1; }";

    /// A vertex pipeline with `DepthBoundsTestEnable` (the depth test
    /// always passing and not writing, so only the bounds gate the draw),
    /// made by the stream the host pipelines use, loads back from the
    /// library, and its draws pass or are culled by the bounds the draw
    /// state gives the list (`crate::draw::State::set_depth_bounds`), on the
    /// real host (WARP).
    #[test]
    fn depth_bounds_pipeline_loads_and_filters_draws() {
        use std::sync::atomic::Ordering;
        let Some(host) = host() else {
            return; // no host GPU here
        };
        if !host.depth_bounds_supported {
            eprintln!("no host depth bounds support; skipped");
            return;
        }
        let Some(dxc) = dxc() else {
            eprintln!("no PC dxc.exe; skipped");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let vs = compile(&dxc, dir.path(), VS, "vs_6_5");
        let ps = compile(&dxc, dir.path(), PS, "ps_6_5");
        let (root, root_blob) = root_signature(&host.device, &[], &[], false).unwrap();

        let mut g = GraphicsState::default();
        #[repr(C, align(8))]
        struct Sub21(u32, D3D12_DEPTH_STENCIL_DESC1);
        let sub = Sub21(
            21,
            D3D12_DEPTH_STENCIL_DESC1 {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                DepthBoundsTestEnable: true.into(),
                ..Default::default()
            },
        );
        // SAFETY: `sub` is a whole subobject 21.
        unsafe { g.read(21, (&raw const sub).cast()) };
        assert!(g.depth_bounds);
        g.rasterizer.CullMode = D3D12_CULL_MODE_NONE;
        g.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        g.targets.NumRenderTargets = 1;
        g.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        g.depth_format = DXGI_FORMAT_D32_FLOAT;
        let depth1 = g
            .depth_with_bounds(host)
            .expect("a host with the bounds test gets DEPTH_STENCIL1");
        assert!(depth1.DepthBoundsTestEnable.as_bool());

        let name = super::super::library::name("graphics depth-bounds test", &root_blob, &[&vs, &ps], &g.describe());
        let make = || {
            let vertex = VertexStream {
                vs: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, code(&vs)),
                gs: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, D3D12_SHADER_BYTECODE::default()),
                input: Sub(
                    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT,
                    D3D12_INPUT_LAYOUT_DESC::default(),
                ),
                strip_cut: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE, g.strip_cut),
            };
            stream_pipeline(
                &host.device,
                &root,
                vertex,
                code(&ps),
                &g,
                Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, depth1),
                &name,
            )
        };
        let pso = make().expect("depth-bounds pipeline made");
        let loaded = super::super::library::LOADED.load(Ordering::Relaxed);
        make().expect("depth-bounds pipeline made again");
        assert!(
            super::super::library::LOADED.load(Ordering::Relaxed) > loaded,
            "the depth-bounds pipeline was compiled again instead of loaded from the library"
        );

        let texture = |format, flags, state, clear: D3D12_CLEAR_VALUE| {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 4,
                Height: 4,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: flags,
                ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            // SAFETY: valid heap and resource descriptions.
            unsafe {
                host.device
                    .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd, state, Some(&clear), &mut out)
                    .unwrap();
            }
            out.unwrap()
        };
        let color = texture(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] },
            },
        );
        let depth = texture(
            DXGI_FORMAT_D32_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_D32_FLOAT,
                Anonymous: D3D12_CLEAR_VALUE_0 {
                    DepthStencil: D3D12_DEPTH_STENCIL_VALUE { Depth: 0.5, Stencil: 0 },
                },
            },
        );
        let readback = {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 1024,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            // SAFETY: a valid read-back buffer description.
            unsafe {
                host.device
                    .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut out)
                    .unwrap();
            }
            out.unwrap()
        };

        // One draw over a depth buffer cleared to 0.5 with the given
        // bounds: the first pixel's red.
        let draw = |min: f32, max: f32| -> u8 {
            let list = host.new_list().unwrap();
            assert!(list.depth_bounds().is_some(), "the list of a supporting host sets bounds");
            let mut state = crate::draw::State::default();
            state.set_depth_bounds(&list, min, max);
            let rtv = host.rtv(&color, None);
            let dsv = host.dsv(&depth, None);
            let viewport = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: 4.0,
                Height: 4.0,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            let rect = windows::Win32::Foundation::RECT { left: 0, top: 0, right: 4, bottom: 4 };
            // SAFETY: an open list, live resources and views.
            unsafe {
                list.list.ClearRenderTargetView(rtv, &[0.0; 4], None);
                list.list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.5, 0, None);
                list.list.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
                list.list.RSSetViewports(&[viewport]);
                list.list.RSSetScissorRects(&[rect]);
                list.list.SetGraphicsRootSignature(&root);
                list.list.SetPipelineState(&pso);
                list.list.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                );
                list.list.DrawInstanced(3, 1, 0, 0);
                list.list.ResourceBarrier(&[crate::host::transition(
                    &color,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&readback),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                Width: 4,
                                Height: 4,
                                Depth: 1,
                                RowPitch: 256,
                            },
                        },
                    },
                };
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&color),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                list.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                list.list.ResourceBarrier(&[crate::host::transition(
                    &color,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                )]);
            }
            list.close();
            host.execute(&[&list]);
            let mut at = std::ptr::null_mut();
            // SAFETY: `execute` waited for the GPU; the buffer holds 1024 bytes.
            unsafe {
                readback.Map(0, None, Some(&mut at)).unwrap();
                let pixel = *at.cast::<u8>();
                readback.Unmap(0, None);
                pixel
            }
        };

        assert_eq!(draw(0.6, 1.0), 0, "depth 0.5 outside [0.6, 1.0] must be culled");
        assert_eq!(draw(0.4, 0.6), 255, "depth 0.5 inside [0.4, 0.6] must pass");
        assert_eq!(draw(0.0, 0.4), 0, "depth 0.5 outside [0.0, 0.4] must be culled");
        eprintln!("depth bounds readback: culled, passed, culled");
    }

    /// Every binding of a translation, without its code.
    fn bindings(t: &Translated) -> (victory_shader::emit::HostShader, Vec<victory_shader::UsedResource>) {
        let host = victory_shader::emit::HostShader {
            dxil: Vec::new(),
            ..t.host.clone()
        };
        (host, t.used.clone())
    }

    /// The normal entry, `HostPipeline::new`, given console containers of a
    /// mesh program over 32 KiB standalone (128 lanes, 16 `float4`
    /// outputs: 34,816 bytes) and a pixel program reading `TEXCOORD0.xy`
    /// and `TEXCOORD15.zw`: the host pipeline's stages, and the root
    /// parameters made from them, are the linked translations (both stages,
    /// 4,096 bytes of mesh outputs, the standalone translations' bindings
    /// and console resources), the same ones again for a second pipeline;
    /// a mesh dispatch through its root signature and pipeline writes
    /// (1, 0.5, 0.25, 1) over the whole target. On the process's host GPU
    /// (hardware, or WARP where that is the host); skipped without mesh
    /// shaders.
    #[test]
    fn host_pipeline_new_binds_and_draws_the_linked_mesh_and_pixel_programs() {
        use victory_shader::emit::{MESH_OUTPUT_LIMIT, mesh_output_bytes};
        use victory_shader::native::fixture;
        let Some(host) = host() else {
            return; // no host GPU here
        };
        if !host.mesh_shaders() {
            eprintln!("no host mesh shaders; skipped");
            return;
        }
        let ms = Arc::new(Shader::new("mesh", &fixture::wide_mesh(128, 128, 128)));
        let reads = [(0, 0), (0, 1), (15, 2), (15, 3)];
        let ps = Arc::new(Shader::new("pixel", &fixture::reading_pixel(16, &reads, true)));
        let alone_ms = ms.translated().expect("the mesh program translates");
        let alone_ps = ps.translated().expect("the pixel program translates");
        let alone_bytes = mesh_output_bytes(&alone_ms.host.dxil).unwrap();
        assert_eq!(alone_bytes, 34_816);
        assert!(alone_bytes > MESH_OUTPUT_LIMIT, "standalone fits; nothing to link");

        let mut g = GraphicsState::default();
        g.rasterizer.CullMode = D3D12_CULL_MODE_NONE;
        g.depth.DepthEnable = false.into();
        g.depth.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        g.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        g.targets.NumRenderTargets = 1;
        g.targets.RTFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        let shaders = [ms.clone(), ps.clone()];
        let made = HostPipeline::new(&shaders, Some(&g)).expect("a host mesh pipeline");
        assert!(made.mesh && !made.compute);
        assert_eq!(made.stages.len(), 2);
        assert!(!Arc::ptr_eq(&made.stages[0].shader, &ms), "the standalone mesh shader is bound");
        assert!(!Arc::ptr_eq(&made.stages[1].shader, &ps), "the standalone pixel shader is bound");
        let (linked_ms, linked_ps) = (made.stages[0].translated(), made.stages[1].translated());
        assert_eq!(linked_ms.host.stage, victory_shader::HwStage::Ms);
        assert_eq!(linked_ps.host.stage, victory_shader::HwStage::Ps);
        assert_eq!(mesh_output_bytes(&linked_ms.host.dxil), Some(8 * 4 * 128));
        assert_ne!(linked_ps.host.dxil, alone_ps.host.dxil);
        assert_eq!(bindings(linked_ms), bindings(alone_ms));
        assert_eq!(bindings(linked_ps), bindings(alone_ps));
        for stage in &made.stages {
            let dwords = stage.translated().host.user_data_dwords();
            assert_eq!(stage.user_data.map(|u| u.1), (dwords > 0).then_some(dwords));
        }
        let again = HostPipeline::new(&shaders, Some(&g)).expect("the host mesh pipeline again");
        assert!(Arc::ptr_eq(&again.stages[0].shader, &made.stages[0].shader));
        assert!(Arc::ptr_eq(&again.stages[1].shader, &made.stages[1].shader));

        let plain = made.marks.is_none()
            && made.stages.iter().all(|s| {
                s.root_cbvs.is_empty()
                    && s.root_srvs.is_empty()
                    && s.root_uavs.is_empty()
                    && s.memory.is_none()
                    && s.memory_dirty.is_none()
                    && s.view_addresses.is_none()
            });
        assert!(plain, "the fixture binds user data at most");
        let texels = readback_16x16(host, &made);
        assert!(
            texels.iter().all(|t| *t == [1.0, 0.5, 0.25, 1.0]),
            "unexpected texel {:?}",
            texels.iter().find(|t| **t != [1.0, 0.5, 0.25, 1.0])
        );
    }

    #[test]
    #[ignore = "requires an isolated physical GPU session and VICTORY_CACHE_DIR"]
    #[cfg(feature = "diagnostics")]
    fn indirect_mesh_reuses_cached_pipeline() {
        use crate::{draw::State, pipeline, shader_cache, shadow, watch};
        use std::time::{Duration, Instant};
        use victory_shader::native::fixture;
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1,
            IDXGIFactory4,
        };

        let cache = std::path::PathBuf::from(std::env::var_os("VICTORY_CACHE_DIR").unwrap());
        pipeline::use_pipeline_cache(cache.join("pipelines"));
        shader_cache::use_shader_cache(cache.join("shaders")).expect("the fixture shader cache");

        let host = host().expect("a Direct3D 12 device");
        assert!(host.mesh_shaders(), "the adapter must support mesh shaders");
        // SAFETY: this queries the adapter associated with the live device.
        let adapter = unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            factory
                .EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid())
                .unwrap()
                .GetDesc1()
                .unwrap()
        };
        assert_eq!(adapter.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32, 0);
        let adapter_name = String::from_utf16_lossy(&adapter.Description);

        let mesh = Arc::new(Shader::new("mesh", &fixture::wide_mesh(128, 128, 128)));
        let channels = [(0, 0), (0, 1), (15, 2), (15, 3)];
        let pixel = Arc::new(Shader::new("pixel", &fixture::reading_pixel(16, &channels, true)));
        let mut graphics = GraphicsState::default();
        graphics.rasterizer.CullMode = D3D12_CULL_MODE_NONE;
        graphics.depth.DepthEnable = false.into();
        graphics.depth.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        graphics.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        graphics.targets = targets(&[DXGI_FORMAT_R32G32B32A32_FLOAT]);

        let mut title_pipeline = std::ptr::null();
        assert_eq!(
            pipeline::hand_out(
                vec![mesh, pixel],
                Some(Box::new(graphics)),
                &[],
                &pipeline::PIPELINE_IIDS[0],
                &mut title_pipeline,
            ),
            windows_sys::Win32::Foundation::S_OK,
        );
        let start = Instant::now();
        while pipeline::host_pipeline(title_pipeline).is_none() {
            assert!(start.elapsed() < Duration::from_secs(10), "mesh pipeline never became ready");
            std::thread::sleep(Duration::from_millis(10));
        }

        let args = watch::tests::pool(1);
        let groups = [1u32, 1, 1];
        // SAFETY: the write-watched page is ours and holds all three words.
        unsafe { std::ptr::copy_nonoverlapping(groups.as_ptr(), args as *mut u32, groups.len()) };
        shadow::register(args, args as u64, watch::PAGE as u64);
        let pixels = render_16x16(host, |list| {
            let mut state = State::default();
            state.set_pipeline(title_pipeline);
            assert!(state.execute_indirect(list, (10, 12, 1), args as u64, None));
        });
        shadow::unregister(args, args as u64);
        watch::tests::free(args);
        crate::object::release(title_pipeline);
        pipeline::save_pipeline_cache();

        assert_eq!(pixels.len(), 256);
        assert!(pixels.iter().all(|pixel| *pixel == [1.0, 0.5, 0.25, 1.0]));
        let (made, loaded) = pipeline::pipeline_counts();
        assert_eq!(made + loaded, 1);
        eprintln!(
            "indirect mesh readback on {} matched 256 pixels. Pipeline library made={made}, loaded={loaded}",
            adapter_name.trim_end_matches('\0')
        );
    }

    fn readback_16x16(host: &crate::host::Host, made: &HostPipeline) -> Vec<[f32; 4]> {
        render_16x16(host, |list| {
            // SAFETY: an open list with live roots and a live pipeline.
            unsafe {
                list.list.SetGraphicsRootSignature(&made.root);
                list.list.SetPipelineState(&made.pso);
                for stage in &made.stages {
                    if let Some((param, dwords)) = stage.user_data {
                        let zeros = vec![0u32; dwords as usize];
                        list.list.SetGraphicsRoot32BitConstants(param, dwords, zeros.as_ptr().cast(), 0);
                    }
                }
                if made.mesh {
                    list.list.cast::<ID3D12GraphicsCommandList6>().unwrap().DispatchMesh(1, 1, 1);
                } else {
                    list.list.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    list.list.DrawInstanced(3, 1, 0, 0);
                }
            }
        })
    }

    fn render_16x16(
        host: &crate::host::Host,
        draw: impl FnOnce(&crate::host::HostList),
    ) -> Vec<[f32; 4]> {
        let (width, height) = (16u32, 16u32);
        let color = {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(width),
                Height: height,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            // SAFETY: valid heap and resource descriptions.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &rd,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        None,
                        &mut out,
                    )
                    .unwrap();
            }
            out.unwrap()
        };
        let pitch = width * 16; // 256 bytes, the copy alignment
        let readback = {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: u64::from(pitch * height),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            // SAFETY: a valid read-back buffer description.
            unsafe {
                host.device
                    .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut out)
                    .unwrap();
            }
            out.unwrap()
        };
        let list = host.new_list().unwrap();
        let rtv = host.rtv(&color, None);
        let viewport = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: width as f32,
            Height: height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let rect = windows::Win32::Foundation::RECT { left: 0, top: 0, right: width as i32, bottom: height as i32 };
        // SAFETY: an open list, live resources, views, root signature and
        // pipeline; root constants of the sizes the root signature declares.
        unsafe {
            list.list.ClearRenderTargetView(rtv, &[0.0; 4], None);
            list.list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.list.RSSetViewports(&[viewport]);
            list.list.RSSetScissorRects(&[rect]);
        }
        draw(&list);
        // SAFETY: an open list and live render target and readback buffer.
        unsafe {
            list.list.ResourceBarrier(&[crate::host::transition(
                &color,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                            Width: width,
                            Height: height,
                            Depth: 1,
                            RowPitch: pitch,
                        },
                    },
                },
            };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&color),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            list.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        }
        list.close();
        host.execute(&[&list]);
        let mut at = std::ptr::null_mut();
        // SAFETY: `execute` waited for the GPU; the buffer holds
        // `pitch * height` bytes of 16-byte texels.
        unsafe {
            readback.Map(0, None, Some(&mut at)).unwrap();
            let all = std::slice::from_raw_parts(at.cast::<[f32; 4]>(), (pitch * height / 16) as usize).to_vec();
            readback.Unmap(0, None);
            all
        }
    }

    #[test]
    #[cfg(feature = "experimental-barycentrics")]
    fn direct_barycentrics_match_a_projective_triangle() {
        use victory_shader::native::fixture;
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1,
            IDXGIFactory4,
        };
        let (Some(host), Some(dxc)) = (host(), dxc()) else {
            eprintln!("skipped barycentric readback: no D3D12 device or PC dxc");
            return;
        };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
        // SAFETY: the output structure has the type and size requested.
        let supported = unsafe {
            host.device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, (&raw mut options).cast(), size_of_val(&options) as u32)
        };
        if supported.is_err() || !options.BarycentricsSupported.as_bool() {
            eprintln!("skipped barycentric readback: unsupported adapter");
            return;
        }
        // After division by W, the three projected vertices lie exactly on
        // the corners (0,0), (16,0), (0,16) of the viewport.
        let dir = tempfile::tempdir().unwrap();
        let vs = compile(&dxc, dir.path(), r#"
            float4 main(uint id : SV_VertexID) : SV_Position {
                float w = id == 0 ? 1.0 : (id == 1 ? 2.0 : 4.0);
                float2 xy = float2(id == 1 ? 1.0 : -1.0, id == 2 ? -1.0 : 1.0);
                return float4(xy * w, 0.5 * w, w);
            }
        "#, "vs_6_5");
        let vs = super::super::depth_bounds_test::host_shader("vertex", victory_shader::HwStage::Vs, &vs);
        let ps = Arc::new(Shader::new("pixel", &fixture::barycentric_pixel_pair()));
        ps.translated().expect("direct perspective/linear center reads must translate");
        let mut state = GraphicsState::default();
        state.rasterizer.CullMode = D3D12_CULL_MODE_NONE;
        state.depth.DepthEnable = false.into();
        state.depth.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        state.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        state.targets = targets(&[DXGI_FORMAT_R32G32B32A32_FLOAT]);
        let made = HostPipeline::new(&[vs, ps], Some(&state)).expect("a barycentric pixel pipeline");
        let texels = readback_16x16(host, &made);
        let mut inside = 0;
        let mut outside = 0;
        for (index, &actual) in texels.iter().enumerate() {
            let (x, y) = (index % 16, index / 16);
            if x + y == 15 {
                continue; // The center lies on the diagonal shared-edge boundary.
            }
            if x + y > 15 {
                assert_eq!(actual, [0.0; 4], "outside the triangle at {x},{y}");
                outside += 1;
                continue;
            }
            let i = (x as f32 + 0.5) / 16.0;
            let j = (y as f32 + 0.5) / 16.0;
            let inv_w = 1.0 - i - j + i / 2.0 + j / 4.0;
            let expected = [i / 2.0 / inv_w, j / 4.0 / inv_w, i, j];
            for (v, e) in actual.into_iter().zip(expected) {
                assert!((v - e).abs() < 0.0005, "pixel {x},{y}: {actual:?}, expected {expected:?}");
            }
            inside += 1;
        }
        assert_eq!((inside, outside), (120, 120));
        // SAFETY: query the adapter associated with this device's LUID.
        let adapter = unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            factory.EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid()).unwrap().GetDesc1().unwrap()
        };
        let name = String::from_utf16_lossy(&adapter.Description);
        let software = adapter.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0;
        eprintln!(
            "direct barycentric readback on {} (software={software}): {inside} interior and {outside} exterior pixels matched",
            name.trim_end_matches('\0')
        );
    }
}
