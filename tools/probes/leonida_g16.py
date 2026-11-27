"""Run four authored G16 pixel shaders through Leonida's translator and Vulkan test renderer."""
import argparse
from pathlib import Path
import math
import shutil
import struct
import subprocess
import sys
import tempfile


def shader(clamp, a16):
    # Leonida's Vulkan fixture provides a 128x128 quadrant texture and a solid cyan mip 1.
    # A horizontal threshold selects mip 1 either through the gradient or through MinLod.
    lines = [
        "v_interp_p1_f32_e32 v2, v0, attr0.x",
        "v_interp_p2_f32_e32 v2, v1, attr0.x",
        "v_interp_p1_f32_e32 v3, v0, attr0.y",
        "v_interp_p2_f32_e32 v3, v1, attr0.y",
        "s_load_dwordx8 s[8:15], s[0:1], 0",
        "s_waitcnt lgkmcnt(0)",
        "s_mov_b32 s16, 0x92",
        "s_mov_b32 s17, 0xc00000",
        "s_mov_b32 s18, 0x14500000",
        "s_mov_b32 s19, 0",
        "v_cmp_le_f32_e32 vcc_lo, 0.5, v2",
        "v_mov_b32_e32 v8, 0x00002000",  # half pair (1/128, 0)
        "v_mov_b32_e32 v9, 0x20000000",  # half pair (0, 1/128)
    ]
    if not clamp:
        lines.extend(["v_mov_b32_e32 v14, 0x00002400",  # (1/64, 0) on the right
                      "v_cndmask_b32_e32 v8, v8, v14, vcc_lo"])
    if a16:
        lines.append("v_cvt_pkrtz_f16_f32 v10, v2, v3")
        last = 10
    else:
        lines.extend(["v_mov_b32_e32 v10, v2", "v_mov_b32_e32 v11, v3"])
        last = 11
    if clamp:
        last += 1
        # An unused high half of 1.0 catches a half-swapped minimum LOD on the left.
        low, high = ("0x3c000000", "0x3c003c00") if a16 else ("0", "0x3f800000")
        lines.extend([f"v_mov_b32_e32 v14, {low}", f"v_mov_b32_e32 v15, {high}",
                      f"v_cndmask_b32_e32 v{last}, v14, v15, vcc_lo"])
    opcode = "image_sample_d_cl_g16" if clamp else "image_sample_d_g16"
    lines.extend([
        f"{opcode} v[4:7], v[8:{last}], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D"
        + (" a16" if a16 else ""),
        "s_waitcnt vmcnt(0)",
        "exp mrt0 v4, v5, v6, v7 done vm",
        "s_endpgm",
    ])
    return lines


def expected(x, y):
    if x >= 32:
        return (0, 1, 1, 1)
    return (1, 0, 0, 1) if y < 32 else (0, 0, 1, 1)


def check_pixels(raw):
    pixels = raw.read_bytes()
    if len(pixels) != 64 * 64 * 16:
        raise ValueError(f"Expected 4096 RGBA32F pixels, received {len(pixels)} bytes.")
    for i, sample in enumerate(struct.iter_unpack("<4f", pixels)):
        x, y = i % 64, i // 64
        want = expected(x, y)
        if any(not math.isclose(v, ref, rel_tol=0, abs_tol=1e-6) for v, ref in zip(sample, want)):
            raise ValueError(f"Pixel ({x}, {y}) read {sample}, expected {want}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leonida", type=Path, help="Checkout containing the G16 translator")
    parser.add_argument("renderer", type=Path, help="Leonida's built vk_texture_test executable")
    parser.add_argument("--spirv-bin", type=Path, help="Directory with spirv-as and spirv-val")
    args = parser.parse_args()
    repo, renderer = args.leonida.resolve(), args.renderer.resolve()
    sys.path.insert(0, str(repo / "tests" / "semantic"))
    import support

    search = str(args.spirv_bin.resolve()) if args.spirv_bin else None
    assembler, validator = (shutil.which(n, path=search) for n in ("spirv-as", "spirv-val"))
    if not assembler or not validator:
        parser.error("Install SPIRV-Tools or supply its directory with --spirv-bin.")
    with tempfile.TemporaryDirectory(prefix="portvi-leonida-") as temp:
        work = Path(temp)

        def translate(name, lines):
            asm, binary = work / (name + ".spvasm"), work / (name + ".spv")
            asm.write_text(support.translate_lines(name, lines), encoding="utf-8")
            subprocess.run([assembler, "--target-env", "vulkan1.2", asm, "-o", binary], check=True)
            subprocess.run([validator, "--target-env", "vulkan1.2", binary], check=True)
            return binary

        vertex = (repo / "runtime/test/shaders/vs_ids.asm").read_text(encoding="utf-8")
        vs = translate("vs_ids", [line for s in vertex.splitlines() if (line := s.partition(";")[0].strip())])
        for clamp in (False, True):
            for a16 in (False, True):
                case = f"ps_sample_d{'_cl' if clamp else ''}_g16_{'a16' if a16 else 'a32'}"
                fs = translate(case, shader(clamp, a16))
                raw = work / (case + ".raw")
                subprocess.run([renderer, vs, fs, raw, "0", "92", "c00000", "14500000", "0",
                                "--two-mip"], check=True)
                check_pixels(raw)
                print(f"{case}: 4096/4096 pixels matched", flush=True)


if __name__ == "__main__":
    main()
